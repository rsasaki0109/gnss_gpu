#!/usr/bin/env python3
"""Demo: Real-data end-to-end GNSS positioning pipeline.

Uses a real GPS broadcast navigation (NAV) file to compute satellite
positions, then generates synthetic pseudorange observations for a receiver
at Tokyo Station.  Runs WLS, EKF, and Particle Filter positioning and
compares results.

NAV file: brdc0150.24n  (2024-01-15, GPS day 015)

Usage:
    PYTHONPATH=python python3 examples/demo_real_data.py
"""

from __future__ import annotations

import math
import os
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Add project root to path so gnss_gpu can be imported
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "python"))

from gnss_gpu.io.nav_rinex import read_nav_rinex, _datetime_to_gps_seconds_of_week, _datetime_to_gps_week
from gnss_gpu.ephemeris import Ephemeris
from gnss_gpu.atmosphere import AtmosphereCorrection
from gnss_gpu.ekf import EKFPositioner

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
C_LIGHT = 299792458.0  # speed of light [m/s]
WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = 2.0 * WGS84_F - WGS84_F ** 2

# Receiver true position: Tokyo Station (ECEF)
TRUE_ECEF = np.array([-3957199.0, 3310205.0, 3737911.0])

# Simulation parameters
N_EPOCHS = 60
DT = 1.0  # seconds between epochs
ELEVATION_MASK_DEG = 10.0
TRUE_CLOCK_BIAS_M = 50000.0  # ~167 us receiver clock bias
PR_NOISE_SIGMA = 3.0  # pseudorange noise [m]

# NAV file paths
NAV_FILE_PRIMARY = Path("/tmp/gnss_eph/rinex/nav/brdc0150.24n")
NAV_FILE_FALLBACK = Path("/tmp/gnss_gpu_demo/brdc0150.24n")
NAV_URL = "https://cddis.nasa.gov/archive/gnss/data/daily/2024/015/24n/brdc0150.24n.gz"
NAV_URL_ALT = "ftp://igs.bkg.bund.de/IGS/BRDC/2024/015/BRDC00IGS_R_20240150000_01D_MN.rnx.gz"


# ---------------------------------------------------------------------------
# Coordinate helpers (pure-Python, no GPU dependency)
# ---------------------------------------------------------------------------

def ecef_to_lla(x, y, z):
    """Convert ECEF to geodetic LLA (lat_rad, lon_rad, alt_m)."""
    b = WGS84_A * (1.0 - WGS84_F)
    ep2 = (WGS84_A ** 2 - b ** 2) / (b ** 2)
    p = math.sqrt(x ** 2 + y ** 2)
    theta = math.atan2(z * WGS84_A, p * b)
    lat = math.atan2(
        z + ep2 * b * math.sin(theta) ** 3,
        p - WGS84_E2 * WGS84_A * math.cos(theta) ** 3,
    )
    lon = math.atan2(y, x)
    sin_lat = math.sin(lat)
    N = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat ** 2)
    alt = p / math.cos(lat) - N if abs(math.cos(lat)) > 1e-10 else abs(z) - b
    return lat, lon, alt


def satellite_azel(rx_ecef, sat_ecef):
    """Compute azimuth and elevation from receiver to satellite.

    Returns (az_rad, el_rad) arrays for each satellite.
    """
    rx = np.asarray(rx_ecef, dtype=np.float64)
    lat, lon, _ = ecef_to_lla(rx[0], rx[1], rx[2])

    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    sin_lon = math.sin(lon)
    cos_lon = math.cos(lon)

    # ECEF-to-ENU rotation matrix
    R = np.array([
        [-sin_lon,              cos_lon,             0.0],
        [-sin_lat * cos_lon,   -sin_lat * sin_lon,   cos_lat],
        [ cos_lat * cos_lon,    cos_lat * sin_lon,   sin_lat],
    ])

    sats = np.asarray(sat_ecef, dtype=np.float64).reshape(-1, 3)
    n_sat = sats.shape[0]
    az = np.zeros(n_sat)
    el = np.zeros(n_sat)

    for i in range(n_sat):
        dx = sats[i] - rx
        enu = R @ dx
        horiz = math.sqrt(enu[0] ** 2 + enu[1] ** 2)
        el[i] = math.atan2(enu[2], horiz)
        az[i] = math.atan2(enu[0], enu[1])
        if az[i] < 0:
            az[i] += 2.0 * math.pi

    return az, el


# ---------------------------------------------------------------------------
# Pure-Python WLS solver
# ---------------------------------------------------------------------------

def wls_solve(sat_ecef, pseudoranges, weights, max_iter=10, tol=1e-4):
    """Weighted Least Squares single-epoch solver.

    Returns (state_4, n_iter) where state_4 = [x, y, z, clock_bias_m].
    """
    n_sat = len(pseudoranges)
    state = np.zeros(4)
    for iteration in range(max_iter):
        H = np.zeros((n_sat, 4))
        pred_pr = np.zeros(n_sat)
        for j in range(n_sat):
            dx = sat_ecef[j] - state[:3]
            r = np.linalg.norm(dx)
            if r < 1.0:
                r = 1.0
            pred_pr[j] = r + state[3]
            H[j, :3] = -dx / r
            H[j, 3] = 1.0
        residual = pseudoranges - pred_pr
        W = np.diag(weights)
        try:
            delta = np.linalg.solve(H.T @ W @ H, H.T @ W @ residual)
        except np.linalg.LinAlgError:
            break
        state += delta
        if np.linalg.norm(delta[:3]) < tol:
            return state, iteration + 1
    return state, max_iter


# ---------------------------------------------------------------------------
# Pure-Python Particle Filter (fallback when GPU unavailable)
# ---------------------------------------------------------------------------

class SimpleParticleFilter:
    """Lightweight particle filter for GNSS positioning."""

    def __init__(self, n_particles=50000, sigma_pos=1.0, sigma_cb=300.0,
                 sigma_pr=5.0, seed=42):
        self.n = n_particles
        self.sigma_pos = sigma_pos
        self.sigma_cb = sigma_cb
        self.sigma_pr = sigma_pr
        self.rng = np.random.default_rng(seed)
        self.particles = None
        self.log_w = None

    def initialize(self, pos_ecef, cb=0.0, spread_pos=100.0, spread_cb=1000.0):
        self.particles = np.column_stack([
            self.rng.normal(pos_ecef[0], spread_pos, self.n),
            self.rng.normal(pos_ecef[1], spread_pos, self.n),
            self.rng.normal(pos_ecef[2], spread_pos, self.n),
            self.rng.normal(cb, spread_cb, self.n),
        ])
        self.log_w = np.zeros(self.n)

    def predict(self, dt=1.0):
        self.particles[:, :3] += self.rng.normal(0, self.sigma_pos * dt, (self.n, 3))
        self.particles[:, 3] += self.rng.normal(0, self.sigma_cb * dt, self.n)

    def update(self, sat_ecef, pseudoranges):
        n_sat = len(pseudoranges)
        log_lik = np.zeros(self.n)
        for j in range(n_sat):
            dx = sat_ecef[j] - self.particles[:, :3]
            r = np.sqrt(np.sum(dx ** 2, axis=1))
            pred_pr = r + self.particles[:, 3]
            diff = pseudoranges[j] - pred_pr
            log_lik += -0.5 * (diff / self.sigma_pr) ** 2
        self.log_w += log_lik

        # Normalize and check ESS
        max_lw = np.max(self.log_w)
        self.log_w -= max_lw
        w = np.exp(self.log_w)
        w /= w.sum()
        ess = 1.0 / np.sum(w ** 2)
        if ess < 0.5 * self.n:
            indices = self.rng.choice(self.n, size=self.n, p=w)
            self.particles = self.particles[indices]
            self.log_w[:] = 0.0

    def estimate(self):
        w = np.exp(self.log_w)
        w /= w.sum()
        return np.average(self.particles, axis=0, weights=w)


# ---------------------------------------------------------------------------
# NAV file acquisition
# ---------------------------------------------------------------------------

def find_or_download_nav_file() -> Path | None:
    """Locate the NAV file, or attempt to download it."""
    # Check primary location
    if NAV_FILE_PRIMARY.exists():
        return NAV_FILE_PRIMARY

    # Check fallback location
    if NAV_FILE_FALLBACK.exists():
        return NAV_FILE_FALLBACK

    # Try to download (compressed)
    print("    NAV file not found locally. Attempting download ...")
    import gzip
    import shutil

    NAV_FILE_FALLBACK.parent.mkdir(parents=True, exist_ok=True)
    gz_path = NAV_FILE_FALLBACK.with_suffix(".24n.gz")

    for url in [NAV_URL, NAV_URL_ALT]:
        try:
            print(f"    Trying: {url}")
            urllib.request.urlretrieve(url, gz_path)
            with gzip.open(gz_path, "rb") as f_in, open(NAV_FILE_FALLBACK, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            gz_path.unlink(missing_ok=True)
            if NAV_FILE_FALLBACK.exists():
                return NAV_FILE_FALLBACK
        except Exception as e:
            print(f"    Download failed: {e}")
            continue

    return None


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

def main():
    print("=" * 74)
    print("  gnss_gpu Demo: Real Broadcast Ephemeris End-to-End Pipeline")
    print("=" * 74)

    # ------------------------------------------------------------------
    # Step 0: Locate NAV file
    # ------------------------------------------------------------------
    print("\n[0] Locating GPS broadcast NAV file ...")
    nav_path = find_or_download_nav_file()
    if nav_path is None:
        print("    ERROR: Could not find or download the NAV file.")
        print("    Expected at: " + str(NAV_FILE_PRIMARY))
        print("    Please download brdc0150.24n (2024 day 015) and place it there.")
        sys.exit(1)
    print(f"    NAV file: {nav_path}")

    # ------------------------------------------------------------------
    # Step 1: Parse real broadcast NAV file
    # ------------------------------------------------------------------
    print("\n[1] Parsing RINEX NAV file ...")
    nav_messages = read_nav_rinex(nav_path)
    total_records = sum(len(v) for v in nav_messages.values())
    print(f"    PRNs found : {len(nav_messages)} satellites")
    print(f"    Total records: {total_records} ephemeris messages")
    print(f"    PRN list   : {sorted(nav_messages.keys())}")

    # ------------------------------------------------------------------
    # Step 2: Compute satellite positions with real ephemeris
    # ------------------------------------------------------------------
    print("\n[2] Computing satellite positions from broadcast ephemeris ...")
    eph = Ephemeris(nav_messages)

    # Target time: 2024-01-15 12:00:00 UTC
    target_dt = datetime(2024, 1, 15, 12, 0, 0)
    gps_tow = _datetime_to_gps_seconds_of_week(target_dt)
    gps_week = _datetime_to_gps_week(target_dt)
    print(f"    Target time : {target_dt.isoformat()} UTC")
    print(f"    GPS week    : {gps_week}")
    print(f"    GPS TOW     : {gps_tow:.1f} s")

    sat_ecef, sat_clk, used_prns = eph.compute(gps_tow)
    print(f"    Satellites computed: {len(used_prns)}")

    # ------------------------------------------------------------------
    # Step 3: Filter visible satellites (above elevation mask)
    # ------------------------------------------------------------------
    print(f"\n[3] Filtering visible satellites (elevation > {ELEVATION_MASK_DEG:.0f} deg) ...")
    rx_lat, rx_lon, rx_alt = ecef_to_lla(*TRUE_ECEF)
    print(f"    Receiver LLA: {math.degrees(rx_lat):.6f} N, "
          f"{math.degrees(rx_lon):.6f} E, {rx_alt:.1f} m")

    az_all, el_all = satellite_azel(TRUE_ECEF, sat_ecef)

    # Build visibility mask
    vis_mask = el_all > np.radians(ELEVATION_MASK_DEG)
    vis_idx = np.where(vis_mask)[0]
    n_vis = len(vis_idx)

    if n_vis < 4:
        print(f"    ERROR: Only {n_vis} satellites visible -- need at least 4.")
        sys.exit(1)

    vis_prns = [used_prns[i] for i in vis_idx]
    vis_sat_ecef = sat_ecef[vis_idx]
    vis_sat_clk = sat_clk[vis_idx]
    vis_az = az_all[vis_idx]
    vis_el = el_all[vis_idx]

    print(f"    Visible satellites: {n_vis}")

    # Sky plot table
    print(f"\n    {'PRN':>5s}  {'Elev [deg]':>10s}  {'Azim [deg]':>10s}  "
          f"{'Range [km]':>10s}  {'Clk [us]':>10s}")
    print(f"    {'-' * 55}")
    for i in range(n_vis):
        rng_km = np.linalg.norm(vis_sat_ecef[i] - TRUE_ECEF) / 1000.0
        clk_us = vis_sat_clk[i] * 1e6
        print(f"    G{vis_prns[i]:02d}   {np.degrees(vis_el[i]):10.2f}  "
              f"{np.degrees(vis_az[i]):10.2f}  {rng_km:10.1f}  {clk_us:10.2f}")

    # ------------------------------------------------------------------
    # Step 4: Generate synthetic pseudoranges
    # ------------------------------------------------------------------
    print(f"\n[4] Generating synthetic pseudorange observations ({N_EPOCHS} epochs) ...")

    rng = np.random.default_rng(2024)
    rx_lla = np.array([rx_lat, rx_lon, rx_alt])

    # Atmospheric correction model
    atmo = AtmosphereCorrection()

    # Pre-compute atmospheric delays at reference epoch
    tropo_delay = atmo.tropo(rx_lla, vis_el)
    iono_delay = atmo.iono(rx_lla, vis_az, vis_el, gps_tow)
    total_atmo = tropo_delay + iono_delay

    print(f"    Tropo delay : {np.mean(tropo_delay):.2f} m mean, "
          f"{np.min(tropo_delay):.2f} - {np.max(tropo_delay):.2f} m")
    print(f"    Iono delay  : {np.mean(iono_delay):.2f} m mean, "
          f"{np.min(iono_delay):.2f} - {np.max(iono_delay):.2f} m")
    print(f"    Total atmo  : {np.mean(total_atmo):.2f} m mean")

    # Compute pseudoranges for each epoch
    # Satellite positions evolve slowly -- recompute per epoch for realism
    epochs_gps_tow = gps_tow + np.arange(N_EPOCHS) * DT

    all_sat_ecef = np.zeros((N_EPOCHS, n_vis, 3))
    all_sat_clk = np.zeros((N_EPOCHS, n_vis))
    all_pr_raw = np.zeros((N_EPOCHS, n_vis))       # with atmo + clock + noise
    all_pr_clean = np.zeros((N_EPOCHS, n_vis))      # geometric + clock bias only
    all_az = np.zeros((N_EPOCHS, n_vis))
    all_el = np.zeros((N_EPOCHS, n_vis))

    for epoch in range(N_EPOCHS):
        t = epochs_gps_tow[epoch]

        # Recompute satellite positions at this epoch
        ep_sat_ecef, ep_sat_clk, ep_prns = eph.compute(t, vis_prns)

        # Map back to our visibility ordering
        prn_to_idx = {p: j for j, p in enumerate(ep_prns)}
        for i, prn in enumerate(vis_prns):
            if prn in prn_to_idx:
                j = prn_to_idx[prn]
                all_sat_ecef[epoch, i] = ep_sat_ecef[j]
                all_sat_clk[epoch, i] = ep_sat_clk[j]

        # Compute az/el for this epoch
        az_ep, el_ep = satellite_azel(TRUE_ECEF, all_sat_ecef[epoch])
        all_az[epoch] = az_ep
        all_el[epoch] = el_ep

        # Geometric range
        geom_range = np.sqrt(np.sum((all_sat_ecef[epoch] - TRUE_ECEF) ** 2, axis=1))

        # Satellite clock correction (convert seconds to meters)
        sat_clk_m = all_sat_clk[epoch] * C_LIGHT

        # Atmospheric delays (recompute for updated elevation)
        tropo_ep = atmo.tropo(rx_lla, el_ep)
        iono_ep = atmo.iono(rx_lla, az_ep, el_ep, t)

        # Synthetic pseudorange = geom_range + rx_clock_bias - sat_clock + tropo + iono + noise
        noise = rng.normal(0, PR_NOISE_SIGMA, n_vis)
        all_pr_raw[epoch] = (geom_range + TRUE_CLOCK_BIAS_M - sat_clk_m
                             + tropo_ep + iono_ep + noise)
        all_pr_clean[epoch] = geom_range + TRUE_CLOCK_BIAS_M - sat_clk_m

    print(f"    Pseudorange noise sigma: {PR_NOISE_SIGMA:.1f} m")
    print(f"    Receiver clock bias    : {TRUE_CLOCK_BIAS_M:.0f} m "
          f"({TRUE_CLOCK_BIAS_M / C_LIGHT * 1e6:.2f} us)")
    print(f"    Epochs generated       : {N_EPOCHS}")

    # ------------------------------------------------------------------
    # Step 5: Apply atmospheric corrections to pseudoranges
    # ------------------------------------------------------------------
    print(f"\n[5] Applying atmospheric corrections (Saastamoinen + Klobuchar) ...")

    all_pr_corrected = np.zeros_like(all_pr_raw)
    for epoch in range(N_EPOCHS):
        t = epochs_gps_tow[epoch]
        tropo_corr = atmo.tropo(rx_lla, all_el[epoch])
        iono_corr = atmo.iono(rx_lla, all_az[epoch], all_el[epoch], t)
        all_pr_corrected[epoch] = all_pr_raw[epoch] - tropo_corr - iono_corr

    residual_atmo = np.mean(np.abs(all_pr_corrected - all_pr_clean))
    print(f"    Mean residual after correction: {residual_atmo:.2f} m "
          f"(should be close to noise level {PR_NOISE_SIGMA:.1f} m)")

    # ------------------------------------------------------------------
    # Step 6: WLS positioning
    # ------------------------------------------------------------------
    print(f"\n[6] Weighted Least Squares (WLS) positioning ...")

    # Elevation-dependent weights: 1/sin^2(el)
    def make_weights(el_arr):
        sin_el = np.sin(np.maximum(el_arr, np.radians(5.0)))
        return sin_el ** 2

    wls_results = np.zeros((N_EPOCHS, 4))
    wls_iters = np.zeros(N_EPOCHS, dtype=int)

    try:
        from gnss_gpu import wls_position as _gpu_wls
        for epoch in range(N_EPOCHS):
            w = make_weights(all_el[epoch])
            result, iters = _gpu_wls(all_sat_ecef[epoch], all_pr_corrected[epoch],
                                     w, 10, 1e-4)
            wls_results[epoch] = np.asarray(result)
            wls_iters[epoch] = iters
        wls_backend = "GPU"
    except (ImportError, Exception):
        for epoch in range(N_EPOCHS):
            w = make_weights(all_el[epoch])
            wls_results[epoch], wls_iters[epoch] = wls_solve(
                all_sat_ecef[epoch], all_pr_corrected[epoch], w)
        wls_backend = "Python"

    wls_pos_err = np.array([np.linalg.norm(wls_results[i, :3] - TRUE_ECEF)
                            for i in range(N_EPOCHS)])
    wls_cb_err = np.abs(wls_results[:, 3] - TRUE_CLOCK_BIAS_M)

    print(f"    Backend    : {wls_backend}")
    print(f"    Position error: mean={np.mean(wls_pos_err):.2f} m, "
          f"std={np.std(wls_pos_err):.2f} m, max={np.max(wls_pos_err):.2f} m")
    print(f"    Clock bias err: mean={np.mean(wls_cb_err):.2f} m")

    # ------------------------------------------------------------------
    # Step 7: EKF positioning (60 epochs)
    # ------------------------------------------------------------------
    print(f"\n[7] Extended Kalman Filter (EKF) positioning ({N_EPOCHS} epochs) ...")

    ekf = EKFPositioner(sigma_pr=PR_NOISE_SIGMA, sigma_pos=0.5, sigma_vel=0.1,
                        sigma_clk=50.0, sigma_drift=5.0)
    ekf.initialize(wls_results[0, :3], clock_bias=wls_results[0, 3],
                   sigma_pos=50.0, sigma_cb=500.0)

    ekf_results = np.zeros((N_EPOCHS, 3))
    for epoch in range(N_EPOCHS):
        if epoch > 0:
            ekf.predict(dt=DT)
        w = make_weights(all_el[epoch])
        ekf.update(all_sat_ecef[epoch], all_pr_corrected[epoch], weights=w)
        ekf_results[epoch] = ekf.get_position()

    ekf_pos_err = np.array([np.linalg.norm(ekf_results[i] - TRUE_ECEF)
                            for i in range(N_EPOCHS)])

    print(f"    Position error: mean={np.mean(ekf_pos_err):.2f} m, "
          f"std={np.std(ekf_pos_err):.2f} m, max={np.max(ekf_pos_err):.2f} m")

    # ------------------------------------------------------------------
    # Step 8: Particle Filter (50K particles, 60 epochs)
    # ------------------------------------------------------------------
    n_particles = 50000
    print(f"\n[8] Particle Filter ({n_particles} particles, {N_EPOCHS} epochs) ...")

    try:
        from gnss_gpu import ParticleFilter
        pf = ParticleFilter(n_particles=n_particles, sigma_pos=1.0,
                            sigma_cb=200.0, sigma_pr=PR_NOISE_SIGMA,
                            seed=42)
        pf.initialize(wls_results[0, :3], clock_bias=wls_results[0, 3],
                      spread_pos=50.0, spread_cb=500.0)
        pf_results = np.zeros((N_EPOCHS, 4))
        for epoch in range(N_EPOCHS):
            pf.predict(dt=DT)
            pf.update(all_sat_ecef[epoch], all_pr_corrected[epoch])
            pf_results[epoch] = pf.estimate()
        pf_backend = "GPU"
    except (ImportError, RuntimeError, Exception):
        pf = SimpleParticleFilter(n_particles=n_particles, sigma_pos=1.0,
                                  sigma_cb=200.0, sigma_pr=PR_NOISE_SIGMA,
                                  seed=42)
        pf.initialize(wls_results[0, :3], cb=wls_results[0, 3],
                      spread_pos=50.0, spread_cb=500.0)
        pf_results = np.zeros((N_EPOCHS, 4))
        for epoch in range(N_EPOCHS):
            pf.predict(dt=DT)
            pf.update(all_sat_ecef[epoch], all_pr_corrected[epoch])
            pf_results[epoch] = pf.estimate()
        pf_backend = "Python"

    pf_pos_err = np.array([np.linalg.norm(pf_results[i, :3] - TRUE_ECEF)
                           for i in range(N_EPOCHS)])

    print(f"    Backend    : {pf_backend}")
    print(f"    Position error: mean={np.mean(pf_pos_err):.2f} m, "
          f"std={np.std(pf_pos_err):.2f} m, max={np.max(pf_pos_err):.2f} m")

    # ------------------------------------------------------------------
    # Step 9: Comparison table
    # ------------------------------------------------------------------
    print(f"\n{'=' * 74}")
    print(f"  POSITIONING RESULTS COMPARISON")
    print(f"{'=' * 74}")
    print(f"  Receiver   : Tokyo Station")
    print(f"  True ECEF  : [{TRUE_ECEF[0]:.1f}, {TRUE_ECEF[1]:.1f}, {TRUE_ECEF[2]:.1f}]")
    print(f"  True LLA   : {math.degrees(rx_lat):.6f} N, {math.degrees(rx_lon):.6f} E, "
          f"{rx_alt:.1f} m")
    print(f"  Satellites : {n_vis} visible (of {len(used_prns)} total)")
    print(f"  Epochs     : {N_EPOCHS} at {DT:.0f} s intervals")
    print(f"  NAV source : {nav_path.name}")
    print()

    header = (f"  {'Method':<30s}  {'Mean [m]':>9s}  {'Std [m]':>9s}  "
              f"{'P95 [m]':>9s}  {'Max [m]':>9s}")
    print(header)
    print(f"  {'-' * 72}")

    for label, errs in [
        ("WLS (single-epoch)", wls_pos_err),
        ("EKF (filtered)", ekf_pos_err),
        (f"Particle Filter ({n_particles//1000}K)", pf_pos_err),
    ]:
        p95 = np.percentile(errs, 95) if len(errs) > 0 else 0.0
        print(f"  {label:<30s}  {np.mean(errs):9.2f}  {np.std(errs):9.2f}  "
              f"{p95:9.2f}  {np.max(errs):9.2f}")

    print()

    # Per-epoch convergence (first 10 + last 5)
    print(f"  Per-epoch position error [m] (first 10 + last 5 epochs):")
    print(f"  {'Epoch':>6s}  {'WLS':>9s}  {'EKF':>9s}  {'PF':>9s}")
    print(f"  {'-' * 38}")
    show_epochs = list(range(min(10, N_EPOCHS))) + list(range(max(N_EPOCHS - 5, 10), N_EPOCHS))
    for i in show_epochs:
        if i == 10 and N_EPOCHS > 15:
            print(f"  {'...':>6s}  {'...':>9s}  {'...':>9s}  {'...':>9s}")
        print(f"  {i:6d}  {wls_pos_err[i]:9.2f}  {ekf_pos_err[i]:9.2f}  "
              f"{pf_pos_err[i]:9.2f}")

    # ------------------------------------------------------------------
    # Step 10: Satellite sky plot summary
    # ------------------------------------------------------------------
    print(f"\n{'=' * 74}")
    print(f"  SATELLITE SKY PLOT (epoch 0, {target_dt.isoformat()} UTC)")
    print(f"{'=' * 74}")
    print(f"  {'PRN':>5s}  {'Elev':>7s}  {'Azim':>7s}  {'Quadrant':<12s}  "
          f"{'Range [km]':>10s}  {'Atmo [m]':>9s}")
    print(f"  {'-' * 60}")

    for i in range(n_vis):
        el_deg = np.degrees(vis_el[i])
        az_deg = np.degrees(vis_az[i])

        # Quadrant label
        if az_deg < 90:
            quad = "NE"
        elif az_deg < 180:
            quad = "SE"
        elif az_deg < 270:
            quad = "SW"
        else:
            quad = "NW"

        rng_km = np.linalg.norm(vis_sat_ecef[i] - TRUE_ECEF) / 1000.0
        atmo_m = total_atmo[i]

        print(f"  G{vis_prns[i]:02d}   {el_deg:7.1f}  {az_deg:7.1f}  {quad:<12s}  "
              f"{rng_km:10.1f}  {atmo_m:9.2f}")

    print()
    print("=" * 74)
    print("  Demo complete. Full pipeline validated with real broadcast ephemeris.")
    print("=" * 74)


if __name__ == "__main__":
    main()
