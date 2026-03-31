#!/usr/bin/env python3
"""Demo: Full gnss_gpu pipeline -- urban multipath scenario.

This script demonstrates the complete positioning pipeline:
  1. Generate a synthetic urban scenario (buildings + satellites + trajectory)
  2. Create a BuildingModel with several buildings
  3. Evaluate GNSS quality with VulnerabilityMap
  4. Corrupt pseudoranges with MultipathSimulator
  5. Run WLS positioning (affected by multipath)
  6. Run ParticleFilter positioning (better multipath handling)
  7. Compare accuracy and print summary

No external data files are required.
"""

from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Coordinate helpers (pure-Python)
# ---------------------------------------------------------------------------

WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = 2.0 * WGS84_F - WGS84_F ** 2


def _lla_to_ecef(lat_rad, lon_rad, alt):
    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    sin_lon = np.sin(lon_rad)
    cos_lon = np.cos(lon_rad)
    N = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat ** 2)
    x = (N + alt) * cos_lat * cos_lon
    y = (N + alt) * cos_lat * sin_lon
    z = (N * (1.0 - WGS84_E2) + alt) * sin_lat
    return np.array([x, y, z])


def _ecef_to_lla(x, y, z):
    b = WGS84_A * (1.0 - WGS84_F)
    p = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(z * WGS84_A, p * b)
    lat = np.arctan2(
        z + WGS84_E2 / (1.0 - WGS84_E2) * b * np.sin(theta) ** 3,
        p - WGS84_E2 * WGS84_A * np.cos(theta) ** 3,
    )
    lon = np.arctan2(y, x)
    sin_lat = np.sin(lat)
    N = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat ** 2)
    alt = p / np.cos(lat) - N
    return np.degrees(lat), np.degrees(lon), alt


def _enu_to_ecef_rotation(lat_rad, lon_rad):
    """ENU-to-ECEF rotation matrix."""
    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    sin_lon = np.sin(lon_rad)
    cos_lon = np.cos(lon_rad)
    return np.array([
        [-sin_lon, -sin_lat * cos_lon, cos_lat * cos_lon],
        [ cos_lon, -sin_lat * sin_lon, cos_lat * sin_lon],
        [ 0.0,      cos_lat,           sin_lat],
    ])


# ---------------------------------------------------------------------------
# Scenario definition
# ---------------------------------------------------------------------------

# Origin: Shinjuku Station area (35.6896 N, 139.7006 E, 40 m)
ORIGIN_LLA_DEG = (35.6896, 139.7006, 40.0)
ORIGIN_LLA_RAD = (np.radians(ORIGIN_LLA_DEG[0]), np.radians(ORIGIN_LLA_DEG[1]), ORIGIN_LLA_DEG[2])
ORIGIN_ECEF = _lla_to_ecef(*ORIGIN_LLA_RAD)

# ENU-to-ECEF rotation and origin
R_ENU2ECEF = _enu_to_ecef_rotation(ORIGIN_LLA_RAD[0], ORIGIN_LLA_RAD[1])

# GPS satellite positions in ECEF (8 satellites)
SAT_ECEF = np.array([
    [-14985000.0, -3988000.0,  21474000.0],
    [ -9575000.0, 15498000.0,  19457000.0],
    [  7624000.0, -16218000.0, 19843000.0],
    [ 16305000.0, 12037000.0,  17183000.0],
    [-20889000.0, 13759000.0,   8291000.0],
    [  5463000.0, 24413000.0,   8934000.0],
    [ 22169000.0,  3975000.0,  13781000.0],
    [-11527000.0, -19421000.0, 13682000.0],
], dtype=np.float64)

N_SAT = len(SAT_ECEF)
N_EPOCHS = 30
DT = 1.0
TRUE_CB_M = 3000.0

# Buildings in ENU (east, north, width_e, depth_n, height)
BUILDINGS_ENU = [
    (  30.0,   20.0,  25.0,  40.0, 100.0),  # tall building to the east
    ( -40.0,  -10.0,  30.0,  30.0,  60.0),   # medium building to the west
    (  10.0,  -50.0,  20.0,  20.0,  80.0),   # building to the south
    ( -20.0,   60.0,  35.0,  25.0, 120.0),   # very tall building to the north
]


# ---------------------------------------------------------------------------
# Pure-Python WLS solver
# ---------------------------------------------------------------------------

def _wls_solve_py(sat_ecef, pseudoranges, weights, max_iter=10, tol=1e-4):
    n_sat = len(pseudoranges)
    state = np.zeros(4)
    for iteration in range(max_iter):
        H = np.zeros((n_sat, 4))
        pred_pr = np.zeros(n_sat)
        for j in range(n_sat):
            dx = sat_ecef[j] - state[:3]
            r = np.linalg.norm(dx)
            pred_pr[j] = r + state[3]
            H[j, :3] = -dx / r
            H[j, 3] = 1.0
        residual = pseudoranges - pred_pr
        W = np.diag(weights)
        delta = np.linalg.solve(H.T @ W @ H, H.T @ W @ residual)
        state += delta
        if np.linalg.norm(delta[:3]) < tol:
            return state, iteration + 1
    return state, max_iter


# ---------------------------------------------------------------------------
# Pure-Python Particle Filter
# ---------------------------------------------------------------------------

class _SimpleParticleFilter:
    def __init__(self, n_particles=10000, sigma_pos=1.0, sigma_cb=300.0,
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

    def predict(self, velocity_ecef=None, dt=1.0):
        if velocity_ecef is not None:
            self.particles[:, :3] += velocity_ecef * dt
        self.particles[:, :3] += self.rng.normal(0, self.sigma_pos * dt, (self.n, 3))
        self.particles[:, 3] += self.rng.normal(0, self.sigma_cb * dt, self.n)

    def update(self, sat_ecef, pseudoranges, sat_weights=None):
        n_sat = len(pseudoranges)
        log_lik = np.zeros(self.n)
        for j in range(n_sat):
            dx = sat_ecef[j] - self.particles[:, :3]
            r = np.sqrt(np.sum(dx ** 2, axis=1))
            pred_pr = r + self.particles[:, 3]
            diff = pseudoranges[j] - pred_pr
            w = 1.0 if sat_weights is None else sat_weights[j]
            log_lik += -0.5 * w * (diff / self.sigma_pr) ** 2
        self.log_w += log_lik
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
# Pure-Python multipath corruption
# ---------------------------------------------------------------------------

def _corrupt_pseudoranges_py(clean_pr, rx_positions, sat_ecef_batch,
                              buildings_enu, rng, max_error=15.0):
    """Add realistic-looking multipath errors based on satellite elevation.

    Low-elevation satellites near buildings get larger errors (positive bias).
    """
    n_epoch, n_sat = clean_pr.shape
    errors = np.zeros_like(clean_pr)

    for i in range(n_epoch):
        rx = rx_positions[i]
        for j in range(n_sat):
            sat = sat_ecef_batch[i, j]
            dx = sat - rx
            r = np.linalg.norm(dx)
            # approximate elevation angle
            up = rx / np.linalg.norm(rx)  # radial direction ~ up
            sin_el = np.dot(dx / r, up)
            el_deg = np.degrees(np.arcsin(np.clip(sin_el, -1, 1)))

            # Multipath error model: larger at low elevation
            if el_deg < 15:
                mp_err = rng.uniform(3.0, max_error)
            elif el_deg < 30:
                mp_err = rng.uniform(1.0, 8.0)
            elif el_deg < 50:
                mp_err = rng.uniform(0.0, 3.0)
            else:
                mp_err = rng.uniform(0.0, 0.5)

            errors[i, j] = mp_err

    corrupted = clean_pr + errors
    return corrupted, errors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  gnss_gpu Demo: Full Urban Positioning Pipeline")
    print("=" * 70)
    rng = np.random.default_rng(42)

    # ------------------------------------------------------------------
    # Step 1: Define the urban scenario
    # ------------------------------------------------------------------
    print(f"\n[1] Urban scenario")
    print(f"    Origin  : {ORIGIN_LLA_DEG[0]:.4f} N, {ORIGIN_LLA_DEG[1]:.4f} E, "
          f"{ORIGIN_LLA_DEG[2]:.0f} m (Shinjuku area)")
    print(f"    Buildings: {len(BUILDINGS_ENU)}")
    for i, (ce, cn, w, d, h) in enumerate(BUILDINGS_ENU):
        print(f"      [{i}] center=({ce:.0f}, {cn:.0f}) m, "
              f"size=({w:.0f} x {d:.0f} x {h:.0f}) m")
    print(f"    Satellites: {N_SAT}")
    print(f"    Epochs    : {N_EPOCHS} at {DT:.0f} s intervals")

    # ------------------------------------------------------------------
    # Step 2: Create building model
    # ------------------------------------------------------------------
    print(f"\n[2] Creating building model ...")
    try:
        from gnss_gpu import BuildingModel
        buildings = [
            BuildingModel.create_box(
                center=[ce, cn, h / 2.0], width=w, depth=d, height=h
            )
            for ce, cn, w, d, h in BUILDINGS_ENU
        ]
        # Merge triangles
        all_tris = np.concatenate([b.triangles for b in buildings], axis=0)
        building_model = BuildingModel(all_tris)
        bm_source = "gnss_gpu"
        print(f"    BuildingModel created: {all_tris.shape[0]} triangles")
    except (ImportError, Exception) as e:
        print(f"    BuildingModel: {e}")
        building_model = None
        bm_source = "unavailable"
        print(f"    Using elevation-based multipath model instead")

    # ------------------------------------------------------------------
    # Step 3: Evaluate GNSS vulnerability map
    # ------------------------------------------------------------------
    print(f"\n[3] GNSS vulnerability map ...")
    try:
        from gnss_gpu import VulnerabilityMap
        vmap = VulnerabilityMap(origin_lla=ORIGIN_LLA_DEG, grid_size_m=200,
                                resolution_m=20, height_m=1.5)
        quality = vmap.evaluate(SAT_ECEF, elevation_mask_deg=10.0)
        vm_source = "GPU"
        hdop = quality["hdop"]
        n_vis = quality["n_visible"]
        print(f"    Grid     : {vmap.n_side} x {vmap.n_side}")
        print(f"    HDOP     : min={np.min(hdop):.2f}, max={np.max(hdop):.2f}, "
              f"mean={np.mean(hdop):.2f}")
        print(f"    Visible  : min={np.min(n_vis)}, max={np.max(n_vis)}")
    except (ImportError, RuntimeError, Exception) as e:
        print(f"    VulnerabilityMap unavailable ({e}), skipping")
        vm_source = "unavailable"

    # ------------------------------------------------------------------
    # Step 4: Generate receiver trajectory (pedestrian walk in ENU)
    # ------------------------------------------------------------------
    print(f"\n[4] Generating receiver trajectory ...")
    # Walk east at ~1.2 m/s then turn north
    traj_enu = np.zeros((N_EPOCHS, 3))
    speed = 1.2  # m/s
    for i in range(N_EPOCHS):
        if i < N_EPOCHS // 2:
            traj_enu[i, 0] = speed * i * DT  # east
            traj_enu[i, 1] = 0.0
        else:
            traj_enu[i, 0] = speed * (N_EPOCHS // 2) * DT
            traj_enu[i, 1] = speed * (i - N_EPOCHS // 2) * DT  # north
        traj_enu[i, 2] = 1.5  # height above ground

    # Convert ENU trajectory to ECEF
    traj_ecef = np.zeros((N_EPOCHS, 3))
    for i in range(N_EPOCHS):
        traj_ecef[i] = ORIGIN_ECEF + R_ENU2ECEF @ traj_enu[i]

    dist_total = np.sum(np.sqrt(np.sum(np.diff(traj_enu, axis=0) ** 2, axis=1)))
    print(f"    Total distance: {dist_total:.1f} m")
    print(f"    Start (ENU)   : ({traj_enu[0, 0]:.1f}, {traj_enu[0, 1]:.1f}) m")
    print(f"    End   (ENU)   : ({traj_enu[-1, 0]:.1f}, {traj_enu[-1, 1]:.1f}) m")

    # ------------------------------------------------------------------
    # Step 5: Generate pseudoranges (clean + multipath-corrupted)
    # ------------------------------------------------------------------
    print(f"\n[5] Generating pseudoranges ...")
    sat_batch = np.tile(SAT_ECEF, (N_EPOCHS, 1, 1))
    clean_pr = np.zeros((N_EPOCHS, N_SAT))
    for i in range(N_EPOCHS):
        ranges = np.sqrt(np.sum((SAT_ECEF - traj_ecef[i]) ** 2, axis=1))
        clean_pr[i] = ranges + TRUE_CB_M

    # Add measurement noise
    noise_sigma = 3.0
    noisy_pr = clean_pr + rng.normal(0, noise_sigma, clean_pr.shape)

    # Apply multipath corruption
    try:
        from gnss_gpu import MultipathSimulator
        mp_sim = MultipathSimulator.from_building_boxes(
            BUILDINGS_ENU, origin_lla=ORIGIN_LLA_RAD
        )
        corrupted_pr, mp_errors = mp_sim.corrupt_pseudoranges(
            noisy_pr, traj_ecef, sat_batch
        )
        mp_source = "GPU"
    except (ImportError, RuntimeError, Exception) as e:
        print(f"    GPU MultipathSimulator unavailable ({e}), using elevation model")
        corrupted_pr, mp_errors = _corrupt_pseudoranges_py(
            noisy_pr, traj_ecef, sat_batch, BUILDINGS_ENU, rng
        )
        mp_source = "Python"

    print(f"    Multipath source  : {mp_source}")
    print(f"    Mean MP error     : {np.mean(mp_errors):.2f} m")
    print(f"    Max  MP error     : {np.max(mp_errors):.2f} m")

    # ------------------------------------------------------------------
    # Step 6: WLS positioning -- clean vs corrupted
    # ------------------------------------------------------------------
    print(f"\n[6] WLS positioning ...")
    weights = np.ones((N_EPOCHS, N_SAT))

    # Try GPU WLS
    try:
        from gnss_gpu import wls_batch
        wls_clean, _ = wls_batch(sat_batch, noisy_pr, weights)
        wls_mp, _ = wls_batch(sat_batch, corrupted_pr, weights)
        wls_source = "GPU"
    except (ImportError, Exception) as e:
        print(f"    GPU wls_batch unavailable ({e}), using Python fallback")
        wls_clean = np.zeros((N_EPOCHS, 4))
        wls_mp = np.zeros((N_EPOCHS, 4))
        for i in range(N_EPOCHS):
            wls_clean[i], _ = _wls_solve_py(sat_batch[i], noisy_pr[i], weights[i])
            wls_mp[i], _ = _wls_solve_py(sat_batch[i], corrupted_pr[i], weights[i])
        wls_source = "Python"

    wls_clean_err = np.array([np.linalg.norm(wls_clean[i, :3] - traj_ecef[i])
                              for i in range(N_EPOCHS)])
    wls_mp_err = np.array([np.linalg.norm(wls_mp[i, :3] - traj_ecef[i])
                           for i in range(N_EPOCHS)])

    print(f"    WLS source: {wls_source}")
    print(f"    WLS (clean):     mean={np.mean(wls_clean_err):.2f} m, "
          f"max={np.max(wls_clean_err):.2f} m")
    print(f"    WLS (multipath): mean={np.mean(wls_mp_err):.2f} m, "
          f"max={np.max(wls_mp_err):.2f} m")

    # ------------------------------------------------------------------
    # Step 7: Particle Filter on corrupted pseudoranges
    # ------------------------------------------------------------------
    print(f"\n[7] Particle Filter positioning (on multipath-corrupted data) ...")
    n_particles = 50000

    try:
        from gnss_gpu import ParticleFilter
        pf = ParticleFilter(n_particles=n_particles, sigma_pos=2.0,
                            sigma_cb=300.0, sigma_pr=10.0, seed=42)
        pf.initialize(wls_mp[0, :3], clock_bias=wls_mp[0, 3],
                       spread_pos=50.0, spread_cb=500.0)
        pf_results = np.zeros((N_EPOCHS, 4))
        for i in range(N_EPOCHS):
            pf.predict(dt=DT)
            pf.update(sat_batch[i], corrupted_pr[i])
            pf_results[i] = pf.estimate()
        pf_source = "GPU"
    except (ImportError, RuntimeError, Exception) as e:
        print(f"    GPU ParticleFilter unavailable ({e}), using Python fallback")
        pf = _SimpleParticleFilter(n_particles=n_particles, sigma_pos=2.0,
                                   sigma_cb=300.0, sigma_pr=10.0, seed=42)
        pf.initialize(wls_mp[0, :3], cb=wls_mp[0, 3],
                       spread_pos=50.0, spread_cb=500.0)
        pf_results = np.zeros((N_EPOCHS, 4))
        for i in range(N_EPOCHS):
            vel_ecef = None
            if i > 0:
                vel_ecef = (traj_ecef[i] - traj_ecef[i - 1]) / DT
            pf.predict(velocity_ecef=vel_ecef, dt=DT)
            pf.update(sat_batch[i], corrupted_pr[i])
            pf_results[i] = pf.estimate()
        pf_source = "Python"

    pf_err = np.array([np.linalg.norm(pf_results[i, :3] - traj_ecef[i])
                       for i in range(N_EPOCHS)])

    print(f"    PF source   : {pf_source}")
    print(f"    PF particles: {n_particles}")
    print(f"    PF error    : mean={np.mean(pf_err):.2f} m, "
          f"max={np.max(pf_err):.2f} m")

    # ------------------------------------------------------------------
    # Step 8: Summary comparison
    # ------------------------------------------------------------------
    print(f"\n[8] Accuracy comparison")
    print(f"    {'Method':<30s} {'Mean [m]':>10s} {'Std [m]':>10s} "
          f"{'95th [m]':>10s} {'Max [m]':>10s}")
    print(f"    {'-' * 72}")

    for label, errs in [
        ("WLS (clean PR)", wls_clean_err),
        ("WLS (multipath PR)", wls_mp_err),
        ("Particle Filter (multipath)", pf_err),
    ]:
        print(f"    {label:<30s} {np.mean(errs):10.2f} {np.std(errs):10.2f} "
              f"{np.percentile(errs, 95):10.2f} {np.max(errs):10.2f}")

    improvement = np.mean(wls_mp_err) - np.mean(pf_err)
    print(f"\n    PF improvement over WLS (multipath): {improvement:+.2f} m mean error")

    # ------------------------------------------------------------------
    # Step 9: Save results
    # ------------------------------------------------------------------
    tmpdir = tempfile.mkdtemp(prefix="gnss_gpu_pipeline_")
    csv_path = Path(tmpdir) / "pipeline_results.csv"
    print(f"\n[9] Saving results to {csv_path}")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch",
                         "true_x", "true_y", "true_z",
                         "wls_clean_err", "wls_mp_err", "pf_err",
                         "mean_mp_error"])
        for i in range(N_EPOCHS):
            writer.writerow([
                i,
                f"{traj_ecef[i, 0]:.4f}", f"{traj_ecef[i, 1]:.4f}", f"{traj_ecef[i, 2]:.4f}",
                f"{wls_clean_err[i]:.4f}", f"{wls_mp_err[i]:.4f}", f"{pf_err[i]:.4f}",
                f"{np.mean(mp_errors[i]):.4f}",
            ])
    print(f"    Written {N_EPOCHS} rows.")

    print("\n" + "=" * 70)
    print("  Demo complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
