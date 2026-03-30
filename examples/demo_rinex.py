#!/usr/bin/env python3
"""Demo: RINEX observation parsing, WLS positioning, and particle filter.

This script demonstrates the full positioning pipeline:
  1. Generate a synthetic RINEX 3.x observation file
  2. Parse it with gnss_gpu.read_rinex_obs()
  3. Run WLS batch positioning on all epochs
  4. Run particle filter on the same data
  5. Compare WLS vs particle filter results
  6. Print summary statistics and save results to CSV

No external data files are required -- a synthetic dataset is created inline.
"""

from __future__ import annotations

import csv
import os
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Coordinate helpers (pure-Python fallbacks)
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


# ---------------------------------------------------------------------------
# Synthetic RINEX generation
# ---------------------------------------------------------------------------

# True receiver: Tokyo Station (~35.6812 N, 139.7671 E, 40 m)
TRUE_LLA = (35.6812, 139.7671, 40.0)
TRUE_ECEF = _lla_to_ecef(np.radians(TRUE_LLA[0]), np.radians(TRUE_LLA[1]), TRUE_LLA[2])
TRUE_CB_M = 3000.0  # clock bias in metres (~10 us)

# Realistic GPS satellite ECEF positions (altitude ~20 200 km)
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

SAT_IDS = ["G01", "G03", "G06", "G09", "G11", "G14", "G17", "G22"]
N_EPOCHS = 60
EPOCH_INTERVAL = 1.0  # seconds
PR_NOISE_SIGMA = 3.0  # metres


def _generate_synthetic_rinex(filepath: Path, rng: np.random.Generator) -> None:
    """Write a minimal RINEX 3.04 observation file with synthetic pseudoranges."""

    ranges = np.sqrt(np.sum((SAT_ECEF - TRUE_ECEF) ** 2, axis=1))
    clean_pr = ranges + TRUE_CB_M

    start = datetime(2024, 6, 15, 0, 0, 0)

    with open(filepath, "w") as f:
        # --- header (each line padded to 60 chars before the label) ---
        def _hdr(content: str, label: str) -> str:
            return f"{content:60s}{label:20s}\n"

        f.write(_hdr("     3.04           OBSERVATION DATA    G", "RINEX VERSION / TYPE"))
        f.write(_hdr("gnss_gpu demo", "MARKER NAME"))
        pos_str = f"{TRUE_ECEF[0]:14.4f}{TRUE_ECEF[1]:14.4f}{TRUE_ECEF[2]:14.4f}"
        f.write(_hdr(pos_str, "APPROX POSITION XYZ"))
        # SYS / # / OBS TYPES: col 0 = system, col 3-5 = n_types, col 7+ = types
        f.write(_hdr("G    1 C1C", "SYS / # / OBS TYPES"))
        f.write(_hdr(f"{EPOCH_INTERVAL:10.3f}", "INTERVAL"))
        f.write(_hdr("", "END OF HEADER"))

        # --- epochs ---
        for ep in range(N_EPOCHS):
            t = start + timedelta(seconds=ep * EPOCH_INTERVAL)
            n_sat = len(SAT_IDS)
            f.write(f"> {t.year:4d} {t.month:2d} {t.day:2d} {t.hour:2d} {t.minute:2d}"
                    f" {t.second:2d}.0000000  0 {n_sat:2d}\n")
            for s_idx, sat_id in enumerate(SAT_IDS):
                noise = rng.normal(0, PR_NOISE_SIGMA)
                pr_val = clean_pr[s_idx] + noise
                f.write(f"{sat_id}{pr_val:14.3f}\n")


# ---------------------------------------------------------------------------
# Pure-Python WLS solver (fallback when CUDA is unavailable)
# ---------------------------------------------------------------------------

def _wls_solve_py(sat_ecef, pseudoranges, weights, max_iter=10, tol=1e-4):
    """Weighted Least Squares GNSS positioning (iterative Gauss-Newton)."""
    n_sat = len(pseudoranges)
    state = np.zeros(4)  # x, y, z, cb

    for iteration in range(max_iter):
        pred_pr = np.zeros(n_sat)
        H = np.zeros((n_sat, 4))
        for j in range(n_sat):
            dx = sat_ecef[j] - state[:3]
            r = np.linalg.norm(dx)
            pred_pr[j] = r + state[3]
            H[j, :3] = -dx / r
            H[j, 3] = 1.0
        residual = pseudoranges - pred_pr
        W = np.diag(weights)
        HTWH = H.T @ W @ H
        delta = np.linalg.solve(HTWH, H.T @ W @ residual)
        state += delta
        if np.linalg.norm(delta[:3]) < tol:
            return state, iteration + 1
    return state, max_iter


def _wls_batch_py(sat_batch, pr_batch, w_batch):
    n_epoch = pr_batch.shape[0]
    results = np.zeros((n_epoch, 4))
    iters = np.zeros(n_epoch, dtype=int)
    for i in range(n_epoch):
        mask = ~np.isnan(pr_batch[i])
        results[i], iters[i] = _wls_solve_py(
            sat_batch[i][mask], pr_batch[i][mask], w_batch[i][mask]
        )
    return results, iters


# ---------------------------------------------------------------------------
# Pure-Python Particle Filter (fallback)
# ---------------------------------------------------------------------------

class _SimpleParticleFilter:
    """Minimal particle filter for demonstration when CUDA is unavailable."""

    def __init__(self, n_particles=10000, sigma_pos=1.0, sigma_cb=300.0,
                 sigma_pr=5.0, seed=42):
        self.n = n_particles
        self.sigma_pos = sigma_pos
        self.sigma_cb = sigma_cb
        self.sigma_pr = sigma_pr
        self.rng = np.random.default_rng(seed)
        self.particles = None  # (n, 4): x, y, z, cb
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
            if np.isnan(pseudoranges[j]):
                continue
            dx = sat_ecef[j] - self.particles[:, :3]
            r = np.sqrt(np.sum(dx ** 2, axis=1))
            pred_pr = r + self.particles[:, 3]
            diff = pseudoranges[j] - pred_pr
            log_lik += -0.5 * (diff / self.sigma_pr) ** 2
        self.log_w += log_lik
        # normalise
        max_lw = np.max(self.log_w)
        self.log_w -= max_lw
        # resample if ESS low
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
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  gnss_gpu Demo: RINEX Parsing + WLS + Particle Filter")
    print("=" * 70)
    rng = np.random.default_rng(42)

    # --- Step 1: generate synthetic RINEX file ---
    tmpdir = tempfile.mkdtemp(prefix="gnss_gpu_demo_")
    rinex_path = Path(tmpdir) / "synthetic.24o"
    print(f"\n[1] Generating synthetic RINEX file: {rinex_path}")
    _generate_synthetic_rinex(rinex_path, rng)
    print(f"    {N_EPOCHS} epochs, {len(SAT_IDS)} satellites, interval={EPOCH_INTERVAL}s")

    # --- Step 2: parse RINEX ---
    print("\n[2] Parsing RINEX observation file ...")
    from gnss_gpu.io import read_rinex_obs
    obs = read_rinex_obs(str(rinex_path))
    print(f"    Header  : version={obs.header.version}, marker={obs.header.marker_name}")
    print(f"    Approx  : {obs.header.approx_position}")
    print(f"    Epochs  : {len(obs.epochs)}")

    times, pr_array, sat_ids = obs.pseudoranges("C1C")
    print(f"    PR shape: {pr_array.shape} (epochs x max_sats)")

    # --- Step 3: prepare satellite positions (constant for demo) ---
    print("\n[3] Using hardcoded satellite ECEF positions (constant across epochs)")
    n_epoch = pr_array.shape[0]
    n_sat = pr_array.shape[1]
    sat_batch = np.tile(SAT_ECEF[:n_sat], (n_epoch, 1, 1))
    w_batch = np.ones_like(pr_array)

    # --- Step 4: WLS positioning ---
    print("\n[4] Running WLS batch positioning ...")
    try:
        from gnss_gpu import wls_batch
        wls_results, wls_iters = wls_batch(sat_batch, pr_array, w_batch)
        wls_source = "GPU"
    except (ImportError, Exception) as e:
        print(f"    CUDA wls_batch unavailable ({e}), using Python fallback")
        wls_results, wls_iters = _wls_batch_py(sat_batch, pr_array, w_batch)
        wls_source = "Python"

    wls_errors = np.array([
        np.linalg.norm(wls_results[i, :3] - TRUE_ECEF) for i in range(n_epoch)
    ])
    print(f"    Source : {wls_source}")
    print(f"    Mean 3D error : {np.mean(wls_errors):8.3f} m")
    print(f"    Std  3D error : {np.std(wls_errors):8.3f} m")
    print(f"    Max  3D error : {np.max(wls_errors):8.3f} m")

    # --- Step 5: Particle filter positioning ---
    print("\n[5] Running Particle Filter positioning ...")
    n_particles = 50000
    try:
        from gnss_gpu import ParticleFilter
        pf = ParticleFilter(n_particles=n_particles, sigma_pos=1.0,
                            sigma_cb=300.0, sigma_pr=5.0, seed=42)
        pf.initialize(wls_results[0, :3], clock_bias=wls_results[0, 3],
                       spread_pos=50.0, spread_cb=500.0)
        pf_source = "GPU"
        pf_results = np.zeros((n_epoch, 4))
        for i in range(n_epoch):
            pf.predict(dt=EPOCH_INTERVAL)
            mask = ~np.isnan(pr_array[i])
            pf.update(sat_batch[i][mask], pr_array[i][mask])
            pf_results[i] = pf.estimate()
    except (ImportError, RuntimeError, Exception) as e:
        print(f"    CUDA ParticleFilter unavailable ({e}), using Python fallback")
        pf = _SimpleParticleFilter(n_particles=n_particles, sigma_pos=1.0,
                                   sigma_cb=300.0, sigma_pr=5.0, seed=42)
        pf.initialize(wls_results[0, :3], cb=wls_results[0, 3],
                       spread_pos=50.0, spread_cb=500.0)
        pf_source = "Python"
        pf_results = np.zeros((n_epoch, 4))
        for i in range(n_epoch):
            pf.predict(dt=EPOCH_INTERVAL)
            pf.update(sat_batch[i], pr_array[i])
            pf_results[i] = pf.estimate()

    pf_errors = np.array([
        np.linalg.norm(pf_results[i, :3] - TRUE_ECEF) for i in range(n_epoch)
    ])
    print(f"    Source : {pf_source}")
    print(f"    Particles    : {n_particles}")
    print(f"    Mean 3D error: {np.mean(pf_errors):8.3f} m")
    print(f"    Std  3D error: {np.std(pf_errors):8.3f} m")
    print(f"    Max  3D error: {np.max(pf_errors):8.3f} m")

    # --- Step 6: comparison ---
    print("\n[6] Comparison: WLS vs Particle Filter")
    print(f"    {'Metric':<25s} {'WLS':>10s} {'PF':>10s}")
    print(f"    {'-' * 47}")
    print(f"    {'Mean 3D error [m]':<25s} {np.mean(wls_errors):10.3f} {np.mean(pf_errors):10.3f}")
    print(f"    {'Std  3D error [m]':<25s} {np.std(wls_errors):10.3f} {np.std(pf_errors):10.3f}")
    print(f"    {'Median 3D error [m]':<25s} {np.median(wls_errors):10.3f} {np.median(pf_errors):10.3f}")
    print(f"    {'95th pct error [m]':<25s} {np.percentile(wls_errors, 95):10.3f} {np.percentile(pf_errors, 95):10.3f}")

    # --- Step 7: convert best WLS epoch to LLA ---
    best_idx = int(np.argmin(wls_errors))
    lat, lon, alt = _ecef_to_lla(*wls_results[best_idx, :3])
    print(f"\n    Best WLS epoch #{best_idx}: "
          f"lat={lat:.6f}, lon={lon:.6f}, alt={alt:.1f} m  "
          f"(error={wls_errors[best_idx]:.3f} m)")

    # --- Step 8: save CSV ---
    csv_path = Path(tmpdir) / "results.csv"
    print(f"\n[7] Saving results to {csv_path}")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "time",
                         "wls_x", "wls_y", "wls_z", "wls_cb", "wls_err_3d",
                         "pf_x", "pf_y", "pf_z", "pf_cb", "pf_err_3d"])
        for i in range(n_epoch):
            writer.writerow([
                i, times[i].isoformat(),
                f"{wls_results[i, 0]:.4f}", f"{wls_results[i, 1]:.4f}",
                f"{wls_results[i, 2]:.4f}", f"{wls_results[i, 3]:.4f}",
                f"{wls_errors[i]:.4f}",
                f"{pf_results[i, 0]:.4f}", f"{pf_results[i, 1]:.4f}",
                f"{pf_results[i, 2]:.4f}", f"{pf_results[i, 3]:.4f}",
                f"{pf_errors[i]:.4f}",
            ])
    print(f"    Written {n_epoch} rows.")

    print("\n" + "=" * 70)
    print("  Demo complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
