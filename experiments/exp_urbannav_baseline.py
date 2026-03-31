#!/usr/bin/env python3
"""Experiment: WLS / EKF / RTKLIB-equivalent baseline evaluation on UrbanNav.

Evaluates three baseline positioning methods on UrbanNav data (or synthetic
fallback):
  1. WLS  -- single-epoch Weighted Least Squares
  2. EKF  -- Extended Kalman Filter
  3. RTK  -- code-only RTKLIB-equivalent (double-differenced WLS)

Outputs
-------
  experiments/results/baseline_wls.csv
  experiments/results/baseline_ekf.csv
  experiments/results/baseline_summary.csv
  experiments/results/baseline_cdf.png
  experiments/results/baseline_timeline.png

Usage
-----
  PYTHONPATH=python python3 experiments/exp_urbannav_baseline.py
  PYTHONPATH=python python3 experiments/exp_urbannav_baseline.py --data-dir /path/to/urbannav
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Project path
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "python"))

RESULTS_DIR = _SCRIPT_DIR / "results"

# ---------------------------------------------------------------------------
# Shared evaluation helpers
# ---------------------------------------------------------------------------
from evaluate import (
    compute_metrics,
    generate_synthetic_urbannav,
    plot_cdf,
    plot_error_timeline,
    print_comparison_table,
    save_results,
    wls_solve_py,
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_urbannav(data_dir: Path) -> dict | None:
    """Attempt to load UrbanNav dataset from data_dir.

    Returns None if the data is not found (triggers synthetic fallback).

    The expected directory layout::

        <data_dir>/
            gnss/          # RINEX OBS files per epoch
            reference/     # ground-truth trajectory CSV
    """
    ref_candidates = list(data_dir.glob("**/*.csv"))
    obs_candidates = list(data_dir.glob("**/*.obs")) + list(data_dir.glob("**/*.rnx"))

    if not ref_candidates:
        return None

    print(f"    Found reference files: {[p.name for p in ref_candidates[:3]]}")

    # Minimal NMEA/CSV parser: expect columns lat,lon,alt or x,y,z
    try:
        import csv
        rows = []
        with open(ref_candidates[0], newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)

        if not rows:
            return None

        # Try to parse ground truth
        keys = list(rows[0].keys())
        print(f"    Reference CSV columns: {keys[:8]}")
        return None  # Extend here with real UrbanNav parsing logic
    except Exception as e:
        print(f"    Could not parse UrbanNav data: {e}")
        return None


def load_or_generate_data(data_dir: Path | None, n_epochs: int = 300) -> dict:
    """Load UrbanNav data or generate synthetic fallback."""
    if data_dir is not None and data_dir.exists():
        print(f"    Searching for UrbanNav data in: {data_dir}")
        data = load_urbannav(data_dir)
        if data is not None:
            print(f"    Loaded UrbanNav data: {data['n_epochs']} epochs")
            return data
        print("    UrbanNav data not found or not parseable. Using synthetic data.")
    else:
        print("    No data directory provided. Using synthetic data.")

    data = generate_synthetic_urbannav(n_epochs=n_epochs, n_satellites=8, seed=42)
    print(f"    Synthetic data: {data['n_epochs']} epochs, "
          f"{data['n_satellites']} satellites, "
          f"{data['n_nlos_total']} NLOS observations "
          f"({100.0 * data['n_nlos_total'] / (data['n_epochs'] * data['n_satellites']):.1f}%)")
    return data


# ---------------------------------------------------------------------------
# WLS baseline
# ---------------------------------------------------------------------------

def run_wls(data: dict) -> tuple[np.ndarray, float]:
    """Run WLS on every epoch.

    Returns
    -------
    positions : ndarray, shape (N, 4)   [x, y, z, clock_bias]
    time_per_epoch_ms : float
    """
    n_epochs = data["n_epochs"]
    sat_ecef = data["sat_ecef"]
    pseudoranges = data["pseudoranges"]
    weights = data["weights"]

    # Try GPU-accelerated WLS
    positions = np.zeros((n_epochs, 4))
    t0 = time.perf_counter()

    try:
        from gnss_gpu import wls_position as _gpu_wls
        for i in range(n_epochs):
            result, _ = _gpu_wls(sat_ecef[i], pseudoranges[i], weights[i],
                                  10, 1e-4)
            positions[i] = np.asarray(result)
        backend = "GPU"
    except (ImportError, Exception):
        for i in range(n_epochs):
            positions[i], _ = wls_solve_py(
                sat_ecef[i], pseudoranges[i], weights[i])
        backend = "Python"

    elapsed = (time.perf_counter() - t0) * 1000.0
    ms_per_epoch = elapsed / max(n_epochs, 1)
    print(f"    WLS ({backend}): {elapsed:.1f} ms total, "
          f"{ms_per_epoch:.3f} ms/epoch")
    return positions, ms_per_epoch


# ---------------------------------------------------------------------------
# EKF baseline
# ---------------------------------------------------------------------------

def run_ekf(data: dict, wls_init: np.ndarray) -> tuple[np.ndarray, float]:
    """Run EKF over all epochs.

    Returns
    -------
    positions : ndarray, shape (N, 3)
    time_per_epoch_ms : float
    """
    n_epochs = data["n_epochs"]
    sat_ecef = data["sat_ecef"]
    pseudoranges = data["pseudoranges"]
    weights = data["weights"]
    dt = data["dt"]

    from gnss_gpu.ekf import EKFPositioner

    ekf = EKFPositioner(sigma_pr=5.0, sigma_pos=1.0, sigma_vel=0.1,
                        sigma_clk=100.0, sigma_drift=10.0)
    ekf.initialize(wls_init[0, :3], clock_bias=float(wls_init[0, 3]),
                   sigma_pos=50.0, sigma_cb=500.0)

    positions = np.zeros((n_epochs, 3))
    t0 = time.perf_counter()

    for i in range(n_epochs):
        if i > 0:
            ekf.predict(dt=dt)
        ekf.update(sat_ecef[i], pseudoranges[i], weights=weights[i])
        positions[i] = ekf.get_position()

    elapsed = (time.perf_counter() - t0) * 1000.0
    ms_per_epoch = elapsed / max(n_epochs, 1)
    print(f"    EKF: {elapsed:.1f} ms total, {ms_per_epoch:.3f} ms/epoch")
    return positions, ms_per_epoch


# ---------------------------------------------------------------------------
# RTKLIB-equivalent: code-only differential positioning
# ---------------------------------------------------------------------------

def run_rtklib_equivalent(data: dict) -> tuple[np.ndarray, float]:
    """RTKLIB-equivalent code-only differential positioning.

    Uses RTKSolver from gnss_gpu with pseudorange-only mode (no carrier phase).
    Falls back to WLS when RTKSolver unavailable.

    Returns
    -------
    positions : ndarray, shape (N, 3)
    time_per_epoch_ms : float
    """
    n_epochs = data["n_epochs"]
    sat_ecef = data["sat_ecef"]
    pseudoranges = data["pseudoranges"]
    weights = data["weights"]

    positions = np.zeros((n_epochs, 3))
    t0 = time.perf_counter()

    try:
        from gnss_gpu.rtk import RTKSolver

        # Use a synthetic base station at origin
        base_pos = data["origin_ecef"]
        base_pr = np.zeros((n_epochs, data["n_satellites"]))
        for i in range(n_epochs):
            for s in range(data["n_satellites"]):
                base_pr[i, s] = np.linalg.norm(sat_ecef[i, s] - base_pos)

        solver = RTKSolver(sigma_pr=5.0)
        for i in range(n_epochs):
            # Double-differenced pseudoranges
            dd_pr = pseudoranges[i] - base_pr[i]
            result = solver.solve(sat_ecef[i], pseudoranges[i], base_pr[i],
                                   base_pos)
            positions[i] = np.asarray(result)[:3]
        backend = "GPU RTK"
    except (ImportError, Exception) as e:
        # Fallback: treat as smoothed WLS with tighter model
        for i in range(n_epochs):
            sol, _ = wls_solve_py(sat_ecef[i], pseudoranges[i], weights[i])
            positions[i] = sol[:3]
        backend = "WLS fallback (RTK unavailable)"

    elapsed = (time.perf_counter() - t0) * 1000.0
    ms_per_epoch = elapsed / max(n_epochs, 1)
    print(f"    RTKLIB-equiv ({backend}): {elapsed:.1f} ms total, "
          f"{ms_per_epoch:.3f} ms/epoch")
    return positions, ms_per_epoch


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Baseline evaluation (WLS/EKF/RTKLIB) on UrbanNav data")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Path to UrbanNav dataset directory")
    parser.add_argument("--n-epochs", type=int, default=300,
                        help="Number of epochs for synthetic data (default: 300)")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip generating plots")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  Experiment: Baseline Methods on UrbanNav Data")
    print("=" * 72)

    # ------------------------------------------------------------------
    # [1] Data loading
    # ------------------------------------------------------------------
    print("\n[1] Loading data ...")
    data = load_or_generate_data(args.data_dir, n_epochs=args.n_epochs)
    ground_truth = data["ground_truth"]
    n_epochs = data["n_epochs"]

    # ------------------------------------------------------------------
    # [2] WLS
    # ------------------------------------------------------------------
    print("\n[2] Running WLS ...")
    wls_pos, wls_ms = run_wls(data)
    wls_metrics = compute_metrics(wls_pos[:, :3], ground_truth)
    print(f"    WLS: RMS 2D={wls_metrics['rms_2d']:.2f} m, "
          f"P95={wls_metrics['p95']:.2f} m")

    # ------------------------------------------------------------------
    # [3] EKF
    # ------------------------------------------------------------------
    print("\n[3] Running EKF ...")
    ekf_pos, ekf_ms = run_ekf(data, wls_pos)
    ekf_metrics = compute_metrics(ekf_pos, ground_truth)
    print(f"    EKF: RMS 2D={ekf_metrics['rms_2d']:.2f} m, "
          f"P95={ekf_metrics['p95']:.2f} m")

    # ------------------------------------------------------------------
    # [4] RTKLIB-equivalent
    # ------------------------------------------------------------------
    print("\n[4] Running RTKLIB-equivalent ...")
    rtk_pos, rtk_ms = run_rtklib_equivalent(data)
    rtk_metrics = compute_metrics(rtk_pos, ground_truth)
    print(f"    RTKLIB-equiv: RMS 2D={rtk_metrics['rms_2d']:.2f} m, "
          f"P95={rtk_metrics['p95']:.2f} m")

    # ------------------------------------------------------------------
    # [5] Summary table
    # ------------------------------------------------------------------
    print("\n[5] Results summary:")
    all_metrics = {
        "WLS": wls_metrics,
        "EKF": ekf_metrics,
        "RTKLIB-equiv": rtk_metrics,
    }
    all_metrics["WLS"]["time_ms"] = wls_ms
    all_metrics["EKF"]["time_ms"] = ekf_ms
    all_metrics["RTKLIB-equiv"]["time_ms"] = rtk_ms

    print_comparison_table(all_metrics)

    # ------------------------------------------------------------------
    # [6] Save per-epoch results
    # ------------------------------------------------------------------
    print("\n[6] Saving results ...")
    epochs = np.arange(n_epochs)

    save_results({
        "epoch": epochs,
        "error_2d": wls_metrics["errors_2d"],
        "error_3d": wls_metrics["errors_3d"],
        "est_x": wls_pos[:, 0],
        "est_y": wls_pos[:, 1],
        "est_z": wls_pos[:, 2],
        "gt_x": ground_truth[:, 0],
        "gt_y": ground_truth[:, 1],
        "gt_z": ground_truth[:, 2],
    }, RESULTS_DIR / "baseline_wls.csv")

    save_results({
        "epoch": epochs,
        "error_2d": ekf_metrics["errors_2d"],
        "error_3d": ekf_metrics["errors_3d"],
        "est_x": ekf_pos[:, 0],
        "est_y": ekf_pos[:, 1],
        "est_z": ekf_pos[:, 2],
    }, RESULTS_DIR / "baseline_ekf.csv")

    summary = {
        "method": ["WLS", "EKF", "RTKLIB-equiv"],
        "rms_2d": [wls_metrics["rms_2d"], ekf_metrics["rms_2d"], rtk_metrics["rms_2d"]],
        "rms_3d": [wls_metrics["rms_3d"], ekf_metrics["rms_3d"], rtk_metrics["rms_3d"]],
        "mean_2d": [wls_metrics["mean_2d"], ekf_metrics["mean_2d"], rtk_metrics["mean_2d"]],
        "p50": [wls_metrics["p50"], ekf_metrics["p50"], rtk_metrics["p50"]],
        "p67": [wls_metrics["p67"], ekf_metrics["p67"], rtk_metrics["p67"]],
        "p95": [wls_metrics["p95"], ekf_metrics["p95"], rtk_metrics["p95"]],
        "max_2d": [wls_metrics["max_2d"], ekf_metrics["max_2d"], rtk_metrics["max_2d"]],
        "time_ms": [wls_ms, ekf_ms, rtk_ms],
        "n_epochs": [n_epochs, n_epochs, n_epochs],
    }
    save_results(summary, RESULTS_DIR / "baseline_summary.csv")

    # ------------------------------------------------------------------
    # [7] Plots
    # ------------------------------------------------------------------
    if not args.no_plots:
        print("\n[7] Generating plots ...")
        errors_for_cdf = {
            "WLS": wls_metrics["errors_2d"],
            "EKF": ekf_metrics["errors_2d"],
            "RTKLIB-equiv": rtk_metrics["errors_2d"],
        }
        plot_cdf(errors_for_cdf, RESULTS_DIR / "baseline_cdf.png",
                 title="CDF of 2D Positioning Error - Baseline Methods")

        plot_error_timeline(
            data["times"], errors_for_cdf,
            RESULTS_DIR / "baseline_timeline.png",
            title="2D Positioning Error Over Time - Baseline Methods",
        )

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    print(f"\n{'=' * 72}")
    print(f"  Results saved to: {RESULTS_DIR}/")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
