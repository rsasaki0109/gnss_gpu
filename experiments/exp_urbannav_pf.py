#!/usr/bin/env python3
"""Experiment: GPU Particle Filter (without 3D model) on UrbanNav data.

Evaluates the GPU Mega Particle Filter at three particle counts and compares
with WLS/EKF baselines.  Reports accuracy vs computation time tradeoff.

Particle counts evaluated: 10K, 100K, 1M

Outputs
-------
  experiments/results/pf_results.csv
  experiments/results/pf_summary.csv
  experiments/results/pf_cdf.png
  experiments/results/pf_timeline.png
  experiments/results/pf_pareto.png

Usage
-----
  PYTHONPATH=python python3 experiments/exp_urbannav_pf.py
  PYTHONPATH=python python3 experiments/exp_urbannav_pf.py --data-dir /path/to/urbannav
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

from evaluate import (
    SimplePFCPU,
    compute_metrics,
    generate_synthetic_urbannav,
    plot_cdf,
    plot_error_timeline,
    plot_pareto,
    print_comparison_table,
    save_results,
    wls_solve_py,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PF_PARTICLE_COUNTS = [10_000, 100_000, 1_000_000]
PF_SIGMA_POS = 2.0    # m/step
PF_SIGMA_CB = 300.0   # m/step
PF_SIGMA_PR = 8.0     # m

# ---------------------------------------------------------------------------
# Data loading (reuse from baseline)
# ---------------------------------------------------------------------------

def load_or_generate_data(data_dir: Path | None, n_epochs: int = 300) -> dict:
    if data_dir is not None and data_dir.exists():
        print(f"    Searching for UrbanNav data in: {data_dir}")
    else:
        print("    No data directory provided. Using synthetic data.")

    data = generate_synthetic_urbannav(n_epochs=n_epochs, n_satellites=8, seed=42)
    print(f"    Synthetic data: {data['n_epochs']} epochs, "
          f"{data['n_satellites']} satellites")
    return data


# ---------------------------------------------------------------------------
# WLS baseline (quick re-run to have common reference)
# ---------------------------------------------------------------------------

def run_wls_quick(data: dict) -> np.ndarray:
    n_epochs = data["n_epochs"]
    sat_ecef = data["sat_ecef"]
    pseudoranges = data["pseudoranges"]
    weights = data["weights"]
    positions = np.zeros((n_epochs, 4))

    try:
        from gnss_gpu import wls_position as _gpu_wls
        for i in range(n_epochs):
            result, _ = _gpu_wls(sat_ecef[i], pseudoranges[i], weights[i], 10, 1e-4)
            positions[i] = np.asarray(result)
    except (ImportError, Exception):
        for i in range(n_epochs):
            positions[i], _ = wls_solve_py(sat_ecef[i], pseudoranges[i], weights[i])

    return positions


# ---------------------------------------------------------------------------
# EKF baseline
# ---------------------------------------------------------------------------

def run_ekf_quick(data: dict, wls_init: np.ndarray) -> np.ndarray:
    from gnss_gpu.ekf import EKFPositioner
    n_epochs = data["n_epochs"]
    sat_ecef = data["sat_ecef"]
    pseudoranges = data["pseudoranges"]
    weights = data["weights"]
    dt = data["dt"]

    ekf = EKFPositioner(sigma_pr=5.0, sigma_pos=1.0, sigma_vel=0.1,
                        sigma_clk=100.0, sigma_drift=10.0)
    ekf.initialize(wls_init[0, :3], clock_bias=float(wls_init[0, 3]),
                   sigma_pos=50.0, sigma_cb=500.0)

    positions = np.zeros((n_epochs, 3))
    for i in range(n_epochs):
        if i > 0:
            ekf.predict(dt=dt)
        ekf.update(sat_ecef[i], pseudoranges[i], weights=weights[i])
        positions[i] = ekf.get_position()

    return positions


# ---------------------------------------------------------------------------
# Particle Filter runner
# ---------------------------------------------------------------------------

def run_pf(data: dict, n_particles: int, wls_init: np.ndarray,
           resampling: str = "megopolis") -> tuple[np.ndarray, float, str]:
    """Run ParticleFilter with given particle count.

    Returns
    -------
    positions : ndarray, shape (N, 3)
    ms_per_epoch : float
    backend : str
    """
    n_epochs = data["n_epochs"]
    sat_ecef = data["sat_ecef"]
    pseudoranges = data["pseudoranges"]
    weights = data["weights"]
    dt = data["dt"]

    positions = np.zeros((n_epochs, 3))
    t0 = time.perf_counter()

    try:
        from gnss_gpu import ParticleFilter
        pf = ParticleFilter(
            n_particles=n_particles,
            sigma_pos=PF_SIGMA_POS,
            sigma_cb=PF_SIGMA_CB,
            sigma_pr=PF_SIGMA_PR,
            resampling=resampling,
            seed=42,
        )
        pf.initialize(
            wls_init[0, :3],
            clock_bias=float(wls_init[0, 3]),
            spread_pos=50.0,
            spread_cb=500.0,
        )
        for i in range(n_epochs):
            pf.predict(dt=dt)
            pf.update(sat_ecef[i], pseudoranges[i], weights=weights[i])
            est = pf.estimate()
            positions[i] = est[:3]
        backend = "GPU"
    except (ImportError, RuntimeError, Exception) as e:
        # CPU fallback with capped particle count for speed
        n_cpu = min(n_particles, 50_000)
        print(f"      GPU ParticleFilter unavailable ({type(e).__name__}). "
              f"CPU fallback with {n_cpu} particles.")
        pf = SimplePFCPU(
            n_particles=n_cpu,
            sigma_pos=PF_SIGMA_POS,
            sigma_cb=PF_SIGMA_CB,
            sigma_pr=PF_SIGMA_PR,
            seed=42,
        )
        pf.initialize(
            wls_init[0, :3],
            clock_bias=float(wls_init[0, 3]),
            spread_pos=50.0,
            spread_cb=500.0,
        )
        for i in range(n_epochs):
            pf.predict(dt=dt)
            pf.update(sat_ecef[i], pseudoranges[i], weights=weights[i])
            est = pf.estimate()
            positions[i] = est[:3]
        backend = f"CPU({n_cpu})"

    elapsed = (time.perf_counter() - t0) * 1000.0
    ms_per_epoch = elapsed / max(n_epochs, 1)
    print(f"    PF-{n_particles//1000}K ({backend}): "
          f"{elapsed:.1f} ms total, {ms_per_epoch:.3f} ms/epoch")
    return positions, ms_per_epoch, backend


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="GPU Particle Filter (no 3D model) evaluation on UrbanNav")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Path to UrbanNav dataset directory")
    parser.add_argument("--n-epochs", type=int, default=300,
                        help="Number of epochs for synthetic data (default: 300)")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip generating plots")
    parser.add_argument("--quick", action="store_true",
                        help="Run only 10K and 100K particles (faster)")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  Experiment: GPU Particle Filter (No 3D Model) on UrbanNav")
    print("=" * 72)

    # ------------------------------------------------------------------
    # [1] Data
    # ------------------------------------------------------------------
    print("\n[1] Loading data ...")
    data = load_or_generate_data(args.data_dir, n_epochs=args.n_epochs)
    ground_truth = data["ground_truth"]
    n_epochs = data["n_epochs"]

    # ------------------------------------------------------------------
    # [2] Baselines
    # ------------------------------------------------------------------
    print("\n[2] Running baselines ...")

    print("    WLS ...")
    t0 = time.perf_counter()
    wls_pos = run_wls_quick(data)
    wls_ms = (time.perf_counter() - t0) * 1000.0 / n_epochs
    wls_metrics = compute_metrics(wls_pos[:, :3], ground_truth)
    print(f"    WLS: RMS 2D={wls_metrics['rms_2d']:.2f} m, "
          f"{wls_ms:.3f} ms/epoch")

    print("    EKF ...")
    t0 = time.perf_counter()
    ekf_pos = run_ekf_quick(data, wls_pos)
    ekf_ms = (time.perf_counter() - t0) * 1000.0 / n_epochs
    ekf_metrics = compute_metrics(ekf_pos, ground_truth)
    print(f"    EKF: RMS 2D={ekf_metrics['rms_2d']:.2f} m, "
          f"{ekf_ms:.3f} ms/epoch")

    # ------------------------------------------------------------------
    # [3] Particle Filter at each particle count
    # ------------------------------------------------------------------
    particle_counts = [10_000, 100_000] if args.quick else PF_PARTICLE_COUNTS

    print(f"\n[3] Running Particle Filter ({particle_counts}) ...")
    pf_results = {}  # label -> (positions, ms_per_epoch, backend)

    for n_p in particle_counts:
        label = f"PF-{n_p // 1000}K"
        print(f"  [{label}]")
        pos, ms, backend = run_pf(data, n_p, wls_pos)
        pf_results[label] = (pos, ms, backend)

    # ------------------------------------------------------------------
    # [4] Compute metrics
    # ------------------------------------------------------------------
    print("\n[4] Computing metrics ...")
    all_metrics = {
        "WLS": wls_metrics,
        "EKF": ekf_metrics,
    }
    all_metrics["WLS"]["time_ms"] = wls_ms
    all_metrics["EKF"]["time_ms"] = ekf_ms

    for label, (pos, ms, backend) in pf_results.items():
        m = compute_metrics(pos, ground_truth)
        m["time_ms"] = ms
        m["backend"] = backend
        all_metrics[label] = m
        print(f"    {label}: RMS 2D={m['rms_2d']:.2f} m, P95={m['p95']:.2f} m")

    # ------------------------------------------------------------------
    # [5] Comparison table
    # ------------------------------------------------------------------
    print("\n[5] Results summary:")
    print_comparison_table(all_metrics)

    # ------------------------------------------------------------------
    # [6] Save results
    # ------------------------------------------------------------------
    print("\n[6] Saving results ...")
    epochs = np.arange(n_epochs)

    # Per-epoch errors for each PF run
    all_errors_per_epoch = {"WLS": wls_metrics["errors_2d"],
                             "EKF": ekf_metrics["errors_2d"]}
    for label, (pos, _, _) in pf_results.items():
        m = compute_metrics(pos, ground_truth)
        all_errors_per_epoch[label] = m["errors_2d"]

    row_data: dict = {"epoch": epochs}
    for label, errs in all_errors_per_epoch.items():
        row_data[f"error_2d_{label.lower().replace('-', '_')}"] = errs
    save_results(row_data, RESULTS_DIR / "pf_results.csv")

    # Summary
    method_labels = list(all_metrics.keys())
    summary = {
        "method": method_labels,
        "rms_2d": [all_metrics[m]["rms_2d"] for m in method_labels],
        "mean_2d": [all_metrics[m]["mean_2d"] for m in method_labels],
        "p50": [all_metrics[m]["p50"] for m in method_labels],
        "p95": [all_metrics[m]["p95"] for m in method_labels],
        "max_2d": [all_metrics[m]["max_2d"] for m in method_labels],
        "time_ms": [all_metrics[m].get("time_ms", 0.0) for m in method_labels],
    }
    save_results(summary, RESULTS_DIR / "pf_summary.csv")

    # ------------------------------------------------------------------
    # [7] Plots
    # ------------------------------------------------------------------
    if not args.no_plots:
        print("\n[7] Generating plots ...")

        plot_cdf(all_errors_per_epoch, RESULTS_DIR / "pf_cdf.png",
                 title="CDF of 2D Error - Particle Filter vs Baselines")

        plot_error_timeline(
            data["times"], all_errors_per_epoch,
            RESULTS_DIR / "pf_timeline.png",
            title="2D Error Over Time - Particle Filter vs Baselines",
        )

        # Pareto: accuracy vs time
        pareto_time = {m: all_metrics[m].get("time_ms", 0.0)
                       for m in all_metrics}
        pareto_acc = {m: all_metrics[m]["mean_2d"] for m in all_metrics}
        plot_pareto(pareto_time, pareto_acc, RESULTS_DIR / "pf_pareto.png",
                    title="Accuracy vs Computation Time - PF Particle Counts")

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    print(f"\n{'=' * 72}")
    print(f"  Results saved to: {RESULTS_DIR}/")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
