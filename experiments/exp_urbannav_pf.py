#!/usr/bin/env python3
"""Experiment: GPU Particle Filter (without 3D model) on PPC / UrbanNav / synthetic data.

Evaluates the GPU particle filter at multiple particle counts and compares it
with WLS/EKF baselines. Supports PPC-Dataset real data via the shared baseline
loader and falls back to synthetic data when no real dataset is provided.

Outputs
-------
  experiments/results/<prefix>_results.csv
  experiments/results/<prefix>_summary.csv
  experiments/results/<prefix>_cdf.png
  experiments/results/<prefix>_timeline.png
  experiments/results/<prefix>_pareto.png

Usage
-----
  PYTHONPATH=python python3 experiments/exp_urbannav_pf.py
  PYTHONPATH=python python3 experiments/exp_urbannav_pf.py \
      --data-dir /tmp/PPC-real/PPC-Dataset/tokyo/run1 --systems G
  PYTHONPATH=python python3 experiments/exp_urbannav_pf.py \
      --data-dir /path/to/UrbanNav/Odaiba --systems G --urban-rover ublox
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
    plot_cdf,
    plot_error_timeline,
    plot_pareto,
    print_comparison_table,
    save_results,
)
from exp_urbannav_baseline import load_or_generate_data, run_ekf, run_wls

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PF_PARTICLE_COUNTS = [10_000, 100_000, 1_000_000]
PF_SIGMA_POS = 2.0    # m/step
PF_SIGMA_CB = 300.0   # m/step
PF_SIGMA_PR = 8.0     # m


def _epoch_dt(data: dict, i: int) -> float:
    """Return a positive step size for epoch *i*."""
    if i <= 0:
        return float(data.get("dt", 1.0))

    times = np.asarray(data.get("times", []), dtype=np.float64)
    if len(times) > i:
        dt = float(times[i] - times[i - 1])
        if dt > 0.0:
            return dt
    return float(data.get("dt", 1.0))


def _longest_segment(mask: np.ndarray, times: np.ndarray) -> tuple[int, float]:
    longest_epochs = 0
    longest_duration = 0.0
    start = None

    for i, flagged in enumerate(mask):
        if flagged and start is None:
            start = i
        elif not flagged and start is not None:
            end = i - 1
            n_epochs = end - start + 1
            duration = float(times[end] - times[start]) if end > start else 0.0
            if n_epochs > longest_epochs:
                longest_epochs = n_epochs
                longest_duration = duration
            start = None

    if start is not None:
        end = len(mask) - 1
        n_epochs = end - start + 1
        duration = float(times[end] - times[start]) if end > start else 0.0
        if n_epochs > longest_epochs:
            longest_epochs = n_epochs
            longest_duration = duration

    return longest_epochs, longest_duration


def _augment_tail_metrics(metrics: dict, times: np.ndarray) -> dict:
    errors = np.asarray(metrics["errors_2d"], dtype=np.float64)
    outlier_mask = errors > 100.0
    catastrophic_mask = errors > 500.0
    longest_epochs, longest_duration = _longest_segment(outlier_mask, times)
    metrics["outlier_rate_pct"] = 100.0 * float(np.mean(outlier_mask))
    metrics["catastrophic_rate_pct"] = 100.0 * float(np.mean(catastrophic_mask))
    metrics["longest_outlier_segment_epochs"] = float(longest_epochs)
    metrics["longest_outlier_segment_s"] = float(longest_duration)
    return metrics


def run_pf(
    data: dict,
    n_particles: int,
    wls_init: np.ndarray,
    resampling: str = "megopolis",
) -> tuple[np.ndarray, float, str]:
    """Run ParticleFilter with the given particle count."""
    n_epochs = data["n_epochs"]
    sat_ecef = data["sat_ecef"]
    pseudoranges = data["pseudoranges"]
    weights = data["weights"]

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
            pf.predict(dt=_epoch_dt(data, i))
            pf.update(sat_ecef[i], pseudoranges[i], weights=weights[i])
            positions[i] = pf.estimate()[:3]
        backend = "GPU"
    except (ImportError, RuntimeError, Exception) as e:
        n_cpu = min(n_particles, 50_000)
        print(
            f"      GPU ParticleFilter unavailable ({type(e).__name__}: {e}). "
            f"CPU fallback with {n_cpu} particles."
        )
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
            pf.predict(dt=_epoch_dt(data, i))
            pf.update(sat_ecef[i], pseudoranges[i], weights=weights[i])
            positions[i] = pf.estimate()[:3]
        backend = f"CPU({n_cpu})"

    elapsed = (time.perf_counter() - t0) * 1000.0
    ms_per_epoch = elapsed / max(n_epochs, 1)
    print(
        f"    PF-{n_particles//1000}K ({backend}): "
        f"{elapsed:.1f} ms total, {ms_per_epoch:.3f} ms/epoch"
    )
    return positions, ms_per_epoch, backend


def main():
    parser = argparse.ArgumentParser(
        description="GPU Particle Filter (no 3D model) evaluation on PPC or synthetic data"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Path to PPC/UrbanNav run directory, dataset root, or none for synthetic",
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=300,
        help="Number of epochs for synthetic fallback (default: 300)",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Optional cap on real-data epochs",
    )
    parser.add_argument(
        "--start-epoch",
        type=int,
        default=0,
        help="Skip this many usable real-data epochs before evaluation",
    )
    parser.add_argument(
        "--systems",
        type=str,
        default="G",
        help="Comma-separated constellations for real data, e.g. G or G,E",
    )
    parser.add_argument(
        "--urban-rover",
        type=str,
        default="ublox",
        help="UrbanNav rover observation source, e.g. ublox or trimble",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating plots",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run only 10K and 100K particles (faster)",
    )
    parser.add_argument(
        "--results-prefix",
        type=str,
        default="pf",
        help="Output filename prefix under experiments/results/",
    )
    args = parser.parse_args()
    systems = tuple(part.strip().upper() for part in args.systems.split(",") if part.strip())

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  Experiment: GPU Particle Filter (No 3D Model) on PPC / Synthetic")
    print("=" * 72)

    # ------------------------------------------------------------------
    # [1] Data
    # ------------------------------------------------------------------
    print("\n[1] Loading data ...")
    data = load_or_generate_data(
        args.data_dir,
        n_epochs=args.n_epochs,
        max_real_epochs=args.max_epochs,
        start_epoch=args.start_epoch,
        systems=systems,
        urban_rover=args.urban_rover,
    )
    ground_truth = data["ground_truth"]
    n_epochs = data["n_epochs"]
    times = np.asarray(data["times"], dtype=np.float64)
    if "dataset_name" in data:
        print(f"    Dataset: {data['dataset_name']}")

    # ------------------------------------------------------------------
    # [2] Baselines
    # ------------------------------------------------------------------
    print("\n[2] Running baselines ...")

    print("    WLS ...")
    wls_pos, wls_ms = run_wls(data)
    wls_metrics = _augment_tail_metrics(compute_metrics(wls_pos[:, :3], ground_truth), times)
    wls_metrics["time_ms"] = wls_ms
    print(
        f"    WLS: RMS 2D={wls_metrics['rms_2d']:.2f} m, "
        f"P95={wls_metrics['p95']:.2f} m"
    )

    print("    EKF ...")
    ekf_pos, ekf_ms = run_ekf(data, wls_pos)
    ekf_metrics = _augment_tail_metrics(compute_metrics(ekf_pos, ground_truth), times)
    ekf_metrics["time_ms"] = ekf_ms
    print(
        f"    EKF: RMS 2D={ekf_metrics['rms_2d']:.2f} m, "
        f"P95={ekf_metrics['p95']:.2f} m"
    )

    # ------------------------------------------------------------------
    # [3] Particle Filter at each particle count
    # ------------------------------------------------------------------
    particle_counts = [10_000, 100_000] if args.quick else PF_PARTICLE_COUNTS

    print(f"\n[3] Running Particle Filter ({particle_counts}) ...")
    pf_results: dict[str, tuple[np.ndarray, float, str]] = {}

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
    position_by_label = {
        "WLS": wls_pos[:, :3],
        "EKF": ekf_pos,
    }

    for label, (pos, ms, backend) in pf_results.items():
        m = _augment_tail_metrics(compute_metrics(pos, ground_truth), times)
        m["time_ms"] = ms
        m["backend"] = backend
        all_metrics[label] = m
        position_by_label[label] = pos
        print(
            f"    {label}: RMS 2D={m['rms_2d']:.2f} m, "
            f"P95={m['p95']:.2f} m, "
            f">100m={m['outlier_rate_pct']:.2f}%"
        )

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
    all_errors_per_epoch = {label: all_metrics[label]["errors_2d"] for label in all_metrics}

    row_data: dict[str, object] = {
        "epoch": epochs,
        "gps_tow": times,
        "satellite_count": np.asarray(data.get("satellite_counts", np.full(n_epochs, data.get("n_satellites", 0))), dtype=np.int32),
    }
    for label, errs in all_errors_per_epoch.items():
        key = label.lower().replace("-", "_")
        row_data[f"error_2d_{key}"] = errs
        pos = np.asarray(position_by_label[label], dtype=np.float64)
        row_data[f"est_x_{key}"] = pos[:, 0]
        row_data[f"est_y_{key}"] = pos[:, 1]
        row_data[f"est_z_{key}"] = pos[:, 2]

    save_results(row_data, RESULTS_DIR / f"{args.results_prefix}_results.csv")

    method_labels = list(all_metrics.keys())
    summary = {
        "method": method_labels,
        "rms_2d": [all_metrics[m]["rms_2d"] for m in method_labels],
        "mean_2d": [all_metrics[m]["mean_2d"] for m in method_labels],
        "p50": [all_metrics[m]["p50"] for m in method_labels],
        "p95": [all_metrics[m]["p95"] for m in method_labels],
        "max_2d": [all_metrics[m]["max_2d"] for m in method_labels],
        "outlier_rate_pct": [all_metrics[m]["outlier_rate_pct"] for m in method_labels],
        "catastrophic_rate_pct": [all_metrics[m]["catastrophic_rate_pct"] for m in method_labels],
        "longest_outlier_segment_epochs": [
            all_metrics[m]["longest_outlier_segment_epochs"] for m in method_labels
        ],
        "longest_outlier_segment_s": [
            all_metrics[m]["longest_outlier_segment_s"] for m in method_labels
        ],
        "time_ms": [all_metrics[m].get("time_ms", 0.0) for m in method_labels],
        "backend": [all_metrics[m].get("backend", "-") for m in method_labels],
        "n_epochs": [n_epochs for _ in method_labels],
    }
    save_results(summary, RESULTS_DIR / f"{args.results_prefix}_summary.csv")

    # ------------------------------------------------------------------
    # [7] Plots
    # ------------------------------------------------------------------
    if not args.no_plots:
        print("\n[7] Generating plots ...")

        plot_cdf(
            all_errors_per_epoch,
            RESULTS_DIR / f"{args.results_prefix}_cdf.png",
            title="CDF of 2D Error - Particle Filter vs Baselines",
        )

        plot_error_timeline(
            times,
            all_errors_per_epoch,
            RESULTS_DIR / f"{args.results_prefix}_timeline.png",
            title="2D Error Over Time - Particle Filter vs Baselines",
        )

        pareto_time = {m: all_metrics[m].get("time_ms", 0.0) for m in all_metrics}
        pareto_acc = {m: all_metrics[m]["mean_2d"] for m in all_metrics}
        plot_pareto(
            pareto_time,
            pareto_acc,
            RESULTS_DIR / f"{args.results_prefix}_pareto.png",
            title="Accuracy vs Computation Time - PF Particle Counts",
        )

    print(f"\n{'=' * 72}")
    print(f"  Results saved to: {RESULTS_DIR}/")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
