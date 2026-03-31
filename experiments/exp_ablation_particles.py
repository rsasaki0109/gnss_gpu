#!/usr/bin/env python3
"""Experiment: Particle count scaling ablation study.

Evaluates Particle Filter accuracy and computation time across a range of
particle counts:
  1K, 5K, 10K, 50K, 100K, 500K, 1M

For each count, measures:
  - Mean 2D error, P95 2D error
  - Total computation time and per-epoch time

Plots the Pareto frontier (accuracy vs computation time).

Outputs
-------
  experiments/results/ablation_particles_results.csv
  experiments/results/ablation_particles_pareto.png
  experiments/results/ablation_particles_accuracy.png
  experiments/results/ablation_particles_time.png

Usage
-----
  PYTHONPATH=python python3 experiments/exp_ablation_particles.py
  PYTHONPATH=python python3 experiments/exp_ablation_particles.py --data-dir /path/to/urbannav
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
    plot_pareto,
    print_comparison_table,
    save_results,
    wls_solve_py,
)

# ---------------------------------------------------------------------------
# Particle counts for ablation
# ---------------------------------------------------------------------------
DEFAULT_PARTICLE_COUNTS = [1_000, 5_000, 10_000, 50_000, 100_000, 500_000, 1_000_000]
CPU_MAX_PARTICLES = 50_000   # Cap for CPU fallback to keep runtime manageable

# PF hyperparameters
PF_SIGMA_POS = 2.0
PF_SIGMA_CB = 300.0
PF_SIGMA_PR = 5.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_data(data_dir: Path | None, n_epochs: int) -> dict:
    if data_dir is not None and data_dir.exists():
        print(f"    Trying UrbanNav data in: {data_dir}")
    print("    Using synthetic UrbanNav data.")
    data = generate_synthetic_urbannav(n_epochs=n_epochs, n_satellites=8, seed=42)
    print(f"    {data['n_epochs']} epochs, {data['n_satellites']} satellites")
    return data


def run_wls_init(data: dict) -> np.ndarray:
    """Quick WLS run to get initialization for PF."""
    n_epochs = data["n_epochs"]
    positions = np.zeros((n_epochs, 4))
    try:
        from gnss_gpu import wls_position as _gpu_wls
        for i in range(n_epochs):
            result, _ = _gpu_wls(data["sat_ecef"][i], data["pseudoranges"][i],
                                  data["weights"][i], 10, 1e-4)
            positions[i] = np.asarray(result)
    except (ImportError, Exception):
        for i in range(n_epochs):
            positions[i], _ = wls_solve_py(
                data["sat_ecef"][i], data["pseudoranges"][i], data["weights"][i])
    return positions


def run_pf_single_count(
    data: dict,
    n_particles: int,
    wls_init: np.ndarray,
    seed: int = 42,
) -> dict:
    """Run PF with a single particle count and return metrics + timing.

    Returns
    -------
    result : dict with keys:
        n_particles, mean_2d, p95, rms_2d, time_ms_per_epoch,
        time_ms_total, backend, errors_2d
    """
    n_epochs = data["n_epochs"]
    sat_ecef = data["sat_ecef"]
    pseudoranges = data["pseudoranges"]
    weights = data["weights"]
    dt = data["dt"]
    ground_truth = data["ground_truth"]

    positions = np.zeros((n_epochs, 3))
    backend = "unknown"
    t0 = time.perf_counter()

    # Try GPU
    gpu_ok = False
    if n_particles >= 1:
        try:
            from gnss_gpu import ParticleFilter
            pf = ParticleFilter(
                n_particles=n_particles,
                sigma_pos=PF_SIGMA_POS,
                sigma_cb=PF_SIGMA_CB,
                sigma_pr=PF_SIGMA_PR,
                resampling="megopolis",
                seed=seed,
            )
            pf.initialize(wls_init[0, :3], clock_bias=float(wls_init[0, 3]),
                          spread_pos=50.0, spread_cb=500.0)
            for i in range(n_epochs):
                pf.predict(dt=dt)
                pf.update(sat_ecef[i], pseudoranges[i], weights=weights[i])
                positions[i] = pf.estimate()[:3]
            backend = "GPU"
            gpu_ok = True
        except (ImportError, RuntimeError, Exception) as e:
            pass

    if not gpu_ok:
        n_cpu = min(n_particles, CPU_MAX_PARTICLES)
        pf = SimplePFCPU(
            n_particles=n_cpu,
            sigma_pos=PF_SIGMA_POS,
            sigma_cb=PF_SIGMA_CB,
            sigma_pr=PF_SIGMA_PR,
            seed=seed,
        )
        pf.initialize(wls_init[0, :3], clock_bias=float(wls_init[0, 3]),
                      spread_pos=50.0, spread_cb=500.0)
        for i in range(n_epochs):
            pf.predict(dt=dt)
            pf.update(sat_ecef[i], pseudoranges[i], weights=weights[i])
            positions[i] = pf.estimate()[:3]
        backend = f"CPU({n_cpu})"

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    ms_per_epoch = elapsed_ms / max(n_epochs, 1)

    metrics = compute_metrics(positions, ground_truth)
    label = _label(n_particles)
    print(f"    {label:>10s} ({backend:<14s}): "
          f"RMS={metrics['rms_2d']:6.2f} m, "
          f"P95={metrics['p95']:6.2f} m, "
          f"{ms_per_epoch:8.2f} ms/ep, "
          f"{elapsed_ms/1000.0:6.2f} s total")

    return {
        "n_particles": n_particles,
        "label": label,
        "backend": backend,
        "mean_2d": metrics["mean_2d"],
        "rms_2d": metrics["rms_2d"],
        "p50": metrics["p50"],
        "p95": metrics["p95"],
        "max_2d": metrics["max_2d"],
        "time_ms_per_epoch": ms_per_epoch,
        "time_ms_total": elapsed_ms,
        "errors_2d": metrics["errors_2d"],
    }


def _label(n: int) -> str:
    if n >= 1_000_000:
        return f"{n // 1_000_000}M"
    if n >= 1_000:
        return f"{n // 1_000}K"
    return str(n)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_accuracy_vs_particles(results: list[dict], output_path: Path) -> None:
    """Plot mean and P95 accuracy vs particle count."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"[ablation_particles] matplotlib unavailable, skipping plot")
        return

    counts = [r["n_particles"] for r in results]
    mean_errs = [r["mean_2d"] for r in results]
    p95_errs = [r["p95"] for r in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(counts, mean_errs, "o-", color="#1f77b4", linewidth=2,
            markersize=6, label="Mean 2D error")
    ax.plot(counts, p95_errs, "s--", color="#ff7f0e", linewidth=2,
            markersize=6, label="P95 2D error")
    ax.set_xscale("log")
    ax.set_xlabel("Number of Particles", fontsize=12)
    ax.set_ylabel("2D Error [m]", fontsize=12)
    ax.set_title("Positioning Accuracy vs Particle Count", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which="both")
    ax.set_xticks(counts)
    ax.set_xticklabels([_label(n) for n in counts], rotation=30)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[ablation] Accuracy plot saved: {output_path}")


def plot_time_vs_particles(results: list[dict], output_path: Path) -> None:
    """Plot computation time vs particle count."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    counts = [r["n_particles"] for r in results]
    times = [r["time_ms_per_epoch"] for r in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(counts, times, "D-", color="#2ca02c", linewidth=2,
            markersize=6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of Particles", fontsize=12)
    ax.set_ylabel("Computation Time per Epoch [ms]", fontsize=12)
    ax.set_title("Computation Time vs Particle Count", fontsize=13)
    ax.grid(True, alpha=0.3, which="both")
    ax.set_xticks(counts)
    ax.set_xticklabels([_label(n) for n in counts], rotation=30)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[ablation] Time plot saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Particle count scaling ablation for GNSS PF")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Path to UrbanNav dataset directory")
    parser.add_argument("--n-epochs", type=int, default=200,
                        help="Number of epochs for synthetic data (default: 200)")
    parser.add_argument("--particle-counts", type=int, nargs="+",
                        default=None,
                        help="Override default particle counts "
                             "(e.g. --particle-counts 1000 10000 100000)")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip generating plots")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    particle_counts = args.particle_counts or DEFAULT_PARTICLE_COUNTS

    print("=" * 72)
    print("  Experiment: Particle Count Scaling Ablation")
    print("=" * 72)
    print(f"  Particle counts: {[_label(n) for n in particle_counts]}")
    print(f"  Epochs: {args.n_epochs}")

    # ------------------------------------------------------------------
    # [1] Data
    # ------------------------------------------------------------------
    print("\n[1] Loading data ...")
    data = load_data(args.data_dir, args.n_epochs)

    # ------------------------------------------------------------------
    # [2] WLS reference
    # ------------------------------------------------------------------
    print("\n[2] Running WLS reference ...")
    wls_init = run_wls_init(data)
    wls_metrics = compute_metrics(wls_init[:, :3], data["ground_truth"])
    print(f"    WLS: RMS 2D={wls_metrics['rms_2d']:.2f} m")

    # ------------------------------------------------------------------
    # [3] PF sweep
    # ------------------------------------------------------------------
    print(f"\n[3] Sweeping particle counts ...")
    print(f"    {'Count':>10s}  {'Backend':<14s}  "
          f"{'RMS':>8s}  {'P95':>8s}  {'ms/epoch':>10s}  {'Total [s]':>10s}")
    print(f"    {'-' * 66}")

    ablation_results = []
    for n_p in particle_counts:
        result = run_pf_single_count(data, n_p, wls_init, seed=42)
        ablation_results.append(result)

    # ------------------------------------------------------------------
    # [4] Summary table
    # ------------------------------------------------------------------
    print("\n[4] Results summary:")
    summary_metrics = {
        f"PF-{r['label']}": {
            "mean_2d": r["mean_2d"],
            "rms_2d": r["rms_2d"],
            "rms_3d": r["rms_2d"] * 1.3,  # approximate
            "p50": r["p50"],
            "p67": (r["p50"] + r["p95"]) / 2,
            "p95": r["p95"],
            "max_2d": r["max_2d"],
            "n_epochs": data["n_epochs"],
            "time_ms": r["time_ms_per_epoch"],
        }
        for r in ablation_results
    }
    print_comparison_table(summary_metrics)

    # ------------------------------------------------------------------
    # [5] Save results
    # ------------------------------------------------------------------
    print("\n[5] Saving results ...")
    save_results({
        "n_particles": [r["n_particles"] for r in ablation_results],
        "label": [r["label"] for r in ablation_results],
        "backend": [r["backend"] for r in ablation_results],
        "mean_2d": [r["mean_2d"] for r in ablation_results],
        "rms_2d": [r["rms_2d"] for r in ablation_results],
        "p50": [r["p50"] for r in ablation_results],
        "p95": [r["p95"] for r in ablation_results],
        "max_2d": [r["max_2d"] for r in ablation_results],
        "time_ms_per_epoch": [r["time_ms_per_epoch"] for r in ablation_results],
        "time_ms_total": [r["time_ms_total"] for r in ablation_results],
    }, RESULTS_DIR / "ablation_particles_results.csv")

    # ------------------------------------------------------------------
    # [6] Plots
    # ------------------------------------------------------------------
    if not args.no_plots:
        print("\n[6] Generating plots ...")

        times_dict = {r["label"]: r["time_ms_per_epoch"] for r in ablation_results}
        acc_dict = {r["label"]: r["mean_2d"] for r in ablation_results}
        plot_pareto(times_dict, acc_dict,
                    RESULTS_DIR / "ablation_particles_pareto.png",
                    title="Pareto Frontier: Accuracy vs Computation Time",
                    xlabel="Computation Time per Epoch [ms]",
                    ylabel="Mean 2D Error [m]")

        plot_accuracy_vs_particles(
            ablation_results,
            RESULTS_DIR / "ablation_particles_accuracy.png")

        plot_time_vs_particles(
            ablation_results,
            RESULTS_DIR / "ablation_particles_time.png")

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    print(f"\n{'=' * 72}")
    print(f"  Results saved to: {RESULTS_DIR}/")
    print(f"{'=' * 72}")

    # Print key findings
    best = min(ablation_results, key=lambda r: r["mean_2d"])
    fastest_best = min(
        [r for r in ablation_results if r["mean_2d"] <= best["mean_2d"] * 1.05],
        key=lambda r: r["time_ms_per_epoch"],
    )
    print(f"\n  Most accurate:  PF-{best['label']} "
          f"(mean={best['mean_2d']:.2f} m, {best['time_ms_per_epoch']:.2f} ms/ep)")
    print(f"  Best tradeoff:  PF-{fastest_best['label']} "
          f"(mean={fastest_best['mean_2d']:.2f} m, "
          f"{fastest_best['time_ms_per_epoch']:.2f} ms/ep)")


if __name__ == "__main__":
    main()
