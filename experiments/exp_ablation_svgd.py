#!/usr/bin/env python3
"""Experiment: SVGD vs resampling ablation study.

Compares three particle filter update strategies:
  1. Systematic resampling   (standard PF)
  2. Megopolis resampling    (MegaParticles-style)
  3. SVGD update             (Stein Variational Gradient Descent)

For each method, measures:
  - Positioning accuracy (mean 2D, P95, RMS)
  - Effective Sample Size (ESS) trajectory
  - Computation time per epoch

Outputs
-------
  experiments/results/ablation_svgd_results.csv
  experiments/results/ablation_svgd_summary.csv
  experiments/results/ablation_svgd_cdf.png
  experiments/results/ablation_svgd_ess.png
  experiments/results/ablation_svgd_timeline.png

Usage
-----
  PYTHONPATH=python python3 experiments/exp_ablation_svgd.py
  PYTHONPATH=python python3 experiments/exp_ablation_svgd.py --data-dir /path/to/urbannav
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
    print_comparison_table,
    save_results,
    wls_solve_py,
)

# ---------------------------------------------------------------------------
# Shared hyperparameters
# ---------------------------------------------------------------------------
N_PARTICLES = 100_000
PF_SIGMA_POS = 2.0
PF_SIGMA_CB = 300.0
PF_SIGMA_PR = 5.0

SVGD_STEPS = 5
SVGD_STEP_SIZE = 0.1
SVGD_N_NEIGHBORS = 32


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(data_dir: Path | None, n_epochs: int) -> dict:
    if data_dir is not None and data_dir.exists():
        print(f"    Trying UrbanNav data in: {data_dir}")
    print("    Using synthetic UrbanNav data.")
    data = generate_synthetic_urbannav(n_epochs=n_epochs, n_satellites=8, seed=42)
    print(f"    {data['n_epochs']} epochs, {data['n_satellites']} satellites")
    return data


def run_wls_init(data: dict) -> np.ndarray:
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


# ---------------------------------------------------------------------------
# Systematic resampling PF
# ---------------------------------------------------------------------------

def run_pf_systematic(
    data: dict,
    n_particles: int,
    wls_init: np.ndarray,
) -> tuple[np.ndarray, list[float], float, str]:
    """Run PF with systematic resampling.

    Returns positions, ess_trajectory, ms_per_epoch, backend.
    """
    n_epochs = data["n_epochs"]
    sat_ecef = data["sat_ecef"]
    pseudoranges = data["pseudoranges"]
    weights = data["weights"]
    dt = data["dt"]
    positions = np.zeros((n_epochs, 3))
    ess_traj = []
    t0 = time.perf_counter()

    try:
        from gnss_gpu import ParticleFilter
        pf = ParticleFilter(
            n_particles=n_particles,
            sigma_pos=PF_SIGMA_POS,
            sigma_cb=PF_SIGMA_CB,
            sigma_pr=PF_SIGMA_PR,
            resampling="systematic",
            ess_threshold=0.5,
            seed=42,
        )
        pf.initialize(wls_init[0, :3], clock_bias=float(wls_init[0, 3]),
                      spread_pos=50.0, spread_cb=500.0)
        for i in range(n_epochs):
            pf.predict(dt=dt)
            pf.update(sat_ecef[i], pseudoranges[i], weights=weights[i])
            est = pf.estimate()
            positions[i] = est[:3]
            ess_traj.append(float(pf.get_ess()))
        backend = "GPU-Systematic"
    except (ImportError, RuntimeError, Exception) as e:
        print(f"      GPU PF (systematic) unavailable ({type(e).__name__}), "
              f"using CPU fallback")
        n_cpu = min(n_particles, 50_000)
        pf = SimplePFCPU(
            n_particles=n_cpu,
            sigma_pos=PF_SIGMA_POS,
            sigma_cb=PF_SIGMA_CB,
            sigma_pr=PF_SIGMA_PR,
            resampling="systematic",
            seed=42,
        )
        pf.initialize(wls_init[0, :3], clock_bias=float(wls_init[0, 3]),
                      spread_pos=50.0, spread_cb=500.0)
        for i in range(n_epochs):
            pf.predict(dt=dt)
            pf.update(sat_ecef[i], pseudoranges[i], weights=weights[i])
            positions[i] = pf.estimate()[:3]
            ess_traj.append(pf.get_ess())
        backend = f"CPU-Systematic({n_cpu})"

    ms = (time.perf_counter() - t0) * 1000.0 / n_epochs
    return positions, ess_traj, ms, backend


# ---------------------------------------------------------------------------
# Megopolis resampling PF
# ---------------------------------------------------------------------------

def run_pf_megopolis(
    data: dict,
    n_particles: int,
    wls_init: np.ndarray,
) -> tuple[np.ndarray, list[float], float, str]:
    """Run PF with Megopolis resampling."""
    n_epochs = data["n_epochs"]
    sat_ecef = data["sat_ecef"]
    pseudoranges = data["pseudoranges"]
    weights = data["weights"]
    dt = data["dt"]
    positions = np.zeros((n_epochs, 3))
    ess_traj = []
    t0 = time.perf_counter()

    try:
        from gnss_gpu import ParticleFilter
        pf = ParticleFilter(
            n_particles=n_particles,
            sigma_pos=PF_SIGMA_POS,
            sigma_cb=PF_SIGMA_CB,
            sigma_pr=PF_SIGMA_PR,
            resampling="megopolis",
            ess_threshold=0.5,
            seed=42,
        )
        pf.initialize(wls_init[0, :3], clock_bias=float(wls_init[0, 3]),
                      spread_pos=50.0, spread_cb=500.0)
        for i in range(n_epochs):
            pf.predict(dt=dt)
            pf.update(sat_ecef[i], pseudoranges[i], weights=weights[i])
            est = pf.estimate()
            positions[i] = est[:3]
            ess_traj.append(float(pf.get_ess()))
        backend = "GPU-Megopolis"
    except (ImportError, RuntimeError, Exception) as e:
        print(f"      GPU PF (megopolis) unavailable ({type(e).__name__}), "
              f"using CPU fallback")
        n_cpu = min(n_particles, 50_000)
        pf = SimplePFCPU(
            n_particles=n_cpu,
            sigma_pos=PF_SIGMA_POS,
            sigma_cb=PF_SIGMA_CB,
            sigma_pr=PF_SIGMA_PR,
            seed=42,
        )
        pf.initialize(wls_init[0, :3], clock_bias=float(wls_init[0, 3]),
                      spread_pos=50.0, spread_cb=500.0)
        for i in range(n_epochs):
            pf.predict(dt=dt)
            pf.update(sat_ecef[i], pseudoranges[i], weights=weights[i])
            positions[i] = pf.estimate()[:3]
            ess_traj.append(pf.get_ess())
        backend = f"CPU-Megopolis({n_cpu})"

    ms = (time.perf_counter() - t0) * 1000.0 / n_epochs
    return positions, ess_traj, ms, backend


# ---------------------------------------------------------------------------
# SVGD PF
# ---------------------------------------------------------------------------

def run_pf_svgd(
    data: dict,
    n_particles: int,
    wls_init: np.ndarray,
) -> tuple[np.ndarray, list[float], float, str]:
    """Run SVGDParticleFilter.

    SVGD does not maintain explicit weights so ESS is reported as n_particles
    (uniform).
    """
    n_epochs = data["n_epochs"]
    sat_ecef = data["sat_ecef"]
    pseudoranges = data["pseudoranges"]
    weights = data["weights"]
    dt = data["dt"]
    positions = np.zeros((n_epochs, 3))
    ess_traj = []
    t0 = time.perf_counter()

    try:
        from gnss_gpu import SVGDParticleFilter
        pf = SVGDParticleFilter(
            n_particles=n_particles,
            sigma_pos=PF_SIGMA_POS,
            sigma_cb=PF_SIGMA_CB,
            sigma_pr=PF_SIGMA_PR,
            svgd_steps=SVGD_STEPS,
            step_size=SVGD_STEP_SIZE,
            n_neighbors=SVGD_N_NEIGHBORS,
            seed=42,
        )
        pf.initialize(wls_init[0, :3], clock_bias=float(wls_init[0, 3]),
                      spread_pos=50.0, spread_cb=500.0)
        for i in range(n_epochs):
            pf.predict(dt=dt)
            pf.update(sat_ecef[i], pseudoranges[i], weights=weights[i])
            est = pf.estimate()
            positions[i] = est[:3]
            # SVGD uses uniform weights after each step
            ess_traj.append(float(n_particles))
        backend = "GPU-SVGD"
    except (ImportError, RuntimeError, Exception) as e:
        print(f"      GPU SVGD unavailable ({type(e).__name__}: {e}). "
              f"Falling back to CPU systematic PF.")
        n_cpu = min(n_particles, 50_000)
        pf = SimplePFCPU(
            n_particles=n_cpu,
            sigma_pos=PF_SIGMA_POS,
            sigma_cb=PF_SIGMA_CB,
            sigma_pr=PF_SIGMA_PR,
            seed=42,
        )
        pf.initialize(wls_init[0, :3], clock_bias=float(wls_init[0, 3]),
                      spread_pos=50.0, spread_cb=500.0)
        for i in range(n_epochs):
            pf.predict(dt=dt)
            pf.update(sat_ecef[i], pseudoranges[i], weights=weights[i])
            positions[i] = pf.estimate()[:3]
            ess_traj.append(pf.get_ess())
        backend = f"CPU-SVGD-fallback({n_cpu})"

    ms = (time.perf_counter() - t0) * 1000.0 / n_epochs
    return positions, ess_traj, ms, backend


# ---------------------------------------------------------------------------
# ESS plot
# ---------------------------------------------------------------------------

def plot_ess_trajectory(
    times: np.ndarray,
    ess_dict: dict[str, list[float]],
    n_particles: int,
    output_path: Path,
) -> None:
    """Plot ESS / N_particles ratio over time."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[ablation_svgd] matplotlib unavailable, skipping ESS plot")
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    linestyles = ["-", "--", "-."]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for idx, (label, ess) in enumerate(ess_dict.items()):
        t = times[:len(ess)]
        ratio = np.asarray(ess) / n_particles
        ax.plot(t, ratio, linestyle=linestyles[idx % 3],
                color=colors[idx % 3], linewidth=1.5, label=label, alpha=0.85)

    ax.axhline(0.5, color="red", linestyle=":", linewidth=0.8, alpha=0.7,
               label="ESS threshold (0.5)")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("ESS / N", fontsize=12)
    ax.set_title("Effective Sample Size Trajectory", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(times[0], times[len(times) - 1])
    ax.set_ylim(0, 1.05)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[ablation_svgd] ESS plot saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="SVGD vs resampling ablation for GNSS PF")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Path to UrbanNav dataset directory")
    parser.add_argument("--n-epochs", type=int, default=300,
                        help="Number of epochs for synthetic data (default: 300)")
    parser.add_argument("--n-particles", type=int, default=N_PARTICLES,
                        help=f"Number of particles (default: {N_PARTICLES})")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip generating plots")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    n_particles = args.n_particles

    print("=" * 72)
    print("  Experiment: SVGD vs Resampling Ablation")
    print("=" * 72)
    print(f"  Particles: {n_particles:,}")
    print(f"  SVGD: {SVGD_STEPS} steps, lr={SVGD_STEP_SIZE}, K={SVGD_N_NEIGHBORS}")

    # ------------------------------------------------------------------
    # [1] Data
    # ------------------------------------------------------------------
    print("\n[1] Loading data ...")
    data = load_data(args.data_dir, args.n_epochs)
    ground_truth = data["ground_truth"]
    n_epochs = data["n_epochs"]

    # ------------------------------------------------------------------
    # [2] WLS reference
    # ------------------------------------------------------------------
    print("\n[2] WLS reference ...")
    wls_init = run_wls_init(data)
    wls_metrics = compute_metrics(wls_init[:, :3], ground_truth)
    print(f"    WLS: RMS 2D={wls_metrics['rms_2d']:.2f} m")

    # ------------------------------------------------------------------
    # [3] Systematic resampling
    # ------------------------------------------------------------------
    print(f"\n[3] Running PF with Systematic resampling ({n_particles:,} particles) ...")
    sys_pos, sys_ess, sys_ms, sys_backend = run_pf_systematic(
        data, n_particles, wls_init)
    sys_metrics = compute_metrics(sys_pos, ground_truth)
    print(f"    Systematic ({sys_backend}): "
          f"RMS 2D={sys_metrics['rms_2d']:.2f} m, "
          f"mean ESS/N={np.mean(sys_ess)/n_particles:.3f}, "
          f"{sys_ms:.3f} ms/epoch")

    # ------------------------------------------------------------------
    # [4] Megopolis resampling
    # ------------------------------------------------------------------
    print(f"\n[4] Running PF with Megopolis resampling ({n_particles:,} particles) ...")
    meg_pos, meg_ess, meg_ms, meg_backend = run_pf_megopolis(
        data, n_particles, wls_init)
    meg_metrics = compute_metrics(meg_pos, ground_truth)
    print(f"    Megopolis ({meg_backend}): "
          f"RMS 2D={meg_metrics['rms_2d']:.2f} m, "
          f"mean ESS/N={np.mean(meg_ess)/n_particles:.3f}, "
          f"{meg_ms:.3f} ms/epoch")

    # ------------------------------------------------------------------
    # [5] SVGD
    # ------------------------------------------------------------------
    print(f"\n[5] Running PF with SVGD ({n_particles:,} particles) ...")
    svgd_pos, svgd_ess, svgd_ms, svgd_backend = run_pf_svgd(
        data, n_particles, wls_init)
    svgd_metrics = compute_metrics(svgd_pos, ground_truth)
    print(f"    SVGD ({svgd_backend}): "
          f"RMS 2D={svgd_metrics['rms_2d']:.2f} m, "
          f"mean ESS/N={np.mean(svgd_ess)/n_particles:.3f}, "
          f"{svgd_ms:.3f} ms/epoch")

    # ------------------------------------------------------------------
    # [6] Comparison table
    # ------------------------------------------------------------------
    print("\n[6] Results summary:")
    all_metrics = {
        "WLS": wls_metrics,
        "PF-Systematic": sys_metrics,
        "PF-Megopolis": meg_metrics,
        "PF-SVGD": svgd_metrics,
    }
    all_metrics["WLS"]["time_ms"] = 0.0
    all_metrics["PF-Systematic"]["time_ms"] = sys_ms
    all_metrics["PF-Megopolis"]["time_ms"] = meg_ms
    all_metrics["PF-SVGD"]["time_ms"] = svgd_ms

    print_comparison_table(all_metrics)

    # ESS summary
    print("\n  Effective Sample Size (mean/N over all epochs):")
    print(f"    Systematic : {np.mean(sys_ess) / n_particles:.4f}")
    print(f"    Megopolis  : {np.mean(meg_ess) / n_particles:.4f}")
    print(f"    SVGD       : {np.mean(svgd_ess) / n_particles:.4f}  "
          f"(uniform by design)")

    # Computation time
    print("\n  Computation time per epoch [ms]:")
    print(f"    Systematic : {sys_ms:.3f}")
    print(f"    Megopolis  : {meg_ms:.3f}")
    print(f"    SVGD       : {svgd_ms:.3f}")

    # ------------------------------------------------------------------
    # [7] Save results
    # ------------------------------------------------------------------
    print("\n[7] Saving results ...")
    epochs = np.arange(n_epochs)

    # Per-epoch errors
    save_results({
        "epoch": epochs,
        "error_2d_systematic": sys_metrics["errors_2d"],
        "error_2d_megopolis": meg_metrics["errors_2d"],
        "error_2d_svgd": svgd_metrics["errors_2d"],
        "ess_systematic": sys_ess,
        "ess_megopolis": meg_ess,
        "ess_svgd": svgd_ess,
    }, RESULTS_DIR / "ablation_svgd_results.csv")

    # Summary
    save_results({
        "method": ["WLS", "PF-Systematic", "PF-Megopolis", "PF-SVGD"],
        "rms_2d": [
            wls_metrics["rms_2d"], sys_metrics["rms_2d"],
            meg_metrics["rms_2d"], svgd_metrics["rms_2d"],
        ],
        "mean_2d": [
            wls_metrics["mean_2d"], sys_metrics["mean_2d"],
            meg_metrics["mean_2d"], svgd_metrics["mean_2d"],
        ],
        "p50": [
            wls_metrics["p50"], sys_metrics["p50"],
            meg_metrics["p50"], svgd_metrics["p50"],
        ],
        "p95": [
            wls_metrics["p95"], sys_metrics["p95"],
            meg_metrics["p95"], svgd_metrics["p95"],
        ],
        "time_ms_per_epoch": [0.0, sys_ms, meg_ms, svgd_ms],
        "mean_ess_ratio": [
            1.0,
            np.mean(sys_ess) / n_particles,
            np.mean(meg_ess) / n_particles,
            np.mean(svgd_ess) / n_particles,
        ],
    }, RESULTS_DIR / "ablation_svgd_summary.csv")

    # ------------------------------------------------------------------
    # [8] Plots
    # ------------------------------------------------------------------
    if not args.no_plots:
        print("\n[8] Generating plots ...")
        errors_for_plot = {
            "WLS": wls_metrics["errors_2d"],
            "PF-Systematic": sys_metrics["errors_2d"],
            "PF-Megopolis": meg_metrics["errors_2d"],
            "PF-SVGD": svgd_metrics["errors_2d"],
        }
        plot_cdf(errors_for_plot, RESULTS_DIR / "ablation_svgd_cdf.png",
                 title="CDF of 2D Error - SVGD vs Resampling Methods")
        plot_error_timeline(
            data["times"], errors_for_plot,
            RESULTS_DIR / "ablation_svgd_timeline.png",
            title="2D Error Over Time - SVGD vs Resampling Methods",
        )

        ess_for_plot = {
            "PF-Systematic": sys_ess,
            "PF-Megopolis": meg_ess,
            "PF-SVGD": svgd_ess,
        }
        plot_ess_trajectory(
            data["times"], ess_for_plot, n_particles,
            RESULTS_DIR / "ablation_svgd_ess.png",
        )

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    print(f"\n{'=' * 72}")
    print(f"  Results saved to: {RESULTS_DIR}/")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
