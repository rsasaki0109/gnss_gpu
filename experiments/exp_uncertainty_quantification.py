#!/usr/bin/env python3
"""Particle-based uncertainty quantification experiment.

Runs PF on UrbanNav with get_particles() to extract per-epoch posterior spread,
then analyzes:
1. Posterior std vs actual error correlation (calibration)
2. Multi-modal epoch detection via kurtosis
3. Posterior particle cloud visualization at selected epochs
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from exp_urbannav_pf3d import (
    PF_SIGMA_CB,
    PF_SIGMA_POS,
    _epoch_dt,
    _select_pf_epoch_measurements,
)
from evaluate import ecef_errors_2d_3d

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"


def run_pf_with_uncertainty(
    data: dict,
    n_particles: int,
    wls_init: np.ndarray,
    sigma_pr: float = 10.0,
    dump_interval: int = 1,
) -> dict:
    """Run PF and record per-epoch posterior statistics."""
    from gnss_gpu import ParticleFilterDevice

    n_epochs = data["n_epochs"]
    sat_ecef = data["sat_ecef"]
    pseudoranges = data["pseudoranges"]
    weights = data["weights"]
    gt_ecef = data["ground_truth"]

    pf = ParticleFilterDevice(
        n_particles=n_particles,
        sigma_pos=PF_SIGMA_POS,
        sigma_cb=PF_SIGMA_CB,
        sigma_pr=sigma_pr,
        resampling="megopolis",
        seed=42,
    )

    init_pos = np.asarray(wls_init[0, :3], dtype=np.float64)
    init_cb = float(wls_init[0, 3])
    pf.initialize(init_pos, clock_bias=init_cb, spread_pos=50.0, spread_cb=500.0)

    sampled_epochs = []
    estimates_all = np.zeros((n_epochs, 3))
    posterior_stats = []
    particle_dumps = {}

    t0 = time.perf_counter()
    for i in range(n_epochs):
        dt = _epoch_dt(data, i)
        sat_i = np.asarray(sat_ecef[i], dtype=np.float64).reshape(-1, 3)
        pr_i = np.asarray(pseudoranges[i], dtype=np.float64).ravel()
        w_i = np.asarray(weights[i], dtype=np.float64).ravel()
        mask = np.isfinite(pr_i) & (pr_i > 0)
        sat_i, pr_i, w_i = sat_i[mask], pr_i[mask], w_i[mask]

        pf.predict(dt=dt)
        if len(pr_i) >= 4:
            pf.update(sat_i, pr_i, weights=w_i)

        estimate = np.asarray(pf.estimate(), dtype=np.float64)
        estimates_all[i] = estimate[:3]

        if i % dump_interval == 0:
            particles = pf.get_particles()  # (N, 4)
            std_x = np.std(particles[:, 0])
            std_y = np.std(particles[:, 1])
            std_z = np.std(particles[:, 2])
            dx = particles[:, 0] - np.mean(particles[:, 0])
            dy = particles[:, 1] - np.mean(particles[:, 1])
            dist_2d = np.sqrt(dx**2 + dy**2)
            std_2d = np.std(dist_2d)
            if std_2d > 0:
                kurtosis_2d = float(np.mean((dist_2d / std_2d) ** 4))
            else:
                kurtosis_2d = 0.0

            posterior_stats.append({
                "epoch": i,
                "std_x": std_x, "std_y": std_y, "std_z": std_z,
                "std_2d": std_2d, "kurtosis_2d": kurtosis_2d,
                "ess": pf.get_ess(), "n_sat": len(pr_i),
            })
            sampled_epochs.append(i)

    elapsed = time.perf_counter() - t0
    print(f"    UQ run: {n_epochs} epochs, {n_particles} particles, {elapsed:.1f}s")

    # Compute proper ENU-based 2D errors
    errors_2d, _ = ecef_errors_2d_3d(estimates_all, gt_ecef)

    # Build results dict with errors at sampled epochs
    results = {
        "epoch": [], "error_2d": [],
        "posterior_std_x": [], "posterior_std_y": [], "posterior_std_z": [],
        "posterior_std_2d": [], "posterior_kurtosis_2d": [],
        "ess": [], "n_sat": [],
    }
    for ps in posterior_stats:
        i = ps["epoch"]
        results["epoch"].append(i)
        results["error_2d"].append(float(errors_2d[i]))
        results["posterior_std_x"].append(ps["std_x"])
        results["posterior_std_y"].append(ps["std_y"])
        results["posterior_std_z"].append(ps["std_z"])
        results["posterior_std_2d"].append(ps["std_2d"])
        results["posterior_kurtosis_2d"].append(ps["kurtosis_2d"])
        results["ess"].append(ps["ess"])
        results["n_sat"].append(ps["n_sat"])

        # Dump particles at high-error epochs
        if float(errors_2d[i]) > 100 and len(particle_dumps) < 10:
            particle_dumps[i] = None  # placeholder

    return {"stats": results, "particle_dumps": particle_dumps}


def save_uq_csv(results: dict, path: Path) -> None:
    stats = results["stats"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "epoch", "error_2d", "posterior_std_2d", "posterior_std_x",
            "posterior_std_y", "posterior_std_z", "kurtosis_2d", "ess", "n_sat",
        ])
        for j in range(len(stats["epoch"])):
            writer.writerow([
                stats["epoch"][j],
                f"{stats['error_2d'][j]:.4f}",
                f"{stats['posterior_std_2d'][j]:.4f}",
                f"{stats['posterior_std_x'][j]:.4f}",
                f"{stats['posterior_std_y'][j]:.4f}",
                f"{stats['posterior_std_z'][j]:.4f}",
                f"{stats['posterior_kurtosis_2d'][j]:.4f}",
                f"{stats['ess'][j]:.1f}",
                stats["n_sat"][j],
            ])
    print(f"    saved {path}")


def plot_uq(results: dict, output_dir: Path, prefix: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stats = results["stats"]
    error = np.array(stats["error_2d"])
    std_2d = np.array(stats["posterior_std_2d"])
    kurtosis = np.array(stats["posterior_kurtosis_2d"])

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Scatter: posterior std vs actual error
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.scatter(std_2d, error, s=2, alpha=0.3, color="#059669")
    max_val = min(max(std_2d.max(), error.max()), 500)
    ax.plot([0, max_val], [0, max_val], "k--", linewidth=1, alpha=0.5, label="y=x")
    ax.set_xlabel("Posterior Std 2D [m]")
    ax.set_ylabel("Actual Error 2D [m]")
    ax.set_title("Uncertainty Calibration")
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.grid(True, alpha=0.25)
    corr = np.corrcoef(std_2d, error)[0, 1]
    ax.text(0.05, 0.95, f"r = {corr:.3f}", transform=ax.transAxes,
            fontsize=11, va="top", ha="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.legend(fontsize=8)

    # 2. Kurtosis vs error
    ax = axes[1]
    ax.scatter(kurtosis, error, s=2, alpha=0.3, color="#7c3aed")
    ax.axvline(3.0, color="red", linestyle="--", linewidth=1, alpha=0.5,
               label="Gaussian kurtosis")
    ax.set_xlabel("Posterior Kurtosis 2D")
    ax.set_ylabel("Actual Error 2D [m]")
    ax.set_title("Multi-modality vs Error")
    ax.set_ylim(0, min(error.max(), 500))
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    # 3. Time series: error and std overlay
    ax = axes[2]
    epochs = np.array(stats["epoch"])
    ax.plot(epochs, error, linewidth=0.5, alpha=0.6, color="#ef4444", label="Error 2D")
    ax.plot(epochs, std_2d, linewidth=0.5, alpha=0.6, color="#059669", label="Posterior Std")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("[m]")
    ax.set_title("Error vs Posterior Spread over Time")
    ax.set_ylim(0, min(max(error.max(), std_2d.max()), 500))
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    fig.suptitle(f"Uncertainty Quantification ({prefix})", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_dir / f"paper_uq_{prefix}.png", dpi=180)
    plt.close(fig)
    print(f"    saved {output_dir / f'paper_uq_{prefix}.png'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="PF uncertainty quantification")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run", type=str, default="Odaiba")
    parser.add_argument("--systems", type=str, default="G,E,J")
    parser.add_argument("--urban-rover", type=str, default="trimble")
    parser.add_argument("--n-particles", type=int, default=10000)
    parser.add_argument("--dump-interval", type=int, default=1,
                        help="Record posterior stats every N epochs")
    args = parser.parse_args()

    sys.path.insert(0, str(PROJECT_ROOT / "experiments"))
    from exp_urbannav_baseline import load_or_generate_data

    systems = tuple(s.strip().upper() for s in args.systems.split(","))
    run_dir = args.data_root / args.run
    data = load_or_generate_data(
        run_dir, systems=systems, urban_rover=args.urban_rover,
    )
    # WLS init
    from gnss_gpu import wls_position
    n_epochs = data["n_epochs"]
    wls_pos = np.zeros((n_epochs, 4))
    for i in range(n_epochs):
        sat_i = np.asarray(data["sat_ecef"][i], dtype=np.float64).reshape(-1, 3)
        pr_i = np.asarray(data["pseudoranges"][i], dtype=np.float64).ravel()
        mask = np.isfinite(pr_i) & (pr_i > 0)
        if mask.sum() >= 4:
            try:
                wls_pos[i] = wls_position(sat_i[mask], pr_i[mask])
            except Exception:
                if i > 0:
                    wls_pos[i] = wls_pos[i - 1]
        elif i > 0:
            wls_pos[i] = wls_pos[i - 1]

    prefix = f"{args.run.lower()}_{args.n_particles}"
    results = run_pf_with_uncertainty(
        data, args.n_particles, wls_pos,
        dump_interval=args.dump_interval,
    )

    save_uq_csv(results, RESULTS_DIR / f"uq_{prefix}.csv")
    plot_uq(results, RESULTS_DIR / "paper_assets", prefix)


if __name__ == "__main__":
    main()
