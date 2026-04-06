#!/usr/bin/env python3
"""Evaluate position_update API: PF + SPP soft constraint.

Uses gnssplusplus corrected pseudoranges (same pipeline as headline results).
Compares PF-only vs PF+position_update(SPP) on Odaiba and Shinjuku.
Tests multiple sigma values to find optimal strength.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "python"))

from evaluate import compute_metrics
from exp_urbannav_baseline import load_or_generate_data

RESULTS_DIR = _SCRIPT_DIR / "results"


def run_pf_gnssplusplus_with_position_update(
    run_dir: Path,
    run_name: str,
    n_particles: int,
    position_update_sigma: float | None,
    rover_source: str = "trimble",
) -> tuple[np.ndarray, np.ndarray, float]:
    """Run PF with gnssplusplus corrections and optional position_update."""
    from libgnsspp import preprocess_spp_file, solve_spp_file
    from gnss_gpu import ParticleFilterDevice
    from exp_urbannav_pf3d import PF_SIGMA_POS, PF_SIGMA_CB

    obs_path = str(run_dir / f"rover_{rover_source}.obs")
    nav_path = str(run_dir / "base.nav")

    epochs = preprocess_spp_file(obs_path, nav_path)
    sol = solve_spp_file(obs_path, nav_path)
    spp_records = [r for r in sol.records() if r.is_valid()]
    spp_lookup = {round(r.time.tow, 1): np.array(r.position_ecef_m) for r in spp_records}

    data = load_or_generate_data(run_dir, systems=("G", "E", "J"), urban_rover=rover_source)
    gt = data["ground_truth"]
    our_times = data["times"]

    pf = ParticleFilterDevice(
        n_particles=n_particles, sigma_pos=PF_SIGMA_POS, sigma_cb=PF_SIGMA_CB,
        sigma_pr=3.0, resampling="megopolis", seed=42,
    )

    first_pos = spp_records[0].position_ecef_m if spp_records else gt[0]
    pf.initialize(np.array(first_pos[:3]), clock_bias=0.0, spread_pos=10.0, spread_cb=100.0)

    all_pf_pos = []
    all_gt = []

    prev_tow = None
    t0 = time.perf_counter()

    for sol_epoch, measurements in epochs:
        if not sol_epoch.is_valid() or len(measurements) < 4:
            continue
        tow = sol_epoch.time.tow
        tow_key = round(tow, 1)
        dt = tow - prev_tow if prev_tow else 0.1

        # Guide velocity from SPP
        velocity = None
        if prev_tow is not None and tow_key in spp_lookup:
            prev_key = round(prev_tow, 1)
            if prev_key in spp_lookup and dt > 0:
                vel = (spp_lookup[tow_key][:3] - spp_lookup[prev_key][:3]) / dt
                if np.all(np.isfinite(vel)) and np.linalg.norm(vel) < 50:
                    velocity = vel

        pf.predict(velocity=velocity, dt=dt)

        sat_ecef = np.array([m.satellite_ecef for m in measurements])
        pr = np.array([m.corrected_pseudorange for m in measurements])
        w = np.array([m.weight for m in measurements])
        pf.update(sat_ecef, pr, weights=w)

        # Position update from SPP
        if position_update_sigma is not None:
            spp_pos = np.array(sol_epoch.position_ecef_m[:3])
            if np.isfinite(spp_pos).all() and np.linalg.norm(spp_pos) > 1e6:
                pf.position_update(spp_pos, sigma_pos=position_update_sigma)

        pf_est = pf.estimate()[:3]

        gt_idx = np.argmin(np.abs(our_times - tow))
        if abs(our_times[gt_idx] - tow) < 0.05:
            all_pf_pos.append(pf_est)
            all_gt.append(gt[gt_idx])

        prev_tow = tow

    elapsed_ms = (time.perf_counter() - t0) * 1000
    return np.array(all_pf_pos), np.array(all_gt), elapsed_ms


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Position update evaluation")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--runs", type=str, default="Odaiba,Shinjuku")
    parser.add_argument("--n-particles", type=int, default=10_000)
    args = parser.parse_args()

    runs = [r.strip() for r in args.runs.split(",")]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    sigmas = [None, 5.0, 10.0, 20.0, 30.0, 50.0, 100.0]

    all_rows = []
    for run_name in runs:
        run_dir = args.data_root / run_name
        print(f"\n{'='*60}")
        print(f"  {run_name} (gnssplusplus corrections)")
        print(f"{'='*60}")

        for sigma in sigmas:
            label = "PF-only" if sigma is None else f"PF+PosUpdate(σ={sigma})"
            print(f"  {label} ...", end=" ", flush=True)

            positions, ground_truth, elapsed_ms = run_pf_gnssplusplus_with_position_update(
                run_dir, run_name, args.n_particles,
                position_update_sigma=sigma,
            )
            metrics = compute_metrics(positions, ground_truth)
            n_ep = len(positions)
            ms_per_epoch = elapsed_ms / n_ep if n_ep > 0 else 0
            outlier_pct = float(np.mean(metrics["errors_2d"] > 100) * 100)

            print(
                f"P50={metrics['p50']:.2f}m  P95={metrics['p95']:.2f}m  "
                f"RMS={metrics['rms_2d']:.2f}m  >100m={outlier_pct:.2f}%  "
                f"({n_ep} ep, {ms_per_epoch:.1f}ms/ep)"
            )

            all_rows.append({
                "run": run_name,
                "method": label,
                "sigma": sigma if sigma is not None else "none",
                "p50": metrics["p50"],
                "p95": metrics["p95"],
                "rms_2d": metrics["rms_2d"],
                "outlier_pct": outlier_pct,
                "n_epochs": n_ep,
                "ms_per_epoch": ms_per_epoch,
            })

    # Print summary table
    print(f"\n{'='*80}")
    print(f"  Summary")
    print(f"{'='*80}")
    print(f"{'Run':<12} {'Method':<25} {'P50':>7} {'P95':>7} {'RMS':>7} {'>100m':>7}")
    print("-" * 80)
    for row in all_rows:
        print(
            f"{row['run']:<12} {row['method']:<25} "
            f"{row['p50']:>6.2f}m {row['p95']:>6.2f}m "
            f"{row['rms_2d']:>6.2f}m {row['outlier_pct']:>6.2f}%"
        )

    import csv
    out_path = RESULTS_DIR / "position_update_eval.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
