#!/usr/bin/env python3
"""Quick evaluation of adaptive per-satellite weighting on UrbanNav."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from exp_urbannav_baseline import load_or_generate_data, run_ekf
from exp_urbannav_pf3d import (
    PF_SIGMA_CB, PF_SIGMA_POS, PF_SIGMA_LOS,
    _epoch_dt, _select_guide_velocity,
)
from evaluate import compute_metrics, ecef_errors_2d_3d
from gnss_gpu.adaptive_weight import compute_adaptive_weights


def run_pf_adaptive_weight(
    data: dict,
    n_particles: int,
    ekf_pos: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Run PF with adaptive per-satellite weighting."""
    from gnss_gpu import ParticleFilterDevice

    n_epochs = data["n_epochs"]
    sat_ecef = data["sat_ecef"]
    pseudoranges = data["pseudoranges"]
    weights = data["weights"]

    pf = ParticleFilterDevice(
        n_particles=n_particles,
        sigma_pos=PF_SIGMA_POS,
        sigma_cb=PF_SIGMA_CB,
        sigma_pr=PF_SIGMA_LOS,
        resampling="megopolis",
        seed=42,
    )

    # Init from EKF
    pf.initialize(ekf_pos[0, :3], clock_bias=0.0, spread_pos=50.0, spread_cb=500.0)

    positions = np.zeros((n_epochs, 3))
    t0 = time.perf_counter()

    for i in range(n_epochs):
        dt = _epoch_dt(data, i)
        sat_i = np.asarray(sat_ecef[i], dtype=np.float64).reshape(-1, 3)
        pr_i = np.asarray(pseudoranges[i], dtype=np.float64).ravel()
        w_i = np.asarray(weights[i], dtype=np.float64).ravel()
        mask = np.isfinite(pr_i) & (pr_i > 0)
        sat_i, pr_i, w_i = sat_i[mask], pr_i[mask], w_i[mask]

        # Guide velocity from EKF
        velocity = None
        if i > 0 and dt > 0:
            vel = (ekf_pos[i, :3] - ekf_pos[i - 1, :3]) / dt
            if np.all(np.isfinite(vel)) and np.linalg.norm(vel) < 100:
                velocity = vel

        pf.predict(velocity=velocity, dt=dt)

        if len(sat_i) >= 4:
            # Adaptive weight from EKF reference (more stable than PF estimate)
            adaptive_w = compute_adaptive_weights(
                sat_i, pr_i, ekf_pos[i], sigma_pr=PF_SIGMA_LOS, base_weights=w_i,
            )
            pf.update(sat_i, pr_i, weights=adaptive_w)

        estimate = pf.estimate()
        positions[i] = estimate[:3]

    elapsed = time.perf_counter() - t0
    ms_per_epoch = elapsed * 1000.0 / n_epochs
    return positions, ms_per_epoch


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run", type=str, default="Odaiba")
    parser.add_argument("--systems", type=str, default="G,E,J")
    parser.add_argument("--urban-rover", type=str, default="trimble")
    parser.add_argument("--n-particles", type=int, default=10000)
    args = parser.parse_args()

    systems = tuple(s.strip().upper() for s in args.systems.split(","))
    run_dir = args.data_root / args.run
    data = load_or_generate_data(run_dir, systems=systems, urban_rover=args.urban_rover)
    gt = data["ground_truth"]

    # WLS init for EKF
    from gnss_gpu import wls_position
    n = data["n_epochs"]
    wls_pos = np.zeros((n, 4))
    for i in range(n):
        sat_i = np.asarray(data["sat_ecef"][i], dtype=np.float64).reshape(-1, 3)
        pr_i = np.asarray(data["pseudoranges"][i], dtype=np.float64).ravel()
        w_i = np.asarray(data["weights"][i], dtype=np.float64).ravel()
        mask = np.isfinite(pr_i) & (pr_i > 0)
        if mask.sum() >= 4:
            try:
                res = wls_position(sat_i[mask], pr_i[mask], w_i[mask])
                wls_pos[i] = np.array(res[0])
            except Exception:
                if i > 0: wls_pos[i] = wls_pos[i - 1]
        elif i > 0:
            wls_pos[i] = wls_pos[i - 1]

    # Run EKF
    print(f"  Running EKF...")
    ekf_pos, ekf_ms = run_ekf(data, wls_pos)
    ekf_metrics = compute_metrics(ekf_pos, gt)
    print(f"  EKF: RMS={ekf_metrics['rms_2d']:.2f} m, P95={ekf_metrics['p95']:.2f} m, "
          f">100m={ekf_metrics.get('outlier_rate_pct', 0):.2f}%")

    # Run PF without adaptive weight (baseline)
    print(f"  Running PF (no adaptive weight)...")
    from gnss_gpu import ParticleFilterDevice
    pf_base = ParticleFilterDevice(
        n_particles=args.n_particles,
        sigma_pos=PF_SIGMA_POS, sigma_cb=PF_SIGMA_CB,
        sigma_pr=PF_SIGMA_LOS, resampling="megopolis", seed=42,
    )
    pf_base.initialize(ekf_pos[0, :3], clock_bias=0.0, spread_pos=50.0, spread_cb=500.0)
    base_positions = np.zeros((n, 3))
    for i in range(n):
        dt = _epoch_dt(data, i)
        sat_i = np.asarray(data["sat_ecef"][i], dtype=np.float64).reshape(-1, 3)
        pr_i = np.asarray(data["pseudoranges"][i], dtype=np.float64).ravel()
        w_i = np.asarray(data["weights"][i], dtype=np.float64).ravel()
        mask = np.isfinite(pr_i) & (pr_i > 0)
        sat_i, pr_i, w_i = sat_i[mask], pr_i[mask], w_i[mask]
        velocity = None
        if i > 0 and dt > 0:
            vel = (ekf_pos[i, :3] - ekf_pos[i - 1, :3]) / dt
            if np.all(np.isfinite(vel)) and np.linalg.norm(vel) < 100:
                velocity = vel
        pf_base.predict(velocity=velocity, dt=dt)
        if len(sat_i) >= 4:
            pf_base.update(sat_i, pr_i, weights=w_i)
        base_positions[i] = pf_base.estimate()[:3]
    base_metrics = compute_metrics(base_positions, gt)
    print(f"  PF (baseline): RMS={base_metrics['rms_2d']:.2f} m, P95={base_metrics['p95']:.2f} m, "
          f">100m={base_metrics.get('outlier_rate_pct', 0):.2f}%")

    # Run PF with adaptive weight
    print(f"  Running PF+AdaptiveWeight ({args.n_particles} particles)...")
    aw_positions, aw_ms = run_pf_adaptive_weight(data, args.n_particles, ekf_pos)
    aw_metrics = compute_metrics(aw_positions, gt)
    print(f"  PF+AdaptiveWeight: RMS={aw_metrics['rms_2d']:.2f} m, P95={aw_metrics['p95']:.2f} m, "
          f">100m={aw_metrics.get('outlier_rate_pct', 0):.2f}%")

    print(f"\n  Summary for {args.run}:")
    print(f"    EKF:              RMS={ekf_metrics['rms_2d']:.2f} m")
    print(f"    PF (baseline):    RMS={base_metrics['rms_2d']:.2f} m")
    print(f"    PF+AdaptiveWeight: RMS={aw_metrics['rms_2d']:.2f} m")


if __name__ == "__main__":
    main()
