#!/usr/bin/env python3
"""Particle-level diagnostics for one PPC PF3D-BVH epoch.

Reruns standard PF and PF3D-BVH up to one target epoch, captures the particle
clouds after the update, and summarizes how the 3D mixture likelihood changes
per-satellite weighting on a sampled subset of particles.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "python"))

RESULTS_DIR = _SCRIPT_DIR / "results"

from evaluate import save_results
from exp_ppc_pf3d_residual_analysis import _reference_state_at_truth
from exp_urbannav_baseline import load_or_generate_data, run_wls
from exp_urbannav_pf3d import (
    PF_NLOS_BIAS,
    PF_SIGMA_CB,
    PF_SIGMA_LOS,
    PF_SIGMA_NLOS,
    PF_SIGMA_POS,
    _build_pf3d_variant,
    _epoch_dt,
    load_plateau_model,
)


def _normalize_log_weights(log_weights: np.ndarray) -> tuple[np.ndarray, bool]:
    logw = np.asarray(log_weights, dtype=np.float64).ravel()
    if logw.size == 0:
        return logw.copy(), True
    if np.allclose(logw, logw[0]):
        return np.full(logw.shape, 1.0 / logw.size, dtype=np.float64), True

    shifted = logw - np.max(logw)
    weights = np.exp(shifted)
    weights /= np.sum(weights)
    return weights, False


def _select_particle_indices(
    posterior_weights: np.ndarray,
    uniform_weights: bool,
    max_particles: int,
) -> np.ndarray:
    n_particles = posterior_weights.size
    if max_particles <= 0 or max_particles >= n_particles:
        return np.arange(n_particles, dtype=np.int32)

    if uniform_weights:
        return np.linspace(0, n_particles - 1, max_particles, dtype=np.int32)

    return np.argsort(posterior_weights)[-max_particles:].astype(np.int32)


def _build_standard_pf(n_particles: int):
    from gnss_gpu import ParticleFilter

    return ParticleFilter(
        n_particles=n_particles,
        sigma_pos=PF_SIGMA_POS,
        sigma_cb=PF_SIGMA_CB,
        sigma_pr=PF_SIGMA_LOS,
        resampling="megopolis",
        seed=42,
    )


def _capture_snapshot(
    data: dict,
    filter_obj,
    target_epoch: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    for epoch_idx in range(target_epoch + 1):
        filter_obj.predict(dt=_epoch_dt(data, epoch_idx))
        filter_obj.update(
            data["sat_ecef"][epoch_idx],
            data["pseudoranges"][epoch_idx],
            weights=data["weights"][epoch_idx],
        )

    particles = np.asarray(filter_obj.get_particles(), dtype=np.float64)
    posterior_weights, uniform_weights = _normalize_log_weights(filter_obj._log_weights)
    estimate = np.asarray(filter_obj.estimate(), dtype=np.float64)
    return particles, posterior_weights, estimate, bool(uniform_weights)


def _compute_particle_tables(
    label: str,
    particles: np.ndarray,
    posterior_weights: np.ndarray,
    uniform_weights: bool,
    sat_ecef: np.ndarray,
    pseudoranges: np.ndarray,
    obs_weights: np.ndarray,
    sat_ids: list[str],
    truth_state: np.ndarray,
    truth_los: np.ndarray,
    sigma_los: float,
    sigma_nlos: float,
    nlos_bias: float,
    blocked_nlos_prob: float,
    clear_nlos_prob: float,
    accelerator,
    max_particles: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    selected_idx = _select_particle_indices(posterior_weights, uniform_weights, max_particles)
    sampled_particles = particles[selected_idx]
    sampled_weights = posterior_weights[selected_idx].copy()
    sampled_weights /= np.sum(sampled_weights)

    n_sample = sampled_particles.shape[0]
    n_sat = sat_ecef.shape[0]
    blocked = np.zeros((n_sample, n_sat), dtype=bool)
    for i, particle in enumerate(sampled_particles):
        blocked[i] = ~np.asarray(accelerator.check_los(particle[:3], sat_ecef), dtype=bool)

    ranges = np.linalg.norm(sat_ecef[None, :, :] - sampled_particles[:, None, :3], axis=2)
    residual = pseudoranges[None, :] - (ranges + sampled_particles[:, 3][:, None])
    los_loglik = -0.5 * obs_weights[None, :] * residual * residual / (sigma_los * sigma_los)
    residual_nlos = residual - np.where(residual > 0.0, nlos_bias, 0.0)
    nlos_loglik = (
        -0.5 * obs_weights[None, :] * residual_nlos * residual_nlos / (sigma_nlos * sigma_nlos)
    )
    p_nlos = np.where(blocked, blocked_nlos_prob, clear_nlos_prob)
    mixed_loglik = np.logaddexp(
        np.log1p(-p_nlos) + los_loglik,
        np.log(p_nlos) + nlos_loglik,
    )
    mixed_minus_los = mixed_loglik - los_loglik

    particle_rows: list[dict[str, object]] = []
    for row_idx, particle_idx in enumerate(selected_idx):
        particle_rows.append(
            {
                "method": label,
                "particle_index": int(particle_idx),
                "posterior_weight": float(sampled_weights[row_idx]),
                "uniform_weights": bool(uniform_weights),
                "x": float(sampled_particles[row_idx, 0]),
                "y": float(sampled_particles[row_idx, 1]),
                "z": float(sampled_particles[row_idx, 2]),
                "cb": float(sampled_particles[row_idx, 3]),
                "position_error_m": float(np.linalg.norm(sampled_particles[row_idx, :3] - truth_state[:3])),
                "n_blocked": int(np.count_nonzero(blocked[row_idx])),
                "weighted_mean_residual": float(np.mean(residual[row_idx])),
                "sum_mixed_minus_los": float(np.sum(mixed_minus_los[row_idx])),
            }
        )

    sat_rows: list[dict[str, object]] = []
    truth_residual = pseudoranges - (np.linalg.norm(sat_ecef - truth_state[:3], axis=1) + truth_state[3])
    for sat_idx, sat_id in enumerate(sat_ids):
        sat_rows.append(
            {
                "method": label,
                "sat_id": sat_id,
                "truth_los": bool(truth_los[sat_idx]),
                "truth_residual_m": float(truth_residual[sat_idx]),
                "blocked_frac": float(np.mean(blocked[:, sat_idx])),
                "weighted_blocked_frac": float(np.sum(sampled_weights * blocked[:, sat_idx])),
                "weighted_mean_residual_m": float(np.sum(sampled_weights * residual[:, sat_idx])),
                "weighted_los_loglik": float(np.sum(sampled_weights * los_loglik[:, sat_idx])),
                "weighted_mixed_loglik": float(np.sum(sampled_weights * mixed_loglik[:, sat_idx])),
                "weighted_mixed_minus_los": float(np.sum(sampled_weights * mixed_minus_los[:, sat_idx])),
                "weighted_nlos_loglik": float(np.sum(sampled_weights * nlos_loglik[:, sat_idx])),
                "sample_count": int(n_sample),
            }
        )

    return particle_rows, sat_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Particle-level PF3D-BVH diagnostics for one PPC epoch")
    parser.add_argument("--data-dir", type=Path, required=True, help="PPC run directory")
    parser.add_argument("--model-dir", type=Path, required=True, help="PLATEAU subset directory")
    parser.add_argument("--plateau-zone", type=int, default=9, help="PLATEAU plane-rect zone")
    parser.add_argument("--start-epoch", type=int, default=0, help="Usable epoch offset within the run")
    parser.add_argument("--max-epochs", type=int, default=100, help="Epochs to load from the segment")
    parser.add_argument("--analyze-epoch", type=int, required=True, help="Segment-relative epoch to diagnose")
    parser.add_argument("--systems", type=str, default="G", help="Comma-separated constellations")
    parser.add_argument("--n-particles", type=int, default=10_000, help="Particle count for PF/PF3D-BVH")
    parser.add_argument("--sample-particles", type=int, default=256, help="Particle count to dump/analyze")
    parser.add_argument("--blocked-nlos-prob", type=float, default=0.05, help="P(NLOS | ray blocked)")
    parser.add_argument("--clear-nlos-prob", type=float, default=0.01, help="P(NLOS | ray clear)")
    parser.add_argument(
        "--results-prefix",
        type=str,
        default="ppc_pf3d_particle_diag",
        help="Output filename prefix under experiments/results/",
    )
    args = parser.parse_args()

    systems = tuple(part.strip().upper() for part in args.systems.split(",") if part.strip())
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  PPC PF3D Particle Diagnostics")
    print("=" * 72)

    print("\n[1] Loading data ...")
    data = load_or_generate_data(
        args.data_dir,
        max_real_epochs=args.max_epochs,
        start_epoch=args.start_epoch,
        systems=systems,
    )
    if not 0 <= args.analyze_epoch < data["n_epochs"]:
        raise ValueError(f"analyze_epoch must be in [0, {data['n_epochs'] - 1}]")

    print("\n[2] Loading building model ...")
    building_model = load_plateau_model(args.model_dir, zone=args.plateau_zone)
    if building_model is None:
        raise RuntimeError("PLATEAU model is required for particle diagnostics")

    print("\n[3] Running WLS initialization ...")
    wls_states, _ = run_wls(data)

    print("\n[4] Capturing PF snapshot ...")
    pf = _build_standard_pf(args.n_particles)
    pf.initialize(
        wls_states[0, :3],
        clock_bias=float(wls_states[0, 3]),
        spread_pos=50.0,
        spread_cb=500.0,
    )
    pf_particles, pf_weights, pf_estimate, pf_uniform = _capture_snapshot(
        data, pf, args.analyze_epoch
    )

    print("\n[5] Capturing PF3D-BVH snapshot ...")
    pf3d_bvh = _build_pf3d_variant(
        building_model,
        args.n_particles,
        "pf3d_bvh",
        PF_SIGMA_LOS,
        PF_SIGMA_LOS,
        PF_SIGMA_NLOS,
        PF_NLOS_BIAS,
        args.blocked_nlos_prob,
        args.clear_nlos_prob,
    )
    pf3d_bvh.initialize(
        wls_states[0, :3],
        clock_bias=float(wls_states[0, 3]),
        spread_pos=50.0,
        spread_cb=500.0,
    )
    pf3d_particles, pf3d_weights, pf3d_estimate, pf3d_uniform = _capture_snapshot(
        data, pf3d_bvh, args.analyze_epoch
    )

    print("\n[6] Building diagnostic tables ...")
    sat_ecef = np.asarray(data["sat_ecef"][args.analyze_epoch], dtype=np.float64)
    pseudoranges = np.asarray(data["pseudoranges"][args.analyze_epoch], dtype=np.float64)
    obs_weights = np.asarray(data["weights"][args.analyze_epoch], dtype=np.float64)
    sat_ids = list(data["used_prns"][args.analyze_epoch])
    truth_state = _reference_state_at_truth(
        sat_ecef,
        pseudoranges,
        obs_weights,
        np.asarray(data["ground_truth"][args.analyze_epoch], dtype=np.float64),
    )
    accelerator = pf3d_bvh.bvh
    truth_los = np.asarray(accelerator.check_los(truth_state[:3], sat_ecef), dtype=bool)

    pf_particle_rows, pf_sat_rows = _compute_particle_tables(
        "pf",
        pf_particles,
        pf_weights,
        pf_uniform,
        sat_ecef,
        pseudoranges,
        obs_weights,
        sat_ids,
        truth_state,
        truth_los,
        PF_SIGMA_LOS,
        PF_SIGMA_NLOS,
        PF_NLOS_BIAS,
        args.blocked_nlos_prob,
        args.clear_nlos_prob,
        accelerator,
        args.sample_particles,
    )
    pf3d_particle_rows, pf3d_sat_rows = _compute_particle_tables(
        "pf3d_bvh",
        pf3d_particles,
        pf3d_weights,
        pf3d_uniform,
        sat_ecef,
        pseudoranges,
        obs_weights,
        sat_ids,
        truth_state,
        truth_los,
        PF_SIGMA_LOS,
        PF_SIGMA_NLOS,
        PF_NLOS_BIAS,
        args.blocked_nlos_prob,
        args.clear_nlos_prob,
        accelerator,
        args.sample_particles,
    )

    meta_rows = [
        {
            "method": "pf",
            "analyze_epoch": int(args.analyze_epoch),
            "gps_tow": float(data["times"][args.analyze_epoch]),
            "n_sat": int(len(sat_ids)),
            "sample_particles": int(len(pf_particle_rows)),
            "uniform_weights": bool(pf_uniform),
            "estimate_x": float(pf_estimate[0]),
            "estimate_y": float(pf_estimate[1]),
            "estimate_z": float(pf_estimate[2]),
            "estimate_cb": float(pf_estimate[3]),
            "position_error_m": float(np.linalg.norm(pf_estimate[:3] - truth_state[:3])),
        },
        {
            "method": "pf3d_bvh",
            "analyze_epoch": int(args.analyze_epoch),
            "gps_tow": float(data["times"][args.analyze_epoch]),
            "n_sat": int(len(sat_ids)),
            "sample_particles": int(len(pf3d_particle_rows)),
            "uniform_weights": bool(pf3d_uniform),
            "estimate_x": float(pf3d_estimate[0]),
            "estimate_y": float(pf3d_estimate[1]),
            "estimate_z": float(pf3d_estimate[2]),
            "estimate_cb": float(pf3d_estimate[3]),
            "position_error_m": float(np.linalg.norm(pf3d_estimate[:3] - truth_state[:3])),
        },
    ]

    particle_rows = pf_particle_rows + pf3d_particle_rows
    sat_rows = pf_sat_rows + pf3d_sat_rows

    meta_path = RESULTS_DIR / f"{args.results_prefix}_meta.csv"
    particle_path = RESULTS_DIR / f"{args.results_prefix}_particles.csv"
    sat_path = RESULTS_DIR / f"{args.results_prefix}_satellites.csv"

    save_results({key: [row[key] for row in meta_rows] for key in meta_rows[0]}, meta_path)
    save_results({key: [row[key] for row in particle_rows] for key in particle_rows[0]}, particle_path)
    save_results({key: [row[key] for row in sat_rows] for key in sat_rows[0]}, sat_path)

    print(f"    Meta: {meta_path}")
    print(f"    Particles: {particle_path}")
    print(f"    Satellites: {sat_path}")


if __name__ == "__main__":
    main()
