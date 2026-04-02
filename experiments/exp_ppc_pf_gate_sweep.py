#!/usr/bin/env python3
"""Evaluate a simple blocked-evidence gate between robust and 3D PF variants."""

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

from evaluate import compute_metrics, save_results
from exp_ppc_pf_ablation_sweep import (
    POSITIVE_SEGMENTS,
    _derive_segment_meshes,
    _empty_building_model,
    _metrics_row,
    _model_dir,
    _run_dir,
    _subset_url_for_city,
)
from exp_ppc_pf3d_particle_diagnostics import _normalize_log_weights, _select_particle_indices
from exp_urbannav_baseline import load_or_generate_data, run_ekf, run_wls
from exp_urbannav_pf3d import (
    PF_NLOS_BIAS,
    PF_SIGMA_LOS,
    PF_SIGMA_NLOS,
    _augment_tail_metrics,
    _build_pf3d_variant,
    _epoch_dt,
    load_plateau_model,
    run_pf_standard,
)
from gnss_gpu.bvh import BVHAccelerator
from scan_ppc_plateau_segments import ensure_subset


def _save_csv_rows(rows: list[dict[str, object]], path: Path) -> None:
    keys = sorted({key for row in rows for key in row})
    save_results({key: [row.get(key, np.nan) for row in rows] for key in keys}, path)


def _parse_segments(raw: str | None):
    if raw is None:
        return POSITIVE_SEGMENTS

    wanted = {part.strip().lower() for part in raw.split(",") if part.strip()}
    selected = [spec for spec in POSITIVE_SEGMENTS if f"{spec.city}/{spec.run}" in wanted]
    missing = sorted(wanted - {f"{spec.city}/{spec.run}" for spec in selected})
    if missing:
        raise ValueError(f"unknown segments requested: {', '.join(missing)}")
    return tuple(selected)


def _parse_thresholds(raw: str) -> tuple[float, ...]:
    values = []
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        value = float(token)
        if value < 0.0 or value > 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {value}")
        values.append(value)
    if not values:
        raise ValueError("at least one threshold must be provided")
    return tuple(values)


def _score_blocked_evidence(filter_obj, accelerator, sat_ecef: np.ndarray, sample_particles: int) -> dict[str, float]:
    particles = np.asarray(filter_obj.get_particles(), dtype=np.float64)
    posterior_weights, uniform_weights = _normalize_log_weights(filter_obj._log_weights)
    selected_idx = _select_particle_indices(posterior_weights, uniform_weights, sample_particles)
    sampled_particles = particles[selected_idx]
    sampled_weights = posterior_weights[selected_idx].astype(np.float64)
    sampled_weights /= np.sum(sampled_weights)

    blocked = np.zeros((sampled_particles.shape[0], sat_ecef.shape[0]), dtype=bool)
    for i, particle in enumerate(sampled_particles):
        blocked[i] = ~np.asarray(accelerator.check_los(particle[:3], sat_ecef), dtype=bool)

    weighted_blocked_frac = np.sum(sampled_weights[:, None] * blocked.astype(np.float64), axis=0)
    return {
        "mean_weighted_blocked_frac": float(np.mean(weighted_blocked_frac)),
        "max_weighted_blocked_frac": float(np.max(weighted_blocked_frac)) if weighted_blocked_frac.size else 0.0,
        "n_sat_blocked_gt_005": float(np.count_nonzero(weighted_blocked_frac > 0.05)),
        "n_sat_blocked_gt_010": float(np.count_nonzero(weighted_blocked_frac > 0.10)),
        "uniform_weights": float(uniform_weights),
    }


def _run_experts_with_scores(
    data: dict,
    building_model,
    n_particles: int,
    wls_states: np.ndarray,
    blocked_nlos_prob: float,
    clear_nlos_prob: float,
    sample_particles: int,
    score_source: str,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]]]:
    robust = _build_pf3d_variant(
        _empty_building_model(),
        n_particles,
        "pf3d",
        PF_SIGMA_LOS,
        PF_SIGMA_LOS,
        PF_SIGMA_NLOS,
        PF_NLOS_BIAS,
        0.0,
        clear_nlos_prob,
    )
    blocked = _build_pf3d_variant(
        building_model,
        n_particles,
        "pf3d_bvh",
        PF_SIGMA_LOS,
        PF_SIGMA_LOS,
        PF_SIGMA_NLOS,
        PF_NLOS_BIAS,
        blocked_nlos_prob,
        0.0,
    )

    for filt in (robust, blocked):
        filt.initialize(
            wls_states[0, :3],
            clock_bias=float(wls_states[0, 3]),
            spread_pos=50.0,
            spread_cb=500.0,
        )

    accelerator = BVHAccelerator.from_building_model(building_model)
    n_epochs = data["n_epochs"]
    robust_positions = np.zeros((n_epochs, 3), dtype=np.float64)
    blocked_positions = np.zeros((n_epochs, 3), dtype=np.float64)
    score_rows: list[dict[str, float]] = []

    for epoch_idx in range(n_epochs):
        dt = _epoch_dt(data, epoch_idx)
        sat_ecef = np.asarray(data["sat_ecef"][epoch_idx], dtype=np.float64)
        pseudoranges = data["pseudoranges"][epoch_idx]
        weights = data["weights"][epoch_idx]

        robust.predict(dt=dt)
        robust.update(sat_ecef, pseudoranges, weights=weights)
        robust_positions[epoch_idx] = np.asarray(robust.estimate(), dtype=np.float64)[:3]

        blocked.predict(dt=dt)
        blocked.update(sat_ecef, pseudoranges, weights=weights)
        blocked_positions[epoch_idx] = np.asarray(blocked.estimate(), dtype=np.float64)[:3]

        score_filter = robust if score_source == "robust" else blocked
        scores = _score_blocked_evidence(score_filter, accelerator, sat_ecef, sample_particles)
        scores["epoch"] = float(epoch_idx)
        score_rows.append(scores)

    return robust_positions, blocked_positions, score_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep a blocked-evidence gate between robust-clear and blocked-only PF variants"
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/tmp/PPC-real/PPC-Dataset"),
        help="PPC-Dataset root directory",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("/tmp/plateau_segment_cache"),
        help="PLATEAU subset cache root",
    )
    parser.add_argument("--systems", type=str, default="G", help="Comma-separated constellations")
    parser.add_argument("--max-epochs", type=int, default=100, help="Epochs per segment")
    parser.add_argument("--n-particles", type=int, default=10_000, help="Particle count")
    parser.add_argument("--sample-particles", type=int, default=128, help="Particle count for gate scoring")
    parser.add_argument(
        "--blocked-nlos-prob",
        type=float,
        default=0.05,
        help="P(NLOS | ray blocked) for the blocked-only expert",
    )
    parser.add_argument(
        "--clear-nlos-prob",
        type=float,
        default=0.01,
        help="P(NLOS | ray clear) for the robust-clear expert",
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        default="0,0.005,0.01,0.02,0.05,0.1",
        help="Comma-separated thresholds on mean weighted blocked fraction",
    )
    parser.add_argument(
        "--results-prefix",
        type=str,
        default="ppc_pf_gate_positive6",
        help="Output filename prefix under experiments/results/",
    )
    parser.add_argument(
        "--segments",
        type=str,
        default=None,
        help="Optional comma-separated subset like tokyo/run2,tokyo/run3",
    )
    parser.add_argument(
        "--score-source",
        type=str,
        choices=("robust", "blocked"),
        default="robust",
        help="Which expert's particles to use for the blocked-evidence gate score",
    )
    args = parser.parse_args()

    systems = tuple(part.strip().upper() for part in args.systems.split(",") if part.strip())
    thresholds = _parse_thresholds(args.thresholds)
    segments = _parse_segments(args.segments)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  PPC PF Gate Sweep")
    print("=" * 72)
    print(f"Thresholds: {', '.join(f'{value:g}' for value in thresholds)}")

    run_rows: list[dict[str, object]] = []
    score_rows: list[dict[str, object]] = []

    for spec in segments:
        run_dir = _run_dir(args.data_root, spec)
        model_dir = _model_dir(args.cache_root, spec)

        print(f"\n[{spec.city}/{spec.run} @ {spec.start_epoch}] Loading data ...")
        data = load_or_generate_data(
            run_dir,
            max_real_epochs=args.max_epochs,
            start_epoch=spec.start_epoch,
            systems=systems,
        )
        if not model_dir.exists():
            meshes = _derive_segment_meshes(np.asarray(data["ground_truth"], dtype=np.float64))
            model_dir = ensure_subset(_subset_url_for_city(spec.city), meshes, args.cache_root)
        building_model = load_plateau_model(model_dir, zone=spec.plateau_zone)
        if building_model is None:
            raise RuntimeError(f"failed to load building model: {model_dir}")

        times = np.asarray(data["times"], dtype=np.float64)
        ground_truth = np.asarray(data["ground_truth"], dtype=np.float64)

        wls_pos, wls_ms = run_wls(data)
        ekf_pos, ekf_ms = run_ekf(data, wls_pos)
        pf_pos, pf_ms, pf_backend = run_pf_standard(
            data,
            args.n_particles,
            wls_pos,
            sigma_pr=PF_SIGMA_LOS,
        )

        robust_positions, blocked_positions, segment_scores = _run_experts_with_scores(
            data,
            building_model,
            args.n_particles,
            wls_pos,
            args.blocked_nlos_prob,
            args.clear_nlos_prob,
            args.sample_particles,
            args.score_source,
        )

        for row in segment_scores:
            row.update(
                {
                    "city": spec.city,
                    "run": spec.run,
                    "segment_label": f"{spec.city}_{spec.run}_seg{spec.start_epoch}",
                    "start_epoch": spec.start_epoch,
                }
            )
            score_rows.append(row)

        baselines = [
            ("WLS", wls_pos, wls_ms, "WLS"),
            ("EKF", ekf_pos, ekf_ms, "EKF"),
            ("PF", pf_pos, pf_ms, pf_backend),
        ]
        for method, positions, time_ms, backend in baselines:
            metrics = compute_metrics(positions, ground_truth)
            metrics = _augment_tail_metrics(metrics, times)
            run_rows.append(_metrics_row(spec, method, metrics, time_ms, backend))

        robust_metrics = compute_metrics(robust_positions, ground_truth)
        robust_metrics = _augment_tail_metrics(robust_metrics, times)
        run_rows.append(_metrics_row(spec, "PF+RobustClear", robust_metrics, np.nan, "GPU-3D"))

        blocked_metrics = compute_metrics(blocked_positions, ground_truth)
        blocked_metrics = _augment_tail_metrics(blocked_metrics, times)
        run_rows.append(_metrics_row(spec, "PF3D-BVH+BlockedOnly", blocked_metrics, np.nan, "GPU-3D-BVH"))

        mean_scores = np.asarray([row["mean_weighted_blocked_frac"] for row in segment_scores], dtype=np.float64)
        for threshold in thresholds:
            use_blocked = mean_scores >= threshold
            gated_positions = robust_positions.copy()
            gated_positions[use_blocked] = blocked_positions[use_blocked]
            metrics = compute_metrics(gated_positions, ground_truth)
            metrics = _augment_tail_metrics(metrics, times)
            row = _metrics_row(
                spec,
                f"PF-Gated(th={threshold:g})",
                metrics,
                np.nan,
                "GPU-hybrid",
            )
            row["gate_threshold"] = float(threshold)
            row["blocked_epoch_frac"] = float(np.mean(use_blocked))
            row["mean_gate_score"] = float(np.mean(mean_scores))
            run_rows.append(row)

    run_path = RESULTS_DIR / f"{args.results_prefix}_runs.csv"
    _save_csv_rows(run_rows, run_path)

    config_rows: list[dict[str, object]] = []
    methods = sorted({row["method"] for row in run_rows})
    for method in methods:
        rows = [row for row in run_rows if row["method"] == method]
        config_rows.append(
            {
                "method": method,
                "n_segments": len(rows),
                "mean_rms_2d": float(np.mean([row["rms_2d"] for row in rows])),
                "mean_p95": float(np.mean([row["p95"] for row in rows])),
                "mean_outlier_rate_pct": float(np.mean([row["outlier_rate_pct"] for row in rows])),
                "mean_catastrophic_rate_pct": float(np.mean([row["catastrophic_rate_pct"] for row in rows])),
                "mean_blocked_epoch_frac": float(
                    np.mean([row.get("blocked_epoch_frac", np.nan) for row in rows])
                ),
                "pf_rms_wins": int(
                    sum(
                        row["rms_2d"] < next(
                            ref["rms_2d"]
                            for ref in run_rows
                            if ref["segment_label"] == row["segment_label"] and ref["method"] == "PF"
                        )
                        for row in rows
                    )
                ),
                "pf_p95_wins": int(
                    sum(
                        row["p95"] < next(
                            ref["p95"]
                            for ref in run_rows
                            if ref["segment_label"] == row["segment_label"] and ref["method"] == "PF"
                        )
                        for row in rows
                    )
                ),
            }
        )

    config_path = RESULTS_DIR / f"{args.results_prefix}_configs.csv"
    _save_csv_rows(config_rows, config_path)

    score_path = RESULTS_DIR / f"{args.results_prefix}_scores.csv"
    _save_csv_rows(score_rows, score_path)

    print(f"\nSaved run-wise results to: {run_path}")
    print(f"Saved config-wise summary to: {config_path}")
    print(f"Saved gate scores to: {score_path}")


if __name__ == "__main__":
    main()
