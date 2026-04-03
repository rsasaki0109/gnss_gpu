#!/usr/bin/env python3
"""Search richer PF gates using blocked score, residual sign, and disagreement."""

from __future__ import annotations

import argparse
import csv
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
    SegmentSpec,
    _derive_segment_meshes,
    _empty_building_model,
    _metrics_row,
    _model_dir,
    _run_dir,
    _subset_url_for_city,
)
from exp_ppc_pf3d_particle_diagnostics import _normalize_log_weights, _select_particle_indices
from exp_ppc_pf3d_residual_analysis import _compute_residuals
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


def _save_rows(rows: list[dict[str, object]], path: Path) -> None:
    keys = sorted({key for row in rows for key in row})
    save_results({key: [row.get(key, np.nan) for row in rows] for key in keys}, path)


def _load_segment_specs(segment_spec_csv: Path | None):
    if segment_spec_csv is None:
        return POSITIVE_SEGMENTS
    rows = list(csv.DictReader(segment_spec_csv.open()))
    specs = []
    for row in rows:
        specs.append(
            SegmentSpec(
                city=str(row["city"]),
                run=str(row["run"]),
                start_epoch=int(row["start_epoch"]),
                nlos_fraction=float(row.get("nlos_fraction", 0.0) or 0.0),
                subset_key=str(row["subset_key"]),
                plateau_zone=int(row.get("plateau_zone", 9)),
            )
        )
    return tuple(specs)


def _parse_segments(raw: str | None, base_specs):
    if raw is None:
        return base_specs

    wanted = {part.strip().lower() for part in raw.split(",") if part.strip()}
    labels = {
        spec: {
            f"{spec.city}/{spec.run}",
            f"{spec.city}/{spec.run}@{spec.start_epoch}",
        }
        for spec in base_specs
    }
    selected = [spec for spec in base_specs if labels[spec] & wanted]
    missing = sorted(wanted - {label for spec in selected for label in labels[spec]})
    if missing:
        raise ValueError(f"unknown segments requested: {', '.join(missing)}")
    return tuple(selected)


def _parse_grid(raw: str) -> tuple[float, ...]:
    values = []
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        values.append(float(token))
    if not values:
        raise ValueError("grid must contain at least one value")
    return tuple(values)


def _compute_blocked_weights(
    filter_obj,
    accelerator,
    sat_ecef: np.ndarray,
    sample_particles: int,
) -> np.ndarray:
    particles = np.asarray(filter_obj.get_particles(), dtype=np.float64)
    posterior_weights, uniform_weights = _normalize_log_weights(filter_obj._log_weights)
    selected_idx = _select_particle_indices(posterior_weights, uniform_weights, sample_particles)
    sampled_particles = particles[selected_idx]
    sampled_weights = posterior_weights[selected_idx].astype(np.float64)
    sampled_weights /= np.sum(sampled_weights)

    blocked = np.zeros((sampled_particles.shape[0], sat_ecef.shape[0]), dtype=bool)
    for i, particle in enumerate(sampled_particles):
        blocked[i] = ~np.asarray(accelerator.check_los(particle[:3], sat_ecef), dtype=bool)

    return np.sum(sampled_weights[:, None] * blocked.astype(np.float64), axis=0)


def _run_experts_and_features(
    data: dict,
    building_model,
    n_particles: int,
    wls_states: np.ndarray,
    blocked_nlos_prob: float,
    clear_nlos_prob: float,
    sample_particles: int,
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
    robust_states = np.zeros((n_epochs, 4), dtype=np.float64)
    blocked_states = np.zeros((n_epochs, 4), dtype=np.float64)
    feature_rows: list[dict[str, float]] = []

    for epoch_idx in range(n_epochs):
        dt = _epoch_dt(data, epoch_idx)
        sat_ecef = np.asarray(data["sat_ecef"][epoch_idx], dtype=np.float64)
        pseudoranges = np.asarray(data["pseudoranges"][epoch_idx], dtype=np.float64)
        weights = np.asarray(data["weights"][epoch_idx], dtype=np.float64)

        robust.predict(dt=dt)
        robust.update(sat_ecef, pseudoranges, weights=weights)
        robust_state = np.asarray(robust.estimate(), dtype=np.float64)
        robust_states[epoch_idx] = robust_state

        blocked.predict(dt=dt)
        blocked.update(sat_ecef, pseudoranges, weights=weights)
        blocked_state = np.asarray(blocked.estimate(), dtype=np.float64)
        blocked_states[epoch_idx] = blocked_state

        blocked_weights = _compute_blocked_weights(robust, accelerator, sat_ecef, sample_particles)
        residual = _compute_residuals(sat_ecef, pseudoranges, weights, robust_state)
        blocked_mass = float(np.sum(blocked_weights))
        positive_mask = residual > 0.0
        positive_gt5_mask = residual > 5.0

        feature_rows.append(
            {
                "epoch": float(epoch_idx),
                "mean_weighted_blocked_frac": float(np.mean(blocked_weights)) if blocked_weights.size else 0.0,
                "max_weighted_blocked_frac": float(np.max(blocked_weights)) if blocked_weights.size else 0.0,
                "n_sat_blocked_gt_005": float(np.count_nonzero(blocked_weights > 0.05)),
                "n_sat_blocked_gt_010": float(np.count_nonzero(blocked_weights > 0.10)),
                "blocked_mass": blocked_mass,
                "blocked_positive_frac": float(
                    np.sum(blocked_weights * positive_mask) / blocked_mass if blocked_mass > 0.0 else 0.0
                ),
                "blocked_positive_frac_gt5": float(
                    np.sum(blocked_weights * positive_gt5_mask) / blocked_mass if blocked_mass > 0.0 else 0.0
                ),
                "blocked_positive_mean_residual": float(
                    np.sum(blocked_weights * np.maximum(residual, 0.0)) / blocked_mass
                    if blocked_mass > 0.0
                    else 0.0
                ),
                "robust_positive_frac": float(np.mean(positive_mask)),
                "robust_positive_frac_gt5": float(np.mean(positive_gt5_mask)),
                "robust_mean_residual": float(np.mean(residual)),
                "robust_mean_abs_residual": float(np.mean(np.abs(residual))),
                "robust_p95_abs_residual": float(np.percentile(np.abs(residual), 95)),
                "disagreement_m": float(np.linalg.norm(blocked_state[:3] - robust_state[:3])),
                "cb_disagreement_m": float(abs(blocked_state[3] - robust_state[3])),
                "satellite_count": float(len(residual)),
            }
        )

    return robust_states[:, :3], blocked_states[:, :3], feature_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search richer gates between PF+RobustClear and PF3D-BVH+BlockedOnly"
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
    parser.add_argument("--segments", type=str, default=None, help="Optional subset like tokyo/run2,nagoya/run2")
    parser.add_argument(
        "--segment-spec-csv",
        type=Path,
        default=None,
        help="Optional CSV with city,run,start_epoch,nlos_fraction,subset_key,plateau_zone",
    )
    parser.add_argument("--max-epochs", type=int, default=100, help="Epochs per segment")
    parser.add_argument("--n-particles", type=int, default=10_000, help="Particle count")
    parser.add_argument("--sample-particles", type=int, default=64, help="Sample size for blocked score")
    parser.add_argument("--blocked-nlos-prob", type=float, default=0.05, help="P(NLOS | blocked)")
    parser.add_argument("--clear-nlos-prob", type=float, default=0.01, help="P(NLOS | clear)")
    parser.add_argument(
        "--blocked-grid",
        type=str,
        default="0,0.001,0.002,0.005,0.01,0.02,0.05",
        help="Blocked-score thresholds",
    )
    parser.add_argument(
        "--positive-grid",
        type=str,
        default="0,0.25,0.5,0.75,0.9",
        help="Blocked-positive-fraction thresholds",
    )
    parser.add_argument(
        "--disagreement-grid",
        type=str,
        default="0,5,10,20,40,80",
        help="Expert disagreement thresholds in meters",
    )
    parser.add_argument(
        "--results-prefix",
        type=str,
        default="ppc_pf_rich_gate_positive6",
        help="Output prefix under experiments/results/",
    )
    args = parser.parse_args()

    systems = tuple(part.strip().upper() for part in args.systems.split(",") if part.strip())
    base_specs = _load_segment_specs(args.segment_spec_csv)
    segments = _parse_segments(args.segments, base_specs)
    blocked_grid = _parse_grid(args.blocked_grid)
    positive_grid = _parse_grid(args.positive_grid)
    disagreement_grid = _parse_grid(args.disagreement_grid)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  PPC PF Rich Gate Search")
    print("=" * 72)
    print(f"Segments: {len(segments)}")
    print(f"Blocked grid: {blocked_grid}")
    print(f"Positive grid: {positive_grid}")
    print(f"Disagreement grid: {disagreement_grid}")

    segment_payloads: list[dict[str, object]] = []
    run_rows: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []
    trajectory_rows: list[dict[str, object]] = []

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
        robust_positions, blocked_positions, segment_features = _run_experts_and_features(
            data,
            building_model,
            args.n_particles,
            wls_pos,
            args.blocked_nlos_prob,
            args.clear_nlos_prob,
            args.sample_particles,
        )

        segment_label = f"{spec.city}_{spec.run}_seg{spec.start_epoch}"
        for row in segment_features:
            row.update(
                {
                    "city": spec.city,
                    "run": spec.run,
                    "segment_label": segment_label,
                    "start_epoch": spec.start_epoch,
                }
            )
            feature_rows.append(row)

        for epoch_idx in range(len(times)):
            trajectory_rows.append(
                {
                    "city": spec.city,
                    "run": spec.run,
                    "segment_label": segment_label,
                    "start_epoch": spec.start_epoch,
                    "epoch": int(epoch_idx),
                    "gps_tow": float(times[epoch_idx]),
                    "truth_x": float(ground_truth[epoch_idx, 0]),
                    "truth_y": float(ground_truth[epoch_idx, 1]),
                    "truth_z": float(ground_truth[epoch_idx, 2]),
                    "robust_x": float(robust_positions[epoch_idx, 0]),
                    "robust_y": float(robust_positions[epoch_idx, 1]),
                    "robust_z": float(robust_positions[epoch_idx, 2]),
                    "blocked_x": float(blocked_positions[epoch_idx, 0]),
                    "blocked_y": float(blocked_positions[epoch_idx, 1]),
                    "blocked_z": float(blocked_positions[epoch_idx, 2]),
                }
            )

        baselines = [
            ("WLS", wls_pos, wls_ms, "WLS"),
            ("EKF", ekf_pos, ekf_ms, "EKF"),
            ("PF", pf_pos, pf_ms, pf_backend),
            ("PF+RobustClear", robust_positions, np.nan, "GPU-3D"),
            ("PF3D-BVH+BlockedOnly", blocked_positions, np.nan, "GPU-3D-BVH"),
        ]
        for method, positions, time_ms, backend in baselines:
            metrics = compute_metrics(positions, ground_truth)
            metrics = _augment_tail_metrics(metrics, times)
            run_rows.append(_metrics_row(spec, method, metrics, time_ms, backend))

        segment_payloads.append(
            {
                "spec": spec,
                "times": times,
                "ground_truth": ground_truth,
                "robust_positions": robust_positions,
                "blocked_positions": blocked_positions,
                "features": segment_features,
            }
        )

    rich_rows: list[dict[str, object]] = []
    config_rows: list[dict[str, object]] = []
    combo_index = 0
    for blocked_th in blocked_grid:
        for positive_th in positive_grid:
            for disagreement_th in disagreement_grid:
                combo_index += 1
                method = (
                    f"PF-RichGate(b={blocked_th:g},p={positive_th:g},d={disagreement_th:g})"
                )
                combo_run_rows: list[dict[str, object]] = []
                for payload in segment_payloads:
                    spec = payload["spec"]
                    feature_array = payload["features"]
                    blocked_score = np.asarray(
                        [row["mean_weighted_blocked_frac"] for row in feature_array], dtype=np.float64
                    )
                    positive_score = np.asarray(
                        [row["blocked_positive_frac_gt5"] for row in feature_array], dtype=np.float64
                    )
                    disagreement = np.asarray(
                        [row["disagreement_m"] for row in feature_array], dtype=np.float64
                    )
                    use_blocked = (
                        (blocked_score >= blocked_th)
                        & (positive_score >= positive_th)
                        & (disagreement >= disagreement_th)
                    )
                    gated_positions = np.asarray(payload["robust_positions"], dtype=np.float64).copy()
                    gated_positions[use_blocked] = np.asarray(payload["blocked_positions"], dtype=np.float64)[use_blocked]

                    metrics = compute_metrics(gated_positions, payload["ground_truth"])
                    metrics = _augment_tail_metrics(metrics, payload["times"])
                    row = _metrics_row(spec, method, metrics, np.nan, "GPU-hybrid")
                    row["blocked_threshold"] = float(blocked_th)
                    row["positive_threshold"] = float(positive_th)
                    row["disagreement_threshold_m"] = float(disagreement_th)
                    row["blocked_epoch_frac"] = float(np.mean(use_blocked))
                    row["mean_blocked_score"] = float(np.mean(blocked_score))
                    row["mean_positive_score"] = float(np.mean(positive_score))
                    row["mean_disagreement_m"] = float(np.mean(disagreement))
                    combo_run_rows.append(row)
                    rich_rows.append(row)

                config_rows.append(
                    {
                        "method": method,
                        "blocked_threshold": float(blocked_th),
                        "positive_threshold": float(positive_th),
                        "disagreement_threshold_m": float(disagreement_th),
                        "n_segments": len(combo_run_rows),
                        "mean_rms_2d": float(np.mean([row["rms_2d"] for row in combo_run_rows])),
                        "mean_p95": float(np.mean([row["p95"] for row in combo_run_rows])),
                        "mean_outlier_rate_pct": float(
                            np.mean([row["outlier_rate_pct"] for row in combo_run_rows])
                        ),
                        "mean_catastrophic_rate_pct": float(
                            np.mean([row["catastrophic_rate_pct"] for row in combo_run_rows])
                        ),
                        "mean_blocked_epoch_frac": float(
                            np.mean([row["blocked_epoch_frac"] for row in combo_run_rows])
                        ),
                        "pf_rms_wins": int(
                            sum(
                                row["rms_2d"]
                                < next(
                                    ref["rms_2d"]
                                    for ref in run_rows
                                    if ref["segment_label"] == row["segment_label"] and ref["method"] == "PF"
                                )
                                for row in combo_run_rows
                            )
                        ),
                        "robust_rms_wins": int(
                            sum(
                                row["rms_2d"]
                                < next(
                                    ref["rms_2d"]
                                    for ref in run_rows
                                    if ref["segment_label"] == row["segment_label"]
                                    and ref["method"] == "PF+RobustClear"
                                )
                                for row in combo_run_rows
                            )
                        ),
                    }
                )

    base_path = RESULTS_DIR / f"{args.results_prefix}_bases.csv"
    feature_path = RESULTS_DIR / f"{args.results_prefix}_features.csv"
    trajectory_path = RESULTS_DIR / f"{args.results_prefix}_trajectories.csv"
    rich_run_path = RESULTS_DIR / f"{args.results_prefix}_runs.csv"
    config_path = RESULTS_DIR / f"{args.results_prefix}_configs.csv"
    best_path = RESULTS_DIR / f"{args.results_prefix}_best.csv"

    _save_rows(run_rows, base_path)
    _save_rows(feature_rows, feature_path)
    _save_rows(trajectory_rows, trajectory_path)
    _save_rows(rich_rows, rich_run_path)
    _save_rows(config_rows, config_path)

    sorted_configs = sorted(config_rows, key=lambda row: (row["mean_rms_2d"], row["mean_p95"]))
    _save_rows(sorted_configs[:20], best_path)

    print(f"\nSaved base metrics to: {base_path}")
    print(f"Saved per-epoch features to: {feature_path}")
    print(f"Saved per-epoch trajectories to: {trajectory_path}")
    print(f"Saved rich-gate runs to: {rich_run_path}")
    print(f"Saved config summary to: {config_path}")
    print(f"Saved top-20 configs to: {best_path}")


if __name__ == "__main__":
    main()
