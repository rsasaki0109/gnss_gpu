#!/usr/bin/env python3
"""Run a fixed UrbanNav evaluation table without retuning on UrbanNav itself."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "python"))

RESULTS_DIR = _SCRIPT_DIR / "results"

from evaluate import compute_metrics, ecef_errors_2d_3d, save_results
from exp_urbannav_baseline import load_or_generate_data, run_ekf, run_wls
from exp_urbannav_pf3d import (
    PF_RESCUE_DISTANCE_M,
    PF_SIGMA_LOS,
    _augment_tail_metrics,
    run_pf3d_variant,
    run_pf_standard,
)
from gnss_gpu.multi_gnss_quality import MultiGNSSQualityVetoConfig
from gnss_gpu.raytrace import BuildingModel

ALL_METHODS = (
    "WLS",
    "WLS+QualityVeto",
    "EKF",
    "PF-10K",
    "PF+EKFGuide-10K",
    "PF+EKFGuideInit-10K",
    "PF+EKFGuideFallback-10K",
    "PF+AdaptiveGuide-10K",
    "PF+EKFRescue-10K",
    "PF+RobustClear-10K",
    "PF+RobustClear+EKFGuide-10K",
    "PF+RobustClear+EKFGuideInit-10K",
    "PF+RobustClear+EKFGuideFallback-10K",
    "PF+RobustClear+EKFRescue-10K",
)


def _empty_building_model() -> BuildingModel:
    return BuildingModel(np.zeros((0, 3, 3), dtype=np.float64))


def _parse_methods(raw: str) -> tuple[str, ...]:
    methods = tuple(part.strip() for part in raw.split(",") if part.strip())
    unknown = sorted(set(methods) - set(ALL_METHODS))
    if unknown:
        raise ValueError(f"unknown methods: {', '.join(unknown)}")
    return methods or ALL_METHODS


def _method_slug(method: str) -> str:
    slug = method.lower().replace("+", "plus").replace("/", "_")
    for ch in (" ", "-", "(", ")", ","):
        slug = slug.replace(ch, "_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_")


def _metrics_row(run_name: str, method: str, metrics: dict, time_ms: float, backend: str) -> dict[str, object]:
    return {
        "run": run_name,
        "method": method,
        "backend": backend,
        "time_ms_per_epoch": float(time_ms),
        "mean_2d": float(metrics["mean_2d"]),
        "rms_2d": float(metrics["rms_2d"]),
        "rms_3d": float(metrics["rms_3d"]),
        "p50": float(metrics["p50"]),
        "p67": float(metrics["p67"]),
        "p95": float(metrics["p95"]),
        "max_2d": float(metrics["max_2d"]),
        "outlier_rate_pct": float(metrics["outlier_rate_pct"]),
        "catastrophic_rate_pct": float(metrics["catastrophic_rate_pct"]),
        "longest_outlier_segment_epochs": float(metrics["longest_outlier_segment_epochs"]),
        "longest_outlier_segment_s": float(metrics["longest_outlier_segment_s"]),
        "n_epochs": int(metrics["n_epochs"]),
    }


def _epoch_rows(
    run_name: str,
    method: str,
    positions: np.ndarray,
    ground_truth: np.ndarray,
    times: np.ndarray,
    satellite_counts: np.ndarray,
    backend: str,
) -> list[dict[str, object]]:
    positions = np.asarray(positions, dtype=np.float64)
    ground_truth = np.asarray(ground_truth, dtype=np.float64)
    error_2d, error_3d = ecef_errors_2d_3d(positions, ground_truth)
    rows: list[dict[str, object]] = []
    for i in range(len(error_2d)):
        rows.append(
            {
                "run": run_name,
                "method": method,
                "backend": backend,
                "epoch_index": int(i),
                "gps_time_s": float(times[i]),
                "satellite_count": int(satellite_counts[i]),
                "est_x": float(positions[i, 0]),
                "est_y": float(positions[i, 1]),
                "est_z": float(positions[i, 2]),
                "gt_x": float(ground_truth[i, 0]),
                "gt_y": float(ground_truth[i, 1]),
                "gt_z": float(ground_truth[i, 2]),
                "error_2d": float(error_2d[i]),
                "error_3d": float(error_3d[i]),
                "outlier_100m": int(error_2d[i] > 100.0),
                "catastrophic_500m": int(error_2d[i] > 500.0),
            }
        )
    return rows


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _build_summary_rows(run_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summary_rows: list[dict[str, object]] = []
    method_names = sorted({str(row["method"]) for row in run_rows})
    for method_name in method_names:
        rows = [row for row in run_rows if row["method"] == method_name]
        wins_vs_wls = 0
        wins_available = True
        for row in rows:
            wls_candidates = [
                ref for ref in run_rows if ref["run"] == row["run"] and ref["method"] == "WLS"
            ]
            if not wls_candidates:
                wins_available = False
                break
            if float(row["rms_2d"]) < float(wls_candidates[0]["rms_2d"]):
                wins_vs_wls += 1
        summary_rows.append(
            {
                "method": method_name,
                "n_runs": len(rows),
                "mean_rms_2d": float(np.mean([float(row["rms_2d"]) for row in rows])),
                "mean_rms_3d": float(np.mean([float(row["rms_3d"]) for row in rows])),
                "mean_p50": float(np.mean([float(row["p50"]) for row in rows])),
                "mean_p95": float(np.mean([float(row["p95"]) for row in rows])),
                "mean_outlier_rate_pct": float(np.mean([float(row["outlier_rate_pct"]) for row in rows])),
                "mean_catastrophic_rate_pct": float(
                    np.mean([float(row["catastrophic_rate_pct"]) for row in rows])
                ),
                "mean_longest_outlier_segment_s": float(
                    np.mean([float(row["longest_outlier_segment_s"]) for row in rows])
                ),
                "mean_time_ms_per_epoch": float(np.mean([float(row["time_ms_per_epoch"]) for row in rows])),
                "wins_vs_wls_rms": float(wins_vs_wls) if wins_available else float("nan"),
            }
        )
    return summary_rows


def _save_rows(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        return
    save_results({key: [row[key] for row in rows] for key in rows[0].keys()}, path)


def _evaluate_methods_for_run(
    run_name: str,
    data: dict,
    methods: tuple[str, ...],
    n_particles: int,
    clear_nlos_prob: float,
    quality_veto_config: MultiGNSSQualityVetoConfig,
    pf_rescue_distance_m: float | None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    times = np.asarray(data["times"], dtype=np.float64)
    ground_truth = np.asarray(data["ground_truth"], dtype=np.float64)
    satellite_counts = np.asarray(data["satellite_counts"], dtype=np.int32)
    empty_model = _empty_building_model()

    need_wls = any(
        method in methods
        for method in (
            "WLS",
            "WLS+QualityVeto",
            "EKF",
            "PF-10K",
            "PF+EKFGuide-10K",
            "PF+EKFGuideInit-10K",
            "PF+EKFGuideFallback-10K",
            "PF+AdaptiveGuide-10K",
            "PF+EKFRescue-10K",
            "PF+RobustClear-10K",
            "PF+RobustClear+EKFGuide-10K",
            "PF+RobustClear+EKFGuideInit-10K",
            "PF+RobustClear+EKFGuideFallback-10K",
            "PF+RobustClear+EKFRescue-10K",
        )
    )
    if not need_wls:
        return [], []

    wls_pos, wls_ms = run_wls(data)
    cached: dict[str, tuple[np.ndarray, float, str]] = {
        "WLS": (wls_pos[:, :3], wls_ms, "WLS"),
    }
    if "WLS+QualityVeto" in methods:
        veto_pos, veto_ms = run_wls(data, quality_veto_config=quality_veto_config)
        cached["WLS+QualityVeto"] = (veto_pos[:, :3], veto_ms, "WLS+QualityVeto")
    need_ekf = any(
        method in methods
        for method in (
            "EKF",
            "PF+EKFGuide-10K",
            "PF+EKFGuideInit-10K",
            "PF+EKFGuideFallback-10K",
            "PF+AdaptiveGuide-10K",
            "PF+EKFRescue-10K",
            "PF+RobustClear+EKFGuide-10K",
            "PF+RobustClear+EKFGuideInit-10K",
            "PF+RobustClear+EKFGuideFallback-10K",
            "PF+RobustClear+EKFRescue-10K",
        )
    )
    if need_ekf:
        ekf_pos, ekf_ms = run_ekf(data, wls_pos)
        cached["EKF"] = (ekf_pos, ekf_ms, "EKF")
    rescue_reference_positions = cached.get("EKF", (None, None, None))[0]
    has_multi_constellations = len(tuple(data.get("constellations", ()))) > 1
    if "PF-10K" in methods:
        pf_pos, pf_ms, pf_backend = run_pf_standard(
            data,
            n_particles,
            wls_pos,
            sigma_pr=PF_SIGMA_LOS,
        )
        cached["PF-10K"] = (pf_pos, pf_ms, pf_backend)
    if "PF+EKFGuide-10K" in methods:
        pf_guide_pos, pf_guide_ms, pf_guide_backend = run_pf_standard(
            data,
            n_particles,
            wls_pos,
            sigma_pr=PF_SIGMA_LOS,
            quality_veto_config=quality_veto_config,
            guide_reference_positions=rescue_reference_positions,
            guide_initial_from_reference=True,
        )
        cached["PF+EKFGuide-10K"] = (pf_guide_pos, pf_guide_ms, pf_guide_backend)
    if "PF+EKFGuideInit-10K" in methods:
        pf_guide_init_pos, pf_guide_init_ms, pf_guide_init_backend = run_pf_standard(
            data,
            n_particles,
            wls_pos,
            sigma_pr=PF_SIGMA_LOS,
            quality_veto_config=quality_veto_config,
            guide_reference_positions=rescue_reference_positions,
            guide_initial_from_reference=True,
            guide_mode="init_only",
        )
        cached["PF+EKFGuideInit-10K"] = (
            pf_guide_init_pos,
            pf_guide_init_ms,
            pf_guide_init_backend,
        )
    if "PF+EKFGuideFallback-10K" in methods:
        pf_guide_fb_pos, pf_guide_fb_ms, pf_guide_fb_backend = run_pf_standard(
            data,
            n_particles,
            wls_pos,
            sigma_pr=PF_SIGMA_LOS,
            quality_veto_config=quality_veto_config,
            guide_reference_positions=rescue_reference_positions,
            guide_initial_from_reference=True,
            guide_mode="fallback_only",
        )
        cached["PF+EKFGuideFallback-10K"] = (
            pf_guide_fb_pos,
            pf_guide_fb_ms,
            pf_guide_fb_backend,
        )
    if "PF+AdaptiveGuide-10K" in methods:
        if has_multi_constellations:
            adaptive_pos, adaptive_ms, adaptive_backend = run_pf3d_variant(
                data,
                empty_model,
                n_particles,
                wls_pos,
                "pf3d",
                sigma_pr=PF_SIGMA_LOS,
                sigma_los=PF_SIGMA_LOS,
                sigma_nlos=PF_SIGMA_LOS * 10.0,
                nlos_bias=20.0,
                blocked_nlos_prob=0.0,
                clear_nlos_prob=clear_nlos_prob,
                quality_veto_config=quality_veto_config,
                guide_reference_positions=rescue_reference_positions,
                guide_initial_from_reference=True,
                guide_mode="init_only",
            )
        else:
            adaptive_pos, adaptive_ms, adaptive_backend = run_pf_standard(
                data,
                n_particles,
                wls_pos,
                sigma_pr=PF_SIGMA_LOS,
                quality_veto_config=quality_veto_config,
                guide_reference_positions=rescue_reference_positions,
                guide_initial_from_reference=True,
                guide_mode="always",
            )
        cached["PF+AdaptiveGuide-10K"] = (
            adaptive_pos,
            adaptive_ms,
            f"{adaptive_backend}+Adaptive",
        )
    if "PF+EKFRescue-10K" in methods:
        pf_rescue_pos, pf_rescue_ms, pf_rescue_backend = run_pf_standard(
            data,
            n_particles,
            wls_pos,
            sigma_pr=PF_SIGMA_LOS,
            quality_veto_config=quality_veto_config,
            rescue_reference_positions=rescue_reference_positions,
            rescue_distance_m=pf_rescue_distance_m,
        )
        cached["PF+EKFRescue-10K"] = (pf_rescue_pos, pf_rescue_ms, pf_rescue_backend)
    if "PF+RobustClear-10K" in methods:
        robust_pos, robust_ms, robust_backend = run_pf3d_variant(
            data,
            empty_model,
            n_particles,
            wls_pos,
            "pf3d",
            sigma_pr=PF_SIGMA_LOS,
            sigma_los=PF_SIGMA_LOS,
            sigma_nlos=PF_SIGMA_LOS * 10.0,
            nlos_bias=20.0,
            blocked_nlos_prob=0.0,
            clear_nlos_prob=clear_nlos_prob,
        )
        cached["PF+RobustClear-10K"] = (robust_pos, robust_ms, robust_backend)
    if "PF+RobustClear+EKFGuide-10K" in methods:
        robust_guide_pos, robust_guide_ms, robust_guide_backend = run_pf3d_variant(
            data,
            empty_model,
            n_particles,
            wls_pos,
            "pf3d",
            sigma_pr=PF_SIGMA_LOS,
            sigma_los=PF_SIGMA_LOS,
            sigma_nlos=PF_SIGMA_LOS * 10.0,
            nlos_bias=20.0,
            blocked_nlos_prob=0.0,
            clear_nlos_prob=clear_nlos_prob,
            quality_veto_config=quality_veto_config,
            guide_reference_positions=rescue_reference_positions,
            guide_initial_from_reference=True,
        )
        cached["PF+RobustClear+EKFGuide-10K"] = (
            robust_guide_pos,
            robust_guide_ms,
            robust_guide_backend,
        )
    if "PF+RobustClear+EKFGuideInit-10K" in methods:
        robust_guide_init_pos, robust_guide_init_ms, robust_guide_init_backend = run_pf3d_variant(
            data,
            empty_model,
            n_particles,
            wls_pos,
            "pf3d",
            sigma_pr=PF_SIGMA_LOS,
            sigma_los=PF_SIGMA_LOS,
            sigma_nlos=PF_SIGMA_LOS * 10.0,
            nlos_bias=20.0,
            blocked_nlos_prob=0.0,
            clear_nlos_prob=clear_nlos_prob,
            quality_veto_config=quality_veto_config,
            guide_reference_positions=rescue_reference_positions,
            guide_initial_from_reference=True,
            guide_mode="init_only",
        )
        cached["PF+RobustClear+EKFGuideInit-10K"] = (
            robust_guide_init_pos,
            robust_guide_init_ms,
            robust_guide_init_backend,
        )
    if "PF+RobustClear+EKFGuideFallback-10K" in methods:
        robust_guide_fb_pos, robust_guide_fb_ms, robust_guide_fb_backend = run_pf3d_variant(
            data,
            empty_model,
            n_particles,
            wls_pos,
            "pf3d",
            sigma_pr=PF_SIGMA_LOS,
            sigma_los=PF_SIGMA_LOS,
            sigma_nlos=PF_SIGMA_LOS * 10.0,
            nlos_bias=20.0,
            blocked_nlos_prob=0.0,
            clear_nlos_prob=clear_nlos_prob,
            quality_veto_config=quality_veto_config,
            guide_reference_positions=rescue_reference_positions,
            guide_initial_from_reference=True,
            guide_mode="fallback_only",
        )
        cached["PF+RobustClear+EKFGuideFallback-10K"] = (
            robust_guide_fb_pos,
            robust_guide_fb_ms,
            robust_guide_fb_backend,
        )
    if "PF+RobustClear+EKFRescue-10K" in methods:
        robust_rescue_pos, robust_rescue_ms, robust_rescue_backend = run_pf3d_variant(
            data,
            empty_model,
            n_particles,
            wls_pos,
            "pf3d",
            sigma_pr=PF_SIGMA_LOS,
            sigma_los=PF_SIGMA_LOS,
            sigma_nlos=PF_SIGMA_LOS * 10.0,
            nlos_bias=20.0,
            blocked_nlos_prob=0.0,
            clear_nlos_prob=clear_nlos_prob,
            quality_veto_config=quality_veto_config,
            rescue_reference_positions=rescue_reference_positions,
            rescue_distance_m=pf_rescue_distance_m,
        )
        cached["PF+RobustClear+EKFRescue-10K"] = (
            robust_rescue_pos,
            robust_rescue_ms,
            robust_rescue_backend,
        )

    run_rows: list[dict[str, object]] = []
    epoch_rows: list[dict[str, object]] = []
    for method_name in methods:
        positions, time_ms, backend = cached[method_name]
        metrics = _augment_tail_metrics(compute_metrics(positions, ground_truth), times)
        run_rows.append(_metrics_row(run_name, method_name, metrics, time_ms, backend))
        epoch_rows.extend(
            _epoch_rows(
                run_name,
                method_name,
                np.asarray(positions, dtype=np.float64),
                ground_truth,
                times,
                satellite_counts,
                backend,
            )
        )
        print(
            f"    {method_name}: RMS 2D={metrics['rms_2d']:.2f} m, "
            f"P95={metrics['p95']:.2f} m, >100m={metrics['outlier_rate_pct']:.2f}%"
        )
    return run_rows, epoch_rows


def _build_subprocess_cmd(
    args: argparse.Namespace,
    run_name: str,
    method_name: str,
    results_prefix: str,
) -> list[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--data-root",
        str(args.data_root),
        "--runs",
        run_name,
        "--systems",
        args.systems,
        "--urban-rover",
        args.urban_rover,
        "--start-epoch",
        str(args.start_epoch),
        "--n-particles",
        str(args.n_particles),
        "--clear-nlos-prob",
        str(args.clear_nlos_prob),
        "--quality-veto-residual-p95-max",
        str(args.quality_veto_residual_p95_max),
        "--quality-veto-residual-max",
        str(args.quality_veto_residual_max),
        "--quality-veto-bias-delta-max",
        str(args.quality_veto_bias_delta_max),
        "--quality-veto-extra-sat-min",
        str(args.quality_veto_extra_sat_min),
        "--pf-rescue-distance",
        str(args.pf_rescue_distance),
        "--results-prefix",
        results_prefix,
        "--methods",
        method_name,
    ]
    if args.max_epochs is not None:
        cmd.extend(["--max-epochs", str(args.max_epochs)])
    if args.save_epoch_errors:
        cmd.append("--save-epoch-errors")
    return cmd


def _run_isolated(
    args: argparse.Namespace,
    runs: list[str],
    methods: tuple[str, ...],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    combined_rows: list[dict[str, object]] = []
    combined_epoch_rows: list[dict[str, object]] = []
    for run_name in runs:
        for method_name in methods:
            temp_prefix = f"{args.results_prefix}__{run_name.lower()}__{_method_slug(method_name)}"
            print(f"\n[{run_name} / {method_name}] Isolated evaluation ...")
            cmd = _build_subprocess_cmd(args, run_name, method_name, temp_prefix)
            subprocess.run(cmd, check=True)
            run_path = RESULTS_DIR / f"{temp_prefix}_runs.csv"
            combined_rows.extend(_read_rows(run_path))
            if args.save_epoch_errors:
                epoch_path = RESULTS_DIR / f"{temp_prefix}_epochs.csv"
                combined_epoch_rows.extend(_read_rows(epoch_path))
    return combined_rows, combined_epoch_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed UrbanNav evaluation table")
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Directory containing UrbanNav run subdirectories such as Odaiba and Shinjuku",
    )
    parser.add_argument(
        "--runs",
        type=str,
        default="Odaiba,Shinjuku",
        help="Comma-separated UrbanNav runs to evaluate",
    )
    parser.add_argument("--systems", type=str, default="G", help="Comma-separated constellations")
    parser.add_argument("--urban-rover", type=str, default="ublox", help="UrbanNav rover source")
    parser.add_argument("--max-epochs", type=int, default=None, help="Optional epoch cap per run")
    parser.add_argument("--start-epoch", type=int, default=0, help="Skip usable epochs before evaluation")
    parser.add_argument("--n-particles", type=int, default=10_000, help="Particle count for PF variants")
    parser.add_argument(
        "--clear-nlos-prob",
        type=float,
        default=0.01,
        help="Fixed robust-clear mixture probability taken from PPC design phase",
    )
    parser.add_argument(
        "--quality-veto-residual-p95-max",
        type=float,
        default=100.0,
        help="Accept multi-GNSS only if epoch residual P95 stays below this threshold [m]",
    )
    parser.add_argument(
        "--quality-veto-residual-max",
        type=float,
        default=250.0,
        help="Accept multi-GNSS only if max absolute residual stays below this threshold [m]",
    )
    parser.add_argument(
        "--quality-veto-bias-delta-max",
        type=float,
        default=100.0,
        help="Accept multi-GNSS only if non-GPS clock offsets remain within this range [m]",
    )
    parser.add_argument(
        "--quality-veto-extra-sat-min",
        type=int,
        default=2,
        help="Accept multi-GNSS only if it adds at least this many satellites over GPS-only",
    )
    parser.add_argument(
        "--pf-rescue-distance",
        type=float,
        default=PF_RESCUE_DISTANCE_M,
        help="Recenter PF on EKF anchor when the PF/EKF gap exceeds this threshold [m]",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default=",".join(ALL_METHODS),
        help="Comma-separated subset of methods to run",
    )
    parser.add_argument(
        "--isolate-methods",
        action="store_true",
        help="Run each (run, method) in a fresh subprocess to avoid long-lived CUDA allocations",
    )
    parser.add_argument(
        "--save-epoch-errors",
        action="store_true",
        help="Save per-epoch positions and error diagnostics to <prefix>_epochs.csv",
    )
    parser.add_argument(
        "--results-prefix",
        type=str,
        default="urbannav_fixed_eval",
        help="Output filename prefix under experiments/results/",
    )
    args = parser.parse_args()

    runs = [part.strip() for part in args.runs.split(",") if part.strip()]
    methods = _parse_methods(args.methods)
    systems = tuple(part.strip().upper() for part in args.systems.split(",") if part.strip())
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  UrbanNav Fixed Evaluation")
    print("=" * 72)
    print(f"Runs: {', '.join(runs)}")
    print(f"Systems: {', '.join(systems)}")
    print(f"Urban rover: {args.urban_rover}")
    print(f"Methods: {', '.join(methods)}")

    quality_veto_config = MultiGNSSQualityVetoConfig(
        residual_p95_max_m=args.quality_veto_residual_p95_max,
        residual_max_abs_m=args.quality_veto_residual_max,
        bias_delta_max_m=args.quality_veto_bias_delta_max,
        extra_satellite_min=args.quality_veto_extra_sat_min,
    )

    if args.isolate_methods and len(runs) * len(methods) > 1:
        run_rows, epoch_rows = _run_isolated(args, runs, methods)
    else:
        run_rows: list[dict[str, object]] = []
        epoch_rows: list[dict[str, object]] = []
        for run_name in runs:
            run_dir = args.data_root / run_name
            print(f"\n[{run_name}] Loading data ...")
            data = load_or_generate_data(
                run_dir,
                max_real_epochs=args.max_epochs,
                start_epoch=args.start_epoch,
                systems=systems,
                urban_rover=args.urban_rover,
            )
            method_run_rows, method_epoch_rows = _evaluate_methods_for_run(
                run_name,
                data,
                methods,
                args.n_particles,
                args.clear_nlos_prob,
                quality_veto_config,
                args.pf_rescue_distance,
            )
            run_rows.extend(method_run_rows)
            epoch_rows.extend(method_epoch_rows)

    if not run_rows:
        raise RuntimeError("no evaluation rows were produced")

    summary_rows = _build_summary_rows(run_rows)
    runs_path = RESULTS_DIR / f"{args.results_prefix}_runs.csv"
    summary_path = RESULTS_DIR / f"{args.results_prefix}_summary.csv"
    _save_rows(run_rows, runs_path)
    _save_rows(summary_rows, summary_path)
    if args.save_epoch_errors and epoch_rows:
        epochs_path = RESULTS_DIR / f"{args.results_prefix}_epochs.csv"
        _save_rows(epoch_rows, epochs_path)
        print(f"Saved epoch-level diagnostics to: {epochs_path}")

    print(f"\nSaved run-level results to: {runs_path}")
    print(f"Saved aggregate summary to: {summary_path}")


if __name__ == "__main__":
    main()
