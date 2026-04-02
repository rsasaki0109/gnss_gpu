#!/usr/bin/env python3
"""Sweep clear-mixture strength on blocked-rich PPC real-PLATEAU segments."""

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
from exp_urbannav_baseline import load_or_generate_data, run_ekf, run_wls
from exp_urbannav_pf3d import (
    PF_SIGMA_LOS,
    _augment_tail_metrics,
    load_plateau_model,
    run_pf3d_variant,
    run_pf_standard,
)
from scan_ppc_plateau_segments import ensure_subset


DEFAULT_SEGMENTS = ("tokyo/run2", "tokyo/run3")


def _format_prob(value: float) -> str:
    return f"{value:.6g}"


def _parse_segments(raw: str) -> tuple[str, ...]:
    parts = tuple(part.strip().lower() for part in raw.split(",") if part.strip())
    if not parts:
        raise ValueError("at least one segment must be provided")
    return parts


def _parse_clear_values(raw: str) -> tuple[float, ...]:
    values = []
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        value = float(token)
        if value < 0.0 or value > 1.0:
            raise ValueError(f"clear_nlos_prob must be in [0, 1], got {value}")
        values.append(value)
    if not values:
        raise ValueError("at least one clear_nlos_prob must be provided")
    return tuple(values)


def _select_segments(segment_names: tuple[str, ...]):
    wanted = set(segment_names)
    selected = []
    for spec in POSITIVE_SEGMENTS:
        name = f"{spec.city}/{spec.run}"
        if name in wanted:
            selected.append(spec)
    missing = sorted(wanted - {f"{spec.city}/{spec.run}" for spec in selected})
    if missing:
        raise ValueError(f"unknown segments requested: {', '.join(missing)}")
    return tuple(selected)


def _method_label(family: str, clear_nlos_prob: float) -> str:
    return f"{family}(clear={_format_prob(clear_nlos_prob)})"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep clear_nlos_prob on blocked-rich PPC PLATEAU segments"
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
    parser.add_argument(
        "--blocked-nlos-prob",
        type=float,
        default=0.05,
        help="P(NLOS | ray blocked) used for the blocked+clear sweep",
    )
    parser.add_argument(
        "--clear-values",
        type=str,
        default="0,0.001,0.002,0.005,0.01",
        help="Comma-separated clear_nlos_prob values to evaluate",
    )
    parser.add_argument(
        "--segments",
        type=str,
        default=",".join(DEFAULT_SEGMENTS),
        help="Comma-separated segment names like tokyo/run2,tokyo/run3",
    )
    parser.add_argument(
        "--results-prefix",
        type=str,
        default="ppc_pf_blocked_clear_sweep_tokyo23",
        help="Output filename prefix under experiments/results/",
    )
    args = parser.parse_args()

    systems = tuple(part.strip().upper() for part in args.systems.split(",") if part.strip())
    clear_values = _parse_clear_values(args.clear_values)
    segment_names = _parse_segments(args.segments)
    segments = _select_segments(segment_names)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  PPC PF Blocked/Clear Sweep")
    print("=" * 72)
    print(f"Segments: {', '.join(segment_names)}")
    print(f"Clear values: {', '.join(_format_prob(value) for value in clear_values)}")
    print(f"Blocked prior: {_format_prob(args.blocked_nlos_prob)}")

    empty_model = _empty_building_model()
    run_rows: list[dict[str, object]] = []

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
            model_dir = ensure_subset(
                _subset_url_for_city(spec.city),
                meshes,
                args.cache_root,
            )
        building_model = load_plateau_model(model_dir, zone=spec.plateau_zone)
        if building_model is None:
            raise RuntimeError(f"failed to load building model: {model_dir}")

        wls_pos, wls_ms = run_wls(data)
        ekf_pos, ekf_ms = run_ekf(data, wls_pos)
        pf_pos, pf_ms, pf_backend = run_pf_standard(
            data,
            args.n_particles,
            wls_pos,
            sigma_pr=PF_SIGMA_LOS,
        )

        times = np.asarray(data["times"], dtype=np.float64)
        ground_truth = np.asarray(data["ground_truth"], dtype=np.float64)

        baselines = (
            ("WLS", wls_pos, wls_ms, "WLS", np.nan),
            ("EKF", ekf_pos, ekf_ms, "EKF", np.nan),
            ("PF", pf_pos, pf_ms, pf_backend, 0.0),
        )
        for method, positions, time_ms, backend, clear_value in baselines:
            metrics = compute_metrics(positions, ground_truth)
            metrics = _augment_tail_metrics(metrics, times)
            row = _metrics_row(spec, method, metrics, time_ms, backend)
            row["family"] = method
            row["clear_nlos_prob"] = float(clear_value)
            row["blocked_nlos_prob"] = 0.0 if method != "PF" else np.nan
            run_rows.append(row)

        for clear_value in clear_values:
            robust_pos, robust_ms, robust_backend = run_pf3d_variant(
                data,
                empty_model,
                args.n_particles,
                wls_pos,
                "pf3d",
                sigma_pr=PF_SIGMA_LOS,
                sigma_los=PF_SIGMA_LOS,
                sigma_nlos=PF_SIGMA_LOS * 10.0,
                nlos_bias=20.0,
                blocked_nlos_prob=0.0,
                clear_nlos_prob=clear_value,
            )
            blocked_mix_pos, blocked_mix_ms, blocked_mix_backend = run_pf3d_variant(
                data,
                building_model,
                args.n_particles,
                wls_pos,
                "pf3d_bvh",
                sigma_pr=PF_SIGMA_LOS,
                sigma_los=PF_SIGMA_LOS,
                sigma_nlos=PF_SIGMA_LOS * 10.0,
                nlos_bias=20.0,
                blocked_nlos_prob=args.blocked_nlos_prob,
                clear_nlos_prob=clear_value,
            )

            variants = (
                (
                    _method_label("PF+RobustClear", clear_value),
                    "PF+RobustClear",
                    robust_pos,
                    robust_ms,
                    robust_backend,
                    clear_value,
                    0.0,
                ),
                (
                    _method_label("PF3D-BVH+BlockedClear", clear_value),
                    "PF3D-BVH+BlockedClear",
                    blocked_mix_pos,
                    blocked_mix_ms,
                    blocked_mix_backend,
                    clear_value,
                    args.blocked_nlos_prob,
                ),
            )

            for method, family, positions, time_ms, backend, clear_prob, blocked_prob in variants:
                metrics = compute_metrics(positions, ground_truth)
                metrics = _augment_tail_metrics(metrics, times)
                row = _metrics_row(spec, method, metrics, time_ms, backend)
                row["family"] = family
                row["clear_nlos_prob"] = float(clear_prob)
                row["blocked_nlos_prob"] = float(blocked_prob)
                run_rows.append(row)

    run_path = RESULTS_DIR / f"{args.results_prefix}_runs.csv"
    save_results({key: [row[key] for row in run_rows] for key in run_rows[0]}, run_path)

    pf_lookup = {
        row["segment_label"]: row
        for row in run_rows
        if row["method"] == "PF"
    }
    robust_lookup = {
        (row["segment_label"], float(row["clear_nlos_prob"])): row
        for row in run_rows
        if row["family"] == "PF+RobustClear"
    }

    config_rows: list[dict[str, object]] = []
    methods = sorted({row["method"] for row in run_rows})
    for method in methods:
        rows = [row for row in run_rows if row["method"] == method]
        family = rows[0]["family"]
        clear_value = rows[0]["clear_nlos_prob"]
        blocked_value = rows[0]["blocked_nlos_prob"]
        summary = {
            "method": method,
            "family": family,
            "clear_nlos_prob": clear_value,
            "blocked_nlos_prob": blocked_value,
            "n_segments": len(rows),
            "mean_rms_2d": float(np.mean([row["rms_2d"] for row in rows])),
            "mean_p95": float(np.mean([row["p95"] for row in rows])),
            "mean_outlier_rate_pct": float(np.mean([row["outlier_rate_pct"] for row in rows])),
            "mean_catastrophic_rate_pct": float(np.mean([row["catastrophic_rate_pct"] for row in rows])),
            "mean_time_ms_per_epoch": float(np.mean([row["time_ms_per_epoch"] for row in rows])),
            "pf_rms_wins": int(sum(row["rms_2d"] < pf_lookup[row["segment_label"]]["rms_2d"] for row in rows)),
            "pf_p95_wins": int(sum(row["p95"] < pf_lookup[row["segment_label"]]["p95"] for row in rows)),
        }
        if family == "PF3D-BVH+BlockedClear":
            summary["robust_clear_rms_wins"] = int(
                sum(
                    row["rms_2d"]
                    < robust_lookup[(row["segment_label"], float(row["clear_nlos_prob"]))]["rms_2d"]
                    for row in rows
                )
            )
            summary["robust_clear_p95_wins"] = int(
                sum(
                    row["p95"]
                    < robust_lookup[(row["segment_label"], float(row["clear_nlos_prob"]))]["p95"]
                    for row in rows
                )
            )
            summary["mean_delta_rms_vs_robust_clear"] = float(
                np.mean(
                    [
                        row["rms_2d"]
                        - robust_lookup[(row["segment_label"], float(row["clear_nlos_prob"]))]["rms_2d"]
                        for row in rows
                    ]
                )
            )
            summary["mean_delta_p95_vs_robust_clear"] = float(
                np.mean(
                    [
                        row["p95"]
                        - robust_lookup[(row["segment_label"], float(row["clear_nlos_prob"]))]["p95"]
                        for row in rows
                    ]
                )
            )
        else:
            summary["robust_clear_rms_wins"] = np.nan
            summary["robust_clear_p95_wins"] = np.nan
            summary["mean_delta_rms_vs_robust_clear"] = np.nan
            summary["mean_delta_p95_vs_robust_clear"] = np.nan
        config_rows.append(summary)

    config_path = RESULTS_DIR / f"{args.results_prefix}_configs.csv"
    save_results({key: [row[key] for row in config_rows] for key in config_rows[0]}, config_path)

    best_rows: list[dict[str, object]] = []
    for spec in segments:
        segment_label = f"{spec.city}_{spec.run}_seg{spec.start_epoch}"
        pf_row = pf_lookup[segment_label]
        robust_rows = [
            row
            for row in run_rows
            if row["segment_label"] == segment_label and row["family"] == "PF+RobustClear"
        ]
        blocked_rows = [
            row
            for row in run_rows
            if row["segment_label"] == segment_label and row["family"] == "PF3D-BVH+BlockedClear"
        ]
        best_rows.append(
            {
                "segment_label": segment_label,
                "pf_rms_2d": float(pf_row["rms_2d"]),
                "pf_p95": float(pf_row["p95"]),
                "best_robust_clear": min(robust_rows, key=lambda row: row["rms_2d"])["method"],
                "best_robust_clear_rms_2d": float(min(robust_rows, key=lambda row: row["rms_2d"])["rms_2d"]),
                "best_blocked_clear": min(blocked_rows, key=lambda row: row["rms_2d"])["method"],
                "best_blocked_clear_rms_2d": float(min(blocked_rows, key=lambda row: row["rms_2d"])["rms_2d"]),
                "best_blocked_clear_p95": float(min(blocked_rows, key=lambda row: row["rms_2d"])["p95"]),
            }
        )

    best_path = RESULTS_DIR / f"{args.results_prefix}_best.csv"
    save_results({key: [row[key] for row in best_rows] for key in best_rows[0]}, best_path)

    print(f"\nSaved run-wise results to: {run_path}")
    print(f"Saved config-wise summary to: {config_path}")
    print(f"Saved per-segment best summary to: {best_path}")


if __name__ == "__main__":
    main()
