#!/usr/bin/env python3
"""Run PF-family ablations on the six positive PPC real-PLATEAU segments."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "python"))

RESULTS_DIR = _SCRIPT_DIR / "results"

from evaluate import compute_metrics, ecef_to_lla, save_results
from exp_urbannav_baseline import load_or_generate_data, run_ekf, run_wls
from exp_urbannav_pf3d import (
    PF_SIGMA_LOS,
    _augment_tail_metrics,
    load_plateau_model,
    run_pf3d_variant,
    run_pf_standard,
)
from fetch_plateau_subset import PRESET_URLS, expand_meshes, mesh3_code
from gnss_gpu.raytrace import BuildingModel
from scan_ppc_plateau_segments import ensure_subset


@dataclass(frozen=True)
class SegmentSpec:
    city: str
    run: str
    start_epoch: int
    nlos_fraction: float
    subset_key: str
    plateau_zone: int


POSITIVE_SEGMENTS = (
    SegmentSpec("tokyo", "run1", 1463, 0.023529, "a5844cfc41fbda08", 9),
    SegmentSpec("tokyo", "run2", 808, 0.064935, "5dc816ee3fc0ae64", 9),
    SegmentSpec("tokyo", "run3", 774, 0.013605, "436910dd3dca5c2a", 9),
    SegmentSpec("nagoya", "run1", 0, 0.006711, "938b44fb6b226b9f", 7),
    SegmentSpec("nagoya", "run2", 983, 0.066667, "63e337fab2e5f33f", 7),
    SegmentSpec("nagoya", "run3", 235, 0.019481, "166238c5838382f5", 7),
)


def _run_dir(data_root: Path, spec: SegmentSpec) -> Path:
    return data_root / spec.city / spec.run


def _model_dir(cache_root: Path, spec: SegmentSpec) -> Path:
    return cache_root / spec.subset_key


def _empty_building_model() -> BuildingModel:
    return BuildingModel(np.zeros((0, 3, 3), dtype=np.float64))


def _subset_url_for_city(city: str) -> str:
    if city == "tokyo":
        return PRESET_URLS["tokyo23"]
    if city == "nagoya":
        return PRESET_URLS["nagoya"]
    raise ValueError(f"unknown city: {city}")


def _derive_segment_meshes(ground_truth_ecef: np.ndarray, mesh_radius: int = 1) -> list[str]:
    meshes: set[str] = set()
    for pos in np.asarray(ground_truth_ecef, dtype=np.float64):
        lat_rad, lon_rad, _ = ecef_to_lla(float(pos[0]), float(pos[1]), float(pos[2]))
        meshes.add(mesh3_code(np.degrees(lat_rad), np.degrees(lon_rad)))
    return expand_meshes(sorted(meshes), mesh_radius)


def _metrics_row(
    spec: SegmentSpec,
    method: str,
    metrics: dict,
    time_ms: float,
    backend: str,
) -> dict[str, object]:
    return {
        "city": spec.city,
        "run": spec.run,
        "segment_label": f"{spec.city}_{spec.run}_seg{spec.start_epoch}",
        "start_epoch": spec.start_epoch,
        "nlos_fraction": spec.nlos_fraction,
        "subset_key": spec.subset_key,
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


def main() -> None:
    parser = argparse.ArgumentParser(description="PF ablation sweep on positive PPC PLATEAU segments")
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
    parser.add_argument("--blocked-nlos-prob", type=float, default=0.05, help="P(NLOS | ray blocked)")
    parser.add_argument("--clear-nlos-prob", type=float, default=0.01, help="P(NLOS | ray clear)")
    parser.add_argument(
        "--results-prefix",
        type=str,
        default="ppc_pf_ablation_positive6",
        help="Output filename prefix under experiments/results/",
    )
    args = parser.parse_args()

    systems = tuple(part.strip().upper() for part in args.systems.split(",") if part.strip())
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  PPC PF Ablation Sweep")
    print("=" * 72)

    empty_model = _empty_building_model()
    run_rows: list[dict[str, object]] = []

    for spec in POSITIVE_SEGMENTS:
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
        robust_clear_pos, robust_clear_ms, robust_clear_backend = run_pf3d_variant(
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
            clear_nlos_prob=args.clear_nlos_prob,
        )
        blocked_only_pos, blocked_only_ms, blocked_only_backend = run_pf3d_variant(
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
            clear_nlos_prob=0.0,
        )
        full_mix_pos, full_mix_ms, full_mix_backend = run_pf3d_variant(
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
            clear_nlos_prob=args.clear_nlos_prob,
        )

        times = np.asarray(data["times"], dtype=np.float64)
        ground_truth = np.asarray(data["ground_truth"], dtype=np.float64)

        results = [
            ("WLS", wls_pos, wls_ms, "WLS"),
            ("EKF", ekf_pos, ekf_ms, "EKF"),
            ("PF", pf_pos, pf_ms, pf_backend),
            ("PF+RobustClear", robust_clear_pos, robust_clear_ms, robust_clear_backend),
            ("PF3D-BVH+BlockedOnly", blocked_only_pos, blocked_only_ms, blocked_only_backend),
            ("PF3D-BVH+FullMix", full_mix_pos, full_mix_ms, full_mix_backend),
        ]

        for method, positions, time_ms, backend in results:
            metrics = compute_metrics(positions, ground_truth)
            metrics = _augment_tail_metrics(metrics, times)
            run_rows.append(_metrics_row(spec, method, metrics, time_ms, backend))

    run_path = RESULTS_DIR / f"{args.results_prefix}_runs.csv"
    save_results({key: [row[key] for row in run_rows] for key in run_rows[0]}, run_path)

    methods = sorted({row["method"] for row in run_rows})
    config_rows: list[dict[str, object]] = []
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
                "mean_time_ms_per_epoch": float(np.mean([row["time_ms_per_epoch"] for row in rows])),
                "pf_rms_wins": int(
                    sum(
                        1
                        for row in rows
                        if row["rms_2d"]
                        < next(
                            ref["rms_2d"]
                            for ref in run_rows
                            if ref["segment_label"] == row["segment_label"] and ref["method"] == "PF"
                        )
                    )
                ),
                "pf_p95_wins": int(
                    sum(
                        1
                        for row in rows
                        if row["p95"]
                        < next(
                            ref["p95"]
                            for ref in run_rows
                            if ref["segment_label"] == row["segment_label"] and ref["method"] == "PF"
                        )
                    )
                ),
            }
        )

    config_path = RESULTS_DIR / f"{args.results_prefix}_configs.csv"
    save_results({key: [row[key] for row in config_rows] for key in config_rows[0]}, config_path)

    print(f"\nSaved run-wise results to: {run_path}")
    print(f"Saved config-wise summary to: {config_path}")


if __name__ == "__main__":
    main()
