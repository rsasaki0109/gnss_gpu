#!/usr/bin/env python3
"""Compare simple multi-GNSS WLS stabilization variants on UrbanNav common epochs."""

from __future__ import annotations

import argparse
import csv
import inspect
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
if str(_PROJECT_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "python"))

RESULTS_DIR = _SCRIPT_DIR / "results"

from evaluate import compute_metrics, save_results, wls_solve_py
from exp_urbannav_pf3d import _augment_tail_metrics
from exp_urbannav_baseline import run_ekf, run_wls
from gnss_gpu.io.urbannav import UrbanNavLoader
from gnss_gpu.multi_gnss import MultiGNSSSolver, SYSTEM_GPS

try:
    from gnss_gpu import wls_position as _gpu_wls
    _HAS_GPU_WLS = True
except ImportError:
    _HAS_GPU_WLS = False


@dataclass(frozen=True)
class EpochFeature:
    run: str
    epoch_index: int
    gps_time_s: float
    gps_satellite_count: int
    multi_satellite_count: int
    extra_satellite_count: int
    gps_position: np.ndarray
    multi_position: np.ndarray
    ground_truth: np.ndarray
    gps_residual_p95_abs_m: float
    gps_residual_max_abs_m: float
    multi_residual_p95_abs_m: float
    multi_residual_max_abs_m: float
    multi_bias_range_m: float
    solution_gap_2d_m: float


class VariantProtocol:
    name: str
    style: str

    def use_multi(self, feature: EpochFeature) -> bool:
        raise NotImplementedError

    def parameters(self) -> dict[str, float]:
        raise NotImplementedError


@dataclass(frozen=True)
class AlwaysGPSVariant(VariantProtocol):
    name: str = "gps_only"
    style: str = "constant"

    def use_multi(self, feature: EpochFeature) -> bool:
        del feature
        return False

    def parameters(self) -> dict[str, float]:
        return {}


@dataclass(frozen=True)
class AlwaysMultiVariant(VariantProtocol):
    name: str = "multi_raw"
    style: str = "constant"

    def use_multi(self, feature: EpochFeature) -> bool:
        del feature
        return True

    def parameters(self) -> dict[str, float]:
        return {}


@dataclass(frozen=True)
class ResidualBiasVetoVariant(VariantProtocol):
    residual_p95_max_m: float
    residual_max_abs_m: float
    bias_delta_max_m: float
    extra_sat_min: int

    @property
    def name(self) -> str:
        return "multi_residual_bias_veto"

    @property
    def style(self) -> str:
        return "hard-veto"

    def use_multi(self, feature: EpochFeature) -> bool:
        return (
            feature.extra_satellite_count >= self.extra_sat_min
            and feature.multi_residual_p95_abs_m <= self.residual_p95_max_m
            and feature.multi_residual_max_abs_m <= self.residual_max_abs_m
            and feature.multi_bias_range_m <= self.bias_delta_max_m
        )

    def parameters(self) -> dict[str, float]:
        return {
            "residual_p95_max_m": self.residual_p95_max_m,
            "residual_max_abs_m": self.residual_max_abs_m,
            "bias_delta_max_m": self.bias_delta_max_m,
            "extra_sat_min": float(self.extra_sat_min),
        }


@dataclass(frozen=True)
class ComparativeResidualVariant(VariantProtocol):
    residual_margin_m: float
    residual_max_abs_m: float
    bias_delta_max_m: float
    extra_sat_min: int

    @property
    def name(self) -> str:
        return "multi_comparative_residual"

    @property
    def style(self) -> str:
        return "comparative-veto"

    def use_multi(self, feature: EpochFeature) -> bool:
        return (
            feature.extra_satellite_count >= self.extra_sat_min
            and feature.multi_residual_max_abs_m <= self.residual_max_abs_m
            and feature.multi_bias_range_m <= self.bias_delta_max_m
            and feature.multi_residual_p95_abs_m
            <= feature.gps_residual_p95_abs_m + self.residual_margin_m
        )

    def parameters(self) -> dict[str, float]:
        return {
            "residual_margin_m": self.residual_margin_m,
            "residual_max_abs_m": self.residual_max_abs_m,
            "bias_delta_max_m": self.bias_delta_max_m,
            "extra_sat_min": float(self.extra_sat_min),
        }


@dataclass(frozen=True)
class SolutionGapVetoVariant(VariantProtocol):
    solution_gap_max_m: float
    bias_delta_max_m: float
    extra_sat_min: int

    @property
    def name(self) -> str:
        return "multi_solution_gap_veto"

    @property
    def style(self) -> str:
        return "disagreement-veto"

    def use_multi(self, feature: EpochFeature) -> bool:
        return (
            feature.extra_satellite_count >= self.extra_sat_min
            and feature.solution_gap_2d_m <= self.solution_gap_max_m
            and feature.multi_bias_range_m <= self.bias_delta_max_m
        )

    def parameters(self) -> dict[str, float]:
        return {
            "solution_gap_max_m": self.solution_gap_max_m,
            "bias_delta_max_m": self.bias_delta_max_m,
            "extra_sat_min": float(self.extra_sat_min),
        }


def _save_rows(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        return
    save_results({key: [row[key] for row in rows] for key in rows[0].keys()}, path)


def _readability_proxy(variant: VariantProtocol) -> tuple[int, int, float]:
    source = inspect.getsource(variant.__class__)
    loc = len([line for line in source.splitlines() if line.strip()])
    branch_count = sum(source.count(token) for token in (" if ", " and ", " or "))
    score = max(0.0, 100.0 - 1.5 * loc - 4.0 * branch_count)
    return loc, branch_count, score


def _extensibility_proxy(variant: VariantProtocol) -> tuple[int, float]:
    param_count = len(variant.parameters())
    score = 100.0 - 3.0 * param_count
    return param_count, max(0.0, score)


def _residual_stats(
    position: np.ndarray,
    sat_ecef: np.ndarray,
    pseudoranges: np.ndarray,
    system_ids: np.ndarray,
    bias_by_system: dict[int, float],
) -> tuple[float, float]:
    ranges = np.linalg.norm(sat_ecef - position.reshape(1, 3), axis=1)
    pred = np.array(
        [ranges[i] + float(bias_by_system.get(int(system_ids[i]), 0.0)) for i in range(len(ranges))],
        dtype=np.float64,
    )
    residual = pseudoranges - pred
    abs_residual = np.abs(residual)
    return float(np.percentile(abs_residual, 95)), float(np.max(abs_residual))


def _gps_epoch_solution(
    sat_ecef: np.ndarray,
    pseudoranges: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    if _HAS_GPU_WLS:
        result, _ = _gpu_wls(sat_ecef, pseudoranges, weights, 10, 1e-4)
        return np.asarray(result, dtype=np.float64)
    result, _ = wls_solve_py(sat_ecef, pseudoranges, weights)
    return np.asarray(result, dtype=np.float64)


def _common_epoch_features(
    run_name: str,
    data_gps: dict,
    data_multi: dict,
) -> list[EpochFeature]:
    solver = MultiGNSSSolver(
        systems=sorted({int(sid) for epoch_ids in data_multi["system_ids"] for sid in epoch_ids})
    )
    map_gps = {float(t): i for i, t in enumerate(data_gps["times"])}
    map_multi = {float(t): i for i, t in enumerate(data_multi["times"])}
    common_times = sorted(set(map_gps) & set(map_multi))
    features: list[EpochFeature] = []

    t0 = time.perf_counter()
    for epoch_index, gps_time_s in enumerate(common_times):
        i_gps = map_gps[gps_time_s]
        i_multi = map_multi[gps_time_s]

        sat_gps = np.asarray(data_gps["sat_ecef"][i_gps], dtype=np.float64)
        pr_gps = np.asarray(data_gps["pseudoranges"][i_gps], dtype=np.float64)
        w_gps = np.asarray(data_gps["weights"][i_gps], dtype=np.float64)
        sat_multi = np.asarray(data_multi["sat_ecef"][i_multi], dtype=np.float64)
        pr_multi = np.asarray(data_multi["pseudoranges"][i_multi], dtype=np.float64)
        w_multi = np.asarray(data_multi["weights"][i_multi], dtype=np.float64)
        sys_multi = np.asarray(data_multi["system_ids"][i_multi], dtype=np.int32)
        gt = np.asarray(data_gps["ground_truth"][i_gps], dtype=np.float64)

        gps_solution = _gps_epoch_solution(sat_gps, pr_gps, w_gps)
        multi_position, multi_biases, _ = solver.solve(sat_multi, pr_multi, sys_multi, w_multi)

        gps_bias = float(gps_solution[3])
        gps_res_p95, gps_res_max = _residual_stats(
            gps_solution[:3],
            sat_gps,
            pr_gps,
            np.full(len(pr_gps), SYSTEM_GPS, dtype=np.int32),
            {SYSTEM_GPS: gps_bias},
        )
        multi_res_p95, multi_res_max = _residual_stats(
            multi_position,
            sat_multi,
            pr_multi,
            sys_multi,
            multi_biases,
        )
        gps_cb = float(multi_biases.get(SYSTEM_GPS, gps_bias))
        used_systems = {int(sys_id) for sys_id in sys_multi if int(sys_id) != SYSTEM_GPS}
        if used_systems:
            multi_bias_range = float(
                max(abs(float(multi_biases.get(system_id, gps_cb)) - gps_cb) for system_id in used_systems)
            )
        else:
            multi_bias_range = 0.0

        solution_gap = float(np.linalg.norm(gps_solution[:2] - multi_position[:2]))
        features.append(
            EpochFeature(
                run=run_name,
                epoch_index=epoch_index,
                gps_time_s=float(gps_time_s),
                gps_satellite_count=len(sat_gps),
                multi_satellite_count=len(sat_multi),
                extra_satellite_count=len(sat_multi) - len(sat_gps),
                gps_position=np.asarray(gps_solution[:3], dtype=np.float64),
                multi_position=np.asarray(multi_position, dtype=np.float64),
                ground_truth=gt,
                gps_residual_p95_abs_m=gps_res_p95,
                gps_residual_max_abs_m=gps_res_max,
                multi_residual_p95_abs_m=multi_res_p95,
                multi_residual_max_abs_m=multi_res_max,
                multi_bias_range_m=multi_bias_range,
                solution_gap_2d_m=solution_gap,
            )
        )

    elapsed = (time.perf_counter() - t0) * 1000.0
    print(
        f"    Built common-epoch feature dump for {run_name}: "
        f"{len(features)} epochs, {elapsed / max(len(features), 1):.3f} ms/epoch"
    )
    return features


def _default_variants() -> list[VariantProtocol]:
    variants: list[VariantProtocol] = [AlwaysGPSVariant(), AlwaysMultiVariant()]
    for residual_p95_max_m in (80.0, 100.0, 120.0):
        for residual_max_abs_m in (250.0, 350.0):
            for bias_delta_max_m in (60.0, 100.0):
                variants.append(
                    ResidualBiasVetoVariant(
                        residual_p95_max_m=residual_p95_max_m,
                        residual_max_abs_m=residual_max_abs_m,
                        bias_delta_max_m=bias_delta_max_m,
                        extra_sat_min=2,
                    )
                )
    for residual_margin_m in (0.0, 20.0, 40.0):
        for residual_max_abs_m in (250.0, 350.0):
            for bias_delta_max_m in (60.0, 100.0):
                variants.append(
                    ComparativeResidualVariant(
                        residual_margin_m=residual_margin_m,
                        residual_max_abs_m=residual_max_abs_m,
                        bias_delta_max_m=bias_delta_max_m,
                        extra_sat_min=2,
                    )
                )
    for solution_gap_max_m in (40.0, 80.0, 120.0):
        for bias_delta_max_m in (60.0, 100.0):
            variants.append(
                SolutionGapVetoVariant(
                    solution_gap_max_m=solution_gap_max_m,
                    bias_delta_max_m=bias_delta_max_m,
                    extra_sat_min=2,
                )
            )
    return variants


def _evaluate_variant(variant: VariantProtocol, features_by_run: dict[str, list[EpochFeature]]) -> tuple[list[dict[str, object]], dict[str, object]]:
    run_rows: list[dict[str, object]] = []
    for run_name, features in features_by_run.items():
        chosen = np.vstack(
            [feat.multi_position if variant.use_multi(feat) else feat.gps_position for feat in features]
        )
        truth = np.vstack([feat.ground_truth for feat in features])
        times = np.array([feat.gps_time_s for feat in features], dtype=np.float64)
        metrics = _augment_tail_metrics(compute_metrics(chosen, truth), times)
        use_multi_frac = float(np.mean([variant.use_multi(feat) for feat in features]))
        run_rows.append(
            {
                "variant": variant.name,
                "style": variant.style,
                "parameters": ",".join(f"{k}={v:g}" for k, v in variant.parameters().items()),
                "run": run_name,
                "n_epochs": len(features),
                "use_multi_frac": use_multi_frac,
                "rms_2d": float(metrics["rms_2d"]),
                "p95": float(metrics["p95"]),
                "outlier_rate_pct": float(metrics["outlier_rate_pct"]),
                "catastrophic_rate_pct": float(metrics["catastrophic_rate_pct"]),
                "longest_outlier_segment_epochs": float(metrics["longest_outlier_segment_epochs"]),
                "longest_outlier_segment_s": float(metrics["longest_outlier_segment_s"]),
            }
        )

    loc, branch_count, readability = _readability_proxy(variant)
    param_count, extensibility = _extensibility_proxy(variant)
    summary = {
        "variant": variant.name,
        "style": variant.style,
        "parameters": ",".join(f"{k}={v:g}" for k, v in variant.parameters().items()),
        "n_runs": len(run_rows),
        "mean_rms_2d": float(np.mean([row["rms_2d"] for row in run_rows])),
        "mean_p95": float(np.mean([row["p95"] for row in run_rows])),
        "mean_outlier_rate_pct": float(np.mean([row["outlier_rate_pct"] for row in run_rows])),
        "mean_catastrophic_rate_pct": float(np.mean([row["catastrophic_rate_pct"] for row in run_rows])),
        "mean_longest_outlier_segment_s": float(
            np.mean([row["longest_outlier_segment_s"] for row in run_rows])
        ),
        "mean_use_multi_frac": float(np.mean([row["use_multi_frac"] for row in run_rows])),
        "readability_loc": float(loc),
        "readability_branch_count": float(branch_count),
        "readability_proxy": float(readability),
        "extensibility_param_count": float(param_count),
        "extensibility_proxy": float(extensibility),
    }
    return run_rows, summary


def _evaluate_ekf_reference(
    features_by_run: dict[str, list[EpochFeature]],
    gps_data_by_run: dict[str, dict],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    run_rows: list[dict[str, object]] = []
    for run_name, features in features_by_run.items():
        gps_data = gps_data_by_run[run_name]
        common_times = np.array([feat.gps_time_s for feat in features], dtype=np.float64)
        time_to_index = {float(t): i for i, t in enumerate(gps_data["times"])}
        indices = [time_to_index[float(t)] for t in common_times]
        wls_g, _ = run_wls(gps_data)
        ekf_g, _ = run_ekf(gps_data, wls_g)
        truth = np.asarray(gps_data["ground_truth"], dtype=np.float64)[indices]
        ekf_common = np.asarray(ekf_g, dtype=np.float64)[indices]
        metrics = _augment_tail_metrics(compute_metrics(ekf_common, truth), common_times)
        run_rows.append(
            {
                "variant": "gps_ekf_reference",
                "style": "reference",
                "parameters": "",
                "run": run_name,
                "n_epochs": len(features),
                "use_multi_frac": 0.0,
                "rms_2d": float(metrics["rms_2d"]),
                "p95": float(metrics["p95"]),
                "outlier_rate_pct": float(metrics["outlier_rate_pct"]),
                "catastrophic_rate_pct": float(metrics["catastrophic_rate_pct"]),
                "longest_outlier_segment_epochs": float(metrics["longest_outlier_segment_epochs"]),
                "longest_outlier_segment_s": float(metrics["longest_outlier_segment_s"]),
            }
        )

    return run_rows, {
        "variant": "gps_ekf_reference",
        "style": "reference",
        "parameters": "",
        "n_runs": len(run_rows),
        "mean_rms_2d": float(np.mean([row["rms_2d"] for row in run_rows])),
        "mean_p95": float(np.mean([row["p95"] for row in run_rows])),
        "mean_outlier_rate_pct": float(np.mean([row["outlier_rate_pct"] for row in run_rows])),
        "mean_catastrophic_rate_pct": float(np.mean([row["catastrophic_rate_pct"] for row in run_rows])),
        "mean_longest_outlier_segment_s": float(
            np.mean([row["longest_outlier_segment_s"] for row in run_rows])
        ),
        "mean_use_multi_frac": 0.0,
        "readability_loc": 0.0,
        "readability_branch_count": 0.0,
        "readability_proxy": 0.0,
        "extensibility_param_count": 0.0,
        "extensibility_proxy": 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="UrbanNav multi-GNSS WLS stabilization lab")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--runs", type=str, default="Odaiba,Shinjuku")
    parser.add_argument("--urban-rover", type=str, default="trimble")
    parser.add_argument("--gps-systems", type=str, default="G")
    parser.add_argument("--multi-systems", type=str, default="G,E,J")
    parser.add_argument(
        "--results-prefix",
        type=str,
        default="urbannav_multignss_stabilization",
    )
    args = parser.parse_args()

    runs = [part.strip() for part in args.runs.split(",") if part.strip()]
    gps_systems = tuple(part.strip().upper() for part in args.gps_systems.split(",") if part.strip())
    multi_systems = tuple(part.strip().upper() for part in args.multi_systems.split(",") if part.strip())
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  UrbanNav Multi-GNSS WLS Stabilization Lab")
    print("=" * 72)
    print(f"Runs: {', '.join(runs)}")
    print(f"Urban rover: {args.urban_rover}")
    print(f"GPS systems: {', '.join(gps_systems)}")
    print(f"Multi systems: {', '.join(multi_systems)}")

    features_by_run: dict[str, list[EpochFeature]] = {}
    gps_data_by_run: dict[str, dict] = {}
    epoch_rows: list[dict[str, object]] = []
    for run_name in runs:
        print(f"\n[{run_name}] Loading common-epoch data ...")
        loader = UrbanNavLoader(args.data_root / run_name)
        data_gps = loader.load_experiment_data(systems=gps_systems, rover_source=args.urban_rover)
        data_multi = loader.load_experiment_data(systems=multi_systems, rover_source=args.urban_rover)
        gps_data_by_run[run_name] = data_gps
        features = _common_epoch_features(run_name, data_gps, data_multi)
        features_by_run[run_name] = features
        for feat in features:
            epoch_rows.append(
                {
                    "run": feat.run,
                    "epoch_index": feat.epoch_index,
                    "gps_time_s": feat.gps_time_s,
                    "gps_satellite_count": feat.gps_satellite_count,
                    "multi_satellite_count": feat.multi_satellite_count,
                    "extra_satellite_count": feat.extra_satellite_count,
                    "gps_residual_p95_abs_m": feat.gps_residual_p95_abs_m,
                    "gps_residual_max_abs_m": feat.gps_residual_max_abs_m,
                    "multi_residual_p95_abs_m": feat.multi_residual_p95_abs_m,
                    "multi_residual_max_abs_m": feat.multi_residual_max_abs_m,
                    "multi_bias_range_m": feat.multi_bias_range_m,
                    "solution_gap_2d_m": feat.solution_gap_2d_m,
                }
            )

    run_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    ekf_run_rows, ekf_summary = _evaluate_ekf_reference(features_by_run, gps_data_by_run)
    run_rows.extend(ekf_run_rows)
    summary_rows.append(ekf_summary)
    for variant in _default_variants():
        variant_run_rows, summary = _evaluate_variant(variant, features_by_run)
        run_rows.extend(variant_run_rows)
        summary_rows.append(summary)

    summary_rows.sort(
        key=lambda row: (
            float(row["mean_catastrophic_rate_pct"]),
            float(row["mean_p95"]),
            float(row["mean_rms_2d"]),
        )
    )
    best_rows = summary_rows[:5]

    features_path = RESULTS_DIR / f"{args.results_prefix}_features.csv"
    runs_path = RESULTS_DIR / f"{args.results_prefix}_runs.csv"
    summary_path = RESULTS_DIR / f"{args.results_prefix}_summary.csv"
    best_path = RESULTS_DIR / f"{args.results_prefix}_best.csv"
    _save_rows(epoch_rows, features_path)
    _save_rows(run_rows, runs_path)
    _save_rows(summary_rows, summary_path)
    _save_rows(best_rows, best_path)

    print(f"\nSaved epoch features to: {features_path}")
    print(f"Saved run summary to: {runs_path}")
    print(f"Saved variant summary to: {summary_path}")
    print(f"Saved best variants to: {best_path}")
    print("\nTop variants:")
    for row in best_rows:
        print(
            f"  {row['variant']} [{row['style']}] {row['parameters'] or '(default)'} "
            f"-> mean catastrophic {row['mean_catastrophic_rate_pct']:.3f}%, "
            f"mean P95 {row['mean_p95']:.2f} m, mean RMS {row['mean_rms_2d']:.2f} m"
        )


if __name__ == "__main__":
    main()
