#!/usr/bin/env python3
"""Cross-validate PF gate strategy families on tuned and holdout dumps."""

from __future__ import annotations

import argparse
import copy
import csv
import inspect
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_EXPERIMENTS_DIR = _SCRIPT_DIR.parent
_PROJECT_ROOT = _EXPERIMENTS_DIR.parent
if str(_EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_DIR))
if str(_PROJECT_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "python"))

from evaluate import compute_metrics, save_results
from exp_urbannav_pf3d import _augment_tail_metrics
from pf_strategy_lab.interfaces import StrategyContext
from pf_strategy_lab.strategies import (
    AlwaysBlockedStrategy,
    AlwaysRobustStrategy,
    BranchAwareHysteresisQualityVetoRegimeGateStrategy,
    ClockVetoGateStrategy,
    DisagreementGateStrategy,
    DualModeRegimeGateStrategy,
    EntryVetoNegativeExitRescueBranchAwareHysteresisQualityVetoRegimeGateStrategy,
    HysteresisQualityVetoRegimeGateStrategy,
    ModeAwareHysteresisQualityVetoRegimeGateStrategy,
    NegativeExitRescueBranchAwareHysteresisQualityVetoRegimeGateStrategy,
    QualityVetoRegimeGateStrategy,
    RescueBranchAwareHysteresisQualityVetoRegimeGateStrategy,
    RuleChainGateStrategy,
    WeightedScoreGateStrategy,
)


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def _epoch_key(value: str) -> int:
    return int(float(value))


def _save_rows(rows: list[dict[str, object]], path: Path) -> None:
    keys = sorted({key for row in rows for key in row})
    save_results({key: [row.get(key, np.nan) for row in rows] for key in keys}, path)


def _readability_proxy(strategy) -> tuple[int, int, float]:
    source = inspect.getsource(strategy.__class__)
    lines = [
        line
        for line in source.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    loc = len(lines)
    branch_count = sum(source.count(token) for token in (" if ", " and ", " or "))
    score = max(0.0, 100.0 - 1.5 * loc - 6.0 * branch_count - 4.0 * len(strategy.required_features()))
    return loc, branch_count, score


def _extensibility_proxy(strategy) -> tuple[int, float]:
    param_count = len(strategy.parameters())
    feature_count = len(strategy.required_features())
    score = max(0.0, 100.0 - 8.0 * feature_count + 6.0 * param_count)
    return param_count, score


def _load_dump(feature_csv: Path, trajectory_csv: Path) -> tuple[dict, dict]:
    feature_rows = _load_csv(feature_csv)
    trajectory_rows = _load_csv(trajectory_csv)

    feature_map = {
        (row["segment_label"], _epoch_key(row["epoch"])): {
            key: float(value)
            for key, value in row.items()
            if key not in {"city", "run", "segment_label", "start_epoch", "epoch"}
        }
        for row in feature_rows
    }

    trajectory_by_segment: dict[str, list[dict[str, str]]] = {}
    for row in trajectory_rows:
        trajectory_by_segment.setdefault(row["segment_label"], []).append(row)
    for rows in trajectory_by_segment.values():
        rows.sort(key=lambda row: _epoch_key(row["epoch"]))
    return feature_map, trajectory_by_segment


def _evaluate_strategy(strategy, feature_map: dict, trajectory_by_segment: dict[str, list[dict[str, str]]]):
    run_rows: list[dict[str, object]] = []
    for segment_label, segment_rows in trajectory_by_segment.items():
        segment_strategy = copy.deepcopy(strategy)
        chosen_positions = []
        truth_positions = []
        times = []
        blocked_flags = []

        for row in segment_rows:
            epoch = _epoch_key(row["epoch"])
            context = StrategyContext(
                segment_label=segment_label,
                epoch=epoch,
                features=feature_map[(segment_label, epoch)],
            )
            decision = segment_strategy.decide(context)
            blocked_flags.append(bool(decision.use_blocked))
            chosen_positions.append(
                [
                    float(row["blocked_x"]) if decision.use_blocked else float(row["robust_x"]),
                    float(row["blocked_y"]) if decision.use_blocked else float(row["robust_y"]),
                    float(row["blocked_z"]) if decision.use_blocked else float(row["robust_z"]),
                ]
            )
            truth_positions.append(
                [float(row["truth_x"]), float(row["truth_y"]), float(row["truth_z"])]
            )
            times.append(float(row["gps_tow"]))

        chosen = np.asarray(chosen_positions, dtype=np.float64)
        truth = np.asarray(truth_positions, dtype=np.float64)
        time_array = np.asarray(times, dtype=np.float64)
        metrics = compute_metrics(chosen, truth)
        metrics = _augment_tail_metrics(metrics, time_array)
        run_rows.append(
            {
                "segment_label": segment_label,
                "rms_2d": float(metrics["rms_2d"]),
                "p95": float(metrics["p95"]),
                "outlier_rate_pct": float(metrics["outlier_rate_pct"]),
                "blocked_epoch_frac": float(np.mean(blocked_flags)),
            }
        )
    return run_rows


def _mean(rows: list[dict[str, object]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def _family_members():
    for strategy in (AlwaysRobustStrategy(), AlwaysBlockedStrategy()):
        yield strategy.name, strategy

    for disagreement_threshold_m in (40.0, 60.0, 80.0, 100.0):
        strategy = DisagreementGateStrategy(disagreement_threshold_m=disagreement_threshold_m)
        yield "disagreement_gate", strategy

    for disagreement_threshold_m in (30.0, 50.0, 70.0, 85.0):
        for cb_disagreement_threshold_m in (10.0, 20.0, 30.0):
            for blocked_ceiling in (0.02, 0.03, 0.05):
                strategy = ClockVetoGateStrategy(
                    disagreement_threshold_m=disagreement_threshold_m,
                    cb_disagreement_threshold_m=cb_disagreement_threshold_m,
                    blocked_ceiling=blocked_ceiling,
                )
                yield "clock_veto_gate", strategy

    for blocked_threshold in (0.001, 0.01, 0.05):
        for positive_threshold in (0.10, 0.25, 0.50):
            for disagreement_threshold_m in (20.0, 50.0, 80.0):
                strategy = RuleChainGateStrategy(
                    blocked_threshold=blocked_threshold,
                    positive_threshold=positive_threshold,
                    disagreement_threshold_m=disagreement_threshold_m,
                )
                yield "rule_chain_gate", strategy

    for blocked_scale in (0.02, 0.05, 0.10):
        for positive_scale in (0.25, 0.50, 0.75):
            for disagreement_scale_m in (40.0, 80.0, 120.0):
                for threshold in (1.20, 1.60, 2.00):
                    strategy = WeightedScoreGateStrategy(
                        blocked_scale=blocked_scale,
                        positive_scale=positive_scale,
                        disagreement_scale_m=disagreement_scale_m,
                        threshold=threshold,
                    )
                    yield "weighted_score_gate", strategy

    for close_blocked_low in (0.10, 0.15):
        for close_blocked_high in (0.35, 0.50):
            if close_blocked_high <= close_blocked_low:
                continue
            for close_disagreement_max_m in (30.0, 40.0):
                for close_cb_max_m in (15.0, 20.0):
                    for close_residual_max_m in (18.0, 22.0):
                        for far_blocked_max in (0.005, 0.01):
                            for far_positive_min in (0.15, 0.20):
                                for far_disagreement_min_m in (80.0, 90.0):
                                    for far_cb_min_m in (45.0, 55.0):
                                        strategy = DualModeRegimeGateStrategy(
                                            close_blocked_low=close_blocked_low,
                                            close_blocked_high=close_blocked_high,
                                            close_disagreement_max_m=close_disagreement_max_m,
                                            close_cb_max_m=close_cb_max_m,
                                            close_residual_max_m=close_residual_max_m,
                                            far_blocked_max=far_blocked_max,
                                            far_positive_min=far_positive_min,
                                            far_disagreement_min_m=far_disagreement_min_m,
                                            far_cb_min_m=far_cb_min_m,
                                        )
                                        yield "dual_mode_regime_gate", strategy

    for close_satellite_max in (7.0, 8.0, 9.0, 10.0):
        for close_p95_abs_residual_max_m in (35.0, 40.0, 45.0, 50.0, 55.0, 60.0):
            strategy = QualityVetoRegimeGateStrategy(
                close_blocked_low=0.10,
                close_blocked_high=0.50,
                close_disagreement_max_m=40.0,
                close_cb_max_m=20.0,
                close_residual_max_m=22.0,
                close_satellite_max=close_satellite_max,
                close_p95_abs_residual_max_m=close_p95_abs_residual_max_m,
                far_blocked_max=0.01,
                far_positive_min=0.15,
                far_disagreement_min_m=90.0,
                far_cb_min_m=45.0,
            )
            yield "quality_veto_regime_gate", strategy

    for close_satellite_max in (7.0, 8.0, 9.0):
        for close_p95_abs_residual_max_m in (45.0, 50.0, 55.0, 60.0):
            for enter_confirm_epochs in (1, 2, 3, 4):
                for exit_confirm_epochs in (1, 2, 3):
                    strategy = HysteresisQualityVetoRegimeGateStrategy(
                        close_blocked_low=0.10,
                        close_blocked_high=0.50,
                        close_disagreement_max_m=40.0,
                        close_cb_max_m=20.0,
                        close_residual_max_m=22.0,
                        close_satellite_max=close_satellite_max,
                        close_p95_abs_residual_max_m=close_p95_abs_residual_max_m,
                        far_blocked_max=0.01,
                        far_positive_min=0.15,
                        far_disagreement_min_m=90.0,
                        far_cb_min_m=45.0,
                        enter_confirm_epochs=enter_confirm_epochs,
                        exit_confirm_epochs=exit_confirm_epochs,
                    )
                    yield "hysteresis_quality_veto_regime_gate", strategy

    for close_satellite_max in (8.0, 9.0):
        for close_p95_abs_residual_max_m in (50.0, 55.0, 60.0):
            for enter_confirm_close_epochs in (2, 3, 4):
                for enter_confirm_far_epochs in (1, 2):
                    for exit_confirm_epochs in (2, 3, 4):
                        strategy = ModeAwareHysteresisQualityVetoRegimeGateStrategy(
                            close_blocked_low=0.10,
                            close_blocked_high=0.50,
                            close_disagreement_max_m=40.0,
                            close_cb_max_m=20.0,
                            close_residual_max_m=22.0,
                            close_satellite_max=close_satellite_max,
                            close_p95_abs_residual_max_m=close_p95_abs_residual_max_m,
                            far_blocked_max=0.01,
                            far_positive_min=0.15,
                            far_disagreement_min_m=90.0,
                            far_cb_min_m=45.0,
                            enter_confirm_close_epochs=enter_confirm_close_epochs,
                            enter_confirm_far_epochs=enter_confirm_far_epochs,
                            exit_confirm_epochs=exit_confirm_epochs,
                        )
                        yield "mode_aware_hysteresis_quality_veto_regime_gate", strategy

    for close_satellite_max in (8.0, 9.0):
        for close_p95_abs_residual_max_m in (50.0, 55.0, 60.0):
            for enter_confirm_close_epochs in (2, 3, 4):
                for enter_confirm_far_epochs in (1, 2):
                    for exit_confirm_close_epochs in (2, 3):
                        for exit_confirm_far_epochs in (3, 4, 5):
                            strategy = BranchAwareHysteresisQualityVetoRegimeGateStrategy(
                                close_blocked_low=0.10,
                                close_blocked_high=0.50,
                                close_disagreement_max_m=40.0,
                                close_cb_max_m=20.0,
                                close_residual_max_m=22.0,
                                close_satellite_max=close_satellite_max,
                                close_p95_abs_residual_max_m=close_p95_abs_residual_max_m,
                                far_blocked_max=0.01,
                                far_positive_min=0.15,
                                far_disagreement_min_m=90.0,
                                far_cb_min_m=45.0,
                                enter_confirm_close_epochs=enter_confirm_close_epochs,
                                enter_confirm_far_epochs=enter_confirm_far_epochs,
                                exit_confirm_close_epochs=exit_confirm_close_epochs,
                                exit_confirm_far_epochs=exit_confirm_far_epochs,
                            )
                            yield "branch_aware_hysteresis_quality_veto_regime_gate", strategy

    for close_satellite_max in (8.0, 9.0):
        for close_p95_abs_residual_max_m in (50.0, 55.0):
            for enter_confirm_close_epochs in (2, 3):
                for enter_confirm_far_epochs in (1, 2):
                    for exit_confirm_close_epochs in (2, 3):
                        for exit_confirm_far_epochs in (4, 5):
                            for close_rescue_satellite_max in (7.0, 8.0, 9.0):
                                for close_rescue_p95_abs_residual_max_m in (45.0, 50.0, 55.0):
                                    for close_rescue_cb_min_m in (16.0, 18.0, 20.0):
                                        strategy = RescueBranchAwareHysteresisQualityVetoRegimeGateStrategy(
                                            close_blocked_low=0.10,
                                            close_blocked_high=0.50,
                                            close_disagreement_max_m=40.0,
                                            close_cb_max_m=20.0,
                                            close_residual_max_m=22.0,
                                            close_satellite_max=close_satellite_max,
                                            close_p95_abs_residual_max_m=close_p95_abs_residual_max_m,
                                            far_blocked_max=0.01,
                                            far_positive_min=0.15,
                                            far_disagreement_min_m=90.0,
                                            far_cb_min_m=45.0,
                                            enter_confirm_close_epochs=enter_confirm_close_epochs,
                                            enter_confirm_far_epochs=enter_confirm_far_epochs,
                                            exit_confirm_close_epochs=exit_confirm_close_epochs,
                                            exit_confirm_far_epochs=exit_confirm_far_epochs,
                                            close_rescue_satellite_max=close_rescue_satellite_max,
                                            close_rescue_p95_abs_residual_max_m=close_rescue_p95_abs_residual_max_m,
                                            close_rescue_cb_min_m=close_rescue_cb_min_m,
                                        )
                                        yield "rescue_branch_aware_hysteresis_quality_veto_regime_gate", strategy

    for close_satellite_max in (9.0,):
        for close_p95_abs_residual_max_m in (55.0,):
            for enter_confirm_close_epochs in (3,):
                for enter_confirm_far_epochs in (1,):
                    for exit_confirm_close_epochs in (3,):
                        for exit_confirm_far_epochs in (5,):
                            for close_rescue_satellite_max in (8.0,):
                                for close_rescue_p95_abs_residual_max_m in (50.0,):
                                    for close_rescue_cb_min_m in (16.0, 18.0):
                                        for negative_exit_disagreement_min_m in (42.0, 45.0, 999.0):
                                            for negative_exit_cb_min_m in (25.0, 999.0):
                                                for negative_exit_p95_abs_residual_min_m in (50.0, 52.0, 55.0):
                                                    for negative_exit_hits_required in (1, 2):
                                                        strategy = NegativeExitRescueBranchAwareHysteresisQualityVetoRegimeGateStrategy(
                                                            close_blocked_low=0.10,
                                                            close_blocked_high=0.50,
                                                            close_disagreement_max_m=40.0,
                                                            close_cb_max_m=20.0,
                                                            close_residual_max_m=22.0,
                                                            close_satellite_max=close_satellite_max,
                                                            close_p95_abs_residual_max_m=close_p95_abs_residual_max_m,
                                                            far_blocked_max=0.01,
                                                            far_positive_min=0.15,
                                                            far_disagreement_min_m=90.0,
                                                            far_cb_min_m=45.0,
                                                            enter_confirm_close_epochs=enter_confirm_close_epochs,
                                                            enter_confirm_far_epochs=enter_confirm_far_epochs,
                                                            exit_confirm_close_epochs=exit_confirm_close_epochs,
                                                            exit_confirm_far_epochs=exit_confirm_far_epochs,
                                                            close_rescue_satellite_max=close_rescue_satellite_max,
                                                            close_rescue_p95_abs_residual_max_m=close_rescue_p95_abs_residual_max_m,
                                                            close_rescue_cb_min_m=close_rescue_cb_min_m,
                                                            negative_exit_disagreement_min_m=negative_exit_disagreement_min_m,
                                                            negative_exit_cb_min_m=negative_exit_cb_min_m,
                                                            negative_exit_p95_abs_residual_min_m=negative_exit_p95_abs_residual_min_m,
                                                            negative_exit_hits_required=negative_exit_hits_required,
                                                        )
                                                        yield "negative_exit_rescue_branch_aware_hysteresis_quality_veto_regime_gate", strategy

    for close_satellite_max in (9.0,):
        for close_p95_abs_residual_max_m in (55.0,):
            for enter_confirm_close_epochs in (3,):
                for enter_confirm_far_epochs in (1,):
                    for exit_confirm_close_epochs in (3,):
                        for exit_confirm_far_epochs in (5,):
                            for close_rescue_satellite_max in (8.0,):
                                for close_rescue_p95_abs_residual_max_m in (50.0,):
                                    for close_rescue_cb_min_m in (16.0,):
                                        for close_entry_p95_abs_residual_max_m in (50.0, 48.0, 45.0, 52.0, 55.0):
                                            for negative_exit_disagreement_min_m in (42.0, 45.0, 999.0):
                                                for negative_exit_cb_min_m in (25.0, 999.0):
                                                    for negative_exit_p95_abs_residual_min_m in (50.0, 52.0, 55.0):
                                                        for negative_exit_hits_required in (1, 2):
                                                            strategy = EntryVetoNegativeExitRescueBranchAwareHysteresisQualityVetoRegimeGateStrategy(
                                                                close_blocked_low=0.10,
                                                                close_blocked_high=0.50,
                                                                close_disagreement_max_m=40.0,
                                                                close_cb_max_m=20.0,
                                                                close_residual_max_m=22.0,
                                                                close_satellite_max=close_satellite_max,
                                                                close_p95_abs_residual_max_m=close_p95_abs_residual_max_m,
                                                                far_blocked_max=0.01,
                                                                far_positive_min=0.15,
                                                                far_disagreement_min_m=90.0,
                                                                far_cb_min_m=45.0,
                                                                enter_confirm_close_epochs=enter_confirm_close_epochs,
                                                                enter_confirm_far_epochs=enter_confirm_far_epochs,
                                                                exit_confirm_close_epochs=exit_confirm_close_epochs,
                                                                exit_confirm_far_epochs=exit_confirm_far_epochs,
                                                                close_rescue_satellite_max=close_rescue_satellite_max,
                                                                close_rescue_p95_abs_residual_max_m=close_rescue_p95_abs_residual_max_m,
                                                                close_rescue_cb_min_m=close_rescue_cb_min_m,
                                                                close_entry_p95_abs_residual_max_m=close_entry_p95_abs_residual_max_m,
                                                                negative_exit_disagreement_min_m=negative_exit_disagreement_min_m,
                                                                negative_exit_cb_min_m=negative_exit_cb_min_m,
                                                                negative_exit_p95_abs_residual_min_m=negative_exit_p95_abs_residual_min_m,
                                                                negative_exit_hits_required=negative_exit_hits_required,
                                                            )
                                                            yield "entry_veto_negative_exit_rescue_branch_aware_hysteresis_quality_veto_regime_gate", strategy


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-validate strategy families on tuned and holdout dumps")
    parser.add_argument("--train-feature-csv", type=Path, required=True, help="Tuned/train feature CSV")
    parser.add_argument("--train-trajectory-csv", type=Path, required=True, help="Tuned/train trajectory CSV")
    parser.add_argument("--holdout-feature-csv", type=Path, required=True, help="Holdout feature CSV")
    parser.add_argument("--holdout-trajectory-csv", type=Path, required=True, help="Holdout trajectory CSV")
    parser.add_argument(
        "--results-prefix",
        type=str,
        default="pf_strategy_family_cv",
        help="Output prefix under experiments/results/",
    )
    args = parser.parse_args()

    results_dir = _EXPERIMENTS_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    train_dump = _load_dump(args.train_feature_csv, args.train_trajectory_csv)
    holdout_dump = _load_dump(args.holdout_feature_csv, args.holdout_trajectory_csv)

    baseline_strategy = AlwaysRobustStrategy()
    train_baseline = _evaluate_strategy(baseline_strategy, *train_dump)
    holdout_baseline = _evaluate_strategy(baseline_strategy, *holdout_dump)
    train_baseline_rms = _mean(train_baseline, "rms_2d")
    holdout_baseline_rms = _mean(holdout_baseline, "rms_2d")
    train_baseline_p95 = _mean(train_baseline, "p95")
    holdout_baseline_p95 = _mean(holdout_baseline, "p95")

    config_rows: list[dict[str, object]] = []
    family_best_rows: list[dict[str, object]] = []
    per_family: dict[str, list[dict[str, object]]] = {}

    for family, strategy in _family_members():
        train_rows = _evaluate_strategy(strategy, *train_dump)
        holdout_rows = _evaluate_strategy(strategy, *holdout_dump)
        loc, branch_count, readability = _readability_proxy(strategy)
        param_count, extensibility = _extensibility_proxy(strategy)
        params = strategy.parameters()
        label = strategy.name
        if params:
            label = f"{strategy.name}(" + ",".join(f"{k}={v:g}" for k, v in params.items()) + ")"
        row = {
            "family": family,
            "strategy": strategy.name,
            "strategy_label": label,
            "style": strategy.style,
            "required_features": ",".join(strategy.required_features()),
            "parameters": ",".join(params.keys()),
            "parameter_values": ",".join(f"{k}={v:g}" for k, v in params.items()),
            "train_mean_rms_2d": _mean(train_rows, "rms_2d"),
            "train_mean_p95": _mean(train_rows, "p95"),
            "train_mean_outlier_rate_pct": _mean(train_rows, "outlier_rate_pct"),
            "train_mean_blocked_epoch_frac": _mean(train_rows, "blocked_epoch_frac"),
            "holdout_mean_rms_2d": _mean(holdout_rows, "rms_2d"),
            "holdout_mean_p95": _mean(holdout_rows, "p95"),
            "holdout_mean_outlier_rate_pct": _mean(holdout_rows, "outlier_rate_pct"),
            "holdout_mean_blocked_epoch_frac": _mean(holdout_rows, "blocked_epoch_frac"),
            "train_rms_delta_vs_robust": _mean(train_rows, "rms_2d") - train_baseline_rms,
            "holdout_rms_delta_vs_robust": _mean(holdout_rows, "rms_2d") - holdout_baseline_rms,
            "train_p95_delta_vs_robust": _mean(train_rows, "p95") - train_baseline_p95,
            "holdout_p95_delta_vs_robust": _mean(holdout_rows, "p95") - holdout_baseline_p95,
            "generalization_gap_rms": _mean(holdout_rows, "rms_2d") - _mean(train_rows, "rms_2d"),
            "generalization_gap_p95": _mean(holdout_rows, "p95") - _mean(train_rows, "p95"),
            "holdout_survives_robust": int(_mean(holdout_rows, "rms_2d") <= holdout_baseline_rms),
            "readability_loc": loc,
            "readability_branch_count": branch_count,
            "readability_proxy": readability,
            "extensibility_param_count": param_count,
            "extensibility_proxy": extensibility,
        }
        config_rows.append(row)
        per_family.setdefault(family, []).append(row)

    for family, rows in per_family.items():
        best = sorted(
            rows,
            key=lambda row: (
                row["holdout_mean_rms_2d"],
                row["holdout_mean_p95"],
                row["train_mean_rms_2d"],
            ),
        )[0]
        family_best_rows.append(best)

    config_path = results_dir / f"{args.results_prefix}_configs.csv"
    family_best_path = results_dir / f"{args.results_prefix}_family_best.csv"
    _save_rows(config_rows, config_path)
    _save_rows(family_best_rows, family_best_path)

    print(f"Saved config CV results to: {config_path}")
    print(f"Saved best-per-family CV results to: {family_best_path}")


if __name__ == "__main__":
    main()
