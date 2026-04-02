#!/usr/bin/env python3
"""Sweep a narrow neighborhood around the current entry-veto family."""

from __future__ import annotations

import argparse
import copy
import csv
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
    EntryVetoNegativeExitRescueBranchAwareHysteresisQualityVetoRegimeGateStrategy,
)


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def _epoch_key(value: str) -> int:
    return int(float(value))


def _load_dump(feature_csv: Path, trajectory_csv: Path) -> tuple[dict, dict[str, list[dict[str, str]]]]:
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


def _evaluate_strategy(strategy, dump) -> list[dict[str, object]]:
    feature_map, trajectory_by_segment = dump
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

        metrics = compute_metrics(
            np.asarray(chosen_positions, dtype=np.float64),
            np.asarray(truth_positions, dtype=np.float64),
        )
        metrics = _augment_tail_metrics(metrics, np.asarray(times, dtype=np.float64))
        run_rows.append(
            {
                "segment_label": segment_label,
                "rms_2d": float(metrics["rms_2d"]),
                "p95": float(metrics["p95"]),
                "blocked_epoch_frac": float(np.mean(blocked_flags)),
            }
        )
    return run_rows


def _save_rows(rows: list[dict[str, object]], path: Path) -> None:
    keys = sorted({key for row in rows for key in row})
    save_results({key: [row.get(key, np.nan) for row in rows] for key in keys}, path)


def _mean(rows: list[dict[str, object]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep the entry-veto family neighborhood")
    parser.add_argument("--train-feature-csv", type=Path, required=True)
    parser.add_argument("--train-trajectory-csv", type=Path, required=True)
    parser.add_argument("--holdout-feature-csv", type=Path, required=True)
    parser.add_argument("--holdout-trajectory-csv", type=Path, required=True)
    parser.add_argument(
        "--results-prefix",
        type=str,
        default="pf_strategy_entry_veto_freeze",
        help="Output prefix under experiments/results/",
    )
    args = parser.parse_args()

    results_dir = _EXPERIMENTS_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    train_dump = _load_dump(args.train_feature_csv, args.train_trajectory_csv)
    holdout_dump = _load_dump(args.holdout_feature_csv, args.holdout_trajectory_csv)

    config_rows: list[dict[str, object]] = []
    for close_rescue_cb_min_m in (16.0, 18.0, 20.0):
        for close_entry_p95_abs_residual_max_m in (45.0, 48.0, 50.0, 52.0, 55.0):
            for exit_confirm_close_epochs in (2, 3, 4):
                for exit_confirm_far_epochs in (4, 5, 6):
                    for negative_exit_disagreement_min_m in (42.0, 45.0, 999.0):
                        for negative_exit_cb_min_m in (25.0, 999.0):
                            for negative_exit_p95_abs_residual_min_m in (50.0, 52.0, 55.0):
                                strategy = EntryVetoNegativeExitRescueBranchAwareHysteresisQualityVetoRegimeGateStrategy(
                                    close_blocked_low=0.10,
                                    close_blocked_high=0.50,
                                    close_disagreement_max_m=40.0,
                                    close_cb_max_m=20.0,
                                    close_residual_max_m=22.0,
                                    close_satellite_max=9.0,
                                    close_p95_abs_residual_max_m=55.0,
                                    far_blocked_max=0.01,
                                    far_positive_min=0.15,
                                    far_disagreement_min_m=90.0,
                                    far_cb_min_m=45.0,
                                    enter_confirm_close_epochs=3,
                                    enter_confirm_far_epochs=1,
                                    exit_confirm_close_epochs=exit_confirm_close_epochs,
                                    exit_confirm_far_epochs=exit_confirm_far_epochs,
                                    close_rescue_satellite_max=8.0,
                                    close_rescue_p95_abs_residual_max_m=50.0,
                                    close_rescue_cb_min_m=close_rescue_cb_min_m,
                                    close_entry_p95_abs_residual_max_m=close_entry_p95_abs_residual_max_m,
                                    negative_exit_disagreement_min_m=negative_exit_disagreement_min_m,
                                    negative_exit_cb_min_m=negative_exit_cb_min_m,
                                    negative_exit_p95_abs_residual_min_m=negative_exit_p95_abs_residual_min_m,
                                    negative_exit_hits_required=1,
                                )
                                train_rows = _evaluate_strategy(strategy, train_dump)
                                holdout_rows = _evaluate_strategy(strategy, holdout_dump)
                                config_rows.append(
                                    {
                                        "close_rescue_cb_min_m": close_rescue_cb_min_m,
                                        "close_entry_p95_abs_residual_max_m": close_entry_p95_abs_residual_max_m,
                                        "exit_confirm_close_epochs": exit_confirm_close_epochs,
                                        "exit_confirm_far_epochs": exit_confirm_far_epochs,
                                        "negative_exit_disagreement_min_m": negative_exit_disagreement_min_m,
                                        "negative_exit_cb_min_m": negative_exit_cb_min_m,
                                        "negative_exit_p95_abs_residual_min_m": negative_exit_p95_abs_residual_min_m,
                                        "train_mean_rms_2d": _mean(train_rows, "rms_2d"),
                                        "train_mean_p95": _mean(train_rows, "p95"),
                                        "holdout_mean_rms_2d": _mean(holdout_rows, "rms_2d"),
                                        "holdout_mean_p95": _mean(holdout_rows, "p95"),
                                    }
                                )

    config_rows.sort(
        key=lambda row: (
            row["holdout_mean_rms_2d"],
            row["holdout_mean_p95"],
            row["train_mean_rms_2d"],
            row["train_mean_p95"],
        )
    )
    best_rows = config_rows[:10]

    config_path = results_dir / f"{args.results_prefix}_configs.csv"
    best_path = results_dir / f"{args.results_prefix}_best.csv"
    _save_rows(config_rows, config_path)
    _save_rows(best_rows, best_path)

    print(f"Saved config sweep to: {config_path}")
    print(f"Saved top configs to: {best_path}")


if __name__ == "__main__":
    main()
