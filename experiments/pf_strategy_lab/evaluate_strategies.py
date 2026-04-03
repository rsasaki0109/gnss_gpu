#!/usr/bin/env python3
"""Evaluate experimental PF selection strategies on a common feature/trajectory dump."""

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
from pf_strategy_lab.strategies import default_strategies


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PF strategy variants on a shared dump")
    parser.add_argument("--feature-csv", type=Path, required=True, help="Feature dump from rich gate search")
    parser.add_argument("--trajectory-csv", type=Path, required=True, help="Trajectory dump from rich gate search")
    parser.add_argument("--base-csv", type=Path, default=None, help="Optional base metrics CSV")
    parser.add_argument(
        "--results-prefix",
        type=str,
        default="pf_strategy_lab",
        help="Output prefix under experiments/results/",
    )
    args = parser.parse_args()

    results_dir = _EXPERIMENTS_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    feature_rows = _load_csv(args.feature_csv)
    trajectory_rows = _load_csv(args.trajectory_csv)
    base_rows = _load_csv(args.base_csv) if args.base_csv is not None else []

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

    base_pf_lookup = {
        row["segment_label"]: float(row["rms_2d"])
        for row in base_rows
        if row["method"] == "PF"
    }

    decision_rows: list[dict[str, object]] = []
    run_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for strategy in default_strategies():
        for segment_label, segment_rows in trajectory_by_segment.items():
            segment_strategy = copy.deepcopy(strategy)
            chosen_positions = []
            truth_positions = []
            times = []

            for row in segment_rows:
                epoch = _epoch_key(row["epoch"])
                context = StrategyContext(
                    segment_label=segment_label,
                    epoch=epoch,
                    features=feature_map[(segment_label, epoch)],
                )
                decision = segment_strategy.decide(context)
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
                decision_rows.append(
                    {
                        "strategy": strategy.name,
                        "style": strategy.style,
                        "segment_label": segment_label,
                        "epoch": epoch,
                        "use_blocked": bool(decision.use_blocked),
                        "score": float(decision.score),
                        "rationale": decision.rationale,
                        **feature_map[(segment_label, epoch)],
                    }
                )

            chosen = np.asarray(chosen_positions, dtype=np.float64)
            truth = np.asarray(truth_positions, dtype=np.float64)
            time_array = np.asarray(times, dtype=np.float64)
            metrics = compute_metrics(chosen, truth)
            metrics = _augment_tail_metrics(metrics, time_array)
            blocked_epoch_frac = float(np.mean([row["use_blocked"] for row in decision_rows if row["strategy"] == strategy.name and row["segment_label"] == segment_label]))

            run_rows.append(
                {
                    "strategy": strategy.name,
                    "style": strategy.style,
                    "segment_label": segment_label,
                    "rms_2d": float(metrics["rms_2d"]),
                    "p95": float(metrics["p95"]),
                    "outlier_rate_pct": float(metrics["outlier_rate_pct"]),
                    "catastrophic_rate_pct": float(metrics["catastrophic_rate_pct"]),
                    "blocked_epoch_frac": blocked_epoch_frac,
                }
            )

        strategy_runs = [row for row in run_rows if row["strategy"] == strategy.name]
        loc, branch_count, readability = _readability_proxy(strategy)
        param_count, extensibility = _extensibility_proxy(strategy)
        summary_rows.append(
            {
                "strategy": strategy.name,
                "style": strategy.style,
                "required_features": ",".join(strategy.required_features()),
                "parameters": ",".join(strategy.parameters().keys()),
                "mean_rms_2d": float(np.mean([row["rms_2d"] for row in strategy_runs])),
                "mean_p95": float(np.mean([row["p95"] for row in strategy_runs])),
                "mean_outlier_rate_pct": float(np.mean([row["outlier_rate_pct"] for row in strategy_runs])),
                "mean_blocked_epoch_frac": float(np.mean([row["blocked_epoch_frac"] for row in strategy_runs])),
                "pf_rms_wins": int(
                    sum(
                        row["rms_2d"] < base_pf_lookup.get(row["segment_label"], np.inf)
                        for row in strategy_runs
                    )
                ) if base_pf_lookup else np.nan,
                "readability_loc": loc,
                "readability_branch_count": branch_count,
                "readability_proxy": float(readability),
                "extensibility_param_count": param_count,
                "extensibility_proxy": float(extensibility),
            }
        )

    run_path = results_dir / f"{args.results_prefix}_runs.csv"
    decision_path = results_dir / f"{args.results_prefix}_decisions.csv"
    summary_path = results_dir / f"{args.results_prefix}_summary.csv"
    _save_rows(run_rows, run_path)
    _save_rows(decision_rows, decision_path)
    _save_rows(summary_rows, summary_path)

    print(f"Saved run-wise metrics to: {run_path}")
    print(f"Saved epoch decisions to: {decision_path}")
    print(f"Saved strategy summary to: {summary_path}")


if __name__ == "__main__":
    main()
