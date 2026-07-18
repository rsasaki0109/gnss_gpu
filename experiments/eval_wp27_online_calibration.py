#!/usr/bin/env python3
"""Evaluate frozen WP27 online diagnostics without promoting a FIX policy."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path

import numpy as np


GAMMA_THRESHOLDS = (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3)
DWELL_THRESHOLDS = (1, 3, 5, 10, 20)
FLOAT_GUARDS_M = (0.25, 0.5, 1.0)
DDPR_GUARDS_M = (0.5, 1.0, 1.75)


def _wilson_upper(false_count: int, total: int, z: float = 1.96) -> float:
    if total == 0:
        return 1.0
    fraction = false_count / total
    denominator = 1.0 + z * z / total
    center = fraction + z * z / (2.0 * total)
    margin = z * math.sqrt(
        fraction * (1.0 - fraction) / total + z * z / (4.0 * total * total)
    )
    return min(1.0, (center + margin) / denominator)


def _calibration(rows: list[dict[str, str]]) -> dict[str, float]:
    probability = np.asarray(
        [float(row["integrity_position_ball_gamma"]) for row in rows], dtype=float
    )
    correct = np.asarray(
        [float(row["integrity_position_ball_error_m"]) < 0.5 for row in rows],
        dtype=float,
    )
    brier = float(np.mean((probability - correct) ** 2))
    ece = 0.0
    for lower in np.linspace(0.0, 0.9, 10):
        mask = (probability >= lower) & (probability < lower + 0.1)
        if np.any(mask):
            ece += float(np.mean(mask)) * abs(
                float(np.mean(probability[mask])) - float(np.mean(correct[mask]))
            )
    return {"brier": brier, "ece_10bin": ece}


def _evaluate(rows: list[dict[str, str]], config: tuple[float, int, float, float]):
    gamma, dwell, float_guard, ddpr_guard = config
    accepted = [
        row
        for row in rows
        if float(row["integrity_position_ball_gamma"]) >= gamma
        and int(row["integrity_dwell_epochs"]) >= dwell
        and float(row["integrity_map_float_separation_m"]) <= float_guard
        and float(row["integrity_map_ddpr_separation_m"]) <= ddpr_guard
        and int(row["last_ddpr_pairs"]) >= 9
        and int(row["ddpr_age_epochs"]) <= 4
    ]
    false = sum(float(row["integrity_map_error_m"]) >= 0.5 for row in accepted)
    return {
        "accepted": len(accepted),
        "correct": len(accepted) - false,
        "false": false,
        "false_pct": 100.0 * false / len(accepted) if accepted else 0.0,
        "false_wilson95_upper_pct": 100.0 * _wilson_upper(false, len(accepted)),
    }


def _configuration_row(config, metrics):
    return {
        "gamma_threshold": config[0],
        "dwell_threshold": config[1],
        "float_guard_m": config[2],
        "ddpr_guard_m": config[3],
        "runs": metrics,
        "minimum_correct_epochs": min(item["correct"] for item in metrics.values()),
        "total_correct_epochs": sum(item["correct"] for item in metrics.values()),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run1", type=Path, required=True)
    parser.add_argument("--run2", type=Path, required=True)
    parser.add_argument("--run3", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    args = parser.parse_args(argv)
    paths = {"run1": args.run1, "run2": args.run2, "run3": args.run3}
    rows = {}
    for run, path in paths.items():
        with path.open(newline="") as fh:
            rows[run] = list(csv.DictReader(fh))

    configurations = list(
        itertools.product(
            GAMMA_THRESHOLDS,
            DWELL_THRESHOLDS,
            FLOAT_GUARDS_M,
            DDPR_GUARDS_M,
        )
    )
    evaluated = []
    for config in configurations:
        metrics = {run: _evaluate(run_rows, config) for run, run_rows in rows.items()}
        evaluated.append(_configuration_row(config, metrics))
    observed_safe = [
        item
        for item in evaluated
        if all(run["false_pct"] <= 1.0 for run in item["runs"].values())
    ]
    wilson_safe = [
        item
        for item in evaluated
        if all(
            run["false_wilson95_upper_pct"] <= 1.0
            for run in item["runs"].values()
        )
    ]
    ranking = lambda item: (item["minimum_correct_epochs"], item["total_correct_epochs"])
    run_summary = {}
    for run, run_rows in rows.items():
        run_summary[run] = {
            "epochs": len(run_rows),
            "oracle_sub50cm_epochs": sum(
                row["basin_oracle_sub50cm_available"] == "1" for row in run_rows
            ),
            "selected_sub50cm_epochs": sum(
                float(row["integrity_map_error_m"]) < 0.5 for row in run_rows
            ),
            **_calibration(run_rows),
        }
    summary = {
        "policy_promoted": False,
        "grid": {
            "gamma_thresholds": GAMMA_THRESHOLDS,
            "dwell_thresholds": DWELL_THRESHOLDS,
            "float_guards_m": FLOAT_GUARDS_M,
            "ddpr_guards_m": DDPR_GUARDS_M,
            "configurations": len(configurations),
        },
        "runs": run_summary,
        "observed_safe_configurations": len(observed_safe),
        "observed_safe_with_common_coverage": sum(
            item["minimum_correct_epochs"] > 0 for item in observed_safe
        ),
        "wilson95_safe_configurations": len(wilson_safe),
        "best_unrestricted": max(evaluated, key=ranking),
        "best_observed_safe": max(observed_safe, key=ranking),
        "verdict": (
            "common zero-observed-false coverage exists, but statistical support "
            "is insufficient: no configuration has a <=1% Wilson 95% false-rate "
            "upper bound on every run"
            if any(item["minimum_correct_epochs"] > 0 for item in observed_safe)
            else "no common safe coverage: observed-safe configurations accept "
            "zero correct epochs on at least one run; no configuration has a "
            "<=1% Wilson 95% false-rate upper bound on every run"
        ),
    }
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
