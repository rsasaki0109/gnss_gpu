#!/usr/bin/env python3
"""Evaluate and aggregate all six PPC routes under one frozen score contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from experiments.evaluate_ppc_official_score import evaluate_route
except ModuleNotFoundError:
    from evaluate_ppc_official_score import evaluate_route  # type: ignore[no-redef]


ROUTES = tuple(
    (city, run, f"{city}_{run}")
    for city in ("tokyo", "nagoya")
    for run in ("run1", "run2", "run3")
)


def aggregate_routes(route_results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if set(route_results) != {route for _, _, route in ROUTES}:
        raise ValueError("suite must contain exactly the six PPC routes")
    pass_distance = sum(float(row["pass_distance_m"]) for row in route_results.values())
    total_distance = sum(float(row["total_distance_m"]) for row in route_results.values())
    fixed = sum(int(row["fixed_epochs"]) for row in route_results.values())
    correct_fix = sum(int(row["correct_fix_epochs"]) for row in route_results.values())
    false_fix = sum(int(row["false_fix_epochs"]) for row in route_results.values())
    severe_false_fix = sum(
        int(row["false_fix_above_1m_epochs"]) for row in route_results.values()
    )
    route_score_mean = sum(
        float(row["ppc_score_pct"]) for row in route_results.values()
    ) / len(route_results)
    pooled_score = 100.0 * pass_distance / total_distance if total_distance else 0.0
    return {
        "schema": "gnss_gpu_ppc_official_suite_score_v1",
        "truth_contract": {
            "production_input_truth": False,
            "truth_usage": "post_estimator_scoring_only",
        },
        "metric": "traveled_distance_with_3d_error_lte_0.5m",
        "forward_only": True,
        "route_count": 6,
        "aggregation": "arithmetic_mean_of_six_route_distance_scores",
        "ppc_score_pct": route_score_mean,
        "pooled_ppc_score_pct_diagnostic": pooled_score,
        "pass_distance_m": pass_distance,
        "total_distance_m": total_distance,
        "fixed_epochs": fixed,
        "correct_fix_epochs": correct_fix,
        "false_fix_epochs": false_fix,
        "false_fix_above_1m_epochs": severe_false_fix,
        "safety_gate_passed": severe_false_fix == 0,
        "targets": {
            "first_70_pct": route_score_mean >= 70.0,
            "public_78_7_pct": route_score_mean > 78.7,
            "stretch_80_pct": route_score_mean >= 80.0,
        },
        "routes": route_results,
    }


def evaluate_suite(
    estimate_root: Path,
    dataset_root: Path,
    estimate_name: str,
    threshold_m: float = 0.5,
) -> dict[str, Any]:
    results = {
        route: evaluate_route(
            estimate_root / route / estimate_name,
            dataset_root / city / run / "reference.csv",
            threshold_m,
        )
        for city, run, route in ROUTES
    }
    return aggregate_routes(results)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--estimate-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--estimate-name", required=True)
    parser.add_argument("--threshold-m", type=float, default=0.5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = evaluate_suite(
        args.estimate_root, args.dataset_root, args.estimate_name, args.threshold_m
    )
    encoded = json.dumps(result, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
