#!/usr/bin/env python3
"""Leave-one-time-block-out CV for RTK surplus-holdout selection policies."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import itertools
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.analyze_wp176_surplus_validation import (  # noqa: E402
    _position_error,
    _read_csv,
    _truth,
)


@dataclass(frozen=True)
class Policy:
    maximum_distance_cycles: float
    minimum_subset_pairs: int
    minimum_ratio: float


@dataclass(frozen=True)
class Example:
    city: str
    block: int
    row: dict[str, str]
    error_m: float


def _selected(example: Example, policy: Policy) -> bool:
    row = example.row
    return (
        float(row["satellite_par_surplus_max_distance_cycles"])
        <= policy.maximum_distance_cycles
        and int(row["satellite_par_subset_size"])
        >= policy.minimum_subset_pairs
        and float(row["satellite_par_ratio"]) >= policy.minimum_ratio
        and float(row["float_update_nis_per_observation"]) <= 3.0
        and float(row["float_update_prefit_residual_rms_m"]) <= 50.0
    )


def _counts(
    examples: list[Example], policy: Policy
) -> tuple[int, int]:
    selected = [example for example in examples if _selected(example, policy)]
    correct = sum(example.error_m < 0.5 for example in selected)
    return correct, len(selected) - correct


def _examples(city: str, integrity: Path, reference: Path) -> list[Example]:
    rows = _read_csv(integrity)
    truth = _truth(reference)
    output = []
    for index, row in enumerate(rows):
        if (
            row.get("library_status") == "4"
            or row.get("satellite_par_surplus_evaluated") != "1"
        ):
            continue
        error = _position_error(row, truth)
        if error is None:
            continue
        output.append(
            Example(
                city=city,
                block=min(4, index * 5 // max(1, len(rows))),
                row=row,
                error_m=error,
            )
        )
    return output


def analyze(examples: list[Example]) -> dict[str, object]:
    policies = [
        Policy(distance, pairs, ratio)
        for distance, pairs, ratio in itertools.product(
            (0.05, 0.075, 0.10),
            (8, 10, 12),
            (1.4, 1.5, 2.0),
        )
    ]
    folds = []
    total_correct = 0
    total_wrong = 0
    for city in ("tokyo", "nagoya"):
        for block in range(5):
            training = [
                example
                for example in examples
                if (example.city, example.block) != (city, block)
            ]
            holdout = [
                example
                for example in examples
                if (example.city, example.block) == (city, block)
            ]
            ranked = []
            for policy in policies:
                correct, wrong = _counts(training, policy)
                ranked.append(
                    (
                        wrong != 0,
                        wrong,
                        -correct,
                        policy.maximum_distance_cycles,
                        -policy.minimum_subset_pairs,
                        -policy.minimum_ratio,
                        policy,
                    )
                )
            policy = min(ranked)[-1]
            train_correct, train_wrong = _counts(training, policy)
            test_correct, test_wrong = _counts(holdout, policy)
            total_correct += test_correct
            total_wrong += test_wrong
            folds.append(
                {
                    "city": city,
                    "block": block,
                    "policy": {
                        "maximum_distance_cycles": (
                            policy.maximum_distance_cycles
                        ),
                        "minimum_subset_pairs": policy.minimum_subset_pairs,
                        "minimum_ratio": policy.minimum_ratio,
                    },
                    "training_correct": train_correct,
                    "training_wrong": train_wrong,
                    "holdout_correct": test_correct,
                    "holdout_wrong": test_wrong,
                }
            )
    return {
        "schema": "gnss_gpu_wp176_surplus_nested_cv_v1",
        "truth_usage": "training folds select policy; disjoint time block is holdout",
        "folds": folds,
        "aggregate_holdout_correct": total_correct,
        "aggregate_holdout_wrong": total_wrong,
        "aggregate_holdout_wrong_rate": (
            total_wrong / (total_correct + total_wrong)
            if total_correct + total_wrong
            else 0.0
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokyo-integrity", type=Path, required=True)
    parser.add_argument("--tokyo-reference", type=Path, required=True)
    parser.add_argument("--nagoya-integrity", type=Path, required=True)
    parser.add_argument("--nagoya-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = analyze(
        _examples("tokyo", args.tokyo_integrity, args.tokyo_reference)
        + _examples("nagoya", args.nagoya_integrity, args.nagoya_reference)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
