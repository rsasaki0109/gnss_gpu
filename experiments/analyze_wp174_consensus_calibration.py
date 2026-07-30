#!/usr/bin/env python3
"""Blocked-CV calibration for full/satellite Lambda position consensus."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.analyze_wp174_ffrt_calibration import BSR_FIELDS  # noqa: E402
from experiments.wp174_ffrt import passes_ffrt  # noqa: E402


@dataclass(frozen=True)
class ConsensusPolicy:
    covariance_scale: int
    minimum_pairs: int
    maximum_second_position_delta_m: float
    maximum_nis_per_observation: float
    maximum_consensus_delta_m: float


def _finite(row: dict[str, str], key: str) -> float:
    try:
        value = float(row.get(key, ""))
    except ValueError:
        return math.nan
    return value if math.isfinite(value) else math.nan


def _read(specification: str) -> list[dict[str, str]]:
    domain, raw_path = specification.split("=", 1)
    with Path(raw_path).open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    return [
        {**row, "_domain": domain}
        for row in rows
        if row.get("lambda_shadow_solved") == "1"
        and row.get("shadow_best_sub50cm") in {"0", "1"}
    ]


def _consensus_delta(row: dict[str, str]) -> float:
    differences = [
        _finite(row, f"lambda_shadow_best_ecef_{axis}")
        - _finite(row, f"lambda_satellite_par_shadow_best_ecef_{axis}")
        for axis in "xyz"
    ]
    return (
        math.sqrt(sum(value * value for value in differences))
        if all(math.isfinite(value) for value in differences)
        else math.nan
    )


def analyze(rows: list[dict[str, str]], purge_blocks: int = 1) -> dict:
    if purge_blocks < 0:
        raise ValueError("purge_blocks must be non-negative")
    folds = sorted(
        {(row["_domain"], int(row["block"])) for row in rows}
    )
    fold_index = {
        fold: index for index, fold in enumerate(folds)
    }
    row_fold = np.array(
        [
            fold_index[(row["_domain"], int(row["block"]))]
            for row in rows
        ],
        dtype=np.int32,
    )
    good = np.array(
        [row["shadow_best_sub50cm"] == "1" for row in rows],
        dtype=bool,
    )
    pairs = np.array([_finite(row, "pair_count") for row in rows])
    spread = np.array(
        [
            _finite(row, "lambda_shadow_second_position_delta_m")
            for row in rows
        ]
    )
    nis = np.array(
        [_finite(row, "float_update_nis_per_observation") for row in rows]
    )
    consensus = np.array([_consensus_delta(row) for row in rows])
    ratio = np.array(
        [_finite(row, "lambda_shadow_ratio") for row in rows]
    )
    ffrt_by_scale = {}
    for scale, field in BSR_FIELDS.items():
        bsr = np.array([_finite(row, field) for row in rows])
        ffrt_by_scale[scale] = np.array(
            [
                (
                    math.isfinite(pairs[index])
                    and math.isfinite(bsr[index])
                    and math.isfinite(ratio[index])
                    and passes_ffrt(
                        int(pairs[index]), bsr[index], ratio[index]
                    )
                )
                for index in range(len(rows))
            ],
            dtype=bool,
        )

    policies = [
        ConsensusPolicy(scale, minimum_pairs, maximum_spread, maximum_nis, delta)
        for scale in BSR_FIELDS
        for minimum_pairs in (4, 6, 8, 10, 12, 16)
        for maximum_spread in (0.03, 0.05, 0.10, 0.25, math.inf)
        for maximum_nis in (1.0, 2.0, 3.0, 5.0, math.inf)
        for delta in (0.005, 0.01, 0.02, 0.03, 0.05, 0.10, 0.25)
    ]
    counts: dict[ConsensusPolicy, tuple[np.ndarray, np.ndarray]] = {}
    for policy in policies:
        accepted = (
            ffrt_by_scale[policy.covariance_scale]
            & (pairs >= policy.minimum_pairs)
            & (spread <= policy.maximum_second_position_delta_m)
            & (nis <= policy.maximum_nis_per_observation)
            & (consensus <= policy.maximum_consensus_delta_m)
        )
        good_counts = np.bincount(
            row_fold[accepted & good], minlength=len(folds)
        )
        bad_counts = np.bincount(
            row_fold[accepted & ~good], minlength=len(folds)
        )
        counts[policy] = (good_counts, bad_counts)

    fold_results = []
    total_good = 0
    total_bad = 0
    by_domain: dict[str, dict[str, int]] = {}
    for test_index, (domain, block) in enumerate(folds):
        excluded = {
            index
            for index, fold in enumerate(folds)
            if fold[0] == domain and abs(fold[1] - block) <= purge_blocks
        }
        train_indices = [
            index for index in range(len(folds)) if index not in excluded
        ]
        zero_bad = []
        for policy, (good_counts, bad_counts) in counts.items():
            train_bad = int(bad_counts[train_indices].sum())
            if train_bad == 0:
                zero_bad.append(
                    (
                        int(good_counts[train_indices].sum()),
                        policy,
                    )
                )
        selected = (
            max(
                zero_bad,
                key=lambda item: (
                    item[0],
                    item[1].covariance_scale,
                    item[1].minimum_pairs,
                    -item[1].maximum_consensus_delta_m,
                    -item[1].maximum_second_position_delta_m,
                    -item[1].maximum_nis_per_observation,
                ),
            )[1]
            if zero_bad
            else None
        )
        test_good = (
            int(counts[selected][0][test_index])
            if selected is not None
            else 0
        )
        test_bad = (
            int(counts[selected][1][test_index])
            if selected is not None
            else 0
        )
        total_good += test_good
        total_bad += test_bad
        domain_metrics = by_domain.setdefault(
            domain,
            {"accepted_good_epochs": 0, "accepted_bad_epochs": 0},
        )
        domain_metrics["accepted_good_epochs"] += test_good
        domain_metrics["accepted_bad_epochs"] += test_bad
        fold_results.append(
            {
                "test_domain": domain,
                "test_block": block,
                "selected_policy": (
                    asdict(selected) if selected is not None else None
                ),
                "test_good_epochs": test_good,
                "test_bad_epochs": test_bad,
            }
        )
    return {
        "schema": "gnss_gpu_wp174_lambda_consensus_blocked_cv_v1",
        "runtime_fgo": False,
        "selection_truth_usage": "training_folds_only",
        "test_truth_usage": "held_out_block_audit_only",
        "promotion_ready": False,
        "candidate_policy_count": len(policies),
        "truth_labeled_epochs": len(rows),
        "out_of_fold": {
            "accepted_good_epochs": total_good,
            "accepted_bad_epochs": total_bad,
            "by_domain": by_domain,
        },
        "folds": fold_results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", action="append", required=True)
    parser.add_argument("--purge-blocks", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = analyze(
        [
            row
            for specification in args.audit
            for row in _read(specification)
        ],
        purge_blocks=args.purge_blocks,
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
