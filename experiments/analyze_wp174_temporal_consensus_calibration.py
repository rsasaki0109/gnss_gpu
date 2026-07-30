#!/usr/bin/env python3
"""Purged blocked-CV for temporal full/satellite Lambda consensus."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.analyze_wp174_consensus_calibration import (  # noqa: E402
    _consensus_delta,
    _finite,
)
from experiments.wp174_ffrt import passes_ffrt  # noqa: E402


@dataclass(frozen=True)
class TemporalConsensusPolicy:
    minimum_pairs: int
    maximum_second_position_delta_m: float
    maximum_consensus_delta_m: float
    maximum_correction_jump_m: float
    acquisition_streak: int


def _read(specification: str) -> list[dict[str, str]]:
    domain, raw_path = specification.split("=", 1)
    with Path(raw_path).open(newline="", encoding="utf-8-sig") as stream:
        return [{**row, "_domain": domain} for row in csv.DictReader(stream)]


def _instantaneous_accepts(
    row: dict[str, str], policy: TemporalConsensusPolicy
) -> bool:
    if row.get("shadow_best_sub50cm") not in {"0", "1"}:
        return False
    pairs = _finite(row, "pair_count")
    return (
        math.isfinite(pairs)
        and passes_ffrt(
            int(pairs),
            _finite(row, "lambda_shadow_bsr_qscale16"),
            _finite(row, "lambda_shadow_ratio"),
        )
        # FFRT can approach one when modelled BSR is nearly one. Retain the
        # solver's pre-existing absolute ratio floor as an independent,
        # non-calibrated guard.
        and _finite(row, "lambda_shadow_ratio") >= 1.5
        and pairs >= policy.minimum_pairs
        and _finite(row, "lambda_shadow_second_position_delta_m")
        <= policy.maximum_second_position_delta_m
        and _finite(row, "float_update_nis_per_observation") <= 3.0
        and _consensus_delta(row) <= policy.maximum_consensus_delta_m
    )


def _correction(row: dict[str, str]) -> tuple[float, float, float]:
    return tuple(
        _finite(row, f"lambda_shadow_best_correction_{axis}")
        for axis in "xyz"
    )


def _declared(
    rows: list[dict[str, str]], policy: TemporalConsensusPolicy
) -> list[bool]:
    decisions: list[bool] = []
    streak = 0
    previous_tow: float | None = None
    previous_correction: tuple[float, float, float] | None = None
    for row in rows:
        tow = float(row["tow"])
        correction = _correction(row)
        accepted = _instantaneous_accepts(row, policy)
        contiguous = (
            previous_tow is not None
            and 0.0 < tow - previous_tow <= 0.21
        )
        correction_close = (
            previous_correction is not None
            and all(
                math.isfinite(value)
                for value in correction + previous_correction
            )
            and math.dist(correction, previous_correction)
            <= policy.maximum_correction_jump_m
        )
        if accepted:
            streak = streak + 1 if contiguous and correction_close else 1
        else:
            streak = 0
        decisions.append(accepted and streak >= policy.acquisition_streak)
        previous_tow = tow
        previous_correction = correction if accepted else None
    return decisions


def analyze(
    rows: list[dict[str, str]], purge_blocks: int = 1
) -> dict[str, Any]:
    if purge_blocks < 0:
        raise ValueError("purge_blocks must be non-negative")
    by_domain: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        if row.get("shadow_best_sub50cm") in {"0", "1"}:
            by_domain.setdefault(row["_domain"], []).append(row)
    for domain_rows in by_domain.values():
        domain_rows.sort(key=lambda row: float(row["tow"]))

    folds = sorted(
        {
            (domain, int(row["block"]))
            for domain, domain_rows in by_domain.items()
            for row in domain_rows
        }
    )
    policies = [
        TemporalConsensusPolicy(
            minimum_pairs=minimum_pairs,
            maximum_second_position_delta_m=spread,
            maximum_consensus_delta_m=consensus,
            maximum_correction_jump_m=jump,
            acquisition_streak=streak,
        )
        for minimum_pairs in (12, 16)
        for spread in (0.05, 0.25)
        for consensus in (0.01, 0.02, 0.03, 0.05, 0.10, 0.25)
        for jump in (0.005, 0.01, 0.02)
        for streak in (3, 5, 8, 12)
    ]
    counts: dict[
        TemporalConsensusPolicy, dict[tuple[str, int], tuple[int, int]]
    ] = {}
    for policy in policies:
        policy_counts: dict[tuple[str, int], list[int]] = {
            fold: [0, 0] for fold in folds
        }
        for domain, domain_rows in by_domain.items():
            for row, declared in zip(
                domain_rows, _declared(domain_rows, policy), strict=True
            ):
                if not declared:
                    continue
                fold = (domain, int(row["block"]))
                good = row["shadow_best_sub50cm"] == "1"
                policy_counts[fold][0 if good else 1] += 1
        counts[policy] = {
            fold: (values[0], values[1])
            for fold, values in policy_counts.items()
        }

    fold_results = []
    total_good = 0
    total_bad = 0
    for test_fold in folds:
        excluded = {
            fold
            for fold in folds
            if fold[0] == test_fold[0]
            and abs(fold[1] - test_fold[1]) <= purge_blocks
        }
        eligible = []
        for policy in policies:
            train_good = sum(
                counts[policy][fold][0]
                for fold in folds
                if fold not in excluded
            )
            train_bad = sum(
                counts[policy][fold][1]
                for fold in folds
                if fold not in excluded
            )
            if train_bad == 0:
                eligible.append((train_good, policy))
        selected = (
            max(
                eligible,
                key=lambda item: (
                    item[0],
                    item[1].minimum_pairs,
                    -item[1].maximum_consensus_delta_m,
                    -item[1].maximum_correction_jump_m,
                    item[1].acquisition_streak,
                    -item[1].maximum_second_position_delta_m,
                ),
            )[1]
            if eligible
            else None
        )
        test_good, test_bad = (
            counts[selected][test_fold] if selected is not None else (0, 0)
        )
        total_good += test_good
        total_bad += test_bad
        fold_results.append(
            {
                "test_domain": test_fold[0],
                "test_block": test_fold[1],
                "selected_policy": (
                    asdict(selected) if selected is not None else None
                ),
                "test_good_epochs": test_good,
                "test_bad_epochs": test_bad,
            }
        )
    return {
        "schema": "gnss_gpu_wp174_temporal_consensus_blocked_cv_v1",
        "runtime_fgo": False,
        "selection_truth_usage": "training_folds_only",
        "test_truth_usage": "held_out_block_audit_only",
        "fixed_structural_guards": {
            "ffrt_covariance_scale": 16.0,
            "minimum_absolute_ratio": 1.5,
            "maximum_nis_per_observation": 3.0,
            "maximum_epoch_gap_s": 0.21,
            "maximum_policy_grid_correction_jump_m": 0.02,
        },
        "candidate_policy_count": len(policies),
        "truth_labeled_epochs": sum(map(len, by_domain.values())),
        "out_of_fold": {
            "accepted_good_epochs": total_good,
            "accepted_bad_epochs": total_bad,
        },
        "folds": fold_results,
        "promotion_ready": False,
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
