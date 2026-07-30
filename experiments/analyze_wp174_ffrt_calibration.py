#!/usr/bin/env python3
"""Blocked, purged exploratory CV for WP174 FFRT covariance calibration."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.wp174_ffrt import passes_ffrt  # noqa: E402


BSR_FIELDS = {
    1: "lambda_shadow_bsr",
    2: "lambda_shadow_bsr_qscale2",
    4: "lambda_shadow_bsr_qscale4",
    8: "lambda_shadow_bsr_qscale8",
    16: "lambda_shadow_bsr_qscale16",
}


@dataclass(frozen=True)
class Policy:
    covariance_scale: int
    minimum_pairs: int
    maximum_second_position_delta_m: float
    maximum_nis_per_observation: float


def _finite(row: dict[str, str], field: str) -> float | None:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _read_audit(specification: str) -> list[dict[str, str]]:
    try:
        domain, raw_path = specification.split("=", 1)
    except ValueError as error:
        raise ValueError("--audit must use DOMAIN=PATH") from error
    path = Path(raw_path)
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    output = []
    for row in rows:
        if row.get("lambda_shadow_solved") != "1":
            continue
        if row.get("shadow_best_sub50cm") not in {"0", "1"}:
            continue
        output.append({**row, "_domain": domain})
    return output


def _accepts(row: dict[str, str], policy: Policy) -> bool:
    pair_count = _finite(row, "pair_count")
    ratio = _finite(row, "lambda_shadow_ratio")
    bsr = _finite(row, BSR_FIELDS[policy.covariance_scale])
    second_delta = _finite(row, "lambda_shadow_second_position_delta_m")
    nis = _finite(row, "float_update_nis_per_observation")
    if None in {pair_count, ratio, bsr, second_delta, nis}:
        return False
    return (
        pair_count >= policy.minimum_pairs
        and second_delta <= policy.maximum_second_position_delta_m
        and nis <= policy.maximum_nis_per_observation
        and passes_ffrt(int(pair_count), bsr, ratio)
    )


def _metrics(
    rows: list[dict[str, str]], policy: Policy
) -> dict[str, Any]:
    accepted = [row for row in rows if _accepts(row, policy)]
    good = sum(row["shadow_best_sub50cm"] == "1" for row in accepted)
    bad = len(accepted) - good
    return {
        "eligible_epochs": len(rows),
        "accepted_epochs": len(accepted),
        "accepted_good_epochs": good,
        "accepted_bad_epochs": bad,
        "accepted_good_rate_pct": (
            100.0 * good / len(rows) if rows else 0.0
        ),
    }


def _confirmed_metrics(
    rows: list[dict[str, str]],
    policy: Policy,
    *,
    minimum_contiguous_epochs: int,
    maximum_correction_jump_m: float,
    maximum_epoch_gap_s: float = 0.21,
    presorted: bool = False,
) -> dict[str, Any]:
    accepted = []
    streak = 0
    previous_tow: float | None = None
    previous_correction: tuple[float, float, float] | None = None
    ordered_rows = (
        rows
        if presorted
        else sorted(rows, key=lambda item: float(item["tow"]))
    )
    for row in ordered_rows:
        correction = tuple(
            _finite(row, f"lambda_shadow_best_correction_{axis}")
            for axis in "xyz"
        )
        tow = _finite(row, "tow")
        qualified = (
            _accepts(row, policy)
            and tow is not None
            and all(value is not None for value in correction)
        )
        if not qualified:
            streak = 0
            previous_tow = None
            previous_correction = None
            continue
        current_correction = tuple(float(value) for value in correction)
        continuous = (
            previous_tow is not None
            and previous_correction is not None
            and tow - previous_tow <= maximum_epoch_gap_s
            and math.dist(current_correction, previous_correction)
            <= maximum_correction_jump_m
        )
        streak = streak + 1 if continuous else 1
        previous_tow = tow
        previous_correction = current_correction
        if streak >= minimum_contiguous_epochs:
            accepted.append(row)
    good = sum(row["shadow_best_sub50cm"] == "1" for row in accepted)
    return {
        "eligible_epochs": len(rows),
        "accepted_epochs": len(accepted),
        "accepted_good_epochs": good,
        "accepted_bad_epochs": len(accepted) - good,
        "accepted_good_rate_pct": (
            100.0 * good / len(rows) if rows else 0.0
        ),
    }


def _policies() -> list[Policy]:
    return [
        Policy(scale, pairs, spread, nis)
        for scale in BSR_FIELDS
        for pairs in (4, 6, 8, 10, 12, 16)
        for spread in (0.03, 0.05, 0.10, 0.25, math.inf)
        for nis in (1.0, 2.0, 3.0, 5.0, math.inf)
    ]


def _select_policy(
    rows: list[dict[str, str]], policies: list[Policy]
) -> tuple[Policy | None, dict[str, Any]]:
    zero_bad = []
    for policy in policies:
        metrics = _metrics(rows, policy)
        if metrics["accepted_bad_epochs"] == 0:
            zero_bad.append((policy, metrics))
    if not zero_bad:
        return None, {
            "eligible_epochs": len(rows),
            "accepted_epochs": 0,
            "accepted_good_epochs": 0,
            "accepted_bad_epochs": 0,
            "accepted_good_rate_pct": 0.0,
        }
    # Maximize correct availability. Ties prefer more conservative covariance,
    # more pairs, and tighter position/NIS bounds.
    return max(
        zero_bad,
        key=lambda item: (
            item[1]["accepted_good_epochs"],
            item[0].covariance_scale,
            item[0].minimum_pairs,
            -item[0].maximum_second_position_delta_m,
            -item[0].maximum_nis_per_observation,
        ),
    )


def analyze(
    rows: list[dict[str, str]], *, purge_blocks: int = 1
) -> dict[str, Any]:
    if purge_blocks < 0:
        raise ValueError("purge_blocks must be non-negative")
    policies = _policies()
    folds = sorted({(row["_domain"], int(row["block"])) for row in rows})
    rows_by_fold = {
        fold: sorted(
            (
                row
                for row in rows
                if (row["_domain"], int(row["block"])) == fold
            ),
            key=lambda item: float(item["tow"]),
        )
        for fold in folds
    }
    # Evaluate every epoch/policy pair once. Fold scoring below then becomes
    # aggregation over cached counts rather than repeating the FFRT power
    # function for every outer fold.
    accepted_by_policy: dict[
        Policy, dict[tuple[str, int], tuple[int, int]]
    ] = {}
    confirmed_by_policy: dict[
        Policy, dict[tuple[str, int], tuple[int, int]]
    ] = {}
    for policy in policies:
        fold_counts = {}
        confirmed_fold_counts = {}
        for fold, fold_rows in rows_by_fold.items():
            accepted = [row for row in fold_rows if _accepts(row, policy)]
            good = sum(
                row["shadow_best_sub50cm"] == "1" for row in accepted
            )
            fold_counts[fold] = (good, len(accepted) - good)
            confirmed = _confirmed_metrics(
                fold_rows,
                policy,
                minimum_contiguous_epochs=2,
                maximum_correction_jump_m=0.03,
                presorted=True,
            )
            confirmed_fold_counts[fold] = (
                confirmed["accepted_good_epochs"],
                confirmed["accepted_bad_epochs"],
            )
        accepted_by_policy[policy] = fold_counts
        confirmed_by_policy[policy] = confirmed_fold_counts

    def metrics_from_counts(
        eligible: int, good: int, bad: int
    ) -> dict[str, Any]:
        return {
            "eligible_epochs": eligible,
            "accepted_epochs": good + bad,
            "accepted_good_epochs": good,
            "accepted_bad_epochs": bad,
            "accepted_good_rate_pct": (
                100.0 * good / eligible if eligible else 0.0
            ),
        }

    fold_results = []
    out_of_fold_good = 0
    out_of_fold_bad = 0
    confirmation_totals = {
        (streak, jump): {"good": 0, "bad": 0}
        for streak in (2, 3, 4)
        for jump in (0.01, 0.03, 0.1, math.inf)
    }
    selected_counts: Counter[str] = Counter()
    for domain, block in folds:
        test_fold = (domain, block)
        excluded = {
            fold
            for fold in folds
            if fold[0] == domain and abs(fold[1] - block) <= purge_blocks
        }
        train_folds = [fold for fold in folds if fold not in excluded]
        train_eligible = sum(len(rows_by_fold[fold]) for fold in train_folds)
        zero_bad = []
        for policy in policies:
            good = sum(
                accepted_by_policy[policy][fold][0]
                for fold in train_folds
            )
            bad = sum(
                accepted_by_policy[policy][fold][1]
                for fold in train_folds
            )
            if bad == 0:
                zero_bad.append(
                    (
                        policy,
                        metrics_from_counts(train_eligible, good, bad),
                    )
                )
        selected, train_metrics = (
            max(
                zero_bad,
                key=lambda item: (
                    item[1]["accepted_good_epochs"],
                    item[0].covariance_scale,
                    item[0].minimum_pairs,
                    -item[0].maximum_second_position_delta_m,
                    -item[0].maximum_nis_per_observation,
                ),
            )
            if zero_bad
            else (
                None,
                metrics_from_counts(train_eligible, 0, 0),
            )
        )
        if selected is None:
            test_metrics = metrics_from_counts(
                len(rows_by_fold[test_fold]), 0, 0
            )
        else:
            test_good, test_bad = accepted_by_policy[selected][test_fold]
            test_metrics = metrics_from_counts(
                len(rows_by_fold[test_fold]), test_good, test_bad
            )
        out_of_fold_good += test_metrics["accepted_good_epochs"]
        out_of_fold_bad += test_metrics["accepted_bad_epochs"]
        confirmation_sweep = {}
        if selected is not None:
            for streak, jump in confirmation_totals:
                confirmed = _confirmed_metrics(
                    rows_by_fold[test_fold],
                    selected,
                    minimum_contiguous_epochs=streak,
                    maximum_correction_jump_m=jump,
                    presorted=True,
                )
                confirmation_totals[(streak, jump)]["good"] += confirmed[
                    "accepted_good_epochs"
                ]
                confirmation_totals[(streak, jump)]["bad"] += confirmed[
                    "accepted_bad_epochs"
                ]
                confirmation_sweep[
                    f"streak{streak}_correction_jump{jump:g}m"
                ] = confirmed
        policy_payload = asdict(selected) if selected is not None else None
        selected_counts[json.dumps(policy_payload, sort_keys=True)] += 1
        fold_results.append(
            {
                "test_domain": domain,
                "test_block": block,
                "purge_blocks": purge_blocks,
                "train": train_metrics,
                "test": test_metrics,
                "test_confirmation_sweep": confirmation_sweep,
                "selected_policy": policy_payload,
            }
        )

    confirmed_policy_folds = []
    confirmed_out_of_fold_good = 0
    confirmed_out_of_fold_bad = 0
    confirmed_selected_counts: Counter[str] = Counter()
    confirmed_safety_envelope = {
        "minimum_covariance_scale": 8,
        "minimum_pairs": 16,
        "maximum_nis_per_observation": 1.0,
        "maximum_second_position_delta_m": 0.05,
    }
    confirmed_policies = [
        policy
        for policy in policies
        if policy.covariance_scale
        >= confirmed_safety_envelope["minimum_covariance_scale"]
        and policy.minimum_pairs
        >= confirmed_safety_envelope["minimum_pairs"]
        and policy.maximum_nis_per_observation
        <= confirmed_safety_envelope["maximum_nis_per_observation"]
        and policy.maximum_second_position_delta_m
        <= confirmed_safety_envelope["maximum_second_position_delta_m"]
    ]
    for domain, block in folds:
        test_fold = (domain, block)
        excluded = {
            fold
            for fold in folds
            if fold[0] == domain and abs(fold[1] - block) <= purge_blocks
        }
        train_folds = [fold for fold in folds if fold not in excluded]
        train_eligible = sum(len(rows_by_fold[fold]) for fold in train_folds)
        zero_bad = []
        for policy in confirmed_policies:
            good = sum(
                confirmed_by_policy[policy][fold][0]
                for fold in train_folds
            )
            bad = sum(
                confirmed_by_policy[policy][fold][1]
                for fold in train_folds
            )
            if bad == 0:
                zero_bad.append(
                    (policy, metrics_from_counts(train_eligible, good, bad))
                )
        selected, train_metrics = (
            max(
                zero_bad,
                key=lambda item: (
                    item[1]["accepted_good_epochs"],
                    item[0].covariance_scale,
                    item[0].minimum_pairs,
                    -item[0].maximum_second_position_delta_m,
                    -item[0].maximum_nis_per_observation,
                ),
            )
            if zero_bad
            else (None, metrics_from_counts(train_eligible, 0, 0))
        )
        if selected is None:
            test_metrics = metrics_from_counts(
                len(rows_by_fold[test_fold]), 0, 0
            )
        else:
            test_good, test_bad = confirmed_by_policy[selected][test_fold]
            test_metrics = metrics_from_counts(
                len(rows_by_fold[test_fold]), test_good, test_bad
            )
        confirmed_out_of_fold_good += test_metrics["accepted_good_epochs"]
        confirmed_out_of_fold_bad += test_metrics["accepted_bad_epochs"]
        payload = asdict(selected) if selected is not None else None
        confirmed_selected_counts[json.dumps(payload, sort_keys=True)] += 1
        confirmed_policy_folds.append(
            {
                "test_domain": domain,
                "test_block": block,
                "purge_blocks": purge_blocks,
                "selection_state_machine": {
                    "minimum_contiguous_epochs": 2,
                    "maximum_correction_jump_m": 0.03,
                    "maximum_epoch_gap_s": 0.21,
                },
                "train": train_metrics,
                "test": test_metrics,
                "selected_policy": payload,
            }
        )

    # Diagnostic arm: let temporal confirmation, rather than a hand-imposed
    # static envelope, provide the abstention. This tests whether the large
    # pool of correct-but-rejected candidates can be recovered safely before
    # implementing a new runtime selector.
    temporal_policy_folds = []
    temporal_out_of_fold_good = 0
    temporal_out_of_fold_bad = 0
    temporal_selected_counts: Counter[str] = Counter()
    for domain, block in folds:
        test_fold = (domain, block)
        excluded = {
            fold
            for fold in folds
            if fold[0] == domain and abs(fold[1] - block) <= purge_blocks
        }
        train_folds = [fold for fold in folds if fold not in excluded]
        train_eligible = sum(len(rows_by_fold[fold]) for fold in train_folds)
        zero_bad = []
        for policy in policies:
            good = sum(
                confirmed_by_policy[policy][fold][0]
                for fold in train_folds
            )
            bad = sum(
                confirmed_by_policy[policy][fold][1]
                for fold in train_folds
            )
            if bad == 0:
                zero_bad.append(
                    (policy, metrics_from_counts(train_eligible, good, bad))
                )
        selected, train_metrics = (
            max(
                zero_bad,
                key=lambda item: (
                    item[1]["accepted_good_epochs"],
                    item[0].covariance_scale,
                    item[0].minimum_pairs,
                    -item[0].maximum_second_position_delta_m,
                    -item[0].maximum_nis_per_observation,
                ),
            )
            if zero_bad
            else (None, metrics_from_counts(train_eligible, 0, 0))
        )
        if selected is None:
            test_metrics = metrics_from_counts(
                len(rows_by_fold[test_fold]), 0, 0
            )
        else:
            test_good, test_bad = confirmed_by_policy[selected][test_fold]
            test_metrics = metrics_from_counts(
                len(rows_by_fold[test_fold]), test_good, test_bad
            )
        temporal_out_of_fold_good += test_metrics["accepted_good_epochs"]
        temporal_out_of_fold_bad += test_metrics["accepted_bad_epochs"]
        payload = asdict(selected) if selected is not None else None
        temporal_selected_counts[json.dumps(payload, sort_keys=True)] += 1
        temporal_policy_folds.append(
            {
                "test_domain": domain,
                "test_block": block,
                "purge_blocks": purge_blocks,
                "selection_state_machine": {
                    "minimum_contiguous_epochs": 2,
                    "maximum_correction_jump_m": 0.03,
                    "maximum_epoch_gap_s": 0.21,
                },
                "train": train_metrics,
                "test": test_metrics,
                "selected_policy": payload,
            }
        )

    all_data_zero_bad = []
    for policy in policies:
        good = sum(
            counts[0] for counts in accepted_by_policy[policy].values()
        )
        bad = sum(
            counts[1] for counts in accepted_by_policy[policy].values()
        )
        if bad == 0:
            all_data_zero_bad.append(
                (policy, metrics_from_counts(len(rows), good, bad))
            )
    final_policy, final_metrics = (
        max(
            all_data_zero_bad,
            key=lambda item: (
                item[1]["accepted_good_epochs"],
                item[0].covariance_scale,
                item[0].minimum_pairs,
                -item[0].maximum_second_position_delta_m,
                -item[0].maximum_nis_per_observation,
            ),
        )
        if all_data_zero_bad
        else (None, metrics_from_counts(len(rows), 0, 0))
    )
    domains = sorted({row["_domain"] for row in rows})
    confirmation_keys = sorted(
        {
            key
            for fold in fold_results
            for key in fold["test_confirmation_sweep"]
        }
    )
    out_of_fold_by_domain = {}
    for domain in domains:
        domain_folds = [
            fold for fold in fold_results if fold["test_domain"] == domain
        ]
        eligible = sum(fold["test"]["eligible_epochs"] for fold in domain_folds)
        good = sum(
            fold["test"]["accepted_good_epochs"] for fold in domain_folds
        )
        bad = sum(
            fold["test"]["accepted_bad_epochs"] for fold in domain_folds
        )
        confirmation = {}
        for key in confirmation_keys:
            confirmed_good = sum(
                fold["test_confirmation_sweep"].get(
                    key, {"accepted_good_epochs": 0}
                )["accepted_good_epochs"]
                for fold in domain_folds
            )
            confirmed_bad = sum(
                fold["test_confirmation_sweep"].get(
                    key, {"accepted_bad_epochs": 0}
                )["accepted_bad_epochs"]
                for fold in domain_folds
            )
            confirmation[key] = metrics_from_counts(
                eligible, confirmed_good, confirmed_bad
            )
        out_of_fold_by_domain[domain] = {
            "instantaneous": metrics_from_counts(eligible, good, bad),
            "confirmation_sweep_diagnostic_only": confirmation,
        }
    return {
        "schema": "gnss_gpu_wp174_ffrt_blocked_cv_v1",
        "selection_truth_usage": "training_folds_only",
        "test_truth_usage": "held_out_block_audit_only",
        "runtime_fgo": False,
        "promotion_ready": False,
        "promotion_blocker": (
            "no untouched route/domain remains; CV is exploratory and must "
            "not be treated as final holdout evidence"
        ),
        "purge_blocks": purge_blocks,
        "candidate_policy_count": len(policies),
        "truth_labeled_epochs": len(rows),
        "out_of_fold": {
            "accepted_good_epochs": out_of_fold_good,
            "accepted_bad_epochs": out_of_fold_bad,
            "accepted_good_rate_pct": (
                100.0 * out_of_fold_good / len(rows) if rows else 0.0
            ),
        },
        "confirmed_policy_out_of_fold": metrics_from_counts(
            len(rows),
            confirmed_out_of_fold_good,
            confirmed_out_of_fold_bad,
        ),
        "confirmed_policy_safety_envelope": confirmed_safety_envelope,
        "confirmed_policy_candidate_count": len(confirmed_policies),
        "confirmed_policy_selected_counts": dict(confirmed_selected_counts),
        "confirmed_policy_folds": confirmed_policy_folds,
        "temporal_policy_diagnostic_only": {
            "candidate_policy_count": len(policies),
            "out_of_fold": metrics_from_counts(
                len(rows),
                temporal_out_of_fold_good,
                temporal_out_of_fold_bad,
            ),
            "selected_counts": dict(temporal_selected_counts),
            "folds": temporal_policy_folds,
        },
        "out_of_fold_confirmation_sweep_diagnostic_only": {
            f"streak{streak}_correction_jump{jump:g}m": metrics_from_counts(
                len(rows), totals["good"], totals["bad"]
            )
            for (streak, jump), totals in confirmation_totals.items()
        },
        "out_of_fold_by_domain": out_of_fold_by_domain,
        "selected_policy_counts": dict(selected_counts),
        "exploratory_all_data_policy": (
            asdict(final_policy) if final_policy is not None else None
        ),
        "exploratory_all_data_metrics": final_metrics,
        "folds": fold_results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit",
        action="append",
        required=True,
        help="repeatable DOMAIN=PATH augmented audit CSV",
    )
    parser.add_argument("--purge-blocks", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [
        row
        for specification in args.audit
        for row in _read_audit(specification)
    ]
    summary = analyze(rows, purge_blocks=args.purge_blocks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
