#!/usr/bin/env python3
"""Blocked leave-one-city/time-block-out CV for the safe basin union."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from experiments.compose_ppc_safe_basin_union import compose_safe_union
    from experiments.run_multisd_fgo_ppc_cv import read_reference
except ModuleNotFoundError:
    from compose_ppc_safe_basin_union import compose_safe_union  # type: ignore[no-redef]
    from run_multisd_fgo_ppc_cv import read_reference  # type: ignore[no-redef]


@dataclass(frozen=True)
class CitySpec:
    name: str
    monitor: Path
    active: Path
    integrity: Path
    tracker: Path
    reference: Path


@dataclass(frozen=True, order=True)
class Policy:
    motion_limit_m: float
    maximum_resets: int
    promotion_streak: int


POLICIES = tuple(
    Policy(motion, resets, streak)
    for motion in (0.20, 0.25, 0.30)
    for resets in (0, 1, 2)
    for streak in (2, 3)
)


def _parse_city(value: str) -> CitySpec:
    fields = value.split("=", 5)
    if len(fields) != 6:
        raise ValueError(
            "city must be NAME=MONITOR=ACTIVE=INTEGRITY=TRACKER=REFERENCE"
        )
    return CitySpec(fields[0], *(Path(field) for field in fields[1:]))


def blocked_cv(specifications: list[CitySpec], block_count: int = 5) -> dict[str, Any]:
    if {spec.name for spec in specifications} != {"tokyo", "nagoya"}:
        raise ValueError("exactly Tokyo and Nagoya specifications are required")
    if block_count < 2:
        raise ValueError("block count must be at least two")

    # Produce every truth-free candidate decision stream before opening any
    # reference. Training truth selects a policy, never estimator decisions.
    decisions: dict[tuple[str, Policy], list[dict[str, object]]] = {}
    for spec in specifications:
        for policy in POLICIES:
            decisions[(spec.name, policy)] = compose_safe_union(
                spec.monitor,
                spec.active,
                spec.integrity,
                spec.tracker,
                motion_innovation_limit_m=policy.motion_limit_m,
                maximum_causal_arc_resets=policy.maximum_resets,
                promotion_streak_epochs=policy.promotion_streak,
            )
    truth = {spec.name: read_reference(spec.reference) for spec in specifications}
    blocks: dict[tuple[str, int], set[float]] = {}
    for spec in specifications:
        tows = [float(row["tow"]) for row in decisions[(spec.name, POLICIES[0])]]
        for index in range(block_count):
            start = index * len(tows) // block_count
            stop = (index + 1) * len(tows) // block_count
            blocks[(spec.name, index)] = set(tows[start:stop])

    def score(policy: Policy, selected_blocks: set[tuple[str, int]]) -> tuple[int, int]:
        fixed = 0
        false = 0
        for spec in specifications:
            allowed = set().union(
                *(blocks[key] for key in selected_blocks if key[0] == spec.name)
            )
            for row in decisions[(spec.name, policy)]:
                tow = float(row["tow"])
                if tow not in allowed or int(row["shadow_fixed"]) != 1:
                    continue
                fixed += 1
                reference = truth[spec.name].get(tow)
                position = tuple(float(row[axis]) for axis in "xyz")
                error = math.inf if reference is None else math.dist(position, reference)
                false += int(error >= 0.5)
        return fixed, false

    all_blocks = set(blocks)
    folds = []
    for holdout in sorted(all_blocks):
        training = all_blocks - {holdout}
        candidates = []
        for policy in POLICIES:
            fixed, false = score(policy, training)
            if false == 0:
                # Prefer availability, then the more conservative policy.
                candidates.append(
                    (
                        -fixed,
                        policy.motion_limit_m,
                        policy.maximum_resets,
                        -policy.promotion_streak,
                        policy,
                    )
                )
        selected = min(candidates)[-1] if candidates else None
        holdout_fixed, holdout_false = (
            score(selected, {holdout}) if selected is not None else (0, 0)
        )
        folds.append(
            {
                "holdout": f"{holdout[0]}/block{holdout[1]}",
                "selected_policy": (
                    {
                        "motion_limit_m": selected.motion_limit_m,
                        "maximum_resets": selected.maximum_resets,
                        "promotion_streak": selected.promotion_streak,
                    }
                    if selected is not None
                    else None
                ),
                "holdout_fixed": holdout_fixed,
                "holdout_false": holdout_false,
            }
        )
    return {
        "schema": "gnss_gpu_ppc_safe_basin_union_blocked_cv_v1",
        "truth_usage": "training_blocks_for_policy_selection_and_holdout_audit_only",
        "estimator_truth_usage": "none",
        "policy_family": [
            {
                "motion_limit_m": policy.motion_limit_m,
                "maximum_resets": policy.maximum_resets,
                "promotion_streak": policy.promotion_streak,
            }
            for policy in POLICIES
        ],
        "folds": folds,
        "holdout_fixed": sum(fold["holdout_fixed"] for fold in folds),
        "holdout_false": sum(fold["holdout_false"] for fold in folds),
        "passed": all(
            fold["selected_policy"] is not None and fold["holdout_false"] == 0
            for fold in folds
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city", action="append", required=True)
    parser.add_argument("--blocks", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = blocked_cv([_parse_city(value) for value in args.city], args.blocks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
