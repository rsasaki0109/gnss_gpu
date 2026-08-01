#!/usr/bin/env python3
"""Audit a fail-closed union of two MultiSD FGO holdout partitions.

Reference truth is opened only after both solver artifacts already exist.  A
production baseline FIX has priority.  For a non-FIX baseline epoch, two shadow
FIX positions are accepted only when they agree within the configured aperture;
otherwise that epoch is rejected as a conflict.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

try:
    from experiments.run_multisd_fgo_ppc_cv import (
        _quantile,
        read_reference,
        read_shadow,
        read_solutions,
    )
except ModuleNotFoundError:  # Direct `python experiments/<script>.py` execution.
    from run_multisd_fgo_ppc_cv import (  # type: ignore[no-redef]
        _quantile,
        read_reference,
        read_shadow,
        read_solutions,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _position(row: dict[str, str]) -> tuple[float, float, float] | None:
    try:
        value = tuple(float(row[axis]) for axis in "xyz")
    except (KeyError, TypeError, ValueError):
        return None
    if not all(math.isfinite(component) for component in value):
        return None
    return value


def _empty_score() -> dict[str, int]:
    return {"fixed": 0, "correct": 0, "false": 0, "above_1m": 0}


def _record_error(score: dict[str, int], error_m: float) -> None:
    score["fixed"] += 1
    score["correct"] += int(error_m < 0.5)
    score["false"] += int(error_m >= 0.5)
    score["above_1m"] += int(error_m > 1.0)


def audit_dual_holdout(
    primary_shadow_path: Path,
    secondary_shadow_path: Path,
    reference_path: Path,
    *,
    baseline_pos_path: Path | None = None,
    maximum_conflict_separation_m: float = 0.1,
) -> dict[str, Any]:
    if maximum_conflict_separation_m < 0.0:
        raise ValueError("maximum conflict separation must be non-negative")

    primary = read_shadow(primary_shadow_path)
    secondary = read_shadow(secondary_shadow_path)
    baseline = read_solutions(baseline_pos_path) if baseline_pos_path else {}

    # Truth is deliberately loaded only after every estimator artifact.
    truth = read_reference(reference_path)
    tows = list(baseline) if baseline else sorted(set(primary) | set(secondary))

    result_score = _empty_score()
    baseline_score = _empty_score()
    primary_rescue_score = _empty_score()
    secondary_rescue_score = _empty_score()
    both_fixed = 0
    primary_only = 0
    secondary_only = 0
    conflicts = 0
    maximum_agreement_separation_m = 0.0
    sequential_runtime_ms: list[float] = []
    parallel_lower_bound_runtime_ms: list[float] = []

    for tow in tows:
        primary_row = primary.get(tow)
        secondary_row = secondary.get(tow)
        if primary_row is not None and secondary_row is not None:
            try:
                primary_runtime = float(primary_row["runtime_ms"])
                secondary_runtime = float(secondary_row["runtime_ms"])
            except (KeyError, TypeError, ValueError):
                pass
            else:
                if math.isfinite(primary_runtime) and math.isfinite(secondary_runtime):
                    sequential_runtime_ms.append(primary_runtime + secondary_runtime)
                    parallel_lower_bound_runtime_ms.append(
                        max(primary_runtime, secondary_runtime)
                    )

        reference = truth.get(tow)
        baseline_row = baseline.get(tow)
        if baseline_row is not None and int(baseline_row["status"]) == 4:
            estimate = tuple(float(baseline_row[axis]) for axis in "xyz")
            error_m = math.inf if reference is None else math.dist(estimate, reference)
            _record_error(baseline_score, error_m)
            _record_error(result_score, error_m)
            continue

        primary_fixed = primary_row is not None and primary_row.get("shadow_fixed") == "1"
        secondary_fixed = (
            secondary_row is not None and secondary_row.get("shadow_fixed") == "1"
        )
        primary_position = _position(primary_row) if primary_fixed else None
        secondary_position = _position(secondary_row) if secondary_fixed else None

        if primary_position is not None:
            error_m = (
                math.inf if reference is None else math.dist(primary_position, reference)
            )
            _record_error(primary_rescue_score, error_m)
        if secondary_position is not None:
            error_m = (
                math.inf if reference is None else math.dist(secondary_position, reference)
            )
            _record_error(secondary_rescue_score, error_m)

        selected: tuple[float, float, float] | None = None
        if primary_position is not None and secondary_position is not None:
            both_fixed += 1
            separation_m = math.dist(primary_position, secondary_position)
            if separation_m > maximum_conflict_separation_m:
                conflicts += 1
                continue
            maximum_agreement_separation_m = max(
                maximum_agreement_separation_m, separation_m
            )
            selected = primary_position
        elif primary_position is not None:
            primary_only += 1
            selected = primary_position
        elif secondary_position is not None:
            secondary_only += 1
            selected = secondary_position

        if selected is not None:
            error_m = math.inf if reference is None else math.dist(selected, reference)
            _record_error(result_score, error_m)

    epochs = len(tows)
    result_score["correct_fix_rate"] = (
        result_score["correct"] / epochs if epochs else 0.0
    )
    result_score["false_per_fixed"] = (
        result_score["false"] / result_score["fixed"]
        if result_score["fixed"]
        else 0.0
    )
    return {
        "schema": "gnss_gpu_multisd_fgo_dual_holdout_audit_v1",
        "epochs": epochs,
        "maximum_conflict_separation_m": maximum_conflict_separation_m,
        "truth_usage": "post_solver_scoring_only",
        "estimator_inputs": "pre-existing solver artifacts only",
        "result": result_score,
        "baseline": baseline_score,
        "primary_rescues": primary_rescue_score,
        "secondary_rescues": secondary_rescue_score,
        "consensus": {
            "both_fixed": both_fixed,
            "primary_only": primary_only,
            "secondary_only": secondary_only,
            "conflicts_rejected": conflicts,
            "maximum_accepted_separation_m": maximum_agreement_separation_m,
        },
        "runtime": {
            "sequential_p95_ms": _quantile(sequential_runtime_ms, 0.95),
            "sequential_max_ms": max(sequential_runtime_ms, default=None),
            "parallel_lower_bound_p95_ms": _quantile(
                parallel_lower_bound_runtime_ms, 0.95
            ),
            "parallel_lower_bound_max_ms": max(
                parallel_lower_bound_runtime_ms, default=None
            ),
        },
        "artifacts": {
            "primary_shadow_sha256": _sha256(primary_shadow_path),
            "secondary_shadow_sha256": _sha256(secondary_shadow_path),
            "reference_sha256": _sha256(reference_path),
            "baseline_pos_sha256": (
                _sha256(baseline_pos_path) if baseline_pos_path else None
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-shadow", type=Path, required=True)
    parser.add_argument("--secondary-shadow", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--baseline-pos", type=Path)
    parser.add_argument("--maximum-conflict-separation", type=float, default=0.1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    audit = audit_dual_holdout(
        args.primary_shadow,
        args.secondary_shadow,
        args.reference,
        baseline_pos_path=args.baseline_pos,
        maximum_conflict_separation_m=args.maximum_conflict_separation,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
