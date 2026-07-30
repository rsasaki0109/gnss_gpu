#!/usr/bin/env python3
"""Audit a causal WP173 + WP174 shadow-candidate FIX union."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.analyze_wp174_ffrt_calibration import (  # noqa: E402
    Policy,
    _accepts,
)


@dataclass(frozen=True)
class StateMachineConfig:
    acquisition_streak: int = 2
    maximum_correction_jump_m: float = 0.03
    maximum_epoch_gap_s: float = 0.21
    maximum_hold_epochs: int = 0
    maximum_hold_correction_jump_m: float = math.inf
    maximum_hold_nis_per_observation: float = math.inf
    maximum_prefit_residual_rms_m: float = math.inf
    maximum_hold_prefit_residual_rms_m: float = math.inf
    maximum_hold_second_position_delta_m: float = math.inf
    minimum_hold_pairs: int = 0
    temporal_acquisition_streak: int = 12
    temporal_maximum_hold_epochs: int = 0


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def _tow(value: str | float) -> float:
    return round(float(value), 3)


def _finite(row: dict[str, str], key: str) -> float | None:
    try:
        value = float(row.get(key, ""))
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _finite_or_infinity(row: dict[str, str], key: str) -> float:
    value = _finite(row, key)
    return value if value is not None else math.inf


def _integer_or_zero(row: dict[str, str], key: str) -> int:
    try:
        return int(float(row.get(key, "0")))
    except ValueError:
        return 0


def _quantile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _union_reacquisition_metrics(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    durations: list[float] = []
    outage_start: float | None = None
    previously_fixed = False
    for row in rows:
        tow = float(row["tow"])
        fixed = bool(row["union_fix"])
        if previously_fixed and not fixed and outage_start is None:
            outage_start = tow
        elif fixed and outage_start is not None:
            durations.append(tow - outage_start)
            outage_start = None
        previously_fixed = fixed
    return {
        "union_reacquisition_events": len(durations),
        "union_reacquisition_p95_s": _quantile(durations, 0.95),
        "union_reacquisition_max_s": max(durations) if durations else None,
        "terminal_outage_open": outage_start is not None,
    }


def _policy_by_fold(
    cv_summary: dict[str, Any],
    domain: str,
    family: str,
) -> dict[int, Policy | None]:
    result: dict[int, Policy | None] = {}
    if family == "confirmed":
        source_folds = cv_summary.get("confirmed_policy_folds")
    elif family == "temporal":
        source_folds = cv_summary.get(
            "temporal_policy_diagnostic_only", {}
        ).get("folds")
    else:
        source_folds = None
    if source_folds is None:
        source_folds = cv_summary["folds"]
    for fold in source_folds:
        if fold["test_domain"] != domain:
            continue
        payload = fold["selected_policy"]
        result[int(fold["test_block"])] = (
            Policy(**payload) if payload is not None else None
        )
    return result


def causal_candidate_declarations(
    audit_rows: list[dict[str, str]],
    policies: dict[int, Policy | None],
    config: StateMachineConfig,
) -> tuple[set[float], dict[str, Any]]:
    if config.acquisition_streak < 1 or config.maximum_hold_epochs < 0:
        raise ValueError("invalid state-machine configuration")
    declared: set[float] = set()
    state = "float"
    streak = 0
    hold_remaining = 0
    previous_tow: float | None = None
    previous_correction: tuple[float, float, float] | None = None
    revoke_tow: float | None = None
    reacquisition_s: list[float] = []
    transitions = {"acquire": 0, "hold": 0, "revoke": 0}

    for row in sorted(audit_rows, key=lambda item: float(item["tow"])):
        tow = _tow(row["tow"])
        policy = policies.get(int(row["block"]))
        correction = tuple(
            _finite(row, f"lambda_shadow_best_correction_{axis}")
            for axis in "xyz"
        )
        decision_fields_present = all(value is not None for value in correction)
        quality_pass = (
            policy is not None
            and decision_fields_present
            and _accepts(row, policy)
            and _finite_or_infinity(
                row, "float_update_prefit_residual_rms_m"
            )
            <= config.maximum_prefit_residual_rms_m
        )
        current_correction = (
            tuple(float(value) for value in correction)
            if decision_fields_present
            else None
        )
        continuous = (
            quality_pass
            and previous_tow is not None
            and previous_correction is not None
            and tow - previous_tow <= config.maximum_epoch_gap_s
            and math.dist(current_correction, previous_correction)
            <= config.maximum_correction_jump_m
        )

        if quality_pass:
            streak = streak + 1 if continuous else 1
            previous_tow = tow
            previous_correction = current_correction
            hold_remaining = config.maximum_hold_epochs
            if streak >= config.acquisition_streak:
                if state != "fix":
                    transitions["acquire"] += 1
                    if revoke_tow is not None:
                        reacquisition_s.append(tow - revoke_tow)
                        revoke_tow = None
                state = "fix"
                declared.add(tow)
            else:
                state = "candidate"
            continue

        hold_guard_pass = (
            state == "fix"
            and hold_remaining > 0
            and current_correction is not None
            and previous_tow is not None
            and previous_correction is not None
            and tow - previous_tow <= config.maximum_epoch_gap_s
            and math.dist(current_correction, previous_correction)
            <= config.maximum_hold_correction_jump_m
            and _finite_or_infinity(
                row, "float_update_nis_per_observation"
            )
            <= config.maximum_hold_nis_per_observation
            and _finite_or_infinity(
                row, "float_update_prefit_residual_rms_m"
            )
            <= config.maximum_hold_prefit_residual_rms_m
            and _finite_or_infinity(
                row, "lambda_shadow_second_position_delta_m"
            )
            <= config.maximum_hold_second_position_delta_m
            and _integer_or_zero(row, "pair_count") >= config.minimum_hold_pairs
        )
        streak = 0
        if hold_guard_pass:
            hold_remaining -= 1
            transitions["hold"] += 1
            declared.add(tow)
            previous_tow = tow
            previous_correction = current_correction
        else:
            if state == "fix":
                transitions["revoke"] += 1
                revoke_tow = tow
            state = "float"
            hold_remaining = 0
            previous_tow = None
            previous_correction = None

    return declared, {
        "transitions": transitions,
        "reacquisition_events": len(reacquisition_s),
        "reacquisition_p95_s": _quantile(reacquisition_s, 0.95),
        "reacquisition_max_s": max(reacquisition_s) if reacquisition_s else None,
    }


def analyze_domain(
    domain: str,
    baseline_rows: list[dict[str, str]],
    audit_rows: list[dict[str, str]],
    cv_summary: dict[str, Any],
    config: StateMachineConfig,
    policy_family: str = "confirmed",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if policy_family == "dual":
        families = ("instantaneous", "confirmed")
    elif policy_family == "triple":
        families = ("instantaneous", "confirmed", "temporal")
    else:
        families = (policy_family,)
    candidate_fix: set[float] = set()
    state_metrics_by_family = {}
    for family in families:
        policies = _policy_by_fold(cv_summary, domain, family)
        family_config = (
            replace(
                config,
                acquisition_streak=config.temporal_acquisition_streak,
                maximum_hold_epochs=config.temporal_maximum_hold_epochs,
            )
            if family == "temporal"
            else config
        )
        family_fix, family_metrics = causal_candidate_declarations(
            audit_rows,
            policies,
            family_config,
        )
        candidate_fix |= family_fix
        state_metrics_by_family[family] = family_metrics
    audit_by_tow = {_tow(row["tow"]): row for row in audit_rows}
    output_rows: list[dict[str, Any]] = []
    for baseline in baseline_rows:
        tow = _tow(baseline["tow"])
        audit = audit_by_tow.get(tow)
        baseline_fix = baseline["fix"] == "1"
        candidate_declared = tow in candidate_fix
        candidate_good = (
            audit is not None and audit["shadow_best_sub50cm"] == "1"
        )
        candidate_bad = (
            audit is not None and audit["shadow_best_sub50cm"] == "0"
        )
        candidate_only = candidate_declared and not baseline_fix
        union_fix = baseline_fix or candidate_declared
        if candidate_only and audit is not None:
            union_ecef = tuple(
                _finite(audit, f"lambda_shadow_best_ecef_{axis}")
                for axis in "xyz"
            )
            union_position_source = "shadow_candidate"
        elif baseline_fix:
            union_ecef = tuple(
                _finite(baseline, f"ecef_{axis}") for axis in "xyz"
            )
            union_position_source = "baseline"
        else:
            union_ecef = (None, None, None)
            union_position_source = ""
        union_position_valid = union_fix and all(
            value is not None for value in union_ecef
        )
        if union_fix and not union_position_valid:
            # A declared FIX without a finite position cannot seed a bridge.
            # Keep the decision audit intact, but make the missing anchor
            # explicit and fail closed in downstream consumers.
            union_position_source = ""
        union_false_fix = (
            baseline["false_fix"] == "1"
            or (candidate_only and candidate_bad)
        )
        baseline_sub50cm = baseline["sub50cm"] == "1"
        union_sub50cm = (
            candidate_good if candidate_only else baseline_sub50cm
        )
        output_rows.append(
            {
                "tow": tow,
                "baseline_fix": int(baseline_fix),
                "candidate_fix": int(candidate_declared),
                "candidate_only_fix": int(candidate_only),
                "union_fix": int(union_fix),
                "union_false_fix": int(union_false_fix),
                "baseline_sub50cm": int(baseline_sub50cm),
                "union_sub50cm": int(union_sub50cm),
                "union_ecef_x": (
                    union_ecef[0] if union_position_valid else ""
                ),
                "union_ecef_y": (
                    union_ecef[1] if union_position_valid else ""
                ),
                "union_ecef_z": (
                    union_ecef[2] if union_position_valid else ""
                ),
                "union_position_source": union_position_source,
            }
        )

    denominator = len(output_rows)
    baseline_fix_count = sum(row["baseline_fix"] for row in output_rows)
    candidate_count = sum(row["candidate_fix"] for row in output_rows)
    candidate_only_count = sum(row["candidate_only_fix"] for row in output_rows)
    union_fix_count = sum(row["union_fix"] for row in output_rows)
    false_fix_count = sum(row["union_false_fix"] for row in output_rows)
    union_sub50cm = sum(row["union_sub50cm"] for row in output_rows)
    union_position_epochs = sum(
        bool(row["union_position_source"]) for row in output_rows
    )
    summary = {
        "domain": domain,
        "denominator_epochs": denominator,
        "baseline_fix_epochs": baseline_fix_count,
        "candidate_fix_epochs": candidate_count,
        "candidate_only_fix_epochs": candidate_only_count,
        "overlap_fix_epochs": candidate_count - candidate_only_count,
        "union_fix_epochs": union_fix_count,
        "union_fix_rate_pct": 100.0 * union_fix_count / denominator,
        "union_false_fix_epochs": false_fix_count,
        "union_sub50cm_epochs": union_sub50cm,
        "union_sub50cm_rate_pct": 100.0 * union_sub50cm / denominator,
        "union_position_epochs": union_position_epochs,
        **_union_reacquisition_metrics(output_rows),
        "state_metrics_by_policy_family": state_metrics_by_family,
    }
    return output_rows, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--cv-summary", type=Path, required=True)
    parser.add_argument("--output-epochs", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--acquisition-streak", type=int, default=2)
    parser.add_argument("--maximum-correction-jump-m", type=float, default=0.03)
    parser.add_argument("--maximum-epoch-gap-s", type=float, default=0.21)
    parser.add_argument("--maximum-hold-epochs", type=int, default=0)
    parser.add_argument(
        "--maximum-hold-correction-jump-m",
        type=float,
        default=math.inf,
    )
    parser.add_argument(
        "--maximum-hold-nis-per-observation",
        type=float,
        default=math.inf,
    )
    parser.add_argument(
        "--maximum-hold-second-position-delta-m",
        type=float,
        default=math.inf,
    )
    parser.add_argument("--minimum-hold-pairs", type=int, default=0)
    parser.add_argument("--temporal-acquisition-streak", type=int, default=12)
    parser.add_argument("--temporal-maximum-hold-epochs", type=int, default=0)
    parser.add_argument(
        "--policy-family",
        choices=("instantaneous", "confirmed", "temporal", "dual", "triple"),
        default="confirmed",
    )
    args = parser.parse_args()
    config = StateMachineConfig(
        acquisition_streak=args.acquisition_streak,
        maximum_correction_jump_m=args.maximum_correction_jump_m,
        maximum_epoch_gap_s=args.maximum_epoch_gap_s,
        maximum_hold_epochs=args.maximum_hold_epochs,
        maximum_hold_correction_jump_m=(
            args.maximum_hold_correction_jump_m
        ),
        maximum_hold_nis_per_observation=(
            args.maximum_hold_nis_per_observation
        ),
        maximum_hold_second_position_delta_m=(
            args.maximum_hold_second_position_delta_m
        ),
        minimum_hold_pairs=args.minimum_hold_pairs,
        temporal_acquisition_streak=args.temporal_acquisition_streak,
        temporal_maximum_hold_epochs=args.temporal_maximum_hold_epochs,
    )
    rows, domain_summary = analyze_domain(
        args.domain,
        _read_csv(args.baseline),
        _read_csv(args.audit),
        json.loads(args.cv_summary.read_text(encoding="utf-8")),
        config,
        policy_family=args.policy_family,
    )
    args.output_epochs.parent.mkdir(parents=True, exist_ok=True)
    with args.output_epochs.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "schema": "gnss_gpu_wp174_safe_union_audit_v1",
        "runtime_fgo": False,
        "truth_usage": "post_selection_audit_only",
        "promotion_ready": False,
        "promotion_blocker": (
            "no untouched route/domain remains; blocked CV is exploratory"
        ),
        "state_machine": {
            "policy_family": args.policy_family,
            "acquisition_streak": config.acquisition_streak,
            "maximum_correction_jump_m": config.maximum_correction_jump_m,
            "maximum_epoch_gap_s": config.maximum_epoch_gap_s,
            "maximum_hold_epochs": config.maximum_hold_epochs,
            "maximum_hold_correction_jump_m": (
                config.maximum_hold_correction_jump_m
            ),
            "maximum_hold_nis_per_observation": (
                config.maximum_hold_nis_per_observation
            ),
            "maximum_hold_second_position_delta_m": (
                config.maximum_hold_second_position_delta_m
            ),
            "minimum_hold_pairs": config.minimum_hold_pairs,
            "temporal_acquisition_streak": (
                config.temporal_acquisition_streak
            ),
            "temporal_maximum_hold_epochs": (
                config.temporal_maximum_hold_epochs
            ),
            "quality_failure_action": "immediate_revoke",
        },
        "result": domain_summary,
    }
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
