#!/usr/bin/env python3
"""Audit the fixed WP174 temporal/strong/change-point policy by domain."""

from __future__ import annotations

import argparse
import csv
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


def _read(specification: str) -> tuple[str, list[dict[str, str]]]:
    domain, raw_path = specification.split("=", 1)
    with Path(raw_path).open(newline="", encoding="utf-8-sig") as stream:
        rows = [
            row
            for row in csv.DictReader(stream)
            if row.get("shadow_best_sub50cm") in {"0", "1"}
        ]
    rows.sort(key=lambda row: float(row["tow"]))
    return domain, rows


def _correction(row: dict[str, str]) -> tuple[float, float, float]:
    return tuple(
        _finite(row, f"lambda_shadow_best_correction_{axis}")
        for axis in "xyz"
    )


def _base_eligible(row: dict[str, str]) -> bool:
    pairs = _finite(row, "pair_count")
    ratio = _finite(row, "lambda_shadow_ratio")
    return (
        math.isfinite(pairs)
        and math.isfinite(ratio)
        and passes_ffrt(
            int(pairs),
            _finite(row, "lambda_shadow_bsr_qscale16"),
            ratio,
        )
        and pairs >= 12
        and _finite(row, "lambda_shadow_second_position_delta_m")
        <= 0.25
        and _finite(row, "float_update_nis_per_observation") <= 3.0
        and _finite(row, "float_update_prefit_residual_rms_m") <= 50.0
        and _consensus_delta(row) <= 0.10
    )


def _strong(row: dict[str, str]) -> bool:
    correction = _correction(row)
    return (
        _base_eligible(row)
        and _finite(row, "lambda_shadow_ratio") >= 10.0
        and _finite(row, "pair_count") >= 20
        and _finite(row, "lambda_shadow_second_position_delta_m")
        <= 0.05
        and _consensus_delta(row) <= 0.01
        and all(math.isfinite(value) for value in correction)
        and math.dist(correction, (0.0, 0.0, 0.0)) <= 0.05
    )


def _change_point_eligible(row: dict[str, str]) -> bool:
    pairs = _finite(row, "pair_count")
    ratio = _finite(row, "lambda_shadow_ratio")
    return (
        math.isfinite(pairs)
        and math.isfinite(ratio)
        and passes_ffrt(
            int(pairs),
            _finite(row, "lambda_shadow_bsr_qscale16"),
            ratio,
        )
        and ratio >= 1.10
        and pairs >= 20
        and _finite(row, "lambda_shadow_second_position_delta_m")
        <= 0.07
        and _finite(row, "float_update_nis_per_observation") <= 3.0
        and _finite(row, "float_update_prefit_residual_rms_m") <= 50.0
        and _consensus_delta(row) <= 0.05
    )


def _decisions(
    rows: list[dict[str, str]],
) -> tuple[list[bool], list[bool], list[bool]]:
    declared: list[bool] = []
    strong_triggers: list[bool] = []
    change_point_triggers: list[bool] = []
    fixed = False
    acquisition_streak = 0
    acquisition_tow: float | None = None
    acquisition_correction: tuple[float, float, float] | None = None
    change_point_armed = False
    change_point_streak = 0
    change_point_tow: float | None = None
    change_point_correction: tuple[float, float, float] | None = None
    for row in rows:
        tow = float(row["tow"])
        correction = _correction(row)
        finite = math.isfinite(tow) and all(
            math.isfinite(value) for value in correction
        )
        normal = (
            _base_eligible(row)
            and _finite(row, "lambda_shadow_ratio") >= 1.4
        )
        row_declared = False
        strong_trigger = False
        change_point_trigger = False

        if fixed:
            change_point_armed = False
            change_point_streak = 0
            change_point_tow = None
            change_point_correction = None
            fixed_contiguous = (
                normal
                and finite
                and acquisition_tow is not None
                and 0.0 < tow - acquisition_tow <= 0.21
                and acquisition_correction is not None
                and math.dist(correction, acquisition_correction) <= 0.02
            )
            if fixed_contiguous:
                row_declared = True
                acquisition_tow = tow
                acquisition_correction = correction
            else:
                fixed = False
                acquisition_streak = 0
                acquisition_tow = None
                acquisition_correction = None
        else:
            if finite:
                cp_contiguous = (
                    change_point_tow is not None
                    and 0.0 < tow - change_point_tow <= 0.21
                    and change_point_correction is not None
                )
                cp_jump = (
                    math.dist(correction, change_point_correction)
                    if change_point_correction is not None
                    else 0.0
                )
                cp_eligible = _change_point_eligible(row)
                if cp_contiguous and cp_jump >= 0.40:
                    change_point_armed = True
                    change_point_streak = 1 if cp_eligible else 0
                elif (
                    change_point_armed
                    and cp_contiguous
                    and cp_eligible
                    and cp_jump <= 0.02
                ):
                    change_point_streak += 1
                else:
                    change_point_streak = 0
                change_point_tow = tow
                change_point_correction = correction

            if change_point_armed and change_point_streak >= 3:
                row_declared = True
                change_point_trigger = True
                fixed = True
                acquisition_streak = change_point_streak
                acquisition_tow = tow
                acquisition_correction = correction
                change_point_armed = False
                change_point_streak = 0
                change_point_tow = None
                change_point_correction = None
            elif normal and finite:
                if _strong(row):
                    row_declared = True
                    strong_trigger = True
                    fixed = True
                    acquisition_streak = 1
                else:
                    contiguous = (
                        acquisition_tow is not None
                        and 0.0 < tow - acquisition_tow <= 0.21
                        and acquisition_correction is not None
                        and math.dist(
                            correction, acquisition_correction
                        )
                        <= 0.02
                    )
                    acquisition_streak = (
                        acquisition_streak + 1 if contiguous else 1
                    )
                    if acquisition_streak >= 3:
                        row_declared = True
                        fixed = True
                acquisition_tow = tow
                acquisition_correction = correction
            else:
                acquisition_streak = 0
                acquisition_tow = None
                acquisition_correction = None

        declared.append(row_declared)
        strong_triggers.append(strong_trigger)
        change_point_triggers.append(change_point_trigger)
    return declared, strong_triggers, change_point_triggers


def analyze(
    domains: list[tuple[str, list[dict[str, str]]]],
) -> dict[str, Any]:
    summaries = []
    total_good = 0
    total_bad = 0
    for domain, rows in domains:
        declared, strong, change_point = _decisions(rows)
        good = sum(
            decision and row["shadow_best_sub50cm"] == "1"
            for row, decision in zip(rows, declared, strict=True)
        )
        bad = sum(
            decision and row["shadow_best_sub50cm"] == "0"
            for row, decision in zip(rows, declared, strict=True)
        )
        strong_good = sum(
            decision and row["shadow_best_sub50cm"] == "1"
            for row, decision in zip(rows, strong, strict=True)
        )
        strong_bad = sum(
            decision and row["shadow_best_sub50cm"] == "0"
            for row, decision in zip(rows, strong, strict=True)
        )
        change_point_good = sum(
            decision and row["shadow_best_sub50cm"] == "1"
            for row, decision in zip(rows, change_point, strict=True)
        )
        change_point_bad = sum(
            decision and row["shadow_best_sub50cm"] == "0"
            for row, decision in zip(rows, change_point, strict=True)
        )
        total_good += good
        total_bad += bad
        formal_target = 0.35 if domain.lower() == "tokyo" else 0.45
        stretch_target = 0.50 if domain.lower() == "tokyo" else 0.60
        declared_rate = good / len(rows) if rows else 0.0
        summaries.append(
            {
                "domain": domain,
                "truth_labeled_epochs": len(rows),
                "declared_good_epochs": good,
                "declared_bad_epochs": bad,
                "strong_instant_good_epochs": strong_good,
                "strong_instant_bad_epochs": strong_bad,
                "change_point_good_triggers": change_point_good,
                "change_point_bad_triggers": change_point_bad,
                "safe_fix_rate": declared_rate,
                "formal_fix_rate_target": formal_target,
                "stretch_fix_rate_target": stretch_target,
                "pass_formal_fix_rate": (
                    bad == 0 and declared_rate >= formal_target
                ),
                "pass_stretch_fix_rate": (
                    bad == 0 and declared_rate >= stretch_target
                ),
            }
        )
    return {
        "schema": "gnss_gpu_wp174_safe_acquisition_policy_audit_v2",
        "runtime_fgo": False,
        "truth_usage": "post_selection_audit_only",
        "policy_generation": "third_generation_exploratory",
        "fixed_policy": {
            "ffrt_covariance_scale": 16.0,
            "minimum_absolute_ratio": 1.4,
            "normal_acquisition_streak": 3,
            "normal_maximum_correction_jump_m": 0.02,
            "strong_minimum_ratio": 10.0,
            "strong_minimum_pairs": 20,
            "strong_maximum_second_position_delta_m": 0.05,
            "strong_maximum_consensus_delta_m": 0.01,
            "strong_maximum_correction_norm_m": 0.05,
            "change_point_minimum_jump_m": 0.40,
            "change_point_stable_streak": 3,
            "change_point_maximum_stable_jump_m": 0.02,
            "change_point_minimum_ratio": 1.10,
            "change_point_minimum_pairs": 20,
            "change_point_maximum_second_position_delta_m": 0.07,
            "change_point_maximum_consensus_delta_m": 0.05,
        },
        "domains": summaries,
        "total_declared_good_epochs": total_good,
        "total_declared_bad_epochs": total_bad,
        "promotion_ready": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = analyze([_read(specification) for specification in args.audit])
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
