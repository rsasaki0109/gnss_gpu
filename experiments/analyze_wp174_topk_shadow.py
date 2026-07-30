#!/usr/bin/env python3
"""Join WP174 top-K shadow telemetry to post-selection position truth."""

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

from experiments.promote_wp172_pf_seeded_rtk_consensus import (  # noqa: E402
    read_gnssplusplus_pos,
)


FEATURES = (
    "ratio",
    "lambda_shadow_ratio",
    "lambda_shadow_runtime_ms",
    "pair_count",
    "lambda_shadow_bsr",
    "lambda_shadow_bsr_qscale2",
    "lambda_shadow_bsr_qscale4",
    "lambda_shadow_bsr_qscale8",
    "lambda_shadow_bsr_qscale16",
    "lambda_shadow_best_mass",
    "lambda_shadow_effective_candidates",
    "lambda_shadow_best_second_disagreements",
    "lambda_shadow_second_position_delta_m",
    "lambda_shadow_position_spread_max_m",
    "lambda_shadow_best_correction_norm_m",
    "float_update_prefit_residual_rms_m",
    "float_update_nis_per_observation",
    "lambda_src_par_shadow_subset_size",
    "lambda_src_par_shadow_bsr",
    "lambda_src_par_shadow_ratio",
    "lambda_src_par_shadow_second_position_delta_m",
    "lambda_src_par_shadow_best_correction_norm_m",
    "lambda_src_par_shadow_runtime_ms",
    "lambda_satellite_par_shadow_subset_size",
    "lambda_satellite_par_shadow_subsets_evaluated",
    "lambda_satellite_par_shadow_dropped_satellites",
    "lambda_satellite_par_shadow_bsr",
    "lambda_satellite_par_shadow_ratio",
    "lambda_satellite_par_shadow_second_position_delta_m",
    "lambda_satellite_par_shadow_best_correction_norm_m",
    "lambda_satellite_par_shadow_runtime_ms",
    "safe_fix_shadow_independent_consensus_delta_m",
)
DERIVED_FEATURES = {
    "lambda_shadow_ratio",
    "lambda_shadow_best_correction_norm_m",
    "lambda_src_par_shadow_best_correction_norm_m",
    "lambda_satellite_par_shadow_best_correction_norm_m",
}


def _tow(value: str | float) -> float:
    return round(float(value), 3)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def _finite(value: str | float | int | None) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _quantile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _feature_profile(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for feature in FEATURES:
        values = [
            value
            for row in rows
            if (value := _finite(row.get(feature))) is not None
        ]
        output[feature] = {
            "count": len(values),
            "p10": _quantile(values, 0.10),
            "p50": _quantile(values, 0.50),
            "p90": _quantile(values, 0.90),
            "min": min(values) if values else None,
            "max": max(values) if values else None,
        }
    return output


def _shadow_ratio(telemetry: dict[str, str]) -> float | None:
    best = _finite(telemetry.get("lambda_shadow_best_cost"))
    second = _finite(telemetry.get("lambda_shadow_second_cost"))
    if best is None or second is None or best <= 0.0 or second < best:
        return None
    return second / best


def _shadow_candidate_error(
    telemetry: dict[str, str],
    reference: tuple[float, float, float],
    *,
    prefix: str = "lambda_shadow_best_ecef",
) -> float | None:
    candidate = tuple(
        _finite(telemetry.get(f"{prefix}_{axis}"))
        for axis in "xyz"
    )
    if any(value is None for value in candidate):
        return None
    return math.sqrt(
        sum(
            (float(candidate[index]) - reference[index]) ** 2
            for index in range(3)
        )
    )


def _vector_norm(
    telemetry: dict[str, str],
    *,
    prefix: str,
) -> float | None:
    components = tuple(
        _finite(telemetry.get(f"{prefix}_{axis}")) for axis in "xyz"
    )
    if any(value is None for value in components):
        return None
    return math.sqrt(sum(float(value) ** 2 for value in components))


def analyze(
    debug_path: Path,
    position_path: Path,
    reference_path: Path,
    *,
    block_count: int = 12,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if block_count < 2:
        raise ValueError("block_count must be at least two")
    debug = {_tow(row["tow"]): row for row in _read_csv(debug_path)}
    positions = read_gnssplusplus_pos(position_path)
    truth = {
        _tow(row["GPS TOW (s)"]): (
            float(row["ECEF X (m)"]),
            float(row["ECEF Y (m)"]),
            float(row["ECEF Z (m)"]),
        )
        for row in _read_csv(reference_path)
    }

    common_tows = sorted(set(debug) & set(truth))
    rows: list[dict[str, Any]] = []
    for index, tow in enumerate(common_tows):
        telemetry = debug[tow]
        position = positions.get(tow)
        reference = truth[tow]
        error_m = (
            math.sqrt(
                sum(
                    (
                        float(position[f"ecef_{axis}"])
                        - reference[axis_index]
                    )
                    ** 2
                    for axis_index, axis in enumerate("xyz")
                )
            )
            if position is not None
            else None
        )
        block = min(block_count - 1, index * block_count // len(common_tows))
        shadow_best_error_m = _shadow_candidate_error(telemetry, reference)
        shadow_second_error_m = _shadow_candidate_error(
            telemetry,
            reference,
            prefix="lambda_shadow_second_ecef",
        )
        src_best_error_m = _shadow_candidate_error(
            telemetry,
            reference,
            prefix="lambda_src_par_shadow_best_ecef",
        )
        satellite_best_error_m = _shadow_candidate_error(
            telemetry,
            reference,
            prefix="lambda_satellite_par_shadow_best_ecef",
        )
        shadow_correction_norm_m = _vector_norm(
            telemetry,
            prefix="lambda_shadow_best_correction",
        )
        src_correction_norm_m = _vector_norm(
            telemetry,
            prefix="lambda_src_par_shadow_best_correction",
        )
        satellite_correction_norm_m = _vector_norm(
            telemetry,
            prefix="lambda_satellite_par_shadow_best_correction",
        )
        rows.append(
            {
                "tow": tow,
                "block": block,
                "status": int(position["status"]) if position is not None else "",
                "num_satellites": (
                    int(position["num_satellites"])
                    if position is not None
                    else int(telemetry["num_sats"])
                ),
                "error_m": error_m if error_m is not None else "",
                "sub50cm": int(error_m < 0.5) if error_m is not None else "",
                "shadow_best_error_m": (
                    shadow_best_error_m
                    if shadow_best_error_m is not None
                    else ""
                ),
                "shadow_best_sub50cm": (
                    int(shadow_best_error_m < 0.5)
                    if shadow_best_error_m is not None
                    else ""
                ),
                "shadow_second_error_m": (
                    shadow_second_error_m
                    if shadow_second_error_m is not None
                    else ""
                ),
                "shadow_second_sub50cm": (
                    int(shadow_second_error_m < 0.5)
                    if shadow_second_error_m is not None
                    else ""
                ),
                "src_par_best_error_m": (
                    src_best_error_m if src_best_error_m is not None else ""
                ),
                "src_par_best_sub50cm": (
                    int(src_best_error_m < 0.5)
                    if src_best_error_m is not None
                    else ""
                ),
                "satellite_par_best_error_m": (
                    satellite_best_error_m
                    if satellite_best_error_m is not None
                    else ""
                ),
                "satellite_par_best_sub50cm": (
                    int(satellite_best_error_m < 0.5)
                    if satellite_best_error_m is not None
                    else ""
                ),
                "lambda_shadow_best_correction_norm_m": (
                    shadow_correction_norm_m
                    if shadow_correction_norm_m is not None
                    else ""
                ),
                "lambda_src_par_shadow_best_correction_norm_m": (
                    src_correction_norm_m
                    if src_correction_norm_m is not None
                    else ""
                ),
                "lambda_satellite_par_shadow_best_correction_norm_m": (
                    satellite_correction_norm_m
                    if satellite_correction_norm_m is not None
                    else ""
                ),
                **{
                    feature: telemetry.get(feature, "")
                    for feature in FEATURES
                    if feature not in DERIVED_FEATURES
                },
                "lambda_shadow_ratio": _shadow_ratio(telemetry) or "",
                "lambda_shadow_attempted": int(
                    telemetry["lambda_shadow_attempted"]
                ),
                "lambda_shadow_solved": int(telemetry["lambda_shadow_solved"]),
                "lambda_shadow_candidate_count": int(
                    telemetry["lambda_shadow_candidate_count"]
                ),
                "lambda_shadow_ffrt_table_supported": int(
                    telemetry.get("lambda_shadow_ffrt_table_supported", 0)
                ),
                "lambda_shadow_ffrt_accepts_any": int(
                    telemetry.get("lambda_shadow_ffrt_accepts_any", 0)
                ),
                "lambda_shadow_ffrt_passed": int(
                    telemetry.get("lambda_shadow_ffrt_passed", 0)
                ),
                "lambda_shadow_ffrt_min_ratio": telemetry.get(
                    "lambda_shadow_ffrt_min_ratio", ""
                ),
                "lambda_src_par_shadow_attempted": int(
                    telemetry.get("lambda_src_par_shadow_attempted", 0)
                ),
                "lambda_src_par_shadow_solved": int(
                    telemetry.get("lambda_src_par_shadow_solved", 0)
                ),
                "lambda_src_par_shadow_ffrt_passed": int(
                    telemetry.get("lambda_src_par_shadow_ffrt_passed", 0)
                ),
                "lambda_src_par_shadow_ffrt_min_ratio": telemetry.get(
                    "lambda_src_par_shadow_ffrt_min_ratio", ""
                ),
                "lambda_satellite_par_shadow_attempted": int(
                    telemetry.get(
                        "lambda_satellite_par_shadow_attempted", 0
                    )
                ),
                "lambda_satellite_par_shadow_solved": int(
                    telemetry.get("lambda_satellite_par_shadow_solved", 0)
                ),
                "lambda_satellite_par_shadow_ffrt_passed": int(
                    telemetry.get(
                        "lambda_satellite_par_shadow_ffrt_passed", 0
                    )
                ),
                "lambda_satellite_par_shadow_ffrt_min_ratio": telemetry.get(
                    "lambda_satellite_par_shadow_ffrt_min_ratio", ""
                ),
                "safe_fix_shadow_enabled": int(
                    telemetry.get("safe_fix_shadow_enabled", 0)
                ),
                "safe_fix_shadow_state": int(
                    telemetry.get("safe_fix_shadow_state", 0)
                ),
                "safe_fix_shadow_declared_fixed": int(
                    telemetry.get("safe_fix_shadow_declared_fixed", 0)
                ),
                "safe_fix_shadow_candidate_accepted": int(
                    telemetry.get(
                        "safe_fix_shadow_candidate_accepted", 0
                    )
                ),
                "safe_fix_shadow_held": int(
                    telemetry.get("safe_fix_shadow_held", 0)
                ),
                "safe_fix_shadow_revoked": int(
                    telemetry.get("safe_fix_shadow_revoked", 0)
                ),
                "safe_fix_shadow_acquisition_streak": int(
                    telemetry.get(
                        "safe_fix_shadow_acquisition_streak", 0
                    )
                ),
                "safe_fix_shadow_hold_epochs": int(
                    telemetry.get("safe_fix_shadow_hold_epochs", 0)
                ),
                **{
                    f"lambda_shadow_best_ecef_{axis}": telemetry.get(
                        f"lambda_shadow_best_ecef_{axis}", ""
                    )
                    for axis in "xyz"
                },
                **{
                    f"lambda_src_par_shadow_best_ecef_{axis}": telemetry.get(
                        f"lambda_src_par_shadow_best_ecef_{axis}", ""
                    )
                    for axis in "xyz"
                },
                **{
                    f"lambda_shadow_best_correction_{axis}": telemetry.get(
                        f"lambda_shadow_best_correction_{axis}", ""
                    )
                    for axis in "xyz"
                },
                **{
                    f"lambda_shadow_second_ecef_{axis}": telemetry.get(
                        f"lambda_shadow_second_ecef_{axis}", ""
                    )
                    for axis in "xyz"
                },
                **{
                    f"lambda_shadow_second_correction_{axis}": telemetry.get(
                        f"lambda_shadow_second_correction_{axis}", ""
                    )
                    for axis in "xyz"
                },
                **{
                    f"lambda_src_par_shadow_best_correction_{axis}": (
                        telemetry.get(
                            f"lambda_src_par_shadow_best_correction_{axis}",
                            "",
                        )
                    )
                    for axis in "xyz"
                },
                **{
                    f"lambda_satellite_par_shadow_best_ecef_{axis}": (
                        telemetry.get(
                            f"lambda_satellite_par_shadow_best_ecef_{axis}",
                            "",
                        )
                    )
                    for axis in "xyz"
                },
                **{
                    f"lambda_satellite_par_shadow_best_correction_{axis}": (
                        telemetry.get(
                            "lambda_satellite_par_shadow_best_correction_"
                            f"{axis}",
                            "",
                        )
                    )
                    for axis in "xyz"
                },
            }
        )

    shadow = [row for row in rows if row["lambda_shadow_solved"]]
    fixed = [row for row in shadow if row["status"] == 4]
    output_labeled = [row for row in shadow if row["sub50cm"] != ""]
    good = [row for row in output_labeled if row["sub50cm"]]
    bad = [row for row in output_labeled if not row["sub50cm"]]
    candidate_labeled = [
        row for row in shadow if row["shadow_best_sub50cm"] != ""
    ]
    candidate_good = [
        row for row in candidate_labeled if row["shadow_best_sub50cm"]
    ]
    candidate_bad = [
        row for row in candidate_labeled if not row["shadow_best_sub50cm"]
    ]
    second_labeled = [
        row for row in shadow if row["shadow_second_sub50cm"] != ""
    ]
    ffrt_passed = [
        row for row in candidate_labeled if row["lambda_shadow_ffrt_passed"]
    ]
    src_labeled = [
        row for row in shadow if row["src_par_best_sub50cm"] != ""
    ]
    src_ffrt_passed = [
        row
        for row in src_labeled
        if row["lambda_src_par_shadow_ffrt_passed"]
    ]
    satellite_labeled = [
        row
        for row in shadow
        if row["satellite_par_best_sub50cm"] != ""
    ]
    satellite_ffrt_passed = [
        row
        for row in satellite_labeled
        if row["lambda_satellite_par_shadow_ffrt_passed"]
    ]
    safe_fix_declared = [
        row for row in rows if row["safe_fix_shadow_declared_fixed"]
    ]
    safe_fix_labeled = [
        row
        for row in safe_fix_declared
        if row["shadow_best_sub50cm"] != ""
    ]
    shadow_runtime_ms = [
        value
        for row in shadow
        if (value := _finite(row.get("lambda_shadow_runtime_ms"))) is not None
    ]
    satellite_runtime_ms = [
        value
        for row in shadow
        if (
            value := _finite(
                row.get("lambda_satellite_par_shadow_runtime_ms")
            )
        )
        is not None
    ]
    block_profiles = []
    for block in range(block_count):
        selected = [row for row in shadow if row["block"] == block]
        block_profiles.append(
            {
                "block": block,
                "epochs": len(selected),
                "output_labeled_epochs": sum(
                    row["sub50cm"] != "" for row in selected
                ),
                "sub50cm_epochs": sum(
                    row["sub50cm"] == 1 for row in selected
                ),
                "fixed_epochs": sum(row["status"] == 4 for row in selected),
                "false_fixed_epochs": sum(
                    row["status"] == 4 and row["sub50cm"] == 0
                    for row in selected
                ),
                "tow_start": selected[0]["tow"] if selected else None,
                "tow_end": selected[-1]["tow"] if selected else None,
            }
        )

    summary = {
        "schema": "gnss_gpu_wp174_topk_shadow_audit_v1",
        "selection_truth_usage": "none",
        "audit_truth_usage": "post_selection_only",
        "runtime_fgo": False,
        "debug_rows": len(debug),
        "position_rows": len(positions),
        "truth_rows": len(truth),
        "joined_rows": len(rows),
        "output_labeled_shadow_epochs": len(output_labeled),
        "shadow_solved_epochs": len(shadow),
        "shadow_candidate_count_values": sorted(
            {row["lambda_shadow_candidate_count"] for row in shadow}
        ),
        "shadow_runtime_ms": {
            "count": len(shadow_runtime_ms),
            "p50": _quantile(shadow_runtime_ms, 0.50),
            "p95": _quantile(shadow_runtime_ms, 0.95),
            "p99": _quantile(shadow_runtime_ms, 0.99),
            "max": max(shadow_runtime_ms) if shadow_runtime_ms else None,
        },
        "satellite_par_runtime_ms": {
            "count": len(satellite_runtime_ms),
            "p50": _quantile(satellite_runtime_ms, 0.50),
            "p95": _quantile(satellite_runtime_ms, 0.95),
            "p99": _quantile(satellite_runtime_ms, 0.99),
            "max": (
                max(satellite_runtime_ms)
                if satellite_runtime_ms
                else None
            ),
        },
        "shadow_sub50cm_epochs": len(good),
        "shadow_bad_position_epochs": len(bad),
        "solver_fixed_epochs": len(fixed),
        "solver_false_fixed_epochs": sum(
            row["sub50cm"] == 0 for row in fixed
        ),
        "shadow_best_truth_labeled_epochs": len(candidate_labeled),
        "shadow_best_sub50cm_epochs": len(candidate_good),
        "shadow_best_not_sub50cm_epochs": len(candidate_bad),
        "shadow_second_truth_labeled_epochs": len(second_labeled),
        "shadow_second_sub50cm_epochs": sum(
            row["shadow_second_sub50cm"] for row in second_labeled
        ),
        "shadow_second_not_sub50cm_epochs": sum(
            not row["shadow_second_sub50cm"] for row in second_labeled
        ),
        "shadow_second_rescues_best_bad_epochs": sum(
            row["shadow_best_sub50cm"] == 0
            and row["shadow_second_sub50cm"] == 1
            for row in second_labeled
        ),
        "shadow_second_harms_best_good_epochs": sum(
            row["shadow_best_sub50cm"] == 1
            and row["shadow_second_sub50cm"] == 0
            for row in second_labeled
        ),
        "ffrt_001_passed_truth_labeled_epochs": len(ffrt_passed),
        "ffrt_001_passed_best_sub50cm_epochs": sum(
            row["shadow_best_sub50cm"] for row in ffrt_passed
        ),
        "ffrt_001_passed_best_not_sub50cm_epochs": sum(
            not row["shadow_best_sub50cm"] for row in ffrt_passed
        ),
        "src_par_truth_labeled_epochs": len(src_labeled),
        "src_par_best_sub50cm_epochs": sum(
            row["src_par_best_sub50cm"] for row in src_labeled
        ),
        "src_par_best_not_sub50cm_epochs": sum(
            not row["src_par_best_sub50cm"] for row in src_labeled
        ),
        "src_par_ffrt_passed_epochs": len(src_ffrt_passed),
        "src_par_ffrt_passed_best_sub50cm_epochs": sum(
            row["src_par_best_sub50cm"] for row in src_ffrt_passed
        ),
        "src_par_ffrt_passed_best_not_sub50cm_epochs": sum(
            not row["src_par_best_sub50cm"] for row in src_ffrt_passed
        ),
        "satellite_par_truth_labeled_epochs": len(satellite_labeled),
        "satellite_par_best_sub50cm_epochs": sum(
            row["satellite_par_best_sub50cm"]
            for row in satellite_labeled
        ),
        "satellite_par_best_not_sub50cm_epochs": sum(
            not row["satellite_par_best_sub50cm"]
            for row in satellite_labeled
        ),
        "satellite_par_ffrt_passed_epochs": len(
            satellite_ffrt_passed
        ),
        "satellite_par_ffrt_passed_best_sub50cm_epochs": sum(
            row["satellite_par_best_sub50cm"]
            for row in satellite_ffrt_passed
        ),
        "satellite_par_ffrt_passed_best_not_sub50cm_epochs": sum(
            not row["satellite_par_best_sub50cm"]
            for row in satellite_ffrt_passed
        ),
        "runtime_safe_fix_shadow": {
            "enabled_epochs": sum(
                row["safe_fix_shadow_enabled"] for row in rows
            ),
            "declared_fix_epochs": len(safe_fix_declared),
            "truth_labeled_fix_epochs": len(safe_fix_labeled),
            "unlabeled_fix_epochs": (
                len(safe_fix_declared) - len(safe_fix_labeled)
            ),
            "sub50cm_fix_epochs": sum(
                row["shadow_best_sub50cm"] for row in safe_fix_labeled
            ),
            "false_fix_epochs": sum(
                not row["shadow_best_sub50cm"] for row in safe_fix_labeled
            ),
            "candidate_accepted_epochs": sum(
                row["safe_fix_shadow_candidate_accepted"] for row in rows
            ),
            "held_epochs": sum(
                row["safe_fix_shadow_held"] for row in rows
            ),
            "revoked_epochs": sum(
                row["safe_fix_shadow_revoked"] for row in rows
            ),
        },
        "feature_profiles": {
            "sub50cm": _feature_profile(good),
            "not_sub50cm": _feature_profile(bad),
            "shadow_best_sub50cm": _feature_profile(candidate_good),
            "shadow_best_not_sub50cm": _feature_profile(candidate_bad),
        },
        "contiguous_block_profiles": block_profiles,
    }
    return rows, summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--debug", type=Path, required=True)
    parser.add_argument("--positions", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--blocks", type=int, default=12)
    parser.add_argument("--output-epochs", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    args = parser.parse_args()
    rows, summary = analyze(
        args.debug,
        args.positions,
        args.reference,
        block_count=args.blocks,
    )
    _write_csv(args.output_epochs, rows)
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
