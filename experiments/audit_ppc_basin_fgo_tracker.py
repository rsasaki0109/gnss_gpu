#!/usr/bin/env python3
"""Score a completed PPC basin tracker artifact in a truth-only audit process."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

try:
    from experiments.run_multisd_fgo_ppc_cv import (
        _quantile,
        read_reference,
        read_solutions,
    )
except ModuleNotFoundError:  # Direct `python experiments/<script>.py` execution.
    from run_multisd_fgo_ppc_cv import (  # type: ignore[no-redef]
        _quantile,
        read_reference,
        read_solutions,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_position(row: dict[str, str]) -> tuple[float, float, float] | None:
    try:
        position = tuple(float(row[axis]) for axis in "xyz")
    except (KeyError, TypeError, ValueError):
        return None
    return position if all(math.isfinite(value) for value in position) else None


def _read_tracker(path: Path) -> dict[float, dict[str, str]]:
    output: dict[float, dict[str, str]] = {}
    with path.open(encoding="utf-8", newline="") as stream:
        for line_number, row in enumerate(csv.DictReader(stream), start=2):
            try:
                tow = round(float(row["tow"]), 3)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid tracker TOW at line {line_number}") from exc
            if tow in output:
                raise ValueError(f"duplicate tracker TOW {tow} at line {line_number}")
            output[tow] = row
    return output


def _distribution(values: Iterable[float]) -> dict[str, float | None]:
    finite = [float(value) for value in values if math.isfinite(value)]
    return {
        "median_m": _quantile(finite, 0.5),
        "p95_m": _quantile(finite, 0.95),
        "maximum_m": max(finite, default=None),
    }


def audit_tracker(
    tracker_csv: Path,
    reference_csv: Path,
    *,
    tracker_summary: Path | None = None,
    baseline_pos: Path | None = None,
    denominator_pos: Path | None = None,
    correct_threshold_m: float = 0.5,
    block_count: int = 5,
) -> dict[str, Any]:
    """Audit FIX safety and availability over the full reference denominator."""

    if not math.isfinite(correct_threshold_m) or correct_threshold_m <= 0.0:
        raise ValueError("correct threshold must be finite and positive")
    if block_count < 2:
        raise ValueError("block count must be at least two")

    # Load and validate every estimator artifact before opening reference truth.
    tracker = _read_tracker(tracker_csv)
    baseline = read_solutions(baseline_pos) if baseline_pos is not None else {}
    denominator = (
        read_solutions(denominator_pos) if denominator_pos is not None else {}
    )
    tracker_hash = _sha256(tracker_csv)
    summary: dict[str, Any] | None = None
    summary_valid = tracker_summary is None
    if tracker_summary is not None:
        summary = json.loads(tracker_summary.read_text(encoding="utf-8"))
        summary_valid = (
            summary.get("schema")
            in {
                "gnss_gpu_ppc_basin_fgo_tracker_v1",
                "gnss_gpu_ppc_imu_safe_output_v1",
            }
            and summary.get("production_input_truth") is False
            and summary.get("truth_usage") == "none"
            and summary.get("output_sha256") == tracker_hash
        )

    # Process boundary: reference truth is first opened here, after artifacts.
    truth = read_reference(reference_csv)
    fixed = 0
    correct = 0
    false = 0
    above_1m = 0
    missing_truth = 0
    fixed_errors: list[float] = []
    ffbsi_errors: list[float] = []
    ffbsi_missing_truth = 0
    tracker_rows_outside_reference = 0
    baseline_fixed = 0
    baseline_correct = 0
    baseline_false = 0
    baseline_above_1m = 0
    rescue_fixed = 0
    rescue_correct = 0
    rescue_false = 0
    rescue_above_1m = 0
    for tow, row in tracker.items():
        if tow not in truth:
            tracker_rows_outside_reference += 1
        try:
            is_fixed = int(row.get("shadow_fixed", "0")) == 1
        except ValueError as exc:
            raise ValueError(f"invalid shadow_fixed at TOW {tow}") from exc
        if not is_fixed:
            continue
        fixed += 1
        position = _finite_position(row)
        reference = truth.get(tow)
        error_m = (
            math.inf
            if position is None or reference is None
            else math.dist(position, reference)
        )
        missing_truth += int(reference is None)
        fixed_errors.append(error_m)
        correct += int(error_m < correct_threshold_m)
        false += int(error_m >= correct_threshold_m)
        above_1m += int(error_m > 1.0)

    evaluation_tows = (
        set(baseline)
        if baseline_pos is not None
        else set(denominator)
        if denominator_pos is not None
        else set(truth)
    )
    baseline_rows_without_truth = 0
    for tow in sorted(evaluation_tows):
        reference = truth.get(tow)
        if reference is None:
            baseline_rows_without_truth += 1
            continue
        baseline_row = baseline.get(tow)
        if baseline_row is not None and int(baseline_row["status"]) == 4:
            error = math.dist(
                tuple(float(baseline_row[axis]) for axis in "xyz"), reference
            )
            baseline_fixed += 1
            baseline_correct += int(error < correct_threshold_m)
            baseline_false += int(error >= correct_threshold_m)
            baseline_above_1m += int(error > 1.0)
            continue
        tracker_row = tracker.get(tow)
        if tracker_row is None or int(tracker_row.get("shadow_fixed", "0")) != 1:
            continue
        position = _finite_position(tracker_row)
        error = math.inf if position is None else math.dist(position, reference)
        rescue_fixed += 1
        rescue_correct += int(error < correct_threshold_m)
        rescue_false += int(error >= correct_threshold_m)
        rescue_above_1m += int(error > 1.0)

    for row in tracker.values():
        try:
            valid = int(row.get("ffbsi_valid", "0")) == 1
        except ValueError as exc:
            raise ValueError("invalid ffbsi_valid") from exc
        if not valid:
            continue
        try:
            smoothed_tow = round(float(row["ffbsi_tow"]), 3)
            position = tuple(float(row[f"ffbsi_{axis}"]) for axis in "xyz")
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid delayed FFBSi row") from exc
        reference = truth.get(smoothed_tow)
        if reference is None or not all(math.isfinite(value) for value in position):
            ffbsi_missing_truth += 1
            continue
        ffbsi_errors.append(math.dist(position, reference))

    total_epochs = len(evaluation_tows)
    ordered_evaluation_tows = sorted(evaluation_tows)
    contiguous_time_blocks: dict[str, dict[str, Any]] = {}
    for block_index in range(block_count):
        start = block_index * total_epochs // block_count
        stop = (block_index + 1) * total_epochs // block_count
        block_tows = ordered_evaluation_tows[start:stop]
        block_errors: list[float] = []
        for tow in block_tows:
            row = tracker.get(tow)
            if row is None or int(row.get("shadow_fixed", "0")) != 1:
                continue
            position = _finite_position(row)
            reference = truth.get(tow)
            block_errors.append(
                math.inf
                if position is None or reference is None
                else math.dist(position, reference)
            )
        contiguous_time_blocks[str(block_index)] = {
            "start_tow": block_tows[0] if block_tows else None,
            "end_tow": block_tows[-1] if block_tows else None,
            "epochs": len(block_tows),
            "fixed": len(block_errors),
            "correct_fix": sum(
                error < correct_threshold_m for error in block_errors
            ),
            "false_fix": sum(
                error >= correct_threshold_m for error in block_errors
            ),
            "false_fix_above_1m": sum(error > 1.0 for error in block_errors),
        }
    return {
        "schema": "gnss_gpu_ppc_basin_fgo_tracker_audit_v1",
        "truth_usage": "post_estimator_scoring_only",
        "truth_opened_after_estimator_artifacts": True,
        "correct_threshold_m": correct_threshold_m,
        "denominator_contract": (
            "baseline_solution_epochs"
            if baseline_pos is not None
            else "explicit_solution_epochs"
            if denominator_pos is not None
            else "reference_epochs"
        ),
        "total_epochs": total_epochs,
        "tracker_epochs": len(tracker),
        "fixed": fixed,
        "correct_fix": correct,
        "false_fix": false,
        "false_fix_above_1m": above_1m,
        "fix_rate_full_denominator": correct / total_epochs if total_epochs else 0.0,
        "false_per_fixed": false / fixed if fixed else 0.0,
        "fixed_error": _distribution(fixed_errors),
        "contiguous_time_blocks": contiguous_time_blocks,
        "baseline_priority_union": {
            "baseline_fixed": baseline_fixed,
            "baseline_correct_fix": baseline_correct,
            "baseline_false_fix": baseline_false,
            "baseline_false_fix_above_1m": baseline_above_1m,
            "tracker_rescue_fixed": rescue_fixed,
            "tracker_rescue_correct_fix": rescue_correct,
            "tracker_rescue_false_fix": rescue_false,
            "tracker_rescue_false_fix_above_1m": rescue_above_1m,
            "fixed": baseline_fixed + rescue_fixed,
            "correct_fix": baseline_correct + rescue_correct,
            "false_fix": baseline_false + rescue_false,
            "false_fix_above_1m": baseline_above_1m + rescue_above_1m,
            "correct_fix_rate_full_denominator": (
                (baseline_correct + rescue_correct) / total_epochs
                if total_epochs
                else 0.0
            ),
        },
        "delayed_ffbsi": {
            "evaluated_epochs": len(ffbsi_errors),
            "below_0_5m": sum(error < 0.5 for error in ffbsi_errors),
            "above_1m": sum(error > 1.0 for error in ffbsi_errors),
            "error": _distribution(ffbsi_errors),
            "missing_truth_epochs": ffbsi_missing_truth,
        },
        "integrity": {
            "tracker_summary_valid": summary_valid,
            "tracker_rows_outside_reference": tracker_rows_outside_reference,
            "fixed_rows_without_truth": missing_truth,
            "baseline_rows_without_truth": baseline_rows_without_truth,
            "passed": summary_valid
            and tracker_rows_outside_reference == 0
            and missing_truth == 0
            and baseline_rows_without_truth == 0
            and false == 0
            and above_1m == 0,
        },
        "artifacts": {
            "tracker_csv_sha256": tracker_hash,
            "tracker_summary_sha256": (
                _sha256(tracker_summary) if tracker_summary is not None else None
            ),
            "reference_csv_sha256": _sha256(reference_csv),
            "baseline_pos_sha256": _sha256(baseline_pos) if baseline_pos else None,
            "denominator_pos_sha256": (
                _sha256(denominator_pos) if denominator_pos else None
            ),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracker-csv", type=Path, required=True)
    parser.add_argument("--tracker-summary", type=Path)
    parser.add_argument("--baseline-pos", type=Path)
    parser.add_argument("--denominator-pos", type=Path)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--correct-threshold-m", type=float, default=0.5)
    parser.add_argument("--blocks", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = audit_tracker(
        args.tracker_csv,
        args.reference,
        tracker_summary=args.tracker_summary,
        baseline_pos=args.baseline_pos,
        denominator_pos=args.denominator_pos,
        correct_threshold_m=args.correct_threshold_m,
        block_count=args.blocks,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
