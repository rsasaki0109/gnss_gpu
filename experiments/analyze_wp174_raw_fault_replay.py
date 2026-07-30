#!/usr/bin/env python3
"""Audit solver output from a raw-RINEX WP174 fault replay."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.analyze_wp174_safe_union import _quantile, _tow  # noqa: E402
from experiments.analyze_wp175_library_fix_integrity import (  # noqa: E402
    read_library_pos,
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def _solver_status(row: dict[str, str]) -> int:
    """Read the solver status from legacy debug or integrity telemetry."""

    return int(row.get("status") or row.get("library_status") or 0)


def _runtime_ms(row: dict[str, str]) -> float | None:
    """Read the runtime from legacy shadow or full integrity telemetry."""

    raw = (
        row.get("lambda_shadow_runtime_ms")
        or row.get("processing_runtime_ms")
        or ""
    )
    if not raw:
        return None
    value = float(raw)
    return value if math.isfinite(value) else None


def analyze(
    debug_rows: list[dict[str, str]],
    positions: dict[float, dict[str, float | int]],
    reference_rows: list[dict[str, str]],
    manifest: dict,
    fix_source: str = "solver",
) -> dict:
    if fix_source not in {"solver", "safe_fix_shadow"}:
        raise ValueError("unsupported fix_source")
    truth = {
        _tow(row["GPS TOW (s)"]): (
            float(row["ECEF X (m)"]),
            float(row["ECEF Y (m)"]),
            float(row["ECEF Z (m)"]),
        )
        for row in reference_rows
    }
    by_tow = {_tow(row["tow"]): row for row in debug_rows}
    tows = sorted(by_tow)
    positions_by_tow = {
        _tow(tow): position for tow, position in positions.items()
    }
    output_tows = set(positions_by_tow)
    is_fixed = {
        tow: (
            tow in positions_by_tow and
            int(positions_by_tow[tow]["status"]) == 4
            if fix_source == "solver"
            else row.get("safe_fix_shadow_declared_fixed") == "1"
        )
        for tow, row in by_tow.items()
    }
    windows = [
        (float(event["start_tow"]), float(event["end_tow"]))
        for event in manifest["events"]
    ]
    fixed_during = 0
    false_fixed_during = 0
    unlabeled_fixed_during = 0
    lost_during = 0
    recoveries: list[float] = []
    fixed_error_by_tow: dict[float, float] = {}
    fixed_tows = {
        tow for tow in tows if is_fixed[tow]
    }
    for tow, row in by_tow.items():
        if not is_fixed[tow] or tow not in truth:
            continue
        if fix_source == "solver":
            if tow not in positions_by_tow:
                continue
            position_xyz = tuple(
                float(positions_by_tow[tow][f"ecef_{axis}"])
                for axis in "xyz"
            )
        else:
            try:
                position_xyz = tuple(
                    float(row[f"lambda_shadow_best_ecef_{axis}"])
                    for axis in "xyz"
                )
            except (KeyError, TypeError, ValueError):
                continue
            if not all(math.isfinite(value) for value in position_xyz):
                continue
        fixed_error_by_tow[tow] = math.sqrt(
            sum(
                (position_xyz[axis_index] - truth[tow][axis_index]) ** 2
                for axis_index in range(3)
            )
        )
    for start, end in windows:
        affected_tows = [
            tow
            for tow in tows
            if start - 1e-6 <= tow <= end + 1e-6
        ]
        fixed_during += sum(
            is_fixed[tow] for tow in affected_tows
        )
        false_fixed_during += sum(
            fixed_error_by_tow.get(tow, 0.0) >= 0.5
            for tow in affected_tows
        )
        unlabeled_fixed_during += sum(
            is_fixed[tow] and tow not in fixed_error_by_tow
            for tow in affected_tows
        )
        lost_during += sum(
            tow not in output_tows for tow in affected_tows
        )
        next_fixed = next(
            (
                tow
                for tow in tows
                if tow > end and is_fixed[tow]
            ),
            None,
        )
        if next_fixed is not None:
            recoveries.append(next_fixed - end)

    fixed_errors = list(fixed_error_by_tow.values())
    false_fixed = sum(error >= 0.5 for error in fixed_errors)
    unlabeled_fixed = len(fixed_tows - fixed_error_by_tow.keys())
    continuity_tows = {
        tow
        for tow, row in by_tow.items()
        if row.get("safe_float_continuity_used") == "1"
    }
    continuity_errors = [
        math.sqrt(
            sum(
                (
                    float(positions_by_tow[tow][f"ecef_{axis}"])
                    - truth[tow][axis_index]
                )
                ** 2
                for axis_index, axis in enumerate("xyz")
            )
        )
        for tow in continuity_tows
        if tow in positions_by_tow and tow in truth
    ]
    shadow_runtime_ms = [
        runtime
        for row in debug_rows
        if (runtime := _runtime_ms(row)) is not None
    ]
    shadow_runtime_p95_ms = _quantile(shadow_runtime_ms, 0.95)
    reacquisition_p95 = _quantile(recoveries, 0.95)
    return {
        "schema": "gnss_gpu_wp174_raw_fault_replay_audit_v1",
        "fault": manifest["fault"],
        "fix_source": fix_source,
        "injection_layer": "raw_rinex_observations",
        "runtime_fgo": False,
        "truth_usage": "post_selection_audit_only",
        "event_count": len(windows),
        "debug_epochs": len(debug_rows),
        "solution_epochs": sum(tow in output_tows for tow in tows),
        "lost_epochs": sum(tow not in output_tows for tow in tows),
        "solver_status_solution_epochs": sum(
            _solver_status(row) != 0 for row in debug_rows
        ),
        "solver_status_lost_epochs": sum(
            _solver_status(row) == 0 for row in debug_rows
        ),
        "safe_float_continuity_attempted_epochs": sum(
            row.get("safe_float_continuity_attempted") == "1"
            for row in debug_rows
        ),
        "safe_float_continuity_epochs": sum(
            row.get("safe_float_continuity_used") == "1"
            for row in debug_rows
        ),
        "safe_float_continuity_solver_gap_anchor_epochs": sum(
            row.get("safe_float_continuity_solver_gap_anchor") == "1"
            for row in debug_rows
        ),
        "safe_float_continuity_output_epochs": sum(
            tow in output_tows for tow in continuity_tows
        ),
        "safe_float_continuity_truth_labeled_epochs": len(
            continuity_errors
        ),
        "safe_float_continuity_sub50cm_epochs": sum(
            error < 0.5 for error in continuity_errors
        ),
        "safe_float_continuity_error_p95_m": _quantile(
            continuity_errors, 0.95
        ),
        "safe_float_continuity_error_max_m": (
            max(continuity_errors) if continuity_errors else None
        ),
        "safe_float_continuity_nonfloat_epochs": sum(
            _solver_status(by_tow[tow]) != 3 for tow in continuity_tows
        ),
        "safe_fix_shadow_change_point_acquisition_epochs": sum(
            row.get("safe_fix_shadow_change_point_acquisition") == "1"
            for row in debug_rows
        ),
        "safe_fix_shadow_strong_acquisition_epochs": sum(
            row.get("safe_fix_shadow_strong_acquisition") == "1"
            for row in debug_rows
        ),
        "lambda_shadow_runtime_p95_ms": shadow_runtime_p95_ms,
        "lambda_shadow_runtime_max_ms": (
            max(shadow_runtime_ms) if shadow_runtime_ms else None
        ),
        "fixed_epochs_during_fault": fixed_during,
        "false_fixed_epochs_during_fault": false_fixed_during,
        "unlabeled_fixed_epochs_during_fault": unlabeled_fixed_during,
        "lost_epochs_during_fault": lost_during,
        "recovered_events": len(recoveries),
        "reacquisition_p95_s": reacquisition_p95,
        "reacquisition_max_s": max(recoveries) if recoveries else None,
        "truth_labeled_fixed_epochs": len(fixed_errors),
        "unlabeled_fixed_epochs": unlabeled_fixed,
        "false_fixed_epochs": false_fixed,
        "fixed_error_p95_m": _quantile(fixed_errors, 0.95),
        # Informational only: a truth-correct FIX during a fault is safe.
        "pass_no_fix_during_fault": fixed_during == 0,
        "pass_false_fix_zero": false_fixed == 0 and unlabeled_fixed == 0,
        "pass_fault_window_false_fix_zero": (
            false_fixed_during == 0 and unlabeled_fixed_during == 0
        ),
        "pass_fixed_truth_coverage": unlabeled_fixed == 0,
        "pass_safe_float_continuity_float_only": all(
            _solver_status(by_tow[tow]) == 3 for tow in continuity_tows
        ),
        "pass_safe_float_continuity_output_coverage": all(
            tow in output_tows and tow in truth for tow in continuity_tows
        ),
        "pass_lost_zero": all(tow in output_tows for tow in tows),
        "pass_reacquisition_p95_10s": (
            reacquisition_p95 is not None and reacquisition_p95 <= 10.0
        ),
        "pass_lambda_shadow_runtime_p95_100ms": (
            shadow_runtime_p95_ms is not None
            and shadow_runtime_p95_ms <= 100.0
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--debug", type=Path, required=True)
    parser.add_argument("--positions", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--fix-source",
        choices=("solver", "safe_fix_shadow"),
        default="solver",
    )
    args = parser.parse_args()
    payload = analyze(
        _read_csv(args.debug),
        read_library_pos(args.positions),
        _read_csv(args.reference),
        json.loads(args.manifest.read_text(encoding="utf-8")),
        fix_source=args.fix_source,
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
