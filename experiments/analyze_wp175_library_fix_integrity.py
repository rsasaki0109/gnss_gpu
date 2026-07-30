#!/usr/bin/env python3
"""Audit authoritative library Status=4 against two-family integrity evidence."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from experiments.analyze_wp174_safe_union import _quantile, _tow


def read_library_pos(
    path: Path,
) -> dict[float, dict[str, float | int]]:
    """Read the common first nine columns of gnss_solve or gnss_fuse POS."""

    output: dict[float, dict[str, float | int]] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip() or line.startswith("%"):
                continue
            values = line.split()
            if len(values) < 9:
                raise ValueError(f"malformed gnssplusplus row in {path}")
            output[_tow(values[1])] = {
                "tow": float(values[1]),
                "ecef_x": float(values[2]),
                "ecef_y": float(values[3]),
                "ecef_z": float(values[4]),
                "status": int(values[8]),
            }
    return output


def analyze(
    integrity_rows: list[dict[str, str]],
    positions: dict[float, dict[str, float | int]],
    reference_rows: list[dict[str, str]],
    city: str,
) -> dict[str, object]:
    truth = {
        _tow(row["GPS TOW (s)"]): tuple(
            float(row[f"ECEF {axis.upper()} (m)"]) for axis in "xyz"
        )
        for row in reference_rows
    }
    telemetry = {_tow(row["tow"]): row for row in integrity_rows}
    fixed_errors: list[float] = []
    output_errors: list[float] = []
    fixed_without_budget: list[float] = []
    fixed_without_two_families: list[float] = []
    fixed_without_quality_gate: list[float] = []
    runtime_ms = [
        float(row["processing_runtime_ms"])
        for row in integrity_rows
        if row.get("processing_runtime_ms")
        and math.isfinite(float(row["processing_runtime_ms"]))
    ]
    status_counts: dict[str, int] = {}
    for raw_tow, position in positions.items():
        tow = _tow(raw_tow)
        status = str(int(position["status"]))
        status_counts[status] = status_counts.get(status, 0) + 1
        row = telemetry.get(tow)
        if status == "4":
            if row is None or row.get("failure_budget_passed") != "1":
                fixed_without_budget.append(tow)
            if row is None or int(row.get("independent_families") or 0) < 2:
                fixed_without_two_families.append(tow)
            if row is None or row.get("quality_gate_passed") != "1":
                fixed_without_quality_gate.append(tow)
        if tow not in truth:
            continue
        estimate = tuple(float(position[f"ecef_{axis}"]) for axis in "xyz")
        error = math.dist(estimate, truth[tow])
        output_errors.append(error)
        if status == "4":
            fixed_errors.append(error)

    src_par_declared_errors: list[float] = []
    src_par_nonfixed_declared_errors: list[float] = []
    src_par_hard_declared_errors: list[float] = []
    src_par_hard_nonfixed_declared_errors: list[float] = []
    src_par_hard_separation_threshold_m = 0.25
    for tow, row in telemetry.items():
        if row.get("src_par_consensus_declared_fixed") != "1":
            continue
        if tow not in truth:
            continue
        coordinate_names = (
            "src_par_candidate_ecef_x",
            "src_par_candidate_ecef_y",
            "src_par_candidate_ecef_z",
        )
        if not all(row.get(name) for name in coordinate_names):
            continue
        estimate = tuple(float(row[name]) for name in coordinate_names)
        if not all(math.isfinite(value) for value in estimate):
            continue
        error = math.dist(estimate, truth[tow])
        src_par_declared_errors.append(error)
        was_not_library_fixed = row.get("library_status") != "4"
        if was_not_library_fixed:
            src_par_nonfixed_declared_errors.append(error)
        separation_names = (
            "src_par_partition_a_separation_m",
            "src_par_partition_b_separation_m",
        )
        try:
            separations = tuple(float(row[name]) for name in separation_names)
        except (KeyError, TypeError, ValueError):
            continue
        if (
            all(math.isfinite(value) for value in separations)
            and max(separations) <= src_par_hard_separation_threshold_m
        ):
            src_par_hard_declared_errors.append(error)
            if was_not_library_fixed:
                src_par_hard_nonfixed_declared_errors.append(error)

    total = len(positions)
    target = 0.50 if city.lower() == "tokyo" else 0.60
    stretch = 0.55 if city.lower() == "tokyo" else 0.65
    fixed_rate = len(fixed_errors) / total if total else 0.0
    return {
        "schema": "gnss_gpu_wp175_library_fix_integrity_audit_v1",
        "city": city,
        "fix_kpi_definition": "gnssplusplus final .pos Status == 4",
        "truth_usage": "post_selection_audit_only",
        "runtime_fgo": False,
        "processing_runtime_p95_ms": _quantile(runtime_ms, 0.95),
        "processing_runtime_max_ms": max(runtime_ms) if runtime_ms else None,
        "runtime_p95_100ms_pass": (
            bool(runtime_ms) and _quantile(runtime_ms, 0.95) <= 100.0
        ),
        "integrity_epochs": len(integrity_rows),
        "solution_epochs": total,
        "lost_epochs": max(0, len(integrity_rows) - total),
        "output_status_counts": status_counts,
        "library_fixed_epochs": len(fixed_errors),
        "library_fix_rate": fixed_rate,
        "formal_target": target,
        "stretch_target": stretch,
        "formal_target_pass": fixed_rate >= target,
        "stretch_target_pass": fixed_rate >= stretch,
        "fixed_without_failure_budget_epochs": fixed_without_budget,
        "fixed_without_two_families_epochs": fixed_without_two_families,
        "fixed_without_quality_gate_epochs": fixed_without_quality_gate,
        "every_fixed_has_two_family_budget": (
            not fixed_without_budget and not fixed_without_two_families
        ),
        "every_fixed_passed_quality_gate": not fixed_without_quality_gate,
        "every_fixed_passed_all_integrity_gates": (
            not fixed_without_budget
            and not fixed_without_two_families
            and not fixed_without_quality_gate
        ),
        "observed_false_fixed_epochs": sum(
            error >= 0.5 for error in fixed_errors
        ),
        "src_par_declared_epochs": len(src_par_declared_errors),
        "src_par_declared_false_epochs": sum(
            error >= 0.5 for error in src_par_declared_errors
        ),
        "src_par_nonfixed_declared_epochs": len(
            src_par_nonfixed_declared_errors
        ),
        "src_par_nonfixed_declared_false_epochs": sum(
            error >= 0.5 for error in src_par_nonfixed_declared_errors
        ),
        "src_par_nonfixed_declared_error_p95_m": _quantile(
            src_par_nonfixed_declared_errors, 0.95
        ),
        "src_par_hard_separation_threshold_m": (
            src_par_hard_separation_threshold_m
        ),
        "src_par_hard_declared_epochs": len(src_par_hard_declared_errors),
        "src_par_hard_declared_false_epochs": sum(
            error >= 0.5 for error in src_par_hard_declared_errors
        ),
        "src_par_hard_nonfixed_declared_epochs": len(
            src_par_hard_nonfixed_declared_errors
        ),
        "src_par_hard_nonfixed_declared_false_epochs": sum(
            error >= 0.5 for error in src_par_hard_nonfixed_declared_errors
        ),
        "src_par_hard_nonfixed_declared_error_p95_m": _quantile(
            src_par_hard_nonfixed_declared_errors, 0.95
        ),
        "fixed_error_p95_m": _quantile(fixed_errors, 0.95),
        "output_sub50cm_rate": (
            sum(error < 0.5 for error in output_errors) / len(output_errors)
            if output_errors
            else 0.0
        ),
        "failure_budget_passed_epochs": sum(
            row.get("failure_budget_passed") == "1"
            for row in integrity_rows
        ),
        "inertial_available_epochs": sum(
            row.get("inertial_available") == "1"
            for row in integrity_rows
        ),
        "inertial_healthy_anchor_epochs": sum(
            row.get("inertial_healthy_anchor") == "1"
            for row in integrity_rows
        ),
        "inertial_passed_epochs": sum(
            row.get("inertial_passed") == "1"
            for row in integrity_rows
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city", choices=("tokyo", "nagoya"), required=True)
    parser.add_argument("--integrity", type=Path, required=True)
    parser.add_argument("--positions", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.integrity.open(newline="", encoding="utf-8-sig") as stream:
        integrity_rows = list(csv.DictReader(stream))
    with args.reference.open(newline="", encoding="utf-8-sig") as stream:
        reference_rows = list(csv.DictReader(stream))
    payload = analyze(
        integrity_rows,
        read_library_pos(args.positions),
        reference_rows,
        args.city,
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
