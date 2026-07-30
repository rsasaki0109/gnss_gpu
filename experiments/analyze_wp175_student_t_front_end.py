#!/usr/bin/env python3
"""Audit a truth-blind Student-t RTK front end against an OFF control."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _reference(path: Path) -> dict[tuple[int, float], tuple[float, float, float]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return {
            (int(row["GPS Week"]), round(float(row["GPS TOW (s)"]), 3)): (
                float(row["ECEF X (m)"]),
                float(row["ECEF Y (m)"]),
                float(row["ECEF Z (m)"]),
            )
            for row in csv.DictReader(handle, skipinitialspace=True)
        }


def _position_metrics(
    path: Path,
    reference: dict[tuple[int, float], tuple[float, float, float]],
) -> dict[str, float | int]:
    errors: list[float] = []
    fixed = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or line.startswith("%"):
                continue
            fields = line.split()
            key = (int(fields[0]), round(float(fields[1]), 3))
            truth = reference.get(key)
            if truth is None:
                continue
            solution = tuple(float(value) for value in fields[2:5])
            errors.append(math.dist(solution, truth))
            fixed += int(int(fields[8]) == 4)
    if not errors:
        raise ValueError(f"no reference-matched positions in {path}")
    return {
        "epochs": len(errors),
        "library_fixed_epochs": fixed,
        "library_fix_rate_pct": 100.0 * fixed / len(errors),
        "position_error_median_m": median(errors),
        "position_error_p95_m": _percentile(errors, 95.0),
        "sub50cm_epochs": sum(error < 0.5 for error in errors),
    }


def _telemetry(path: Path) -> dict[str, float | int]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    downweighted = [
        int(row["float_update_student_t_downweighted_rows"]) for row in rows
    ]
    nis = [float(row["float_update_nis_per_observation"]) for row in rows]
    weights = [
        float(row["float_update_student_t_minimum_weight"])
        for row in rows
        if row["float_update_student_t_minimum_weight"] not in {"", "nan"}
    ]
    return {
        "triggered_epochs": sum(value > 0 for value in downweighted),
        "downweighted_rows": sum(downweighted),
        "maximum_downweighted_rows_per_epoch": max(downweighted, default=0),
        "minimum_weight": min(weights, default=1.0),
        "nis_per_observation_p95": _percentile(nis, 95.0),
    }


def analyze(
    control_pos: Path,
    treatment_pos: Path,
    treatment_debug: Path,
    reference_csv: Path,
    domain: str,
) -> dict[str, object]:
    reference = _reference(reference_csv)
    control = _position_metrics(control_pos, reference)
    treatment = _position_metrics(treatment_pos, reference)
    telemetry = _telemetry(treatment_debug)
    return {
        "domain": domain,
        "fix_definition": "gnssplusplus .pos Status == 4",
        "runtime_truth_used": False,
        "configuration": {
            "code_only": True,
            "degrees_of_freedom": 4.0,
            "minimum_weight": 0.05,
            "activation_threshold_sigma": 2.5,
            "default_enabled": False,
        },
        "control": control,
        "student_t": treatment,
        "telemetry": telemetry,
        "delta": {
            "library_fixed_epochs": (
                treatment["library_fixed_epochs"] - control["library_fixed_epochs"]
            ),
            "sub50cm_epochs": (
                treatment["sub50cm_epochs"] - control["sub50cm_epochs"]
            ),
            "position_error_median_m": (
                treatment["position_error_median_m"]
                - control["position_error_median_m"]
            ),
            "position_error_p95_m": (
                treatment["position_error_p95_m"]
                - control["position_error_p95_m"]
            ),
        },
        "passes_non_regression_gate": (
            treatment["library_fixed_epochs"] >= control["library_fixed_epochs"]
            and treatment["sub50cm_epochs"] >= control["sub50cm_epochs"]
            and treatment["position_error_p95_m"]
            <= control["position_error_p95_m"] + 1e-6
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-pos", type=Path, required=True)
    parser.add_argument("--student-t-pos", type=Path, required=True)
    parser.add_argument("--student-t-debug", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = analyze(
        args.control_pos,
        args.student_t_pos,
        args.student_t_debug,
        args.reference,
        args.domain,
    )
    args.output.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()
