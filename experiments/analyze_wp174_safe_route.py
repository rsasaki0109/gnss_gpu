#!/usr/bin/env python3
"""Audit runtime safe-FIX telemetry and ordinary output on one full route."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.analyze_wp174_safe_union import _quantile, _tow  # noqa: E402
from experiments.promote_wp172_pf_seeded_rtk_consensus import (  # noqa: E402
    read_gnssplusplus_pos,
)


def analyze(
    debug_rows: list[dict[str, str]],
    positions: dict[float, dict[str, float | int]],
    reference_rows: list[dict[str, str]],
    domain: str,
    position_sha256: str,
) -> dict:
    truth = {
        _tow(row["GPS TOW (s)"]): tuple(
            float(row[f"ECEF {axis.upper()} (m)"]) for axis in "xyz"
        )
        for row in reference_rows
    }
    positions = {_tow(tow): row for tow, row in positions.items()}
    safe_errors = []
    output_errors = []
    runtime_ms = []
    output_status_counts: dict[str, int] = {}
    for row in debug_rows:
        tow = _tow(row["tow"])
        if row.get("safe_fix_shadow_declared_fixed") == "1" and tow in truth:
            candidate = tuple(
                float(row[f"lambda_shadow_best_ecef_{axis}"])
                for axis in "xyz"
            )
            if all(math.isfinite(value) for value in candidate):
                safe_errors.append(math.dist(candidate, truth[tow]))
        runtime = row.get("lambda_shadow_runtime_ms", "")
        if runtime and math.isfinite(float(runtime)):
            runtime_ms.append(float(runtime))
    for tow, position in positions.items():
        status = str(int(position.get("status", -1)))
        output_status_counts[status] = (
            output_status_counts.get(status, 0) + 1
        )
        if tow not in truth:
            continue
        estimate = tuple(
            float(position[f"ecef_{axis}"]) for axis in "xyz"
        )
        output_errors.append(math.dist(estimate, truth[tow]))
    safe_fix_rate = len(safe_errors) / len(debug_rows) if debug_rows else 0.0
    output_sub50_rate = (
        sum(error < 0.5 for error in output_errors) / len(output_errors)
        if output_errors
        else 0.0
    )
    formal_target = 0.35 if domain.lower() == "tokyo" else 0.45
    stretch_target = 0.50 if domain.lower() == "tokyo" else 0.60
    runtime_p95 = _quantile(runtime_ms, 0.95)
    return {
        "schema": "gnss_gpu_wp174_safe_route_runtime_audit_v1",
        "domain": domain,
        "runtime_fgo": False,
        "truth_usage": "post_selection_audit_only",
        "debug_epochs": len(debug_rows),
        "solution_epochs": len(positions),
        "lost_epochs": max(0, len(debug_rows) - len(positions)),
        "safe_fix_epochs": len(safe_errors),
        "safe_fix_rate": safe_fix_rate,
        "safe_fix_false_epochs": sum(
            error >= 0.5 for error in safe_errors
        ),
        "safe_fix_error_p95_m": _quantile(safe_errors, 0.95),
        "safe_fix_formal_target": formal_target,
        "safe_fix_stretch_target": stretch_target,
        "pass_safe_fix_formal_target": (
            safe_fix_rate >= formal_target
            and all(error < 0.5 for error in safe_errors)
        ),
        "pass_safe_fix_stretch_target": (
            safe_fix_rate >= stretch_target
            and all(error < 0.5 for error in safe_errors)
        ),
        "output_truth_labeled_epochs": len(output_errors),
        "output_status_counts": output_status_counts,
        "output_propagated_epochs": output_status_counts.get("7", 0),
        "output_sub50cm_rate": output_sub50_rate,
        "pass_tokyo_output_sub50cm_46p5112": (
            domain.lower() != "tokyo" or output_sub50_rate >= 0.465112
        ),
        "lambda_shadow_runtime_p95_ms": runtime_p95,
        "lambda_shadow_runtime_max_ms": (
            max(runtime_ms) if runtime_ms else None
        ),
        "pass_runtime_p95_100ms": (
            runtime_p95 is not None and runtime_p95 <= 100.0
        ),
        "change_point_acquisition_epochs": sum(
            row.get("safe_fix_shadow_change_point_acquisition") == "1"
            for row in debug_rows
        ),
        "strong_acquisition_epochs": sum(
            row.get("safe_fix_shadow_strong_acquisition") == "1"
            for row in debug_rows
        ),
        "position_sha256": position_sha256,
        "promotion_ready": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--debug", type=Path, required=True)
    parser.add_argument("--positions", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.debug.open(newline="", encoding="utf-8-sig") as stream:
        debug_rows = list(csv.DictReader(stream))
    with args.reference.open(newline="", encoding="utf-8-sig") as stream:
        reference_rows = list(csv.DictReader(stream))
    payload = analyze(
        debug_rows,
        read_gnssplusplus_pos(args.positions),
        reference_rows,
        args.domain,
        hashlib.sha256(args.positions.read_bytes()).hexdigest(),
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
