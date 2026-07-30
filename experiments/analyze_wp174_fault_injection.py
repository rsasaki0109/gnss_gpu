#!/usr/bin/env python3
"""Inject decision-telemetry faults into the WP174 shadow state machine."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.analyze_wp174_safe_union import (  # noqa: E402
    StateMachineConfig,
    _quantile,
    _read_csv,
    _tow,
    analyze_domain,
)


FAULT_DURATIONS_S = {
    "outage": 5.0,
    "cycle_slip": 0.2,
    "satellite_loss": 3.0,
    "nlos": 5.0,
}


def inject_audit_row(row: dict[str, str], fault: str) -> dict[str, str]:
    output = dict(row)
    if fault == "outage":
        output["pair_count"] = "0"
        output["float_update_nis_per_observation"] = ""
        for axis in "xyz":
            output[f"lambda_shadow_best_correction_{axis}"] = ""
    elif fault == "cycle_slip":
        output["pair_count"] = "0"
        for axis in "xyz":
            output[f"lambda_shadow_best_correction_{axis}"] = ""
    elif fault == "satellite_loss":
        output["pair_count"] = "0"
    elif fault == "nlos":
        output["float_update_nis_per_observation"] = "inf"
    else:
        raise ValueError(f"unsupported fault: {fault}")
    return output


def _event_starts(
    output_rows: list[dict[str, Any]],
    *,
    count: int,
    duration_s: float,
) -> list[float]:
    if count < 1:
        raise ValueError("event count must be positive")
    starts = []
    minimum_spacing_epochs = max(1, math.ceil((duration_s + 20.0) / 0.2))
    last_index = -minimum_spacing_epochs
    targets = [
        round((index + 1) * len(output_rows) / (count + 1))
        for index in range(count)
    ]
    for target in targets:
        for index in range(max(target, last_index + minimum_spacing_epochs), len(output_rows) - 2):
            if output_rows[index - 1]["union_fix"]:
                starts.append(float(output_rows[index]["tow"]))
                last_index = index
                break
    return starts


def analyze_fault(
    *,
    domain: str,
    baseline_rows: list[dict[str, str]],
    audit_rows: list[dict[str, str]],
    cv_summary: dict[str, Any],
    config: StateMachineConfig,
    fault: str,
    event_count: int,
) -> dict[str, Any]:
    duration_s = FAULT_DURATIONS_S[fault]
    clean_output, clean_summary = analyze_domain(
        domain,
        baseline_rows,
        audit_rows,
        cv_summary,
        config,
        policy_family="dual",
    )
    starts = _event_starts(
        clean_output,
        count=event_count,
        duration_s=duration_s,
    )
    windows = [(start, start + duration_s) for start in starts]
    mutated_baseline = copy.deepcopy(baseline_rows)
    mutated_audit = copy.deepcopy(audit_rows)
    for row in mutated_baseline:
        tow = _tow(row["tow"])
        if any(start <= tow < end for start, end in windows):
            row["fix"] = "0"
            row["false_fix"] = "0"
    for index, row in enumerate(mutated_audit):
        tow = _tow(row["tow"])
        if any(start <= tow < end for start, end in windows):
            mutated_audit[index] = inject_audit_row(row, fault)

    fault_output, fault_summary = analyze_domain(
        domain,
        mutated_baseline,
        mutated_audit,
        cv_summary,
        config,
        policy_family="dual",
    )
    by_tow = {_tow(row["tow"]): row for row in fault_output}
    sorted_tows = sorted(by_tow)
    recoveries = []
    fixed_during_fault = 0
    for start, end in windows:
        fixed_during_fault += sum(
            by_tow[tow]["union_fix"]
            for tow in sorted_tows
            if start <= tow < end
        )
        next_fix = next(
            (
                tow
                for tow in sorted_tows
                if tow >= end and by_tow[tow]["union_fix"]
            ),
            None,
        )
        if next_fix is not None:
            recoveries.append(next_fix - end)
    return {
        "schema": "gnss_gpu_wp174_fault_injection_audit_v1",
        "domain": domain,
        "fault": fault,
        "injection_layer": "decision_telemetry_shadow",
        "runtime_fgo": False,
        "promotion_ready": False,
        "event_count_requested": event_count,
        "event_count_injected": len(windows),
        "duration_s": duration_s,
        "fixed_epochs_during_fault": fixed_during_fault,
        "recovered_events": len(recoveries),
        "reacquisition_p95_s": _quantile(recoveries, 0.95),
        "reacquisition_max_s": max(recoveries) if recoveries else None,
        "clean_fix_rate_pct": clean_summary["union_fix_rate_pct"],
        "faulted_fix_rate_pct": fault_summary["union_fix_rate_pct"],
        "faulted_false_fix_epochs": fault_summary["union_false_fix_epochs"],
        "windows": [
            {"start_tow": start, "end_tow": end} for start, end in windows
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--cv-summary", type=Path, required=True)
    parser.add_argument(
        "--fault",
        choices=tuple(FAULT_DURATIONS_S),
        required=True,
    )
    parser.add_argument("--event-count", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = analyze_fault(
        domain=args.domain,
        baseline_rows=_read_csv(args.baseline),
        audit_rows=_read_csv(args.audit),
        cv_summary=json.loads(args.cv_summary.read_text(encoding="utf-8")),
        config=StateMachineConfig(
            acquisition_streak=2,
            maximum_correction_jump_m=0.03,
            maximum_epoch_gap_s=0.21,
            maximum_hold_epochs=100,
            maximum_hold_correction_jump_m=0.06,
            maximum_hold_nis_per_observation=50.0,
            minimum_hold_pairs=4,
        ),
        fault=args.fault,
        event_count=args.event_count,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
