#!/usr/bin/env python3
"""Summarize truth-free library-FIX-anchored fault audits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


FAULTS = ("cycle_slip", "nlos", "satellite_loss", "outage")
CITIES = ("tokyo", "nagoya")


def summarize(audits: dict[tuple[str, str], dict]) -> dict:
    cases: dict[str, dict[str, dict]] = {}
    for city in CITIES:
        cases[city] = {}
        for fault in FAULTS:
            audit = audits[(city, fault)]
            cases[city][fault] = {
                "event_count": audit["event_count"],
                "recovered_events": audit["recovered_events"],
                "library_fixed_epochs": audit[
                    "truth_labeled_fixed_epochs"
                ],
                "observed_false_fixed_epochs": audit[
                    "false_fixed_epochs"
                ],
                "fixed_epochs_during_fault": audit[
                    "fixed_epochs_during_fault"
                ],
                "lost_epochs": audit["lost_epochs"],
                "reacquisition_p95_s": audit[
                    "reacquisition_p95_s"
                ],
                "reacquisition_max_s": audit[
                    "reacquisition_max_s"
                ],
                "runtime_p95_ms": audit[
                    "lambda_shadow_runtime_p95_ms"
                ],
            }
    flat = [
        cases[city][fault] for city in CITIES for fault in FAULTS
    ]
    outage = [
        cases[city]["outage"]["reacquisition_p95_s"]
        for city in CITIES
    ]
    expanded = [
        cases[city][fault]["reacquisition_p95_s"]
        for city in CITIES
        for fault in ("cycle_slip", "nlos", "satellite_loss")
    ]
    return {
        "schema": "gnss_gpu_wp175_library_fix_anchored_fault_matrix_v2",
        "fix_definition": "gnssplusplus final .pos Status == 4",
        "event_selection": (
            "baseline Status=4 for 10 consecutive epochs; "
            "truth-free; 10 s recovery horizon"
        ),
        "runtime_truth_usage": "none",
        "audit_truth_usage": "post-selection labeling only",
        "runtime_fgo": False,
        "cases": cases,
        "totals": {
            "events": sum(case["event_count"] for case in flat),
            "recovered_events": sum(
                case["recovered_events"] for case in flat
            ),
            "observed_false_fixed_epochs": sum(
                case["observed_false_fixed_epochs"] for case in flat
            ),
            "lost_epochs": sum(case["lost_epochs"] for case in flat),
        },
        "maximum_runtime_p95_ms": max(
            case["runtime_p95_ms"] for case in flat
        ),
        "maximum_complete_outage_reacquisition_p95_s": max(outage),
        "maximum_expanded_fault_reacquisition_p95_s": max(expanded),
        "passes_false_fix_zero": all(
            case["observed_false_fixed_epochs"] == 0 for case in flat
        ),
        "passes_nlos_outage_negative_fix_zero": all(
            cases[city][fault]["fixed_epochs_during_fault"] == 0
            for city in CITIES
            for fault in ("nlos", "outage")
        ),
        "passes_lost_zero": all(
            case["lost_epochs"] == 0 for case in flat
        ),
        "passes_all_events_recovered": all(
            case["recovered_events"] == case["event_count"]
            for case in flat
        ),
        "passes_complete_outage_p95_8s": max(outage) <= 8.0,
        "passes_expanded_fault_p95_10s": max(expanded) <= 10.0,
        "passes_runtime_p95_100ms": all(
            case["runtime_p95_ms"] <= 100.0 for case in flat
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audits: dict[tuple[str, str], dict] = {}
    for path in args.audit:
        audit = json.loads(path.read_text(encoding="utf-8"))
        city = next(city for city in CITIES if city in path.name)
        audits[(city, audit["fault"])] = audit
    missing = [
        (city, fault)
        for city in CITIES
        for fault in FAULTS
        if (city, fault) not in audits
    ]
    if missing:
        raise ValueError(f"missing audits: {missing}")
    result = summarize(audits)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
