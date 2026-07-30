#!/usr/bin/env python3
"""Summarize raw-RINEX safe-FIX fault replay audit artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def summarize(audits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    matrix = {}
    total_false = 0
    total_lost = 0
    all_reacquisition_pass = True
    all_truth_coverage_pass = True
    all_runtime_pass = True
    for domain, payloads in audits.items():
        matrix[domain] = {}
        for payload in payloads:
            fault = str(payload["fault"])
            row = {
                "debug_epochs": payload["debug_epochs"],
                "event_count": payload["event_count"],
                "declared_fix_epochs": payload[
                    "truth_labeled_fixed_epochs"
                ],
                "false_fix_epochs": payload["false_fixed_epochs"],
                "false_fix_epochs_during_fault": payload[
                    "false_fixed_epochs_during_fault"
                ],
                "lost_epochs": payload["lost_epochs"],
                "lost_epochs_during_fault": payload[
                    "lost_epochs_during_fault"
                ],
                "reacquisition_p95_s": payload["reacquisition_p95_s"],
                "pass_false_fix_zero": payload["pass_false_fix_zero"],
                "pass_lost_zero": payload["pass_lost_zero"],
                "pass_reacquisition_p95_10s": payload[
                    "pass_reacquisition_p95_10s"
                ],
                "pass_fixed_truth_coverage": payload[
                    "pass_fixed_truth_coverage"
                ],
                "change_point_acquisition_epochs": payload.get(
                    "safe_fix_shadow_change_point_acquisition_epochs", 0
                ),
                "strong_acquisition_epochs": payload.get(
                    "safe_fix_shadow_strong_acquisition_epochs", 0
                ),
                "lambda_shadow_runtime_p95_ms": payload.get(
                    "lambda_shadow_runtime_p95_ms"
                ),
                "lambda_shadow_runtime_max_ms": payload.get(
                    "lambda_shadow_runtime_max_ms"
                ),
                "pass_lambda_shadow_runtime_p95_100ms": payload.get(
                    "pass_lambda_shadow_runtime_p95_100ms", False
                ),
            }
            matrix[domain][fault] = row
            total_false += int(row["false_fix_epochs"])
            total_lost += int(row["lost_epochs"])
            all_reacquisition_pass &= bool(
                row["pass_reacquisition_p95_10s"]
            )
            all_truth_coverage_pass &= bool(
                row["pass_fixed_truth_coverage"]
            )
            all_runtime_pass &= bool(
                row["pass_lambda_shadow_runtime_p95_100ms"]
            )
    return {
        "schema": "gnss_gpu_wp174_raw_fault_matrix_v1",
        "injection_layer": "raw_rinex_observations",
        "fix_source": "safe_fix_shadow",
        "runtime_fgo": False,
        "truth_usage": "post_selection_audit_only",
        "scope": "1000_epoch_prefix_two_distributed_events_per_case",
        "promotion_ready": False,
        "matrix": matrix,
        "totals": {
            "false_fix_epochs": total_false,
            "lost_epochs": total_lost,
        },
        "all_false_fix_zero": total_false == 0,
        "all_lost_zero": total_lost == 0,
        "all_reacquisition_p95_10s": all_reacquisition_pass,
        "all_fixed_truth_coverage": all_truth_coverage_pass,
        "all_lambda_shadow_runtime_p95_100ms": all_runtime_pass,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit",
        action="append",
        required=True,
        help="DOMAIN=JSON_PATH",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audits: dict[str, list[dict[str, Any]]] = {}
    for specification in args.audit:
        domain, raw_path = specification.split("=", 1)
        audits.setdefault(domain, []).append(
            json.loads(Path(raw_path).read_text(encoding="utf-8"))
        )
    payload = summarize(audits)
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
