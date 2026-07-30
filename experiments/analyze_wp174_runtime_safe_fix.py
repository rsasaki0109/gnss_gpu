#!/usr/bin/env python3
"""Audit one fixed runtime-safe-FIX policy across WP174 domains."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.analyze_wp174_ffrt_calibration import Policy  # noqa: E402
from experiments.analyze_wp174_safe_union import (  # noqa: E402
    StateMachineConfig,
    _tow,
    causal_candidate_declarations,
)


def analyze(
    audits: dict[str, list[dict[str, str]]],
    policy: Policy,
    config: StateMachineConfig,
) -> dict[str, Any]:
    domains = {}
    total_good = 0
    total_bad = 0
    for domain, rows in audits.items():
        policy_by_block = {
            int(row["block"]): policy for row in rows
        }
        declared, state_metrics = causal_candidate_declarations(
            rows,
            policy_by_block,
            config,
        )
        by_tow = {_tow(row["tow"]): row for row in rows}
        good = sum(
            by_tow[tow]["shadow_best_sub50cm"] == "1"
            for tow in declared
        )
        bad = sum(
            by_tow[tow]["shadow_best_sub50cm"] == "0"
            for tow in declared
        )
        unlabeled = len(declared) - good - bad
        total_good += good
        total_bad += bad
        domains[domain] = {
            "declared_fix_epochs": len(declared),
            "accepted_good_epochs": good,
            "accepted_bad_epochs": bad,
            "unlabeled_fix_epochs": unlabeled,
            **state_metrics,
        }
    state_machine = asdict(config)
    state_machine = {
        key: (
            None
            if isinstance(value, float) and not math.isfinite(value)
            else value
        )
        for key, value in state_machine.items()
    }
    return {
        "schema": "gnss_gpu_wp174_runtime_safe_fix_audit_v1",
        "runtime_fgo": False,
        "runtime_integration_mode": "shadow_only_default_off",
        "truth_usage": "post_selection_audit_only",
        "selection_status": "posthoc_exploratory_not_promotion_evidence",
        "promotion_ready": False,
        "policy": asdict(policy),
        "state_machine": state_machine,
        "domains": domains,
        "total_accepted_good_epochs": total_good,
        "total_accepted_bad_epochs": total_bad,
    }


def _read(specification: str) -> tuple[str, list[dict[str, str]]]:
    domain, raw_path = specification.split("=", 1)
    with Path(raw_path).open(newline="", encoding="utf-8-sig") as stream:
        return domain, list(csv.DictReader(stream))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = analyze(
        dict(_read(specification) for specification in args.audit),
        Policy(16, 16, 0.05, 3.0),
        StateMachineConfig(
            acquisition_streak=12,
            maximum_correction_jump_m=0.03,
            maximum_epoch_gap_s=0.21,
            maximum_hold_epochs=100,
            maximum_hold_correction_jump_m=0.06,
            maximum_hold_nis_per_observation=50.0,
            maximum_prefit_residual_rms_m=50.0,
            maximum_hold_prefit_residual_rms_m=50.0,
            minimum_hold_pairs=16,
        ),
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
