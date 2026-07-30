#!/usr/bin/env python3
"""Compare fixed safe-FIX geometry/prefit guards on full-route audits."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.analyze_wp174_ffrt_calibration import Policy  # noqa: E402
from experiments.analyze_wp174_runtime_safe_fix import analyze  # noqa: E402
from experiments.analyze_wp174_safe_union import StateMachineConfig  # noqa: E402


def compare(audits: dict[str, list[dict[str, str]]]) -> dict[str, Any]:
    base = {
        "acquisition_streak": 12,
        "maximum_correction_jump_m": 0.03,
        "maximum_epoch_gap_s": 0.21,
        "maximum_hold_epochs": 100,
        "maximum_hold_correction_jump_m": 0.06,
        "maximum_hold_nis_per_observation": 50.0,
    }
    candidates = {
        "prefit50_pairs16_streak2": StateMachineConfig(
            **(base | {"acquisition_streak": 2}),
            maximum_prefit_residual_rms_m=50.0,
            maximum_hold_prefit_residual_rms_m=50.0,
            minimum_hold_pairs=16,
        ),
        "prefit3_pairs4": StateMachineConfig(
            **base,
            maximum_prefit_residual_rms_m=3.0,
            maximum_hold_prefit_residual_rms_m=3.0,
            minimum_hold_pairs=4,
        ),
        "no_prefit_pairs8": StateMachineConfig(
            **base,
            minimum_hold_pairs=8,
        ),
        "no_prefit_pairs16": StateMachineConfig(
            **base,
            minimum_hold_pairs=16,
        ),
        "prefit50_pairs16_selected": StateMachineConfig(
            **base,
            maximum_prefit_residual_rms_m=50.0,
            maximum_hold_prefit_residual_rms_m=50.0,
            minimum_hold_pairs=16,
        ),
    }
    policy = Policy(16, 16, 0.05, 3.0)
    results = {
        name: analyze(audits, policy, config)
        for name, config in candidates.items()
    }
    return {
        "schema": "gnss_gpu_wp174_safe_fix_gate_ablation_v1",
        "runtime_fgo": False,
        "truth_usage": "post_selection_audit_only",
        "selection_status": "posthoc_exploratory_not_promotion_evidence",
        "nlos_counterexample": {
            "false_held_epochs": 4,
            "pair_count": 6,
            "prefit_residual_rms_m_range": [22.11, 22.18],
            "selected_minimum_hold_pairs": 16,
        },
        "selected": "prefit50_pairs16_selected",
        "candidates": results,
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
    payload = compare(dict(_read(specification) for specification in args.audit))
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
