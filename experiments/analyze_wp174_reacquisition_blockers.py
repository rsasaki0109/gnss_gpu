#!/usr/bin/env python3
"""Diagnose truth-free candidate blockers inside long safe-FIX gaps."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.analyze_wp174_ffrt_calibration import (  # noqa: E402
    BSR_FIELDS,
    Policy,
    _finite,
)
from experiments.analyze_wp174_safe_union import (  # noqa: E402
    StateMachineConfig,
    _tow,
    causal_candidate_declarations,
)
from experiments.wp174_ffrt import passes_ffrt  # noqa: E402


POLICY = Policy(16, 16, 0.05, 3.0)
CONFIG = StateMachineConfig(
    acquisition_streak=12,
    maximum_correction_jump_m=0.03,
    maximum_epoch_gap_s=0.21,
    maximum_hold_epochs=100,
    maximum_hold_correction_jump_m=0.06,
    maximum_hold_nis_per_observation=50.0,
    maximum_prefit_residual_rms_m=50.0,
    maximum_hold_prefit_residual_rms_m=50.0,
    minimum_hold_pairs=16,
)


def _failure_reason(row: dict[str, str]) -> str:
    pair_count = _finite(row, "pair_count")
    ratio = _finite(row, "lambda_shadow_ratio")
    bsr = _finite(row, BSR_FIELDS[16])
    second_delta = _finite(row, "lambda_shadow_second_position_delta_m")
    nis = _finite(row, "float_update_nis_per_observation")
    prefit = _finite(row, "float_update_prefit_residual_rms_m")
    if row.get("lambda_shadow_solved") != "1":
        return "lambda_not_solved"
    if pair_count is None or pair_count < POLICY.minimum_pairs:
        return "pairs_below_16"
    if (
        ratio is None
        or bsr is None
        or not passes_ffrt(int(pair_count), bsr, ratio)
    ):
        return "ffrt_failed"
    if second_delta is None or second_delta > 0.05:
        return "second_position_delta_above_5cm"
    if nis is None or nis > 3.0:
        return "nis_above_3"
    if prefit is None or prefit > 50.0:
        return "prefit_above_50m"
    return "eligible_but_temporally_unconfirmed"


def diagnose(rows: list[dict[str, str]], minimum_gap_s: float = 10.0) -> dict[str, Any]:
    policy_by_block = {int(row["block"]): POLICY for row in rows}
    declared, metrics = causal_candidate_declarations(
        rows,
        policy_by_block,
        CONFIG,
    )
    gap_rows: list[dict[str, str]] = []
    current: list[dict[str, str]] = []
    gap_durations: list[float] = []
    for row in sorted(rows, key=lambda item: _tow(item["tow"])):
        if _tow(row["tow"]) in declared:
            if current:
                duration = _tow(current[-1]["tow"]) - _tow(current[0]["tow"])
                if duration >= minimum_gap_s:
                    gap_rows.extend(current)
                    gap_durations.append(duration)
            current = []
        else:
            current.append(row)
    reasons = Counter(_failure_reason(row) for row in gap_rows)
    return {
        "declared_fix_epochs": len(declared),
        "state_metrics": metrics,
        "minimum_gap_s": minimum_gap_s,
        "long_gap_count": len(gap_durations),
        "long_gap_epochs": len(gap_rows),
        "long_gap_max_s": max(gap_durations) if gap_durations else None,
        "blocker_counts": dict(sorted(reasons.items())),
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
    payload = {
        "schema": "gnss_gpu_wp174_reacquisition_blockers_v1",
        "runtime_fgo": False,
        "truth_usage": "none",
        "policy": POLICY.__dict__,
        "domains": {
            domain: diagnose(rows)
            for domain, rows in map(_read, args.audit)
        },
    }
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
