#!/usr/bin/env python3
"""Run mandatory fail-closed negative controls on WP174 FIX declarations."""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
import sys
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.analyze_wp174_safe_union import (  # noqa: E402
    StateMachineConfig,
    _policy_by_fold,
    causal_candidate_declarations,
)


Mutation = Callable[[list[dict[str, str]]], None]


def _blank_correction(rows: list[dict[str, str]]) -> None:
    for row in rows:
        for axis in "xyz":
            row[f"lambda_shadow_best_correction_{axis}"] = ""


def _nonfinite_quality(rows: list[dict[str, str]]) -> None:
    for row in rows:
        row["float_update_nis_per_observation"] = "nan"
        row["lambda_shadow_second_position_delta_m"] = "nan"


def _insufficient_pairs(rows: list[dict[str, str]]) -> None:
    for row in rows:
        row["pair_count"] = "3"


def _ffrt_reject(rows: list[dict[str, str]]) -> None:
    for row in rows:
        row["lambda_shadow_ratio"] = "1"
        for suffix in ("", "_qscale2", "_qscale4", "_qscale8", "_qscale16"):
            row[f"lambda_shadow_bsr{suffix}"] = "0"


def _break_contiguity(rows: list[dict[str, str]]) -> None:
    for index, row in enumerate(
        sorted(rows, key=lambda item: float(item["tow"]))
    ):
        row["tow"] = str(1000.0 + index)


NEGATIVE_CONTROLS: dict[str, Mutation] = {
    "missing_candidate_correction": _blank_correction,
    "nonfinite_quality": _nonfinite_quality,
    "insufficient_ambiguity_pairs": _insufficient_pairs,
    "ffrt_reject_all": _ffrt_reject,
    "noncontiguous_epochs": _break_contiguity,
}


def analyze(
    rows: list[dict[str, str]],
    cv_summary: dict,
    domain: str,
) -> dict:
    results = []
    for control_id, mutation in NEGATIVE_CONTROLS.items():
        mutated = copy.deepcopy(rows)
        mutation(mutated)
        family_counts = {}
        for family, acquisition_streak in (
            ("instantaneous", 2),
            ("confirmed", 2),
            ("temporal", 12),
        ):
            policies = _policy_by_fold(cv_summary, domain, family)
            declared, _ = causal_candidate_declarations(
                mutated,
                policies,
                StateMachineConfig(
                    acquisition_streak=acquisition_streak,
                    maximum_hold_epochs=0,
                ),
            )
            family_counts[family] = len(declared)
        total = len(
            set().union(
                *(
                    causal_candidate_declarations(
                        mutated,
                        _policy_by_fold(cv_summary, domain, family),
                        StateMachineConfig(
                            acquisition_streak=streak,
                            maximum_hold_epochs=0,
                        ),
                    )[0]
                    for family, streak in (
                        ("instantaneous", 2),
                        ("confirmed", 2),
                        ("temporal", 12),
                    )
                )
            )
        )
        results.append(
            {
                "control_id": control_id,
                "family_fix_epochs": family_counts,
                "union_fix_epochs": total,
                "pass": total == 0,
            }
        )
    return {
        "schema": "gnss_gpu_wp174_negative_controls_v1",
        "domain": domain,
        "runtime_fgo": False,
        "truth_usage": "none",
        "controls": results,
        "all_pass": all(result["pass"] for result in results),
        "total_negative_fix_epochs": sum(
            result["union_fix_epochs"] for result in results
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--cv-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.audit.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    payload = analyze(
        rows,
        json.loads(args.cv_summary.read_text(encoding="utf-8")),
        args.domain,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
