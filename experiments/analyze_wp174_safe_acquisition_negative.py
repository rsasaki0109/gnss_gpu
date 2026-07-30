#!/usr/bin/env python3
"""Fail-closed controls for temporal/strong/change-point acquisition."""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.analyze_wp174_strong_instant_policy import _decisions  # noqa: E402


def analyze(rows: list[dict[str, str]], domain: str) -> dict:
    controls: list[tuple[str, list[dict[str, str]]]] = []

    missing = copy.deepcopy(rows)
    for row in missing:
        for axis in "xyz":
            row[f"lambda_shadow_best_correction_{axis}"] = ""
    controls.append(("missing_candidate_correction", missing))

    nonfinite = copy.deepcopy(rows)
    for row in nonfinite:
        row["float_update_nis_per_observation"] = "nan"
        row["float_update_prefit_residual_rms_m"] = "nan"
        row["lambda_shadow_second_position_delta_m"] = "nan"
    controls.append(("nonfinite_quality", nonfinite))

    insufficient = copy.deepcopy(rows)
    for row in insufficient:
        row["pair_count"] = "3"
    controls.append(("insufficient_ambiguity_pairs", insufficient))

    rejected = copy.deepcopy(rows)
    for row in rejected:
        row["lambda_shadow_ratio"] = "1"
        row["lambda_shadow_bsr_qscale16"] = "0"
    controls.append(("ffrt_reject_all", rejected))

    noncontiguous = copy.deepcopy(rows)
    for index, row in enumerate(noncontiguous):
        row["tow"] = str(1000.0 + index)
        row["lambda_shadow_ratio"] = "1.4"
    controls.append(("noncontiguous_without_strong_evidence", noncontiguous))

    results = []
    for control_id, mutated in controls:
        declared, strong, change_point = _decisions(mutated)
        count = sum(declared)
        results.append(
            {
                "control_id": control_id,
                "declared_fix_epochs": count,
                "strong_acquisition_epochs": sum(strong),
                "change_point_acquisition_epochs": sum(change_point),
                "pass": count == 0,
            }
        )
    return {
        "schema": "gnss_gpu_wp174_safe_acquisition_negative_v1",
        "domain": domain,
        "runtime_fgo": False,
        "truth_usage": "none",
        "controls": results,
        "total_negative_fix_epochs": sum(
            result["declared_fix_epochs"] for result in results
        ),
        "all_pass": all(result["pass"] for result in results),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.audit.open(newline="", encoding="utf-8-sig") as stream:
        rows = [
            row
            for row in csv.DictReader(stream)
            if row.get("shadow_best_sub50cm") in {"0", "1"}
        ]
    payload = analyze(rows, args.domain)
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
