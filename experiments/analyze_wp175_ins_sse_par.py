#!/usr/bin/env python3
"""Post-selection truth audit for the shadow INS-SSE-PAR candidate source."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def _quantile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def analyze(
    audit_path: Path, reference_path: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    truth = {
        round(float(row["GPS TOW (s)"]), 3): tuple(
            float(row[f"ECEF {axis} (m)"]) for axis in "XYZ"
        )
        for row in _rows(reference_path)
    }
    rows = _rows(audit_path)
    candidates: list[dict[str, Any]] = []
    for row in rows:
        if row.get("passed") != "1":
            continue
        tow = round(float(row["tow"]), 3)
        candidate = tuple(
            float(row[f"candidate_ecef_{axis}"]) for axis in "xyz"
        )
        reference = truth.get(tow)
        error_m = math.dist(candidate, reference) if reference else None
        candidates.append(
            {
                "tow": tow,
                "fixed_count": int(row["fixed_count"]),
                "ratio": float(row["ratio"]),
                "bsr_qscale16": float(row["bsr_qscale16"]),
                "ffrt_min_ratio": float(row["ffrt_min_ratio"]),
                "sse_statistic_per_dof": float(
                    row["sse_statistic_per_dof"]
                ),
                "position_separation_m": float(
                    row["position_separation_m"]
                ),
                "error_m": error_m if error_m is not None else "",
                "sub50cm": (
                    int(error_m < 0.5) if error_m is not None else ""
                ),
            }
        )
    errors = [
        float(row["error_m"])
        for row in candidates
        if row["error_m"] != ""
    ]
    summary = {
        "schema": "gnss_gpu_wp175_ins_sse_par_audit_v1",
        "runtime_fgo": False,
        "truth_usage": "post_selection_audit_only",
        "audit_epochs": len(rows),
        "joint_posterior_and_anchor_available_epochs": sum(
            row.get("available") == "1" for row in rows
        ),
        "eligible_epochs": sum(
            int(row.get("subsets_evaluated") or 0) > 0 for row in rows
        ),
        "subsets_evaluated": sum(
            int(row.get("subsets_evaluated") or 0) for row in rows
        ),
        "ffrt_passed_subsets": sum(
            int(row.get("ratio_passed_subsets") or 0) for row in rows
        ),
        "sse_rejected_subsets": sum(
            int(row.get("separation_rejected_subsets") or 0)
            for row in rows
        ),
        "candidate_epochs": len(candidates),
        "truth_labeled_candidate_epochs": len(errors),
        "sub50cm_candidate_epochs": sum(error < 0.5 for error in errors),
        "not_sub50cm_candidate_epochs": sum(
            error >= 0.5 for error in errors
        ),
        "candidate_error_m": {
            "p50": _quantile(errors, 0.5),
            "p95": _quantile(errors, 0.95),
            "max": max(errors) if errors else None,
        },
        "single_source_fix_authority": False,
        "promotion_ready": False,
    }
    return candidates, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output-candidates", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    args = parser.parse_args()
    candidates, summary = analyze(args.audit, args.reference)
    args.output_candidates.parent.mkdir(parents=True, exist_ok=True)
    if candidates:
        with args.output_candidates.open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(
                stream, fieldnames=list(candidates[0])
            )
            writer.writeheader()
            writer.writerows(candidates)
    else:
        args.output_candidates.write_text("", encoding="utf-8")
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
