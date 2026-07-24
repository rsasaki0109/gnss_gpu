#!/usr/bin/env python3
"""Select one resampled parent by relative secondary DDPR, then compact primary fit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def _spread(rows: list[dict[str, Any]]) -> float:
    positions = np.asarray([row["position_ecef"] for row in rows], dtype=np.float64)
    return max(
        (
            float(np.linalg.norm(positions[left] - positions[right]))
            for left in range(len(positions))
            for right in range(left + 1, len(positions))
        ),
        default=0.0,
    )


def select_relative_secondary_parent(
    parents: list[tuple[dict[str, Any], dict[str, Any]]],
    *,
    top_k: int = 3,
    min_evidence_epochs: int = 10,
    min_relative_margin: float = 0.075,
    max_primary_spread_m: float = 0.5,
) -> dict[str, Any]:
    if len(parents) < 3:
        raise ValueError("at least three identically resampled parents are required")
    summaries: list[dict[str, Any]] = []
    for parent_index, (candidates, secondary) in enumerate(parents):
        if candidates.get("segment") != secondary.get("segment"):
            raise ValueError("candidate and secondary segments differ")
        if bool(secondary.get("production_input_truth", True)):
            raise ValueError("secondary artifact is not truth-free")
        if secondary.get("pseudorange_family") != "secondary":
            raise ValueError("secondary artifact used another pseudorange family")
        if secondary.get("calibration") is not None:
            raise ValueError("relative selector requires uncalibrated secondary evidence")
        if int(secondary.get("evidence_epochs", 0)) < min_evidence_epochs:
            raise ValueError("secondary evidence is below the fixed minimum")
        candidate_rows = {
            int(row["candidate_id"]): row
            for row in candidates.get("candidates", [])
            if row.get("proposal_kind") == "offset_seed"
        }
        secondary_rows = [
            row
            for row in secondary.get("candidates", [])
            if int(row["candidate_id"]) in candidate_rows
        ]
        secondary_rows.sort(key=lambda row: float(row["ddpr_median_abs_m"]))
        if len(secondary_rows) < top_k:
            raise ValueError("parent has fewer than top-k offset candidates")
        top_secondary = secondary_rows[:top_k]
        summaries.append(
            {
                "parent_index": parent_index,
                "seed_parent_candidate_id": int(candidates["seed_parent_candidate_id"]),
                "secondary_top_ids": [int(row["candidate_id"]) for row in top_secondary],
                "secondary_top_mean_m": float(
                    np.mean([row["ddpr_median_abs_m"] for row in top_secondary])
                ),
                "secondary_top_max_m": float(
                    max(row["ddpr_median_abs_m"] for row in top_secondary)
                ),
                "candidate_rows": candidate_rows,
            }
        )
    summaries.sort(key=lambda row: row["secondary_top_mean_m"])
    winner, runner = summaries[:2]
    relative_margin = float(
        (runner["secondary_top_mean_m"] - winner["secondary_top_mean_m"])
        / winner["secondary_top_mean_m"]
    )
    accepted = relative_margin >= min_relative_margin
    primary_top: list[dict[str, Any]] = []
    primary_spread_m = float("inf")
    position = None
    if accepted:
        primary_top = sorted(
            winner["candidate_rows"].values(),
            key=lambda row: (float(row["final_norm_rms"]), float(row["final_cost"])),
        )[:top_k]
        primary_spread_m = _spread(primary_top)
        accepted = primary_spread_m <= max_primary_spread_m
        if accepted:
            position = np.mean(
                np.asarray([row["position_ecef"] for row in primary_top]), axis=0
            ).tolist()
    public_summaries = [
        {key: value for key, value in row.items() if key != "candidate_rows"}
        for row in summaries
    ]
    return {
        "selected_candidate_id": 0 if accepted else None,
        "reason": (
            "unique_relative_secondary_parent_primary_compact"
            if accepted
            else "relative_secondary_parent_gate_failed"
        ),
        "selected_parent_index": winner["parent_index"] if accepted else None,
        "selected_seed_parent_candidate_id": (
            winner["seed_parent_candidate_id"] if accepted else None
        ),
        "secondary_relative_margin": relative_margin,
        "primary_top_ids": [int(row["candidate_id"]) for row in primary_top],
        "primary_spread_m": primary_spread_m,
        "position_ecef": position,
        "parents": public_summaries,
        "candidates": []
        if not accepted
        else [{"candidate_id": 0, "position_ecef": position}],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-json", type=Path, action="append", required=True)
    parser.add_argument("--secondary-json", type=Path, action="append", required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-evidence-epochs", type=int, default=10)
    parser.add_argument("--min-relative-margin", type=float, default=0.075)
    parser.add_argument("--max-primary-spread-m", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.candidate_json) != len(args.secondary_json):
        parser.error("candidate and secondary artifact counts differ")
    candidate_docs = [json.loads(path.read_text(encoding="utf-8")) for path in args.candidate_json]
    secondary_docs = [json.loads(path.read_text(encoding="utf-8")) for path in args.secondary_json]
    result = select_relative_secondary_parent(
        list(zip(candidate_docs, secondary_docs)),
        top_k=args.top_k,
        min_evidence_epochs=args.min_evidence_epochs,
        min_relative_margin=args.min_relative_margin,
        max_primary_spread_m=args.max_primary_spread_m,
    )
    segment = candidate_docs[0]["segment"]
    if any(document.get("segment") != segment for document in candidate_docs):
        parser.error("parent segments differ")
    result = {
        "schema": "wp34_relative_secondary_parent_development_v1",
        "production_input_truth": False,
        "production_promoted": False,
        "segment": segment,
        "config": {
            "top_k": args.top_k,
            "min_evidence_epochs": args.min_evidence_epochs,
            "min_relative_margin": args.min_relative_margin,
            "max_primary_spread_m": args.max_primary_spread_m,
        },
        "candidate_sha256": [
            hashlib.sha256(path.read_bytes()).hexdigest() for path in args.candidate_json
        ],
        "secondary_sha256": [
            hashlib.sha256(path.read_bytes()).hexdigest() for path in args.secondary_json
        ],
        **result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
