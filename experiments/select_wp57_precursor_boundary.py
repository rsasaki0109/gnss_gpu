#!/usr/bin/env python3
"""Resolve a CP/PR tie from a long accepted right-anchor boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def select(
    source_bytes: bytes,
    cppr_bytes: bytes,
    anchor_bytes: bytes,
    *,
    max_boundary_distance_m: float = 0.2,
    min_runner_margin: float = 0.2,
    min_anchor_epochs: int = 220,
) -> dict[str, Any]:
    source, cppr, anchor = (
        json.loads(source_bytes),
        json.loads(cppr_bytes),
        json.loads(anchor_bytes),
    )
    if any(
        bool(payload.get("production_input_truth", True))
        for payload in (source, cppr, anchor)
    ):
        raise ValueError("all precursor-boundary inputs must be truth-free")
    if cppr.get("input_sha256") != _sha256(source_bytes):
        raise ValueError("CP/PR validation is not linked to the candidate source")
    if not bool(anchor.get("production_promoted", False)) or anchor.get(
        "reason"
    ) != "unique_cppr_rank_consensus":
        raise ValueError("right anchor is not a promoted CP/PR profile")
    segment = [int(value) for value in source["segment"]]
    anchor_segment = [int(value) for value in anchor["segment"]]
    if segment[1] != anchor_segment[0]:
        raise ValueError("candidate block is not directly before the right anchor")
    if anchor_segment[1] - anchor_segment[0] < int(min_anchor_epochs):
        raise ValueError("right anchor is too short for one-hop propagation")
    if not (
        math.isfinite(float(max_boundary_distance_m))
        and max_boundary_distance_m >= 0.0
        and min_runner_margin >= 0.0
    ):
        raise ValueError("boundary thresholds are invalid")

    source_by_id = {int(row["seed_id"]): row for row in source["hypotheses"]}
    absolute = cppr["absolute_gate"]
    family_rank_limit = int(cppr["family_rank_limit"])
    eligible = []
    anchor_boundary = np.asarray(anchor["block_offsets_ecef_m"][0], dtype=float)
    for ranked in cppr["candidates"]:
        candidate_id = int(ranked["candidate_id"])
        row = source_by_id[candidate_id]
        family_pass = max(ranked["family_ranks"].values()) <= family_rank_limit
        absolute_pass = (
            int(ranked["checked_pairs"]) >= int(absolute["min_checked_pairs"])
            and float(ranked["bad_pair_fraction"])
            <= float(absolute["max_bad_pair_fraction"])
            and float(ranked["block_spread_m"])
            <= float(absolute["max_block_spread_m"])
        )
        if not family_pass or not absolute_pass:
            continue
        candidate_boundary = np.asarray(row["block_offsets_ecef_m"][-1], dtype=float)
        eligible.append(
            {
                "candidate_id": candidate_id,
                "boundary_distance_m": float(
                    np.linalg.norm(candidate_boundary - anchor_boundary)
                ),
                "family_ranks": ranked["family_ranks"],
                "cppr_rank_sum": ranked["rank_sum"],
                "checked_pairs": ranked["checked_pairs"],
                "bad_pair_fraction": ranked["bad_pair_fraction"],
                "block_spread_m": ranked["block_spread_m"],
                "offset_ecef_m": row["offset_ecef_m"],
                "block_offsets_ecef_m": row["block_offsets_ecef_m"],
            }
        )
    if len(eligible) < 2:
        raise ValueError("fewer than two candidates pass the frozen CP/PR gates")
    eligible.sort(key=lambda row: (row["boundary_distance_m"], row["candidate_id"]))
    winner, runner = eligible[:2]
    runner_margin = float(
        (runner["boundary_distance_m"] - winner["boundary_distance_m"])
        / max(winner["boundary_distance_m"], 1.0e-9)
    )
    distance_pass = winner["boundary_distance_m"] <= float(
        max_boundary_distance_m
    )
    margin_pass = runner_margin >= float(min_runner_margin)
    accepted = distance_pass and margin_pass
    return {
        "schema": "wp57_precursor_boundary_validation_v1",
        "production_input_truth": False,
        "truth_usage": "none",
        "production_promoted": False,
        "scope": "one block directly before a >=220-epoch promoted CPPR profile",
        "segment": segment,
        "anchor_segment": anchor_segment,
        "selected_candidate_id": winner["candidate_id"] if accepted else None,
        "reason": (
            "unique_long_cppr_precursor_boundary" if accepted else "boundary_gate_failed"
        ),
        "winner": winner,
        "runner": runner,
        "runner_margin": runner_margin,
        "distance_pass": distance_pass,
        "runner_margin_pass": margin_pass,
        "candidate_count": len(eligible),
        "candidates": eligible,
        "config": {
            "max_boundary_distance_m": float(max_boundary_distance_m),
            "min_runner_margin": float(min_runner_margin),
            "min_anchor_epochs": int(min_anchor_epochs),
            "no_recursive_anchor": True,
        },
        "input_sha256": {
            "candidate_source": _sha256(source_bytes),
            "cppr_validation": _sha256(cppr_bytes),
            "right_anchor": _sha256(anchor_bytes),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("cppr_validation", type=Path)
    parser.add_argument("right_anchor", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = select(
        args.source.read_bytes(),
        args.cppr_validation.read_bytes(),
        args.right_anchor.read_bytes(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
