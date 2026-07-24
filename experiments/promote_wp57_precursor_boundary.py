#!/usr/bin/env python3
"""Promote a hash-linked WP57 long-anchor precursor-boundary winner."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def promote(
    source_bytes: bytes,
    cppr_bytes: bytes,
    validation_bytes: bytes,
    anchor_bytes: bytes,
) -> dict[str, Any]:
    source, cppr, validation, anchor = (
        json.loads(source_bytes),
        json.loads(cppr_bytes),
        json.loads(validation_bytes),
        json.loads(anchor_bytes),
    )
    if any(
        bool(payload.get("production_input_truth", True))
        for payload in (source, cppr, validation, anchor)
    ):
        raise ValueError("WP57 promotion inputs must be truth-free")
    expected = validation.get("input_sha256", {})
    actual = {
        "candidate_source": _sha256(source_bytes),
        "cppr_validation": _sha256(cppr_bytes),
        "right_anchor": _sha256(anchor_bytes),
    }
    if expected != actual:
        raise ValueError("WP57 validation input hashes do not match")
    candidate_id = validation.get("selected_candidate_id")
    if (
        candidate_id is None
        or validation.get("reason") != "unique_long_cppr_precursor_boundary"
        or not bool(validation.get("distance_pass", False))
        or not bool(validation.get("runner_margin_pass", False))
    ):
        raise ValueError("WP57 boundary validation did not pass")
    candidates = [
        row
        for row in source.get("hypotheses", [])
        if int(row.get("seed_id", -1)) == int(candidate_id)
    ]
    if len(candidates) != 1:
        raise ValueError("selected WP57 candidate is not unique")
    candidate = candidates[0]
    winner = validation["winner"]
    for key in ("offset_ecef_m", "block_offsets_ecef_m", "block_spread_m"):
        if winner.get(key) != candidate.get(key):
            raise ValueError(f"WP57 winner {key} differs from source")
    return {
        "schema": "wp57_precursor_boundary_profile_v1",
        "production_input_truth": False,
        "truth_usage": "none",
        "production_promoted": True,
        "segment": validation["segment"],
        "selected_candidate_id": int(candidate_id),
        "reason": "unique_long_cppr_precursor_boundary",
        "profile_mode": "linear_bootstrap_centers",
        "offset_ecef_m": candidate["offset_ecef_m"],
        "block_offsets_ecef_m": candidate["block_offsets_ecef_m"],
        "diagnostics": {
            "cp_pr_consistency": candidate["cp_pr_consistency"],
            "carrier_rms_cycles": candidate["carrier_rms_cycles"],
            "block_spread_m": candidate["block_spread_m"],
            "boundary_distance_m": winner["boundary_distance_m"],
            "runner_boundary_distance_m": validation["runner"][
                "boundary_distance_m"
            ],
            "runner_margin": validation["runner_margin"],
        },
        "anchor_lineage": {
            "segment": anchor["segment"],
            "reason": anchor["reason"],
            "may_seed_another_boundary_promotion": False,
        },
        "input_sha256": {
            **actual,
            "boundary_validation": _sha256(validation_bytes),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("cppr_validation", type=Path)
    parser.add_argument("boundary_validation", type=Path)
    parser.add_argument("right_anchor", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = promote(
        args.source.read_bytes(),
        args.cppr_validation.read_bytes(),
        args.boundary_validation.read_bytes(),
        args.right_anchor.read_bytes(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
