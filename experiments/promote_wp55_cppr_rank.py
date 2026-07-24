#!/usr/bin/env python3
"""Promote a hash-linked, truth-free WP55 CP/PR rank winner."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def promote(source_bytes: bytes, validation_bytes: bytes) -> dict[str, Any]:
    source = json.loads(source_bytes)
    validation = json.loads(validation_bytes)
    if bool(source.get("production_input_truth", True)) or bool(
        validation.get("production_input_truth", True)
    ):
        raise ValueError("promotion inputs must be truth-free")
    if validation.get("input_sha256") != _sha256(source_bytes):
        raise ValueError("validation is not hash-linked to the candidate source")
    candidate_id = validation.get("selected_candidate_id")
    if (
        candidate_id is None
        or validation.get("reason") != "unique_cppr_rank_consensus"
        or not bool(validation.get("family_rank_pass", False))
        or not bool(validation.get("runner_margin_pass", False))
        or not bool(validation.get("absolute_gate_pass", False))
    ):
        raise ValueError("CP/PR validation did not pass every production gate")
    candidates = [
        row
        for row in source.get("hypotheses", [])
        if int(row.get("seed_id", -1)) == int(candidate_id)
    ]
    if len(candidates) != 1:
        raise ValueError("selected candidate is not unique in the source")
    candidate = candidates[0]
    winner = validation.get("winner", {})
    for key in (
        "offset_ecef_m",
        "block_offsets_ecef_m",
        "carrier_rms_cycles",
        "block_spread_m",
    ):
        if winner.get(key) != candidate.get(key):
            raise ValueError(f"validation winner {key} differs from source")
    return {
        "schema": "wp55_cppr_rank_profile_v1",
        "production_input_truth": False,
        "truth_usage": "none",
        "production_promoted": True,
        "segment": validation["segment"],
        "selected_candidate_id": int(candidate_id),
        "reason": "unique_cppr_rank_consensus",
        "profile_mode": "linear_bootstrap_centers",
        "offset_ecef_m": candidate["offset_ecef_m"],
        "block_offsets_ecef_m": candidate["block_offsets_ecef_m"],
        "diagnostics": {
            "cp_pr_consistency": candidate["cp_pr_consistency"],
            "carrier_rms_cycles": candidate["carrier_rms_cycles"],
            "block_spread_m": candidate["block_spread_m"],
            "family_ranks": winner["family_ranks"],
            "rank_sum": winner["rank_sum"],
            "runner_margin": validation["runner_margin"],
        },
        "gates": {
            "family_rank": True,
            "runner_margin": True,
            "absolute": True,
            **validation["absolute_gate"],
        },
        "input_sha256": {
            "candidate_source": _sha256(source_bytes),
            "validation": _sha256(validation_bytes),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("validation", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = promote(args.source.read_bytes(), args.validation.read_bytes())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
