#!/usr/bin/env python3
"""Promote a hash-linked WP60 two-block path validation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def promote(
    validation_bytes: bytes,
    left_sources: list[bytes],
    left_cpprs: list[bytes],
    right_source: bytes,
    right_cppr: bytes,
    right_boundary: bytes,
    right_anchor: bytes,
) -> dict[str, Any]:
    validation = json.loads(validation_bytes)
    payloads = [
        *[json.loads(value) for value in left_sources],
        *[json.loads(value) for value in left_cpprs],
        json.loads(right_source),
        json.loads(right_cppr),
        json.loads(right_boundary),
        json.loads(right_anchor),
        validation,
    ]
    if any(bool(payload.get("production_input_truth", True)) for payload in payloads):
        raise ValueError("WP60 promotion inputs must be truth-free")
    actual = {
        "left_sources": [_sha256(value) for value in left_sources],
        "left_cppr_validations": [_sha256(value) for value in left_cpprs],
        "right_source": _sha256(right_source),
        "right_cppr_validation": _sha256(right_cppr),
        "right_boundary_validation": _sha256(right_boundary),
        "right_anchor": _sha256(right_anchor),
    }
    if validation.get("input_sha256") != actual:
        raise ValueError("WP60 validation input hashes do not match")
    if not bool(validation.get("selected", False)) or validation.get(
        "reason"
    ) != "unique_multibasis_two_block_path":
        raise ValueError("WP60 path validation did not pass")
    return {
        "schema": "wp60_two_block_path_profile_v1",
        "production_input_truth": False,
        "truth_usage": "none",
        "production_promoted": True,
        "segment": validation["segment"],
        "reason": "unique_multibasis_two_block_path",
        "profile_mode": "linear_bootstrap_centers",
        "block_offsets_ecef_m": validation["block_offsets_ecef_m"],
        "diagnostics": {
            "left_basis_support": validation["left_consensus"]["basis_support"],
            "max_cross_basis_distance_m": validation["left_consensus"][
                "max_cross_basis_distance_m"
            ],
            "right_candidate_id": validation["right_candidate_id"],
            "interface_distance_m": validation["interface_distance_m"],
            "runner_interface_distance_m": validation[
                "runner_interface_distance_m"
            ],
            "interface_runner_margin": validation["interface_runner_margin"],
        },
        "lineage": {
            "right_anchor_reason": json.loads(right_anchor)["reason"],
            "may_seed_another_path_promotion": False,
        },
        "input_sha256": {
            **actual,
            "path_validation": _sha256(validation_bytes),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("validation", type=Path)
    parser.add_argument("--left-source", action="append", type=Path, required=True)
    parser.add_argument("--left-cppr", action="append", type=Path, required=True)
    parser.add_argument("--right-source", type=Path, required=True)
    parser.add_argument("--right-cppr", type=Path, required=True)
    parser.add_argument("--right-boundary", type=Path, required=True)
    parser.add_argument("--right-anchor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = promote(
        args.validation.read_bytes(),
        [path.read_bytes() for path in args.left_source],
        [path.read_bytes() for path in args.left_cppr],
        args.right_source.read_bytes(),
        args.right_cppr.read_bytes(),
        args.right_boundary.read_bytes(),
        args.right_anchor.read_bytes(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
