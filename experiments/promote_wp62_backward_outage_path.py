#!/usr/bin/env python3
"""Promote a WP62 validation after verifying every hash-linked input."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _flatten_hashes(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [item for child in value for item in _flatten_hashes(child)]
    if isinstance(value, dict):
        return [item for child in value.values() for item in _flatten_hashes(child)]
    raise ValueError("validation input hash structure is invalid")


def promote(validation_bytes: bytes, input_bytes: list[bytes]) -> dict[str, Any]:
    validation = json.loads(validation_bytes)
    inputs = [json.loads(value) for value in input_bytes]
    if bool(validation.get("production_input_truth", True)) or any(
        bool(payload.get("production_input_truth", True)) for payload in inputs
    ):
        raise ValueError("WP62 promotion inputs must be truth-free")
    expected = sorted(_flatten_hashes(validation.get("input_sha256", {})))
    actual = sorted(_sha256(value) for value in input_bytes)
    if expected != actual:
        raise ValueError("WP62 validation input hash multiset does not match")
    if not bool(validation.get("selected", False)) or validation.get(
        "reason"
    ) != "multibasis_leading_instability_backward_outage_recovery":
        raise ValueError("WP62 outage validation did not pass")
    return {
        "schema": "wp62_backward_outage_path_profile_v1",
        "production_input_truth": False,
        "truth_usage": "none",
        "production_promoted": True,
        "segment": validation["segment"],
        "reason": validation["reason"],
        "profile_mode": "linear_bootstrap_centers",
        "block_offsets_ecef_m": validation["block_offsets_ecef_m"],
        "diagnostics": {
            "basis_support": validation["basis_support"],
            "passing_bases": validation["passing_bases"],
            "propagation": validation["config"]["propagation"],
        },
        "lineage": {
            "global_base_path_recomputed": True,
            "may_seed_another_outage_or_path_promotion": False,
        },
        "input_sha256": {
            "validation": _sha256(validation_bytes),
            "verified_inputs": actual,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("validation", type=Path)
    parser.add_argument("--input", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = promote(
        args.validation.read_bytes(), [path.read_bytes() for path in args.input]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
