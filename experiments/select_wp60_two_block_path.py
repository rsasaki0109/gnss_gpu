#!/usr/bin/env python3
"""Select a two-block path from multi-basis CP/PR consensus and a right anchor."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def select(
    left_source_bytes: list[bytes],
    left_cppr_bytes: list[bytes],
    right_source_bytes: bytes,
    right_cppr_bytes: bytes,
    right_boundary_bytes: bytes,
    anchor_bytes: bytes,
    *,
    max_cross_basis_distance_m: float = 0.2,
    min_basis_support: int = 2,
    max_interface_distance_m: float = 0.2,
    min_interface_runner_margin: float = 0.2,
) -> dict[str, Any]:
    if len(left_source_bytes) != len(left_cppr_bytes) or len(left_source_bytes) < 2:
        raise ValueError("left source/validation lists must have equal length >= 2")
    right_source, right_cppr, right_boundary, anchor = (
        json.loads(right_source_bytes),
        json.loads(right_cppr_bytes),
        json.loads(right_boundary_bytes),
        json.loads(anchor_bytes),
    )
    left_sources = [json.loads(payload) for payload in left_source_bytes]
    left_cpprs = [json.loads(payload) for payload in left_cppr_bytes]
    if any(
        bool(payload.get("production_input_truth", True))
        for payload in [*left_sources, *left_cpprs, right_source, right_cppr, right_boundary, anchor]
    ):
        raise ValueError("all two-block path inputs must be truth-free")
    for source_bytes, source, cppr in zip(
        left_source_bytes, left_sources, left_cpprs
    ):
        if cppr.get("input_sha256") != _sha256(source_bytes):
            raise ValueError("a left CP/PR validation is not source-linked")
        if source["segment"] != left_sources[0]["segment"]:
            raise ValueError("left candidate sources cover different segments")
    expected_right = right_boundary.get("input_sha256", {})
    if expected_right != {
        "candidate_source": _sha256(right_source_bytes),
        "cppr_validation": _sha256(right_cppr_bytes),
        "right_anchor": _sha256(anchor_bytes),
    }:
        raise ValueError("right boundary validation is not input-linked")
    if right_boundary.get("selected_candidate_id") is None:
        raise ValueError("right boundary validation did not pass")
    left_segment = [int(value) for value in left_sources[0]["segment"]]
    right_segment = [int(value) for value in right_source["segment"]]
    if left_segment[1] != right_segment[0]:
        raise ValueError("left and right blocks are not adjacent")

    selected_left = []
    for source, cppr in zip(left_sources, left_cpprs):
        candidate_id = cppr.get("selected_candidate_id")
        if candidate_id is None:
            continue
        matches = [
            row for row in source["hypotheses"] if int(row["seed_id"]) == int(candidate_id)
        ]
        if len(matches) != 1:
            raise ValueError("left selected candidate is not unique")
        row = matches[0]
        selected_left.append(
            {
                "carrier_reference_rank": int(source["carrier_reference_rank"]),
                "candidate_id": int(candidate_id),
                "offset_ecef_m": row["offset_ecef_m"],
                "block_offsets_ecef_m": row["block_offsets_ecef_m"],
            }
        )
    if len(selected_left) < int(min_basis_support):
        raise ValueError("too few left bases pass their CP/PR selectors")
    compatible_groups = []
    for size in range(int(min_basis_support), len(selected_left) + 1):
        for indices in itertools.combinations(range(len(selected_left)), size):
            distances = [
                float(
                    np.linalg.norm(
                        np.asarray(selected_left[i]["offset_ecef_m"], dtype=float)
                        - np.asarray(selected_left[j]["offset_ecef_m"], dtype=float)
                    )
                )
                for i, j in itertools.combinations(indices, 2)
            ]
            if distances and max(distances) <= float(max_cross_basis_distance_m):
                compatible_groups.append((indices, max(distances)))
    if not compatible_groups:
        raise ValueError("no left cross-basis consensus exists")
    max_support = max(len(indices) for indices, _distance in compatible_groups)
    maximal = [row for row in compatible_groups if len(row[0]) == max_support]
    unique_memberships = {tuple(row[0]) for row in maximal}
    if len(unique_memberships) != 1:
        raise ValueError("left cross-basis consensus is not unique")
    indices, max_basis_distance = maximal[0]
    members = [selected_left[index] for index in indices]
    profiles = np.asarray([row["block_offsets_ecef_m"] for row in members], dtype=float)
    if len({profile.shape for profile in profiles}) != 1:
        raise ValueError("left consensus profiles have different shapes")
    left_profile = np.median(profiles, axis=0)

    right_winner = right_boundary["winner"]
    right_runner = right_boundary["runner"]
    right_profile = np.asarray(right_winner["block_offsets_ecef_m"], dtype=float)
    runner_profile = np.asarray(right_runner["block_offsets_ecef_m"], dtype=float)
    interface_distance = float(np.linalg.norm(left_profile[-1] - right_profile[0]))
    runner_interface_distance = float(
        np.linalg.norm(left_profile[-1] - runner_profile[0])
    )
    runner_margin = float(
        (runner_interface_distance - interface_distance)
        / max(interface_distance, 1.0e-9)
    )
    interface_pass = interface_distance <= float(max_interface_distance_m)
    margin_pass = runner_margin >= float(min_interface_runner_margin)
    accepted = interface_pass and margin_pass
    return {
        "schema": "wp60_two_block_path_validation_v1",
        "production_input_truth": False,
        "truth_usage": "none",
        "production_promoted": False,
        "segment": [left_segment[0], right_segment[1]],
        "left_segment": left_segment,
        "right_segment": right_segment,
        "selected": accepted,
        "reason": "unique_multibasis_two_block_path" if accepted else "path_gate_failed",
        "left_consensus": {
            "members": members,
            "basis_support": len(members),
            "max_cross_basis_distance_m": max_basis_distance,
            "block_offsets_ecef_m": left_profile.tolist(),
        },
        "right_candidate_id": right_winner["candidate_id"],
        "right_block_offsets_ecef_m": right_profile.tolist(),
        "interface_distance_m": interface_distance,
        "runner_interface_distance_m": runner_interface_distance,
        "interface_runner_margin": runner_margin,
        "interface_pass": interface_pass,
        "interface_runner_margin_pass": margin_pass,
        "profile_mode": "linear_bootstrap_centers",
        "block_offsets_ecef_m": np.concatenate([left_profile, right_profile]).tolist(),
        "config": {
            "max_cross_basis_distance_m": float(max_cross_basis_distance_m),
            "min_basis_support": int(min_basis_support),
            "max_interface_distance_m": float(max_interface_distance_m),
            "min_interface_runner_margin": float(min_interface_runner_margin),
            "may_seed_another_path_promotion": False,
        },
        "input_sha256": {
            "left_sources": [_sha256(payload) for payload in left_source_bytes],
            "left_cppr_validations": [_sha256(payload) for payload in left_cppr_bytes],
            "right_source": _sha256(right_source_bytes),
            "right_cppr_validation": _sha256(right_cppr_bytes),
            "right_boundary_validation": _sha256(right_boundary_bytes),
            "right_anchor": _sha256(anchor_bytes),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-source", action="append", type=Path, required=True)
    parser.add_argument("--left-cppr", action="append", type=Path, required=True)
    parser.add_argument("--right-source", type=Path, required=True)
    parser.add_argument("--right-cppr", type=Path, required=True)
    parser.add_argument("--right-boundary", type=Path, required=True)
    parser.add_argument("--right-anchor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = select(
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
