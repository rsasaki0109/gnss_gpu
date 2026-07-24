#!/usr/bin/env python3
"""Extend a recomputed WP60 path over one leading-instability outage block."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from select_wp60_two_block_path import select as select_base_path  # noqa: E402


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def select(
    predecessor_sources: list[bytes],
    predecessor_cpprs: list[bytes],
    base_validation_bytes: bytes,
    base_left_sources: list[bytes],
    base_left_cpprs: list[bytes],
    base_right_source: bytes,
    base_right_cppr: bytes,
    base_right_boundary: bytes,
    base_anchor: bytes,
    *,
    min_basis_support: int = 2,
    max_tail_spread_m: float = 0.2,
    min_leading_divergence_m: float = 1.0,
) -> dict[str, Any]:
    if len(predecessor_sources) != len(predecessor_cpprs):
        raise ValueError("predecessor source/validation counts differ")
    stored_base = json.loads(base_validation_bytes)
    recomputed_base = select_base_path(
        base_left_sources,
        base_left_cpprs,
        base_right_source,
        base_right_cppr,
        base_right_boundary,
        base_anchor,
    )
    if recomputed_base != stored_base or not bool(recomputed_base.get("selected")):
        raise ValueError("stored WP60 path does not match a fresh recomputation")
    sources = [json.loads(value) for value in predecessor_sources]
    cpprs = [json.loads(value) for value in predecessor_cpprs]
    if any(
        bool(payload.get("production_input_truth", True))
        for payload in [*sources, *cpprs]
    ):
        raise ValueError("predecessor inputs must be truth-free")
    if not sources or any(source["segment"] != sources[0]["segment"] for source in sources):
        raise ValueError("predecessor sources cover different segments")
    segment = [int(value) for value in sources[0]["segment"]]
    if segment[1] - segment[0] != 55 or segment[1] != recomputed_base["segment"][0]:
        raise ValueError("predecessor must be one 55-epoch adjacent block")

    passing = []
    for source_bytes, source, cppr in zip(predecessor_sources, sources, cpprs):
        if cppr.get("input_sha256") != _sha256(source_bytes):
            raise ValueError("a predecessor CP/PR validation is not source-linked")
        absolute = cppr["absolute_gate"]
        shape_pass = (
            bool(cppr.get("family_rank_pass"))
            and bool(cppr.get("runner_margin_pass"))
            and bool(absolute["checked_pairs_pass"])
            and bool(absolute["bad_pair_fraction_pass"])
            and not bool(absolute["block_spread_pass"])
        )
        if not shape_pass:
            continue
        candidate_id = int(cppr["winner"]["candidate_id"])
        matches = [
            row for row in source["hypotheses"] if int(row["seed_id"]) == candidate_id
        ]
        if len(matches) != 1:
            raise ValueError("predecessor winner is not unique")
        blocks = np.asarray(matches[0]["block_offsets_ecef_m"], dtype=float)
        if blocks.shape != (4, 3):
            raise ValueError("predecessor bootstrap profile is not four blocks")
        tail_center = np.median(blocks[1:], axis=0)
        tail_spread = float(np.max(np.linalg.norm(blocks[1:] - tail_center, axis=1)))
        leading_divergence = float(np.linalg.norm(blocks[0] - tail_center))
        if tail_spread <= float(max_tail_spread_m) and leading_divergence >= float(
            min_leading_divergence_m
        ):
            passing.append(
                {
                    "carrier_reference_rank": int(source["carrier_reference_rank"]),
                    "candidate_id": candidate_id,
                    "tail_spread_m": tail_spread,
                    "leading_divergence_m": leading_divergence,
                }
            )
    unique_ranks = {row["carrier_reference_rank"] for row in passing}
    support_pass = len(unique_ranks) >= int(min_basis_support)
    base_profile = np.asarray(recomputed_base["block_offsets_ecef_m"], dtype=float)
    predecessor_profile = np.tile(base_profile[0], (4, 1))
    return {
        "schema": "wp62_backward_outage_path_validation_v1",
        "production_input_truth": False,
        "truth_usage": "none",
        "production_promoted": False,
        "segment": [segment[0], recomputed_base["segment"][1]],
        "predecessor_segment": segment,
        "base_path_segment": recomputed_base["segment"],
        "selected": support_pass,
        "reason": (
            "multibasis_leading_instability_backward_outage_recovery"
            if support_pass
            else "backward_outage_gate_failed"
        ),
        "basis_support": len(unique_ranks),
        "passing_bases": passing,
        "profile_mode": "linear_bootstrap_centers",
        "block_offsets_ecef_m": np.concatenate(
            [predecessor_profile, base_profile]
        ).tolist(),
        "config": {
            "min_basis_support": int(min_basis_support),
            "max_tail_spread_m": float(max_tail_spread_m),
            "min_leading_divergence_m": float(min_leading_divergence_m),
            "propagation": "one_55_epoch_block_constant_from_recomputed_base_path",
            "may_seed_another_outage_or_path_promotion": False,
        },
        "input_sha256": {
            "predecessor_sources": [_sha256(value) for value in predecessor_sources],
            "predecessor_cppr_validations": [
                _sha256(value) for value in predecessor_cpprs
            ],
            "stored_base_validation": _sha256(base_validation_bytes),
            "base_left_sources": [_sha256(value) for value in base_left_sources],
            "base_left_cppr_validations": [
                _sha256(value) for value in base_left_cpprs
            ],
            "base_right_source": _sha256(base_right_source),
            "base_right_cppr": _sha256(base_right_cppr),
            "base_right_boundary": _sha256(base_right_boundary),
            "base_anchor": _sha256(base_anchor),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predecessor-source", action="append", type=Path, required=True)
    parser.add_argument("--predecessor-cppr", action="append", type=Path, required=True)
    parser.add_argument("--base-validation", type=Path, required=True)
    parser.add_argument("--base-left-source", action="append", type=Path, required=True)
    parser.add_argument("--base-left-cppr", action="append", type=Path, required=True)
    parser.add_argument("--base-right-source", type=Path, required=True)
    parser.add_argument("--base-right-cppr", type=Path, required=True)
    parser.add_argument("--base-right-boundary", type=Path, required=True)
    parser.add_argument("--base-anchor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = select(
        [path.read_bytes() for path in args.predecessor_source],
        [path.read_bytes() for path in args.predecessor_cppr],
        args.base_validation.read_bytes(),
        [path.read_bytes() for path in args.base_left_source],
        [path.read_bytes() for path in args.base_left_cppr],
        args.base_right_source.read_bytes(),
        args.base_right_cppr.read_bytes(),
        args.base_right_boundary.read_bytes(),
        args.base_anchor.read_bytes(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
