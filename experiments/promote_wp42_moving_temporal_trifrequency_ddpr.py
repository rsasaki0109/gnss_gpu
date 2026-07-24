#!/usr/bin/env python3
"""Promote one validated WP42 moving offset without reading audit fields."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sanitized_candidates(source: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "candidate_id": int(row["seed_id"]),
            "offset_ecef_m": [float(value) for value in row["offset_ecef_m"]],
            "integer_arcs": int(row["integer_arcs"]),
            "carrier_rows": int(row["carrier_rows"]),
            "carrier_rms_cycles": float(row["carrier_rms_cycles"]),
            "block_spread_m": float(row["block_spread_m"]),
            "block_offsets_ecef_m": [
                [float(value) for value in offset]
                for offset in row["block_offsets_ecef_m"]
            ],
        }
        for row in source["hypotheses"]
    ]


def _candidate_hash(candidates: list[dict[str, Any]]) -> str:
    return hashlib.sha256(
        json.dumps(candidates, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def promote(
    selector: dict[str, Any],
    source: dict[str, Any],
    *,
    max_offset_norm_m: float = 0.5,
) -> dict[str, Any]:
    if bool(selector.get("production_input_truth", True)):
        raise ValueError("WP42 selector is not truth-free")
    if (
        selector.get("reason")
        != "unique_moving_temporal_trifrequency_ddpr_rank_consensus"
    ):
        raise ValueError("WP42 selector did not pass the fixed consensus gate")
    if selector.get("selected_candidate_id") is None:
        raise ValueError("WP42 selector has no selected candidate")
    if selector.get("segment") != source.get("segment"):
        raise ValueError("WP42 selector and candidate segments differ")
    candidates = _sanitized_candidates(source)
    if selector.get("candidate_source_sha256") != _candidate_hash(candidates):
        raise ValueError("WP42 candidate provenance mismatch")
    selected_id = int(selector["selected_candidate_id"])
    matches = [row for row in candidates if row["candidate_id"] == selected_id]
    if len(matches) != 1:
        raise ValueError("WP42 selected candidate is absent or duplicated")
    selected = matches[0]
    winner = selector.get("winner", {})
    if int(winner.get("candidate_id", -1)) != selected_id or not bool(
        winner.get("supply_pass", False)
    ):
        raise ValueError("WP42 winner did not pass upstream supply")
    offset = np.asarray(selected["offset_ecef_m"], dtype=np.float64)
    if offset.shape != (3,) or not np.isfinite(offset).all():
        raise ValueError("WP42 selected offset is invalid")
    norm = float(np.linalg.norm(offset))
    if norm > float(max_offset_norm_m):
        raise ValueError("WP42 selected offset exceeds the boundary-continuity gate")
    block_offsets = np.asarray(selected["block_offsets_ecef_m"], dtype=np.float64)
    if (
        block_offsets.ndim != 2
        or block_offsets.shape[0] < 2
        or block_offsets.shape[1] != 3
        or not np.isfinite(block_offsets).all()
    ):
        raise ValueError("WP42 bootstrap offset profile is invalid")
    max_profile_norm = float(np.max(np.linalg.norm(block_offsets, axis=1)))
    max_profile_deviation = float(
        np.max(np.linalg.norm(block_offsets - offset[None, :], axis=1))
    )
    if max_profile_norm > float(max_offset_norm_m):
        raise ValueError("WP42 bootstrap profile exceeds the boundary-continuity gate")
    if max_profile_deviation > float(selected["block_spread_m"]) + 1e-9:
        raise ValueError("WP42 bootstrap profile contradicts its frozen spread")
    return {
        "schema": "wp42_moving_temporal_trifrequency_ddpr_production_v1",
        "production_input_truth": False,
        "production_promoted": True,
        "segment": [int(value) for value in source["segment"]],
        "selected_candidate_id": selected_id,
        "offset_ecef_m": offset.tolist(),
        "offset_norm_m": norm,
        "profile_mode": "linear_bootstrap_centers",
        "block_offsets_ecef_m": block_offsets.tolist(),
        "max_profile_offset_norm_m": max_profile_norm,
        "max_profile_deviation_m": max_profile_deviation,
        "reason": "unique_moving_temporal_trifrequency_ddpr_rank_consensus",
        "gate": {
            "max_offset_norm_m": float(max_offset_norm_m),
            "family_ranks": winner["family_ranks"],
            "rank_sum": int(winner["rank_sum"]),
            "runner_margin": float(selector["runner_margin"]),
            "family_rank_limit": int(selector["family_rank_limit"]),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("selector_json", type=Path)
    parser.add_argument("candidate_json", type=Path)
    parser.add_argument("--max-offset-norm-m", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selector = json.loads(args.selector_json.read_text(encoding="utf-8"))
    source = json.loads(args.candidate_json.read_text(encoding="utf-8"))
    result = promote(selector, source, max_offset_norm_m=args.max_offset_norm_m)
    result["input_sha256"] = {
        "selector": _sha256(args.selector_json),
        "candidate_source": _sha256(args.candidate_json),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
