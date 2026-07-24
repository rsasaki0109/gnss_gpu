#!/usr/bin/env python3
"""Select an alternate-reference candidate using three-basis consistency only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _find_refit(
    hypotheses: list[dict[str, Any]], seed: np.ndarray
) -> dict[str, Any]:
    ranked = sorted(
        (
            float(np.linalg.norm(np.asarray(row["seed_offset_ecef_m"]) - seed)),
            int(row["seed_id"]),
            row,
        )
        for row in hypotheses
    )
    if not ranked or ranked[0][0] > 0.05:
        raise ValueError("cross-basis refit seed is absent")
    if len(ranked) >= 2 and abs(ranked[1][0] - ranked[0][0]) <= 1.0e-9:
        raise ValueError("cross-basis refit seed is duplicated")
    return ranked[0][2]


def select(
    pool: dict[str, Any],
    source: dict[str, Any],
    cross0: dict[str, Any],
    cross2: dict[str, Any],
    *,
    min_runner_margin: float = 0.20,
    max_cross_refit_disagreement_m: float = 0.05,
    max_carrier_rms_cycles: float = 0.5,
    max_block_spread_m: float = 0.5,
) -> dict[str, Any]:
    for payload in (pool, source, cross0, cross2):
        if bool(payload.get("production_input_truth", True)):
            raise ValueError("cross-basis input is not production-safe")
    source_rank = int(pool.get("source_reference_rank", -1))
    input_ranks = (
        source_rank,
        int(source.get("carrier_reference_rank", -1)),
        int(cross0.get("carrier_reference_rank", -1)),
        int(cross2.get("carrier_reference_rank", -1)),
    )
    if input_ranks[0] != input_ranks[1]:
        raise ValueError("seed pool and source reference ranks differ")
    if set((input_ranks[0], input_ranks[2], input_ranks[3])) != {0, 1, 2}:
        raise ValueError("cross-basis inputs must cover reference ranks 0, 1, and 2")
    segment = source.get("segment")
    if segment != cross0.get("segment") or segment != cross2.get("segment"):
        raise ValueError("cross-basis segments differ")
    source_by_id = {int(row["seed_id"]): row for row in source["hypotheses"]}
    rows = []
    for pool_row in pool["seeds"]:
        source_id = int(pool_row["source_seed_id"])
        source_row = source_by_id[source_id]
        source_offset = np.asarray(source_row["offset_ecef_m"], dtype=np.float64)
        if np.linalg.norm(source_offset - np.asarray(pool_row["offset_ecef_m"])) > 1e-9:
            raise ValueError("seed pool does not match source hypothesis")
        refit0 = _find_refit(cross0["hypotheses"], source_offset)
        refit2 = _find_refit(cross2["hypotheses"], source_offset)
        offset0 = np.asarray(refit0["offset_ecef_m"], dtype=np.float64)
        offset2 = np.asarray(refit2["offset_ecef_m"], dtype=np.float64)
        source_to_0 = float(np.linalg.norm(offset0 - source_offset))
        source_to_2 = float(np.linalg.norm(offset2 - source_offset))
        cross_disagreement = float(np.linalg.norm(offset0 - offset2))
        supply_pass = bool(
            cross_disagreement <= float(max_cross_refit_disagreement_m)
            and max(
                float(source_row["carrier_rms_cycles"]),
                float(refit0["carrier_rms_cycles"]),
                float(refit2["carrier_rms_cycles"]),
            )
            <= float(max_carrier_rms_cycles)
            and max(
                float(source_row["block_spread_m"]),
                float(refit0["block_spread_m"]),
                float(refit2["block_spread_m"]),
            )
            <= float(max_block_spread_m)
        )
        rows.append(
            {
                "source_candidate_id": source_id,
                "source_to_rank0_m": source_to_0,
                "source_to_rank2_m": source_to_2,
                "rank0_to_rank2_m": cross_disagreement,
                "consensus_score_m": source_to_0 + source_to_2 + cross_disagreement,
                "max_carrier_rms_cycles": max(
                    float(source_row["carrier_rms_cycles"]),
                    float(refit0["carrier_rms_cycles"]),
                    float(refit2["carrier_rms_cycles"]),
                ),
                "max_block_spread_m": max(
                    float(source_row["block_spread_m"]),
                    float(refit0["block_spread_m"]),
                    float(refit2["block_spread_m"]),
                ),
                "supply_pass": supply_pass,
                "selected_offset_ecef_m": offset0.tolist(),
                "selected_block_offsets_ecef_m": refit0["block_offsets_ecef_m"],
            }
        )
    eligible = sorted(
        (row for row in rows if row["supply_pass"]),
        key=lambda row: (row["consensus_score_m"], row["source_candidate_id"]),
    )
    winner = eligible[0] if eligible else None
    runner_margin = 0.0
    if len(eligible) >= 2 and float(eligible[1]["consensus_score_m"]) > 0.0:
        runner_margin = (
            float(eligible[1]["consensus_score_m"])
            - float(eligible[0]["consensus_score_m"])
        ) / float(eligible[1]["consensus_score_m"])
    selected = winner if winner is not None and runner_margin >= min_runner_margin else None
    return {
        "schema": "wp53_cross_basis_consensus_validation_v1",
        "production_input_truth": False,
        "truth_usage": "none",
        "production_promoted": False,
        "segment": segment,
        "source_reference_rank": source_rank,
        "cross_reference_ranks": [input_ranks[2], input_ranks[3]],
        "gate": {
            "min_runner_margin": float(min_runner_margin),
            "max_cross_refit_disagreement_m": float(max_cross_refit_disagreement_m),
            "max_carrier_rms_cycles": float(max_carrier_rms_cycles),
            "max_block_spread_m": float(max_block_spread_m),
        },
        "runner_margin": runner_margin,
        "selected_candidate_id": (
            int(selected["source_candidate_id"]) if selected is not None else None
        ),
        "reason": (
            "unique_three_reference_basis_consensus"
            if selected is not None
            else "cross_basis_consensus_gate_failed"
        ),
        "winner": winner,
        "candidates": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--cross-rank0", type=Path, required=True)
    parser.add_argument("--cross-rank2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = [args.pool, args.source, args.cross_rank0, args.cross_rank2]
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    result = select(*payloads)
    result["input_sha256"] = {
        key: _sha256(path)
        for key, path in zip(("pool", "source", "cross_rank0", "cross_rank2"), paths)
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
