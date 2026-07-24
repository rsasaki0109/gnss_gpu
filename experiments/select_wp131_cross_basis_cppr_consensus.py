#!/usr/bin/env python3
"""Fuse cross-reference convergence with independent CP/PR ranks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from select_wp54_cppr_rank import select as cppr_rank


def _dense_ranks(values: list[float]) -> list[int]:
    ordered = {value: rank for rank, value in enumerate(sorted(set(values)), 1)}
    return [ordered[value] for value in values]


def select(
    source: dict[str, Any],
    cross: dict[str, Any],
    *,
    max_cross_refit_disagreement_m: float = 0.10,
    max_carrier_rms_cycles: float = 0.5,
    max_block_spread_m: float = 0.5,
    min_checked_pairs: int = 40,
    max_bad_pair_fraction: float = 0.05,
    within_basis_dedup_m: float = 0.08,
    max_family_rank_fraction: float = 0.2,
    min_runner_margin: float = 0.2,
) -> dict[str, Any]:
    if bool(source.get("production_input_truth", True)):
        raise ValueError("candidate source is not production-safe")
    if bool(cross.get("production_input_truth", True)):
        raise ValueError("cross-basis source is not production-safe")
    if source.get("segment") != cross.get("segment"):
        raise ValueError("candidate and cross-basis segments differ")

    hypotheses = source.get("hypotheses")
    if not isinstance(hypotheses, list) or len(hypotheses) < 2:
        raise ValueError("candidate source has fewer than two hypotheses")
    if any(not isinstance(row.get("cp_pr_consistency"), dict) for row in hypotheses):
        return {
            "schema": "wp131_cross_basis_cppr_consensus_v1",
            "production_input_truth": False,
            "truth_usage": "none",
            "production_promoted": False,
            "segment": source.get("segment"),
            "accepted": False,
            "reason": "cppr_evidence_unavailable",
            "selected_candidate_id": None,
            "mode_count": 0,
            "modes": [],
        }

    cppr = cppr_rank(source)
    cppr_by_id = {int(row["candidate_id"]): row for row in cppr["candidates"]}
    source_by_id = {int(row["seed_id"]): row for row in hypotheses}
    cross_by_id = {
        int(row["source_candidate_id"]): row for row in cross["candidates"]
    }
    if set(source_by_id) != set(cross_by_id):
        raise ValueError("cross-basis candidates do not cover the source pool")

    eligible = []
    for candidate_id in sorted(source_by_id):
        source_row = source_by_id[candidate_id]
        cppr_row = cppr_by_id[candidate_id]
        cross_row = cross_by_id[candidate_id]
        checked_pairs = int(cppr_row["checked_pairs"])
        bad_pair_fraction = float(cppr_row["bad_pair_fraction"])
        if (
            checked_pairs < int(min_checked_pairs)
            or bad_pair_fraction > float(max_bad_pair_fraction)
            or float(source_row["block_spread_m"]) > float(max_block_spread_m)
            or float(cross_row["max_block_spread_m"]) > float(max_block_spread_m)
            or float(cross_row["max_carrier_rms_cycles"])
            > float(max_carrier_rms_cycles)
            or float(cross_row["rank0_to_rank2_m"])
            > float(max_cross_refit_disagreement_m)
        ):
            continue
        eligible.append(
            {
                "candidate_id": candidate_id,
                "offset_ecef_m": source_row["offset_ecef_m"],
                "block_offsets_ecef_m": source_row["block_offsets_ecef_m"],
                "checked_pairs": checked_pairs,
                "bad_pair_fraction": bad_pair_fraction,
                "block_spread_m": float(source_row["block_spread_m"]),
                "cross_consensus_score_m": float(cross_row["consensus_score_m"]),
                "cross_refit_disagreement_m": float(
                    cross_row["rank0_to_rank2_m"]
                ),
                "max_cross_basis_carrier_rms_cycles": float(
                    cross_row["max_carrier_rms_cycles"]
                ),
                "cppr_rank_sum": int(cppr_row["rank_sum"]),
            }
        )

    eligible.sort(key=lambda row: (row["cppr_rank_sum"], row["candidate_id"]))
    modes: list[dict[str, Any]] = []
    for row in eligible:
        offset = np.asarray(row["offset_ecef_m"], dtype=np.float64)
        if all(
            np.linalg.norm(offset - np.asarray(prior["offset_ecef_m"]))
            > float(within_basis_dedup_m)
            for prior in modes
        ):
            modes.append(row)

    if len(modes) < 2:
        return {
            "schema": "wp131_cross_basis_cppr_consensus_v1",
            "production_input_truth": False,
            "truth_usage": "none",
            "production_promoted": False,
            "segment": source.get("segment"),
            "accepted": False,
            "reason": "fewer_than_two_cross_basis_cppr_modes",
            "selected_candidate_id": None,
            "mode_count": len(modes),
            "modes": modes,
        }

    family_values = {
        "cross_consensus": [row["cross_consensus_score_m"] for row in modes],
        "carrier_rms": [
            row["max_cross_basis_carrier_rms_cycles"] for row in modes
        ],
        "cppr": [float(row["cppr_rank_sum"]) for row in modes],
    }
    family_ranks = {
        name: _dense_ranks(values) for name, values in family_values.items()
    }
    for index, row in enumerate(modes):
        row["family_ranks"] = {
            name: ranks[index] for name, ranks in family_ranks.items()
        }
        row["rank_sum"] = int(sum(row["family_ranks"].values()))
    modes.sort(key=lambda row: (row["rank_sum"], row["candidate_id"]))
    winner, runner = modes[:2]
    runner_margin = float(
        (runner["rank_sum"] - winner["rank_sum"])
        / max(winner["rank_sum"], 1)
    )
    family_rank_limit = int(
        math.ceil(len(modes) * float(max_family_rank_fraction))
    )
    family_rank_pass = max(winner["family_ranks"].values()) <= family_rank_limit
    runner_margin_pass = runner_margin >= float(min_runner_margin)
    accepted = family_rank_pass and runner_margin_pass
    return {
        "schema": "wp131_cross_basis_cppr_consensus_v1",
        "production_input_truth": False,
        "truth_usage": "none",
        "production_promoted": False,
        "segment": source.get("segment"),
        "accepted": accepted,
        "reason": (
            "unique_cross_basis_cppr_mode"
            if accepted
            else "cross_basis_cppr_family_or_margin_gate_failed"
        ),
        "selected_candidate_id": winner["candidate_id"] if accepted else None,
        "winner": winner,
        "runner": runner,
        "runner_margin": runner_margin,
        "family_rank_limit": family_rank_limit,
        "family_rank_pass": family_rank_pass,
        "runner_margin_pass": runner_margin_pass,
        "mode_count": len(modes),
        "modes": modes,
        "gate": {
            "max_cross_refit_disagreement_m": float(
                max_cross_refit_disagreement_m
            ),
            "max_carrier_rms_cycles": float(max_carrier_rms_cycles),
            "max_block_spread_m": float(max_block_spread_m),
            "min_checked_pairs": int(min_checked_pairs),
            "max_bad_pair_fraction": float(max_bad_pair_fraction),
            "within_basis_dedup_m": float(within_basis_dedup_m),
            "max_family_rank_fraction": float(max_family_rank_fraction),
            "min_runner_margin": float(min_runner_margin),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--cross-basis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_bytes = args.source.read_bytes()
    cross_bytes = args.cross_basis.read_bytes()
    result = select(
        json.loads(source_bytes.decode("utf-8")),
        json.loads(cross_bytes.decode("utf-8")),
    )
    result["input_sha256"] = {
        "source": hashlib.sha256(source_bytes).hexdigest(),
        "cross_basis": hashlib.sha256(cross_bytes).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
