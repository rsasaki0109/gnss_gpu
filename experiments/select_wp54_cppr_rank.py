#!/usr/bin/env python3
"""Select a moving hypothesis by CP/PR median, tail, and bad-pair ranks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def select(
    source: dict[str, Any],
    *,
    max_family_rank_fraction: float = 0.2,
    min_runner_margin: float = 0.2,
    min_checked_pairs: int = 40,
    max_bad_pair_fraction: float = 0.05,
    max_block_spread_m: float = 0.5,
) -> dict[str, Any]:
    if bool(source.get("production_input_truth", True)):
        raise ValueError("CP/PR source is not production-safe")
    hypotheses = source.get("hypotheses")
    if not isinstance(hypotheses, list) or len(hypotheses) < 2:
        raise ValueError("CP/PR source has fewer than two hypotheses")
    if not 0.0 < float(max_family_rank_fraction) <= 1.0:
        raise ValueError("rank fraction must be in (0, 1]")
    if int(min_checked_pairs) < 1:
        raise ValueError("minimum checked pairs must be positive")
    if not 0.0 <= float(max_bad_pair_fraction) <= 1.0:
        raise ValueError("bad-pair fraction must be in [0, 1]")
    if not math.isfinite(float(max_block_spread_m)) or max_block_spread_m < 0.0:
        raise ValueError("maximum block spread must be finite and nonnegative")
    metric_names = (
        "median_abs_innovation_m",
        "p95_abs_innovation_m",
        "bad_pairs",
    )
    ranks: dict[int, dict[str, int]] = {int(row["seed_id"]): {} for row in hypotheses}
    for metric in metric_names:
        distinct_values = sorted(
            {float(row["cp_pr_consistency"][metric]) for row in hypotheses}
        )
        dense_rank = {value: rank for rank, value in enumerate(distinct_values, 1)}
        ordered = sorted(
            hypotheses,
            key=lambda row: (
                float(row["cp_pr_consistency"][metric]), int(row["seed_id"])
            ),
        )
        for row in ordered:
            value = float(row["cp_pr_consistency"][metric])
            ranks[int(row["seed_id"])][metric] = dense_rank[value]
    rows = []
    source_by_id = {int(row["seed_id"]): row for row in hypotheses}
    for candidate_id in sorted(source_by_id):
        source_row = source_by_id[candidate_id]
        family_ranks = ranks[candidate_id]
        rows.append(
            {
                "candidate_id": candidate_id,
                "family_ranks": family_ranks,
                "rank_sum": int(sum(family_ranks.values())),
                "checked_pairs": int(
                    source_row["cp_pr_consistency"]["checked_pairs"]
                ),
                "bad_pair_fraction": float(
                    source_row["cp_pr_consistency"]["bad_pairs"]
                    / max(int(source_row["cp_pr_consistency"]["checked_pairs"]), 1)
                ),
                "metrics": {
                    metric: source_row["cp_pr_consistency"][metric]
                    for metric in metric_names
                },
                "offset_ecef_m": source_row["offset_ecef_m"],
                "block_offsets_ecef_m": source_row["block_offsets_ecef_m"],
                "carrier_rms_cycles": source_row["carrier_rms_cycles"],
                "block_spread_m": source_row["block_spread_m"],
            }
        )
    rows.sort(key=lambda row: (row["rank_sum"], row["candidate_id"]))
    winner, runner = rows[:2]
    runner_margin = float(
        (int(runner["rank_sum"]) - int(winner["rank_sum"]))
        / max(int(winner["rank_sum"]), 1)
    )
    rank_limit = int(math.ceil(len(rows) * float(max_family_rank_fraction)))
    family_rank_pass = max(winner["family_ranks"].values()) <= rank_limit
    margin_pass = runner_margin >= float(min_runner_margin)
    checked_pairs_pass = winner["checked_pairs"] >= int(min_checked_pairs)
    bad_pair_fraction_pass = (
        winner["bad_pair_fraction"] <= float(max_bad_pair_fraction)
    )
    block_spread_pass = (
        float(winner["block_spread_m"]) <= float(max_block_spread_m)
    )
    absolute_gate_pass = (
        checked_pairs_pass and bad_pair_fraction_pass and block_spread_pass
    )
    accepted = family_rank_pass and margin_pass and absolute_gate_pass
    return {
        "schema": "wp54_cppr_rank_validation_v1",
        "production_input_truth": False,
        "truth_usage": "none",
        "production_promoted": False,
        "segment": source.get("segment"),
        "selected_candidate_id": winner["candidate_id"] if accepted else None,
        "reason": (
            "unique_cppr_rank_consensus" if accepted else "cppr_rank_gate_failed"
        ),
        "winner": winner,
        "runner": runner,
        "runner_margin": runner_margin,
        "family_rank_limit": rank_limit,
        "family_rank_pass": family_rank_pass,
        "runner_margin_pass": margin_pass,
        "absolute_gate_pass": absolute_gate_pass,
        "absolute_gate": {
            "min_checked_pairs": int(min_checked_pairs),
            "max_bad_pair_fraction": float(max_bad_pair_fraction),
            "max_block_spread_m": float(max_block_spread_m),
            "checked_pairs_pass": checked_pairs_pass,
            "bad_pair_fraction_pass": bad_pair_fraction_pass,
            "block_spread_pass": block_spread_pass,
        },
        "candidate_count": len(rows),
        "candidates": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_bytes = args.source.read_bytes()
    result = select(json.loads(source_bytes.decode("utf-8")))
    result["input_sha256"] = hashlib.sha256(source_bytes).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
