#!/usr/bin/env python3
"""Fuse temporal carrier and wide-lane static-grid ranks without reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def select_static_grid_fusion(
    temporal_rows: list[dict[str, Any]],
    widelane_rows: list[dict[str, Any]],
    *,
    evidence_epochs: int,
    candidate_pairs: int,
    fixed_pairs: int,
    min_evidence_epochs: int = 5,
    min_fix_rate: float = 0.5,
    clear_max_median_m: float = 0.5,
    clear_runner_up_ratio: float = 0.6,
    consensus_min_evidence_epochs: int = 10,
    consensus_max_temporal_rank: int = 4,
    consensus_max_widelane_rank: int = 2,
    consensus_max_median_m: float = 2.0,
    consensus_min_rank_gap: int = 1,
    high_evidence_min_epochs: int = 30,
    high_evidence_min_fix_rate: float = 0.45,
    high_evidence_max_temporal_rank: int = 1,
    high_evidence_max_widelane_rank: int = 3,
    high_evidence_min_rank_gap: int = 2,
) -> dict[str, Any]:
    by_temporal = {int(row["candidate_id"]): row for row in temporal_rows}
    by_widelane = {int(row["candidate_id"]): row for row in widelane_rows}
    common = sorted(set(by_temporal) & set(by_widelane))
    fix_rate = float(fixed_pairs) / max(int(candidate_pairs), 1)
    base = {
        "selected_candidate_id": None,
        "reason": "rejected",
        "evidence_epochs": int(evidence_epochs),
        "wide_lane_fix_rate": fix_rate,
    }
    if int(evidence_epochs) < int(min_evidence_epochs):
        return {**base, "reason": "insufficient_widelane_epochs"}
    low_fix_rate = fix_rate < float(min_fix_rate)
    high_evidence_fix_eligible = (
        int(evidence_epochs) >= int(high_evidence_min_epochs)
        and fix_rate >= float(high_evidence_min_fix_rate)
    )
    if low_fix_rate and not high_evidence_fix_eligible:
        return {**base, "reason": "insufficient_widelane_fix_rate"}
    finite_wl = [
        by_widelane[candidate_id]
        for candidate_id in common
        if np.isfinite(float(by_widelane[candidate_id]["widelane_median_abs_m"]))
    ]
    finite_wl.sort(key=lambda row: float(row["widelane_median_abs_m"]))
    if len(finite_wl) < 2:
        return {**base, "reason": "insufficient_widelane_candidates"}
    best_wl = finite_wl[0]
    runner_wl = finite_wl[1]
    best_median = float(best_wl["widelane_median_abs_m"])
    runner_median = float(runner_wl["widelane_median_abs_m"])
    clear_ratio = best_median / max(runner_median, 1.0e-12)
    diagnostics = {
        "best_widelane_candidate_id": int(best_wl["candidate_id"]),
        "best_widelane_median_m": best_median,
        "widelane_runner_up_ratio": clear_ratio,
    }
    if (not low_fix_rate and
        best_median <= float(clear_max_median_m)
        and clear_ratio <= float(clear_runner_up_ratio)
    ):
        return {
            **base,
            **diagnostics,
            "selected_candidate_id": int(best_wl["candidate_id"]),
            "reason": "clear_widelane",
        }
    if int(evidence_epochs) < int(consensus_min_evidence_epochs):
        return {**base, **diagnostics, "reason": "insufficient_consensus_epochs"}

    fused: list[dict[str, Any]] = []
    for candidate_id in common:
        temporal = by_temporal[candidate_id]
        widelane = by_widelane[candidate_id]
        temporal_value = float(temporal["carrier_temporal_window_mean"])
        widelane_value = float(widelane["widelane_median_abs_m"])
        if not np.isfinite(temporal_value) or not np.isfinite(widelane_value):
            continue
        temporal_rank = int(temporal["carrier_temporal_window_mean_rank"])
        widelane_rank = int(widelane["widelane_median_abs_m_rank"])
        fused.append(
            {
                "candidate_id": candidate_id,
                "temporal_rank": temporal_rank,
                "widelane_rank": widelane_rank,
                "widelane_median_abs_m": widelane_value,
                "rank_sum": temporal_rank + widelane_rank,
            }
        )
    fused.sort(key=lambda row: (row["rank_sum"], row["temporal_rank"], row["widelane_rank"]))
    if len(fused) < 2:
        return {**base, **diagnostics, "reason": "insufficient_consensus_candidates"}
    winner, runner = fused[:2]
    rank_gap = int(runner["rank_sum"] - winner["rank_sum"])
    diagnostics.update(
        {
            "consensus_candidate_id": int(winner["candidate_id"]),
            "consensus_temporal_rank": int(winner["temporal_rank"]),
            "consensus_widelane_rank": int(winner["widelane_rank"]),
            "consensus_rank_sum": int(winner["rank_sum"]),
            "consensus_runner_rank_sum": int(runner["rank_sum"]),
            "consensus_rank_gap": rank_gap,
        }
    )
    if low_fix_rate:
        if (
            int(winner["temporal_rank"]) <= int(high_evidence_max_temporal_rank)
            and int(winner["widelane_rank"]) <= int(high_evidence_max_widelane_rank)
            and rank_gap >= int(high_evidence_min_rank_gap)
            and float(winner["widelane_median_abs_m"])
            <= float(consensus_max_median_m)
        ):
            return {
                **base,
                **diagnostics,
                "selected_candidate_id": int(winner["candidate_id"]),
                "reason": "high_evidence_temporal_widelane_consensus",
            }
        return {**base, **diagnostics, "reason": "insufficient_widelane_fix_rate"}
    if int(winner["temporal_rank"]) > int(consensus_max_temporal_rank):
        return {**base, **diagnostics, "reason": "weak_temporal_rank"}
    if int(winner["widelane_rank"]) > int(consensus_max_widelane_rank):
        return {**base, **diagnostics, "reason": "weak_widelane_rank"}
    if float(winner["widelane_median_abs_m"]) > float(consensus_max_median_m):
        return {**base, **diagnostics, "reason": "large_widelane_residual"}
    if rank_gap < int(consensus_min_rank_gap):
        return {**base, **diagnostics, "reason": "ambiguous_rank_sum"}
    return {
        **base,
        **diagnostics,
        "selected_candidate_id": int(winner["candidate_id"]),
        "reason": "temporal_widelane_consensus",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("temporal_json", type=Path)
    parser.add_argument("widelane_json", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    temporal = json.loads(args.temporal_json.read_text(encoding="utf-8"))
    widelane = json.loads(args.widelane_json.read_text(encoding="utf-8"))
    result = select_static_grid_fusion(
        list(temporal["candidates"]),
        list(widelane["candidates"]),
        evidence_epochs=int(widelane["evidence_epochs"]),
        candidate_pairs=int(widelane["candidate_pairs"]),
        fixed_pairs=int(widelane["fixed_pairs"]),
    )
    selected_id = result.get("selected_candidate_id")
    if selected_id is not None:
        source = next(
            row for row in temporal["candidates"] if int(row["candidate_id"]) == int(selected_id)
        )
        result["selected_audit_error_m"] = float(source.get("final_error_m", float("nan")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
