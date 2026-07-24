#!/usr/bin/env python3
"""Pair truth-free local ambiguity candidates across adjacent moving blocks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def _load(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "wp31_moving_block_truth_free_local_pool_v1":
        raise ValueError(f"unexpected pool schema: {path}")
    if payload.get("production_input_truth") is not False:
        raise ValueError(f"pool is not truth-free: {path}")
    return payload


def pair_pools(
    support: dict,
    primary: dict,
    *,
    support_baseline_ddpr_m: float,
    primary_baseline_ddpr_m: float,
    max_carrier_rms_cycles: float = 0.30,
    max_ddpr_ratio: float = 0.80,
    max_offset_distance_m: float = 0.75,
    max_seeds: int = 128,
) -> dict:
    def eligible(rows: list[dict], baseline: float) -> list[dict]:
        return [
            row for row in rows
            if row["integer_arcs"] >= 4
            and row["retained_carrier_rows"] >= 8
            and row["carrier_rms_cycles"] <= max_carrier_rms_cycles
            and row["ddpr_rms_m"] / baseline <= max_ddpr_ratio
        ]

    support_rows = eligible(support["candidates"], support_baseline_ddpr_m)
    primary_rows = eligible(primary["candidates"], primary_baseline_ddpr_m)
    pairs = []
    if support_rows and primary_rows:
        support_xyz = np.asarray([row["offset_ecef_m"] for row in support_rows])
        primary_xyz = np.asarray([row["offset_ecef_m"] for row in primary_rows])
        support_tree = cKDTree(support_xyz); primary_tree = cKDTree(primary_xyz)
        primary_distance, primary_index = support_tree.query(primary_xyz, k=1)
        _support_distance, support_index = primary_tree.query(support_xyz, k=1)
        for p_index, (distance, s_index) in enumerate(zip(primary_distance, primary_index)):
            if distance > max_offset_distance_m or int(support_index[int(s_index)]) != p_index:
                continue
            p_row = primary_rows[p_index]; s_row = support_rows[int(s_index)]
            p_ratio = p_row["ddpr_rms_m"] / primary_baseline_ddpr_m
            s_ratio = s_row["ddpr_rms_m"] / support_baseline_ddpr_m
            pairs.append({
                "offset_distance_m": float(distance),
                "primary_offset_ecef_m": p_row["offset_ecef_m"],
                "support_offset_ecef_m": s_row["offset_ecef_m"],
                "primary_map_translation_xyh_m": p_row.get("map_translation_xyh_m"),
                "support_map_translation_xyh_m": s_row.get("map_translation_xyh_m"),
                "primary_carrier_rms_cycles": p_row["carrier_rms_cycles"],
                "support_carrier_rms_cycles": s_row["carrier_rms_cycles"],
                "primary_ddpr_ratio": p_ratio,
                "support_ddpr_ratio": s_ratio,
                "primary_proposal_score": p_row["proposal_score"],
                "support_proposal_score": s_row["proposal_score"],
            })
    pairs.sort(key=lambda row: (
        max(row["primary_ddpr_ratio"], row["support_ddpr_ratio"]),
        max(row["primary_carrier_rms_cycles"], row["support_carrier_rms_cycles"]),
        row["offset_distance_m"],
        row["primary_proposal_score"] + row["support_proposal_score"],
    ))
    selected = pairs[:max_seeds]
    return {
        "schema": "wp31_moving_block_raw_temporal_pool_v1",
        "production_input_truth": False,
        "support_segment": support["segment"],
        "primary_segment": primary["segment"],
        "gate": {
            "max_carrier_rms_cycles": max_carrier_rms_cycles,
            "max_ddpr_ratio": max_ddpr_ratio,
            "max_offset_distance_m": max_offset_distance_m,
            "mutual_nearest_required": True,
        },
        "support_eligible_candidates": len(support_rows),
        "primary_eligible_candidates": len(primary_rows),
        "mutual_temporal_pairs": len(pairs),
        "selected_seed_count": len(selected),
        "seeds": [
            {"offset_ecef_m": row["primary_offset_ecef_m"], "pair_rank": rank}
            for rank, row in enumerate(selected, start=1)
        ],
        "selected_pairs": selected,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("support_pool", type=Path); parser.add_argument("primary_pool", type=Path)
    parser.add_argument("--support-baseline-ddpr-m", type=float, required=True)
    parser.add_argument("--primary-baseline-ddpr-m", type=float, required=True)
    parser.add_argument("--max-carrier-rms-cycles", type=float, default=0.30)
    parser.add_argument("--max-ddpr-ratio", type=float, default=0.80)
    parser.add_argument("--max-offset-distance-m", type=float, default=0.75)
    parser.add_argument("--max-seeds", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = pair_pools(
        _load(args.support_pool), _load(args.primary_pool),
        support_baseline_ddpr_m=args.support_baseline_ddpr_m,
        primary_baseline_ddpr_m=args.primary_baseline_ddpr_m,
        max_carrier_rms_cycles=args.max_carrier_rms_cycles,
        max_ddpr_ratio=args.max_ddpr_ratio,
        max_offset_distance_m=args.max_offset_distance_m,
        max_seeds=args.max_seeds,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
