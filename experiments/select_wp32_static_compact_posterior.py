#!/usr/bin/env python3
"""Select compact posterior balls without single-link chaining."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def select_compact_posterior(
    candidates: list[dict[str, Any]],
    wide_lane: list[dict[str, Any]],
    temporal: list[dict[str, Any]],
    *,
    ball_radius_m: float = 0.5,
    min_members: int = 3,
    max_spread_m: float = 0.5,
    rank_temperature: float = 10.0,
    min_score: float = 0.5,
) -> dict[str, Any]:
    positions = {
        int(row["candidate_id"]): np.asarray(row["position_ecef"], dtype=np.float64)
        for row in candidates
    }
    wl = {
        int(row["candidate_id"]): int(row["widelane_median_abs_m_rank"])
        for row in wide_lane
    }
    tr = {
        int(row["candidate_id"]): int(row["carrier_temporal_arc_cauchy_mean_rank"])
        for row in temporal
    }
    ids = sorted(set(positions) & set(wl) & set(tr))
    balls: dict[tuple[int, ...], dict[str, Any]] = {}
    for seed in ids:
        members = [
            candidate_id
            for candidate_id in ids
            if float(np.linalg.norm(positions[candidate_id] - positions[seed]))
            <= ball_radius_m
        ]
        for _iteration in range(4):
            if not members:
                break
            weights = np.asarray(
                [
                    np.exp(-(wl[candidate_id] + tr[candidate_id]) / rank_temperature)
                    for candidate_id in members
                ]
            )
            xyz = np.asarray([positions[candidate_id] for candidate_id in members])
            center = np.average(xyz, axis=0, weights=weights)
            retained = [
                candidate_id
                for candidate_id in members
                if float(np.linalg.norm(positions[candidate_id] - center))
                <= max_spread_m
            ]
            if retained == members:
                break
            members = retained
        if not members:
            continue
        weights = np.asarray(
            [
                np.exp(-(wl[candidate_id] + tr[candidate_id]) / rank_temperature)
                for candidate_id in members
            ]
        )
        xyz = np.asarray([positions[candidate_id] for candidate_id in members])
        center = np.average(xyz, axis=0, weights=weights)
        spread = float(np.max(np.linalg.norm(xyz - center, axis=1)))
        key = tuple(sorted(members))
        balls[key] = {
            "member_ids": list(key),
            "members": len(key),
            "score": float(np.sum(weights)),
            "spread_m": spread,
            "position_ecef": center.tolist(),
            "best_rank_sum": min(wl[value] + tr[value] for value in key),
        }
    ranked = sorted(
        balls.values(), key=lambda row: (-float(row["score"]), float(row["spread_m"]))
    )
    eligible = [
        row
        for row in ranked
        if int(row["members"]) >= min_members
        and float(row["spread_m"]) <= max_spread_m
        and float(row["score"]) >= min_score
    ]
    if not eligible:
        return {
            "selected_candidate_ids": [],
            "reason": "no_eligible_compact_posterior",
            "posterior_balls": ranked,
        }
    best = eligible[0]
    return {
        "selected_candidate_ids": best["member_ids"],
        "position_ecef": best["position_ecef"],
        "reason": "compact_rank_posterior_development",
        "selected_score": best["score"],
        "selected_spread_m": best["spread_m"],
        "eligible_posterior_count": len(eligible),
        "posterior_balls": ranked,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidates_json", type=Path)
    parser.add_argument("wide_lane_json", type=Path)
    parser.add_argument("temporal_json", type=Path)
    parser.add_argument("--ball-radius-m", type=float, default=0.5)
    parser.add_argument("--min-members", type=int, default=3)
    parser.add_argument("--max-spread-m", type=float, default=0.5)
    parser.add_argument("--rank-temperature", type=float, default=10.0)
    parser.add_argument("--min-score", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.candidates_json.read_text(encoding="utf-8"))
    wide_lane = json.loads(args.wide_lane_json.read_text(encoding="utf-8"))
    temporal = json.loads(args.temporal_json.read_text(encoding="utf-8"))
    result = select_compact_posterior(
        source["candidates"],
        wide_lane["candidates"],
        temporal["candidates"],
        ball_radius_m=args.ball_radius_m,
        min_members=args.min_members,
        max_spread_m=args.max_spread_m,
        rank_temperature=args.rank_temperature,
        min_score=args.min_score,
    )
    result.update(
        {
            "schema": "wp32_static_compact_posterior_v1",
            "segment": source["segment"],
            "production_input_truth": False,
            "production_promoted": False,
            "config": {
                "ball_radius_m": args.ball_radius_m,
                "min_members": args.min_members,
                "max_spread_m": args.max_spread_m,
                "rank_temperature": args.rank_temperature,
                "min_score": args.min_score,
            },
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "posterior_balls"},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
