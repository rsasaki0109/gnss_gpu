#!/usr/bin/env python3
"""Select a compact static mode cluster from independent rank evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def select_multimode_cluster(
    candidates: list[dict[str, Any]],
    wide_lane: list[dict[str, Any]],
    temporal: list[dict[str, Any]],
    *,
    link_radius_m: float = 0.5,
    min_members: int = 3,
    max_cluster_spread_m: float = 0.5,
    rank_temperature: float = 10.0,
    min_cluster_score: float = 0.5,
) -> dict[str, Any]:
    positions = {
        int(row["candidate_id"]): np.asarray(row["position_ecef"], dtype=np.float64)
        for row in candidates
    }
    wl = {int(row["candidate_id"]): int(row["widelane_median_abs_m_rank"]) for row in wide_lane}
    tr = {
        int(row["candidate_id"]): int(row["carrier_temporal_arc_cauchy_mean_rank"])
        for row in temporal
    }
    ids = sorted(set(positions) & set(wl) & set(tr))
    if not ids:
        return {"selected_candidate_ids": [], "reason": "no_common_candidates"}
    unvisited = set(ids)
    components = []
    while unvisited:
        seed = min(unvisited)
        unvisited.remove(seed)
        stack, members = [seed], []
        while stack:
            candidate_id = stack.pop()
            members.append(candidate_id)
            neighbors = [
                other for other in list(unvisited)
                if float(np.linalg.norm(positions[candidate_id] - positions[other])) <= link_radius_m
            ]
            for other in neighbors:
                unvisited.remove(other)
                stack.append(other)
        weights = np.asarray(
            [np.exp(-(wl[candidate_id] + tr[candidate_id]) / rank_temperature) for candidate_id in members]
        )
        xyz = np.asarray([positions[candidate_id] for candidate_id in members])
        center = np.average(xyz, axis=0, weights=weights)
        spread = float(np.max(np.linalg.norm(xyz - center, axis=1)))
        components.append(
            {
                "member_ids": sorted(members),
                "members": len(members),
                "score": float(np.sum(weights)),
                "spread_m": spread,
                "position_ecef": center.tolist(),
                "best_rank_sum": min(wl[candidate_id] + tr[candidate_id] for candidate_id in members),
            }
        )
    components.sort(key=lambda row: (-float(row["score"]), float(row["spread_m"])))
    eligible = [
        row for row in components
        if int(row["members"]) >= min_members
        and float(row["spread_m"]) <= max_cluster_spread_m
        and float(row["score"]) >= min_cluster_score
    ]
    if not eligible:
        return {
            "selected_candidate_ids": [],
            "reason": "no_eligible_multimode_cluster",
            "clusters": components,
        }
    best = eligible[0]
    return {
        "selected_candidate_ids": best["member_ids"],
        "position_ecef": best["position_ecef"],
        "reason": "compact_multimode_rank_cluster_development",
        "selected_cluster_score": best["score"],
        "selected_cluster_spread_m": best["spread_m"],
        "eligible_cluster_count": len(eligible),
        "clusters": components,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidates_json", type=Path)
    parser.add_argument("wide_lane_json", type=Path)
    parser.add_argument("temporal_json", type=Path)
    parser.add_argument("--link-radius-m", type=float, default=0.5)
    parser.add_argument("--min-members", type=int, default=3)
    parser.add_argument("--max-cluster-spread-m", type=float, default=0.5)
    parser.add_argument("--rank-temperature", type=float, default=10.0)
    parser.add_argument("--min-cluster-score", type=float, default=0.5)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.candidates_json.read_text(encoding="utf-8"))
    wide_lane = json.loads(args.wide_lane_json.read_text(encoding="utf-8"))
    temporal = json.loads(args.temporal_json.read_text(encoding="utf-8"))
    result = select_multimode_cluster(
        source["candidates"], wide_lane["candidates"], temporal["candidates"],
        link_radius_m=args.link_radius_m, min_members=args.min_members,
        max_cluster_spread_m=args.max_cluster_spread_m,
        rank_temperature=args.rank_temperature, min_cluster_score=args.min_cluster_score,
    )
    result.update(
        {
            "schema": "wp31_static_multimode_cluster_v1",
            "segment": source["segment"],
            "production_input_truth": False,
            "production_promoted": False,
            "config": {
                "link_radius_m": args.link_radius_m,
                "min_members": args.min_members,
                "max_cluster_spread_m": args.max_cluster_spread_m,
                "rank_temperature": args.rank_temperature,
                "min_cluster_score": args.min_cluster_score,
            },
        }
    )
    if args.data_dir is not None and result.get("position_ecef") is not None:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
        from gnss_gpu.io.ppc import PPCDatasetLoader

        start, end = (int(value) for value in source["segment"])
        data = PPCDatasetLoader(args.data_dir).load_experiment_data(max_epochs=end)
        segment_truth = np.asarray(data["ground_truth"][start:end], dtype=np.float64)
        segment_truth = segment_truth[np.isfinite(segment_truth).all(axis=1)]
        if not len(segment_truth):
            raise RuntimeError("static segment has no finite audit truth")
        truth_center = np.median(segment_truth, axis=0)
        result["selected_audit_error_m"] = float(
            np.linalg.norm(np.asarray(result["position_ecef"]) - truth_center)
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "clusters"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
