#!/usr/bin/env python3
"""Select a horizontal PF direction by calibrated road offset and carrier stability."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "python"))

from gnss_gpu.io.ppc import PPCDatasetLoader  # noqa: E402


_GRID_SCORE = re.compile(r"^carrier_temporal_m\d+_s\d+$")


def select_direction_consensus(
    shell_candidates: list[dict[str, Any]],
    road_candidates: list[dict[str, Any]],
    integrity_candidates: list[dict[str, Any]],
    calibration_road_distances_m: list[float],
    *,
    road_margin_m: float = 0.15,
    min_calibration_points: int = 3,
    min_grid_scores: int = 12,
    min_winner_fraction: float = 2.0 / 3.0,
    min_win_gap: int = 8,
    max_cluster_spread_m: float = 0.25,
) -> dict[str, Any]:
    base = {"selected_candidate_id": None, "reason": "osm_temporal_direction_rejected"}
    if len(calibration_road_distances_m) < int(min_calibration_points):
        return {**base, "reason": "insufficient_road_calibration"}
    lower = max(0.0, min(calibration_road_distances_m) - float(road_margin_m))
    upper = max(calibration_road_distances_m) + float(road_margin_m)
    shell = {int(row["candidate_id"]): row for row in shell_candidates}
    road = {int(row["candidate_id"]): row for row in road_candidates}
    integrity = {int(row["candidate_id"]): row for row in integrity_candidates}
    if len(shell) != len(shell_candidates) or len(road) != len(road_candidates):
        return {**base, "reason": "duplicate_candidate_id"}
    eligible: list[dict[str, Any]] = []
    for candidate_id, row in integrity.items():
        if candidate_id not in shell or candidate_id not in road:
            return {**base, "reason": "candidate_linkage_missing"}
        shell_position = np.asarray(shell[candidate_id]["position_ecef"], dtype=np.float64)
        if np.linalg.norm(shell_position - np.asarray(road[candidate_id]["position_ecef"])) > 1e-6 or np.linalg.norm(shell_position - np.asarray(row["position_ecef"])) > 1e-6:
            return {**base, "reason": "candidate_position_mismatch"}
        distance = float(road[candidate_id]["road_distance_m"])
        if (
            shell[candidate_id].get("proposal_kind") == "horizontal_ring"
            and lower <= distance <= upper
        ):
            eligible.append(row)
    if not eligible:
        return {**base, "reason": "no_road_calibrated_candidate"}
    score_names = sorted(
        name for name in eligible[0] if _GRID_SCORE.fullmatch(name)
    )
    if len(score_names) < int(min_grid_scores):
        return {**base, "reason": "insufficient_temporal_grid"}
    wins: dict[int, int] = {}
    for score in score_names:
        winner = min(eligible, key=lambda row: float(row[score]))
        direction = int(shell[int(winner["candidate_id"])]["direction_index"])
        wins[direction] = wins.get(direction, 0) + 1
    ranked_wins = sorted(wins.items(), key=lambda item: (-item[1], item[0]))
    if len(ranked_wins) < 2:
        return {**base, "reason": "insufficient_direction_competition"}
    (best_direction, best_wins), (runner_direction, runner_wins) = ranked_wins[:2]
    diagnostics = {
        "calibration_road_distances_m": [float(value) for value in calibration_road_distances_m],
        "road_distance_lower_m": lower,
        "road_distance_upper_m": upper,
        "road_eligible_candidate_ids": sorted(int(row["candidate_id"]) for row in eligible),
        "temporal_grid_score_count": len(score_names),
        "direction_win_counts": {str(key): value for key, value in sorted(wins.items())},
        "winning_direction_index": best_direction,
        "winning_direction_wins": best_wins,
        "runner_direction_index": runner_direction,
        "runner_direction_wins": runner_wins,
        "direction_win_gap": best_wins - runner_wins,
    }
    required_wins = int(math.ceil(float(min_winner_fraction) * len(score_names)))
    if best_wins < required_wins or best_wins - runner_wins < int(min_win_gap):
        return {**base, **diagnostics, "reason": "temporal_direction_not_stable"}
    radii = sorted({float(row["radius_m"]) for row in shell_candidates if row.get("proposal_kind") == "horizontal_ring"})
    cluster = [
        row
        for row in shell_candidates
        if row.get("proposal_kind") == "horizontal_ring"
        and int(row["direction_index"]) == best_direction
    ]
    if len(cluster) != len(radii) or any(int(row["candidate_id"]) not in {int(item["candidate_id"]) for item in eligible} for row in cluster):
        return {**base, **diagnostics, "reason": "winning_direction_incomplete"}
    positions = np.asarray([row["position_ecef"] for row in cluster], dtype=np.float64)
    position = np.mean(positions, axis=0)
    spread = float(np.max(np.linalg.norm(positions - position, axis=1)))
    if spread > float(max_cluster_spread_m):
        return {**base, **diagnostics, "reason": "winning_radius_cluster_too_wide", "cluster_spread_m": spread}
    representative = min(cluster, key=lambda row: float(np.linalg.norm(np.asarray(row["position_ecef"]) - position)))
    return {
        **base,
        **diagnostics,
        "selected_candidate_id": int(representative["candidate_id"]),
        "reason": "gsi_osm_carrier_temporal_direction_consensus",
        "selected_cluster_candidate_ids": [int(row["candidate_id"]) for row in cluster],
        "cluster_spread_m": spread,
        "position_ecef": position.tolist(),
    }


def _calibration(value: str) -> tuple[Path, int]:
    path, candidate_id = value.rsplit(":", 1)
    return Path(path), int(candidate_id)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shell_json", type=Path)
    parser.add_argument("road_json", type=Path)
    parser.add_argument("integrity_json", type=Path)
    parser.add_argument("--road-calibration", action="append", type=_calibration, required=True)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    shell_bytes = args.shell_json.read_bytes()
    shell, road, integrity = json.loads(shell_bytes), json.loads(args.road_json.read_text(encoding="utf-8")), json.loads(args.integrity_json.read_text(encoding="utf-8"))
    if integrity.get("candidate_source_sha256") != hashlib.sha256(shell_bytes).hexdigest():
        parser.error("integrity candidate source SHA-256 mismatch")
    calibration_distances = []
    for path, candidate_id in args.road_calibration:
        source = json.loads(path.read_text(encoding="utf-8"))
        matches = [row for row in source["candidates"] if int(row["candidate_id"]) == candidate_id]
        if len(matches) != 1:
            parser.error(f"road calibration {path}:{candidate_id} is not unique")
        calibration_distances.append(float(matches[0]["road_distance_m"]))
    result = select_direction_consensus(shell["candidates"], road["candidates"], integrity["candidates"], calibration_distances)
    result["segment"] = [int(value) for value in shell["segment"]]
    if result.get("position_ecef") is not None and args.data_dir is not None:
        start, end = result["segment"]
        _times, truth = PPCDatasetLoader(args.data_dir).load_ground_truth()
        values = np.asarray(truth[start:end], dtype=np.float64)
        values = values[np.isfinite(values).all(axis=1)]
        result["selected_audit_error_m"] = float(np.linalg.norm(np.asarray(result["position_ecef"]) - np.median(values, axis=0)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
