#!/usr/bin/env python3
"""Select one refined cube direction by road calibration and carrier-grid consensus."""

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


def select_cube_consensus(
    candidates: list[dict[str, Any]],
    road_candidates: list[dict[str, Any]],
    integrity_candidates: list[dict[str, Any]],
    calibration_road_distances_m: list[float],
    *,
    road_margin_m: float = 0.15,
    min_calibration_points: int = 5,
    min_grid_scores: int = 12,
    min_winner_fraction: float = 0.5,
    min_win_gap: int = 4,
) -> dict[str, Any]:
    base = {"selected_candidate_id": None, "reason": "osm_temporal_cube_rejected"}
    if len(calibration_road_distances_m) < int(min_calibration_points):
        return {**base, "reason": "insufficient_road_calibration"}
    lower = max(0.0, min(calibration_road_distances_m) - float(road_margin_m))
    upper = max(calibration_road_distances_m) + float(road_margin_m)
    source = {int(row["candidate_id"]): row for row in candidates}
    road = {int(row["candidate_id"]): row for row in road_candidates}
    integrity = {int(row["candidate_id"]): row for row in integrity_candidates}
    eligible = []
    for candidate_id, row in integrity.items():
        if candidate_id not in source or candidate_id not in road:
            return {**base, "reason": "candidate_linkage_missing"}
        position = np.asarray(source[candidate_id]["position_ecef"], dtype=np.float64)
        if np.linalg.norm(position - np.asarray(road[candidate_id]["position_ecef"])) > 1e-6 or np.linalg.norm(position - np.asarray(row["position_ecef"])) > 1e-6:
            return {**base, "reason": "candidate_position_mismatch"}
        distance = float(road[candidate_id]["road_distance_m"])
        if lower <= distance <= upper:
            eligible.append(row)
    if len(eligible) < 2:
        return {**base, "reason": "insufficient_road_eligible_candidates"}
    score_names = sorted(name for name in eligible[0] if _GRID_SCORE.fullmatch(name))
    if len(score_names) < int(min_grid_scores):
        return {**base, "reason": "insufficient_temporal_grid"}
    wins: dict[int, int] = {}
    for score in score_names:
        winner = min(eligible, key=lambda row: float(row[score]))
        candidate_id = int(winner["candidate_id"])
        wins[candidate_id] = wins.get(candidate_id, 0) + 1
    ranked = sorted(wins.items(), key=lambda item: (-item[1], item[0]))
    if len(ranked) < 2:
        return {**base, "reason": "insufficient_candidate_competition"}
    (best_id, best_wins), (runner_id, runner_wins) = ranked[:2]
    cauchy_winner = int(min(eligible, key=lambda row: float(row["carrier_cauchy_mean"]))["candidate_id"])
    arc_winner = int(min(eligible, key=lambda row: float(row["carrier_temporal_arc_cauchy_mean"]))["candidate_id"])
    diagnostics = {
        "calibration_road_distances_m": [float(value) for value in calibration_road_distances_m],
        "road_distance_lower_m": lower,
        "road_distance_upper_m": upper,
        "road_eligible_candidate_ids": sorted(int(row["candidate_id"]) for row in eligible),
        "temporal_grid_score_count": len(score_names),
        "candidate_win_counts": {str(key): value for key, value in sorted(wins.items())},
        "winning_candidate_id": best_id,
        "winning_candidate_wins": best_wins,
        "runner_candidate_id": runner_id,
        "runner_candidate_wins": runner_wins,
        "candidate_win_gap": best_wins - runner_wins,
        "carrier_cauchy_winner_id": cauchy_winner,
        "carrier_temporal_arc_winner_id": arc_winner,
    }
    required = int(math.ceil(float(min_winner_fraction) * len(score_names)))
    if best_wins < required or best_wins - runner_wins < int(min_win_gap):
        return {**base, **diagnostics, "reason": "temporal_cube_not_stable"}
    if best_id != cauchy_winner or best_id != arc_winner:
        return {**base, **diagnostics, "reason": "carrier_metrics_disagree"}
    selected = source[best_id]
    if not bool(selected.get("applied")) or str(selected.get("reason")) not in ("converged", "ok"):
        return {**base, **diagnostics, "reason": "static_refinement_not_applied"}
    return {
        **base,
        **diagnostics,
        "selected_candidate_id": best_id,
        "reason": "gsi_osm_carrier_temporal_cube_consensus",
        "position_ecef": list(selected["position_ecef"]),
    }


def _calibration(value: str) -> tuple[Path, int]:
    path, candidate_id = value.rsplit(":", 1)
    return Path(path), int(candidate_id)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidates_json", type=Path)
    parser.add_argument("road_json", type=Path)
    parser.add_argument("integrity_json", type=Path)
    parser.add_argument("--road-calibration", action="append", type=_calibration, required=True)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_bytes = args.candidates_json.read_bytes()
    source, road, integrity = json.loads(source_bytes), json.loads(args.road_json.read_text(encoding="utf-8")), json.loads(args.integrity_json.read_text(encoding="utf-8"))
    if integrity.get("candidate_source_sha256") != hashlib.sha256(source_bytes).hexdigest():
        parser.error("integrity candidate source SHA-256 mismatch")
    distances = []
    for path, candidate_id in args.road_calibration:
        rows = json.loads(path.read_text(encoding="utf-8"))["candidates"]
        matches = [row for row in rows if int(row["candidate_id"]) == candidate_id]
        if len(matches) != 1:
            parser.error(f"road calibration {path}:{candidate_id} is not unique")
        distances.append(float(matches[0]["road_distance_m"]))
    result = select_cube_consensus(source["candidates"], road["candidates"], integrity["candidates"], distances)
    result["segment"] = [int(value) for value in source["segment"]]
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
