#!/usr/bin/env python3
"""Select a static child with a cached GSI ground-height calibration.

The production selector is network-free.  A checked-in JSON cache supplies the
GSI DEM and geoid values; two already accepted static anchors calibrate the
vehicle antenna height above the mapped ground surface.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "python"))

from gnss_gpu.io.nmea_writer import _ecef_to_lla_py  # noqa: E402
from gnss_gpu.io.ppc import PPCDatasetLoader  # noqa: E402


def _ellipsoid_height(position_ecef: list[float]) -> float:
    return float(_ecef_to_lla_py(*np.asarray(position_ecef, dtype=np.float64))[2])


def select_gsi_height_candidate(
    candidates: list[dict[str, Any]],
    calibration_points: list[dict[str, Any]],
    target_point: dict[str, Any],
    *,
    min_calibration_points: int = 2,
    max_antenna_height_spread_m: float = 0.15,
    max_selected_residual_m: float = 0.15,
    min_runner_gap_m: float = 0.10,
    required_dem_source: str = "1m（レーザ）",
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "selected_candidate_id": None,
        "reason": "gsi_height_rejected",
    }
    if len(calibration_points) < int(min_calibration_points):
        return {**base, "reason": "insufficient_calibration_points"}
    all_points = [*calibration_points, target_point]
    if any(point.get("dem_source") != required_dem_source for point in all_points):
        return {**base, "reason": "unsupported_dem_source"}
    geoid_models = {str(point.get("geoid_model")) for point in all_points}
    if len(geoid_models) != 1:
        return {**base, "reason": "mixed_geoid_models"}

    antenna_offsets = []
    for point in calibration_points:
        antenna_height = _ellipsoid_height(point["antenna_position_ecef"])
        ground_height = float(point["elevation_m"]) + float(point["geoid_height_m"])
        antenna_offsets.append(antenna_height - ground_height)
    offset_spread = float(np.ptp(np.asarray(antenna_offsets, dtype=np.float64)))
    if offset_spread > float(max_antenna_height_spread_m):
        return {
            **base,
            "reason": "inconsistent_antenna_height_calibration",
            "antenna_height_offsets_m": antenna_offsets,
            "antenna_height_spread_m": offset_spread,
        }

    target_ground_height = float(target_point["elevation_m"]) + float(
        target_point["geoid_height_m"]
    )
    antenna_offset = float(np.median(np.asarray(antenna_offsets, dtype=np.float64)))
    predicted_height = target_ground_height + antenna_offset
    ranked = sorted(
        (
            {
                "candidate_id": int(row["candidate_id"]),
                "position_ecef": row["position_ecef"],
                "ellipsoid_height_m": _ellipsoid_height(row["position_ecef"]),
            }
            for row in candidates
        ),
        key=lambda row: abs(float(row["ellipsoid_height_m"]) - predicted_height),
    )
    if len(ranked) < 2:
        return {**base, "reason": "insufficient_candidate_count"}
    best, runner = ranked[:2]
    best_residual = abs(float(best["ellipsoid_height_m"]) - predicted_height)
    runner_residual = abs(float(runner["ellipsoid_height_m"]) - predicted_height)
    runner_gap = runner_residual - best_residual
    diagnostics = {
        "geoid_model": next(iter(geoid_models)),
        "antenna_height_offsets_m": antenna_offsets,
        "antenna_height_spread_m": offset_spread,
        "calibrated_antenna_height_m": antenna_offset,
        "target_ground_ellipsoid_height_m": target_ground_height,
        "predicted_antenna_ellipsoid_height_m": predicted_height,
        "best_candidate_id": int(best["candidate_id"]),
        "best_height_residual_m": best_residual,
        "runner_height_residual_m": runner_residual,
        "runner_gap_m": runner_gap,
        "runner_candidate_id": int(runner["candidate_id"]),
    }
    if best_residual > float(max_selected_residual_m):
        return {**base, **diagnostics, "reason": "weak_absolute_height_match"}
    if runner_gap < float(min_runner_gap_m):
        return {**base, **diagnostics, "reason": "height_winner_not_separated"}
    return {
        **base,
        **diagnostics,
        "selected_candidate_id": int(best["candidate_id"]),
        "reason": "gsi_ground_height_calibrated",
        "selected_ellipsoid_height_m": float(best["ellipsoid_height_m"]),
        "position_ecef": list(best["position_ecef"]),
    }


def select_gsi_height_osm_candidate(
    candidates: list[dict[str, Any]],
    road_candidates: list[dict[str, Any]],
    calibration_points: list[dict[str, Any]],
    target_point: dict[str, Any],
    *,
    max_antenna_height_spread_m: float = 0.15,
    max_height_residual_m: float = 0.15,
    max_road_distance_m: float = 1.0,
    required_dem_source: str = "1m（レーザ）",
) -> dict[str, Any]:
    """Require a unique candidate in the conservative GSI-height/OSM set."""
    base = {"selected_candidate_id": None, "reason": "gsi_height_osm_rejected"}
    if len(calibration_points) < 2:
        return {**base, "reason": "insufficient_calibration_points"}
    all_points = [*calibration_points, target_point]
    if any(point.get("dem_source") != required_dem_source for point in all_points):
        return {**base, "reason": "unsupported_dem_source"}
    geoid_models = {str(point.get("geoid_model")) for point in all_points}
    if len(geoid_models) != 1:
        return {**base, "reason": "mixed_geoid_models"}
    offsets = [
        _ellipsoid_height(point["antenna_position_ecef"])
        - float(point["elevation_m"])
        - float(point["geoid_height_m"])
        for point in calibration_points
    ]
    spread = float(np.ptp(np.asarray(offsets, dtype=np.float64)))
    if spread > float(max_antenna_height_spread_m):
        return {
            **base,
            "reason": "inconsistent_antenna_height_calibration",
            "antenna_height_offsets_m": offsets,
            "antenna_height_spread_m": spread,
        }
    antenna_offset = float(np.median(np.asarray(offsets, dtype=np.float64)))
    predicted_height = (
        float(target_point["elevation_m"])
        + float(target_point["geoid_height_m"])
        + antenna_offset
    )
    road_by_id = {int(row["candidate_id"]): row for row in road_candidates}
    eligible = []
    diagnostics = []
    for row in candidates:
        candidate_id = int(row["candidate_id"])
        if candidate_id not in road_by_id:
            return {**base, "reason": "road_candidate_missing"}
        position = np.asarray(row["position_ecef"], dtype=np.float64).reshape(3)
        road_position = np.asarray(
            road_by_id[candidate_id]["position_ecef"], dtype=np.float64
        ).reshape(3)
        if float(np.linalg.norm(position - road_position)) > 1e-6:
            return {**base, "reason": "road_candidate_position_mismatch"}
        height = _ellipsoid_height(row["position_ecef"])
        height_residual = abs(height - predicted_height)
        road_distance = float(road_by_id[candidate_id]["road_distance_m"])
        item = {
            "candidate_id": candidate_id,
            "height_residual_m": height_residual,
            "road_distance_m": road_distance,
        }
        diagnostics.append(item)
        if (
            height_residual <= float(max_height_residual_m)
            and road_distance <= float(max_road_distance_m)
        ):
            eligible.append(item)
    common = {
        "geoid_model": next(iter(geoid_models)),
        "antenna_height_offsets_m": offsets,
        "antenna_height_spread_m": spread,
        "calibrated_antenna_height_m": antenna_offset,
        "predicted_antenna_ellipsoid_height_m": predicted_height,
        "max_height_residual_m": float(max_height_residual_m),
        "max_road_distance_m": float(max_road_distance_m),
        "eligible_candidate_ids": [item["candidate_id"] for item in eligible],
    }
    if len(eligible) != 1:
        return {
            **base,
            **common,
            "reason": "height_osm_gate_not_unique",
            "eligible_candidate_count": len(eligible),
        }
    selected_id = int(eligible[0]["candidate_id"])
    selected = next(row for row in candidates if int(row["candidate_id"]) == selected_id)
    return {
        **base,
        **common,
        "selected_candidate_id": selected_id,
        "reason": "gsi_height_osm_unique_gate",
        "selected_height_residual_m": float(eligible[0]["height_residual_m"]),
        "selected_road_distance_m": float(eligible[0]["road_distance_m"]),
        "position_ecef": list(selected["position_ecef"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidates_json", type=Path)
    parser.add_argument("--cache-json", type=Path, required=True)
    parser.add_argument("--road-json", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.candidates_json.read_text(encoding="utf-8"))
    cache_bytes = args.cache_json.read_bytes()
    cache = json.loads(cache_bytes.decode("utf-8"))
    if args.road_json is None:
        result = select_gsi_height_candidate(
            list(source["candidates"]),
            list(cache["calibration_points"]),
            dict(cache["target_point"]),
        )
    else:
        road = json.loads(args.road_json.read_text(encoding="utf-8"))
        result = select_gsi_height_osm_candidate(
            list(source["candidates"]),
            list(road["candidates"]),
            list(cache["calibration_points"]),
            dict(cache["target_point"]),
        )
    result["segment"] = [int(value) for value in source["segment"]]
    result["height_cache_sha256"] = hashlib.sha256(cache_bytes).hexdigest()
    if result.get("position_ecef") is not None and args.data_dir is not None:
        start, end = result["segment"]
        _times, truth = PPCDatasetLoader(args.data_dir).load_ground_truth()
        segment_truth = np.asarray(truth[start:end], dtype=np.float64)
        segment_truth = segment_truth[np.isfinite(segment_truth).all(axis=1)]
        if not len(segment_truth):
            raise RuntimeError("static segment has no finite audit truth")
        truth_position = np.median(segment_truth, axis=0)
        result["selected_audit_error_m"] = float(
            np.linalg.norm(np.asarray(result["position_ecef"]) - truth_position)
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
