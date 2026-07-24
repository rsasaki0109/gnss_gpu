#!/usr/bin/env python3
"""Select a static candidate with temporal carrier, road, and prior-height evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "python"))

from gnss_gpu.io.nmea_writer import _ecef_to_lla_py  # noqa: E402
from gnss_gpu.io.ppc import PPCDatasetLoader  # noqa: E402
from gnss_gpu.pf_imu_preint_adapter import (  # noqa: E402
    ecef_to_enu_rotation,
    ecef_to_lla_rad,
)


def _selected_position(static_path: Path, fusion_path: Path | None) -> np.ndarray:
    result = json.loads(static_path.read_text(encoding="utf-8"))
    candidates = list(result["candidates"])
    selected_id = int(candidates[0]["candidate_id"])
    if fusion_path is not None:
        fusion = json.loads(fusion_path.read_text(encoding="utf-8"))
        if fusion.get("selected_candidate_id") is None:
            raise RuntimeError("prior fusion did not select a candidate")
        selected_id = int(fusion["selected_candidate_id"])
    matches = [row for row in candidates if int(row["candidate_id"]) == selected_id]
    if len(matches) != 1:
        raise RuntimeError("prior static candidate is absent or duplicated")
    return np.asarray(matches[0]["position_ecef"], dtype=np.float64).reshape(3)


def select_height_temporal_candidate(
    temporal: dict[str, Any],
    road: dict[str, Any],
    prior_positions: list[np.ndarray],
    *,
    max_temporal_ratio: float,
    max_road_distance_m: float,
    min_temporal_arcs: int,
    max_prior_height_spread_m: float,
) -> dict[str, Any]:
    if len(prior_positions) < 2:
        raise RuntimeError("height fusion requires at least two prior static anchors")
    prior_heights = np.asarray(
        [_ecef_to_lla_py(*np.asarray(position, dtype=np.float64))[2] for position in prior_positions]
    )
    spread = float(np.ptp(prior_heights))
    if spread > float(max_prior_height_spread_m):
        raise RuntimeError("prior static heights are not mutually consistent")
    candidates = sorted(
        temporal["candidates"],
        key=lambda row: float(row["carrier_temporal_arc_cauchy_mean"]),
    )
    if len(candidates) < 2:
        raise RuntimeError("temporal result needs a runner-up candidate")
    best, runner = candidates[:2]
    ratio = float(best["carrier_temporal_arc_cauchy_mean"]) / max(
        float(runner["carrier_temporal_arc_cauchy_mean"]), np.finfo(float).eps
    )
    if ratio > float(max_temporal_ratio):
        raise RuntimeError("temporal carrier winner is not separated")
    if int(best.get("carrier_temporal_arcs", 0)) < int(min_temporal_arcs):
        raise RuntimeError("temporal carrier winner has too few arcs")
    road_by_id = {int(row["candidate_id"]): row for row in road["candidates"]}
    selected_id = int(best["candidate_id"])
    road_distance = float(road_by_id[selected_id]["road_distance_m"])
    if road_distance > float(max_road_distance_m):
        raise RuntimeError("temporal carrier winner is too far from an OSM road")
    original = np.asarray(
        road_by_id[selected_id]["position_ecef"], dtype=np.float64
    ).reshape(3)
    target_height = float(np.median(prior_heights))
    original_height = float(_ecef_to_lla_py(*original)[2])
    lat, lon = ecef_to_lla_rad(original)
    up_ecef = ecef_to_enu_rotation(lat, lon)[2]
    corrected = original + (target_height - original_height) * up_ecef
    return {
        "selected_candidate_id": selected_id,
        "reason": "height_temporal_road_consensus",
        "temporal_best_runner_ratio": ratio,
        "temporal_arcs": int(best["carrier_temporal_arcs"]),
        "road_distance_m": road_distance,
        "prior_heights_m": prior_heights.tolist(),
        "prior_height_spread_m": spread,
        "target_height_m": target_height,
        "original_height_m": original_height,
        "height_correction_m": target_height - original_height,
        "original_position_ecef": original.tolist(),
        "position_ecef": corrected.tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("temporal_json", type=Path)
    parser.add_argument("--road-json", type=Path, required=True)
    parser.add_argument("--prior-static-json", type=Path, action="append", required=True)
    parser.add_argument("--prior-fusion-json", type=Path, action="append", default=[])
    parser.add_argument("--max-temporal-ratio", type=float, default=0.95)
    parser.add_argument("--max-road-distance-m", type=float, default=0.5)
    parser.add_argument("--min-temporal-arcs", type=int, default=30)
    parser.add_argument("--max-prior-height-spread-m", type=float, default=1.0)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.prior_fusion_json) > len(args.prior_static_json):
        parser.error("more prior fusion files than prior static files")
    fusion_paths = list(args.prior_fusion_json) + [None] * (
        len(args.prior_static_json) - len(args.prior_fusion_json)
    )
    prior_positions = [
        _selected_position(static_path, fusion_path)
        for static_path, fusion_path in zip(args.prior_static_json, fusion_paths)
    ]
    temporal = json.loads(args.temporal_json.read_text(encoding="utf-8"))
    road = json.loads(args.road_json.read_text(encoding="utf-8"))
    result = select_height_temporal_candidate(
        temporal,
        road,
        prior_positions,
        max_temporal_ratio=float(args.max_temporal_ratio),
        max_road_distance_m=float(args.max_road_distance_m),
        min_temporal_arcs=int(args.min_temporal_arcs),
        max_prior_height_spread_m=float(args.max_prior_height_spread_m),
    )
    result["segment"] = list(temporal["segment"])
    if args.data_dir is not None:
        start, end = (int(value) for value in result["segment"])
        times, truth = PPCDatasetLoader(args.data_dir).load_ground_truth()
        tow = float(times[0]) + 0.2 * (start + end - 1) / 2.0
        truth_position = np.asarray(truth[int(np.argmin(np.abs(times - tow)))])
        result["selected_audit_error_m"] = float(
            np.linalg.norm(np.asarray(result["position_ecef"]) - truth_position)
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
