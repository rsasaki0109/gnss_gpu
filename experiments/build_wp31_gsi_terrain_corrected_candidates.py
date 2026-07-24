#!/usr/bin/env python3
"""Project a full horizontal proposal grid to candidate-local cached GSI terrain."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from pyproj import Transformer

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "python"))

from gnss_gpu.io.nmea_writer import _ecef_to_lla_py  # noqa: E402


def terrain_correct_candidates(source: dict[str, Any], cache: dict[str, Any]) -> dict[str, Any]:
    calibrations = list(cache["calibration_points"])
    offsets = []
    for point in calibrations:
        antenna_height = float(_ecef_to_lla_py(*point["antenna_position_ecef"])[2])
        offsets.append(antenna_height - float(point["elevation_m"]) - float(point["geoid_height_m"]))
    spread = float(np.ptp(offsets))
    if len(offsets) < 2 or spread > 0.15:
        raise ValueError("antenna-height calibration is invalid")
    antenna_offset = float(np.median(offsets))
    points = {int(row["candidate_id"]): row for row in cache["candidate_points"]}
    to_lla = Transformer.from_crs("EPSG:4978", "EPSG:4979", always_xy=True)
    to_ecef = Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True)
    output = []
    for row in source["candidates"]:
        candidate_id = int(row["candidate_id"])
        point = points.get(candidate_id)
        if point is None or point.get("dem_source") != "1m（レーザ）":
            raise ValueError("candidate GSI point is missing or unsupported")
        position = np.asarray(row["position_ecef"], dtype=np.float64)
        longitude, latitude, old_height = to_lla.transform(*position)
        if abs(longitude - float(point["longitude_deg"])) > 1e-6 or abs(latitude - float(point["latitude_deg"])) > 1e-6:
            raise ValueError("candidate/cache coordinate mismatch")
        target_height = float(point["elevation_m"]) + float(point["geoid_height_m"]) + antenna_offset
        corrected = np.asarray(to_ecef.transform(longitude, latitude, target_height))
        output.append({
            **{key: value for key, value in row.items() if key != "position_ecef"},
            "preterrain_position_ecef": position.tolist(),
            "preterrain_ellipsoid_height_m": float(old_height),
            "terrain_elevation_m": float(point["elevation_m"]),
            "target_ellipsoid_height_m": target_height,
            "terrain_height_correction_m": float(target_height - old_height),
            "position_ecef": corrected.tolist(),
        })
    return {
        "schema": "wp31_gsi_terrain_corrected_candidates_v1",
        "segment": [int(value) for value in source["segment"]],
        "calibrated_antenna_height_m": antenna_offset,
        "antenna_height_spread_m": spread,
        "candidates": output,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidates_json", type=Path)
    parser.add_argument("cache_json", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = terrain_correct_candidates(json.loads(args.candidates_json.read_text(encoding="utf-8")), json.loads(args.cache_json.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({**result, "candidates": f"{len(result['candidates'])} candidates"}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
