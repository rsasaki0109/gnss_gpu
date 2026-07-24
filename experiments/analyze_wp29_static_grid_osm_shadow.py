#!/usr/bin/env python3
"""Shadow-rank static-grid candidates with a truth-free OSM road distance."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from pyproj import Transformer
from shapely.geometry import Point
from shapely.ops import nearest_points

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "experiments"))

from analyze_phase67_osm_road_centerline_feasibility import (  # noqa: E402
    _ecef_to_llh,
    _road_union_from_osm,
)


def rank_road_distance(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a copy ordered only by the truth-free road-distance field."""

    return sorted(rows, key=lambda row: float(row["road_distance_m"]))


def road_offset_vector(point: Point, road_geometry: Any) -> tuple[float, float, float]:
    """Return candidate-minus-road east/north vector and its norm."""
    nearest_road, _candidate = nearest_points(road_geometry, point)
    east = float(point.x - nearest_road.x)
    north = float(point.y - nearest_road.y)
    return east, north, float(np.hypot(east, north))


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    source = json.loads(args.candidates_json.read_text(encoding="utf-8"))
    candidates = list(source["candidates"])
    if not candidates:
        raise RuntimeError("candidate result is empty")
    llh = [
        _ecef_to_llh(np.asarray(row["position_ecef"], dtype=np.float64))
        for row in candidates
    ]
    margin = float(args.bbox_margin_deg)
    road_union, _transformer, n_geometries = _road_union_from_osm(
        north=max(row[0] for row in llh) + margin,
        south=min(row[0] for row in llh) - margin,
        east=max(row[1] for row in llh) + margin,
        west=min(row[1] for row in llh) - margin,
        epsg=int(args.epsg),
    )
    ecef_to_map = Transformer.from_crs(
        "EPSG:4978", f"EPSG:{int(args.epsg)}", always_xy=True
    )
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        position = np.asarray(candidate["position_ecef"], dtype=np.float64)
        x, y, _z = ecef_to_map.transform(*position)
        offset_east, offset_north, distance = road_offset_vector(
            Point(x, y), road_union
        )
        rows.append(
            {
                "candidate_id": int(candidate["candidate_id"]),
                "road_distance_m": distance,
                "road_offset_east_m": offset_east,
                "road_offset_north_m": offset_north,
                "map_east_m": float(x),
                "map_north_m": float(y),
                "nearest_road_east_m": float(x - offset_east),
                "nearest_road_north_m": float(y - offset_north),
                "final_error_m": float(candidate.get("final_error_m", float("nan"))),
                "position_ecef": position.tolist(),
            }
        )
    ranked = rank_road_distance(rows)
    for rank, row in enumerate(ranked, start=1):
        row["road_distance_rank"] = rank
    return {
        "epsg": int(args.epsg),
        "bbox_margin_deg": margin,
        "n_road_geometries": int(n_geometries),
        "candidates": ranked,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidates_json", type=Path)
    parser.add_argument("--epsg", type=int, default=32654)
    parser.add_argument("--bbox-margin-deg", type=float, default=0.002)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
