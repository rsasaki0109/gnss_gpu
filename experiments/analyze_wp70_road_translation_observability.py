#!/usr/bin/env python3
"""Gate common-translation road evidence by posterior observability.

The analyzer is truth-free.  It evaluates a fixed XY translation grid against
the calibrated road-distance band and rejects broad or tied posteriors before
they can supply a PF/carrier candidate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from pyproj import Transformer
from shapely.geometry import LineString
from shapely.strtree import STRtree

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "experiments"))

from build_wp31_osm_particle_route_bridge import _road_distances  # noqa: E402


def evaluate_road_translation_observability(
    route_xy: np.ndarray,
    road: Any,
    *,
    radius_m: float,
    step_m: float,
    lower_m: float,
    upper_m: float,
    equivalent_score_tolerance: float = 1e-12,
    distinct_runner_m: float = 1.0,
    max_equivalent_cells: int = 25,
    max_equivalent_extent_m: float = 1.0,
    min_runner_margin: float = 0.2,
) -> dict[str, Any]:
    route_xy = np.asarray(route_xy, dtype=np.float64)
    if route_xy.ndim != 2 or route_xy.shape[1] != 2 or len(route_xy) < 2:
        raise ValueError("route_xy must contain at least two XY positions")
    if radius_m <= 0.0 or step_m <= 0.0 or lower_m < 0.0 or upper_m < lower_m:
        raise ValueError("road translation grid or band is invalid")

    values = np.arange(-radius_m, radius_m + 0.5 * step_m, step_m)
    rows: list[tuple[float, float, float, float, float]] = []
    for dx in values:
        for dy in values:
            distances = _road_distances(
                road, route_xy[:, 0] + float(dx), route_xy[:, 1] + float(dy)
            )
            outside = np.maximum(lower_m - distances, 0.0) + np.maximum(
                distances - upper_m, 0.0
            )
            score = float(np.mean(np.square(outside)))
            rows.append(
                (
                    score,
                    float(dx),
                    float(dy),
                    float(np.median(distances)),
                    float(np.percentile(distances, 95.0)),
                )
            )
    rows.sort()
    winner = rows[0]
    equivalent = [row for row in rows if row[0] <= winner[0] + equivalent_score_tolerance]
    equivalent_xy = np.asarray([[row[1], row[2]] for row in equivalent])
    extent = float(
        max(np.ptp(equivalent_xy[:, 0]), np.ptp(equivalent_xy[:, 1]))
    )
    runner = next(
        (
            row
            for row in rows[1:]
            if np.hypot(row[1] - winner[1], row[2] - winner[2])
            >= distinct_runner_m
        ),
        None,
    )
    runner_margin = (
        float("inf")
        if runner is None
        else float((runner[0] - winner[0]) / max(winner[0], 1e-6))
    )
    cell_pass = len(equivalent) <= max_equivalent_cells
    extent_pass = extent <= max_equivalent_extent_m
    margin_pass = runner_margin >= min_runner_margin
    accepted = cell_pass and extent_pass and margin_pass
    return {
        "accepted": accepted,
        "reason": (
            "unique_road_translation_posterior"
            if accepted
            else "road_translation_posterior_unobservable"
        ),
        "winner": {
            "translation_xy_m": [winner[1], winner[2]],
            "mean_squared_band_violation_m2": winner[0],
            "road_median_m": winner[3],
            "road_p95_m": winner[4],
        },
        "runner": (
            None
            if runner is None
            else {
                "translation_xy_m": [runner[1], runner[2]],
                "mean_squared_band_violation_m2": runner[0],
            }
        ),
        "runner_margin": runner_margin,
        "equivalent_cell_count": len(equivalent),
        "equivalent_extent_m": extent,
        "grid_cell_count": len(rows),
        "gates": {
            "max_equivalent_cells": max_equivalent_cells,
            "max_equivalent_extent_m": max_equivalent_extent_m,
            "min_runner_margin": min_runner_margin,
            "equivalent_cells_pass": cell_pass,
            "equivalent_extent_pass": extent_pass,
            "runner_margin_pass": margin_pass,
        },
    }


def _read_route_xy(
    trajectory: Path, *, start: int, end: int, epsg: int
) -> np.ndarray:
    with trajectory.open(newline="", encoding="utf-8-sig") as fh:
        rows = [
            row
            for row in csv.DictReader(fh)
            if start <= int(row["epoch"]) < end
        ]
    if len(rows) != end - start:
        raise ValueError("trajectory does not fully cover the requested segment")
    positions = np.asarray(
        [
            [float(row["ecef_x"]), float(row["ecef_y"]), float(row["ecef_z"])]
            for row in rows
        ]
    )
    transformer = Transformer.from_crs("EPSG:4978", f"EPSG:{epsg}", always_xy=True)
    x, y, _height = transformer.transform(
        positions[:, 0], positions[:, 1], positions[:, 2]
    )
    return np.column_stack([x, y])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("osm_cache", type=Path)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--radius-m", type=float, default=5.0)
    parser.add_argument("--step-m", type=float, default=0.2)
    parser.add_argument("--road-lower-m", type=float, required=True)
    parser.add_argument("--road-upper-m", type=float, required=True)
    parser.add_argument("--max-equivalent-cells", type=int, default=25)
    parser.add_argument("--max-equivalent-extent-m", type=float, default=1.0)
    parser.add_argument("--min-runner-margin", type=float, default=0.2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cache_bytes = args.osm_cache.read_bytes()
    cache = json.loads(cache_bytes)
    epsg = int(cache["epsg"])
    road = STRtree([LineString(row) for row in cache["projected_road_lines"]])
    route_xy = _read_route_xy(
        args.trajectory, start=args.start, end=args.end, epsg=epsg
    )
    result = evaluate_road_translation_observability(
        route_xy,
        road,
        radius_m=args.radius_m,
        step_m=args.step_m,
        lower_m=args.road_lower_m,
        upper_m=args.road_upper_m,
        max_equivalent_cells=args.max_equivalent_cells,
        max_equivalent_extent_m=args.max_equivalent_extent_m,
        min_runner_margin=args.min_runner_margin,
    )
    result.update(
        {
            "schema": "wp70_road_translation_observability_v1",
            "production_input_truth": False,
            "truth_usage": "none",
            "production_promoted": False,
            "segment": [args.start, args.end],
            "grid": {"radius_m": args.radius_m, "step_m": args.step_m},
            "road_band_m": [args.road_lower_m, args.road_upper_m],
            "epsg": epsg,
            "input_sha256": {
                "trajectory": hashlib.sha256(args.trajectory.read_bytes()).hexdigest(),
                "osm_cache": hashlib.sha256(cache_bytes).hexdigest(),
            },
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
