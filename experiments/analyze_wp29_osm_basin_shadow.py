#!/usr/bin/env python3
"""Shadow-rank ambiguity basins with a truth-free OSM road prior."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from pyproj import Transformer
from shapely.geometry import Point

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from analyze_phase67_osm_road_centerline_feasibility import (  # noqa: E402
    _ecef_to_llh,
    _road_union_from_osm,
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def _reference_by_tow(path: Path) -> dict[float, np.ndarray]:
    result: dict[float, np.ndarray] = {}
    for row in _read_csv(path):
        tow = round(float(row["GPS TOW (s)"]), 3)
        result[tow] = np.array(
            [
                float(row["ECEF X (m)"]),
                float(row["ECEF Y (m)"]),
                float(row["ECEF Z (m)"]),
            ],
            dtype=np.float64,
        )
    return result


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - float(np.max(values))
    weights = np.exp(shifted)
    return weights / float(np.sum(weights))


def _rank_epoch(
    rows: list[dict[str, Any]],
    *,
    sigma_m: float,
    trigger_distance_m: float,
    corridor_half_width_m: float,
) -> tuple[int, float, bool]:
    base_scores = np.asarray([float(row["log_weight"]) for row in rows])
    distances = np.asarray([float(row["road_distance_m"]) for row in rows])
    base_index = int(np.argmax(base_scores))
    triggered = bool(distances[base_index] >= trigger_distance_m)
    scores = base_scores.copy()
    if triggered:
        outside = np.maximum(0.0, distances - corridor_half_width_m)
        scores -= 0.5 * np.square(outside / sigma_m)
    probabilities = _softmax(scores)
    selected = int(np.argmax(scores))
    return selected, float(probabilities[selected]), triggered


def analyze(
    basin_rows: list[dict[str, str]],
    reference: dict[float, np.ndarray],
    *,
    bbox_margin_deg: float,
    epsg: int,
    sigma_m: float,
    trigger_distance_m: float,
    corridor_half_width_m: float,
    gamma_threshold: float,
    fix_streak: int,
    temporal_road_sigma_m: float = 0.2,
    temporal_emission_weight: float = 0.01,
    basin_road_output: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    llh = [
        _ecef_to_llh(
            np.asarray(
                [float(row["ecef_x"]), float(row["ecef_y"]), float(row["ecef_z"])],
                dtype=np.float64,
            )
        )
        for row in basin_rows
    ]
    north = max(item[0] for item in llh) + bbox_margin_deg
    south = min(item[0] for item in llh) - bbox_margin_deg
    east = max(item[1] for item in llh) + bbox_margin_deg
    west = min(item[1] for item in llh) - bbox_margin_deg
    road_union, _transformer, n_road_geometries = _road_union_from_osm(
        north=north,
        south=south,
        east=east,
        west=west,
        epsg=epsg,
    )
    ecef_to_map = Transformer.from_crs("EPSG:4978", f"EPSG:{epsg}", always_xy=True)

    by_epoch: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in basin_rows:
        x, y, _z = ecef_to_map.transform(
            float(row["ecef_x"]), float(row["ecef_y"]), float(row["ecef_z"])
        )
        prepared = dict(row)
        prepared["road_distance_m"] = float(Point(x, y).distance(road_union))
        by_epoch[int(row["epoch"])].append(prepared)
        if basin_road_output is not None:
            basin_road_output.append(
                {
                    "epoch": int(row["epoch"]),
                    "basin_id": str(row["basin_id"]),
                    "road_distance_m": prepared["road_distance_m"],
                }
            )

    epoch_rows: list[dict[str, Any]] = []
    ordered_epochs = sorted(by_epoch)
    temporal_scores: list[np.ndarray] = []
    temporal_backpointers: list[np.ndarray] = []
    for epoch_index, epoch in enumerate(ordered_epochs):
        rows = by_epoch[epoch]
        emission = temporal_emission_weight * np.asarray(
            [float(row["log_weight"]) for row in rows], dtype=np.float64
        )
        if epoch_index == 0:
            temporal_scores.append(emission)
            temporal_backpointers.append(np.full(len(rows), -1, dtype=np.int64))
            continue
        previous_rows = by_epoch[ordered_epochs[epoch_index - 1]]
        previous_distance = np.asarray(
            [float(row["road_distance_m"]) for row in previous_rows]
        )
        current_distance = np.asarray(
            [float(row["road_distance_m"]) for row in rows]
        )
        delta = current_distance.reshape(1, -1) - previous_distance.reshape(-1, 1)
        transition = -np.log1p(np.square(delta / temporal_road_sigma_m))
        total = temporal_scores[-1].reshape(-1, 1) + transition
        backpointer = np.argmax(total, axis=0)
        temporal_scores.append(emission + total[backpointer, np.arange(len(rows))])
        temporal_backpointers.append(backpointer.astype(np.int64))
    temporal_path = [0] * len(ordered_epochs)
    temporal_path[-1] = int(np.argmax(temporal_scores[-1]))
    for epoch_index in range(len(ordered_epochs) - 1, 0, -1):
        temporal_path[epoch_index - 1] = int(
            temporal_backpointers[epoch_index][temporal_path[epoch_index]]
        )
    temporal_index_by_epoch = dict(zip(ordered_epochs, temporal_path))
    streak = 0
    for epoch in sorted(by_epoch):
        rows = by_epoch[epoch]
        selected, gamma, triggered = _rank_epoch(
            rows,
            sigma_m=sigma_m,
            trigger_distance_m=trigger_distance_m,
            corridor_half_width_m=corridor_half_width_m,
        )
        base = max(range(len(rows)), key=lambda idx: float(rows[idx]["log_weight"]))
        nearest = min(range(len(rows)), key=lambda idx: float(rows[idx]["road_distance_m"]))
        temporal = temporal_index_by_epoch[epoch]
        truth = reference[round(float(rows[0]["tow"]), 3)]

        def error(index: int) -> float:
            return float(
                np.linalg.norm(
                    np.asarray(
                        [
                            float(rows[index]["ecef_x"]),
                            float(rows[index]["ecef_y"]),
                            float(rows[index]["ecef_z"]),
                        ]
                    )
                    - truth
                )
            )

        errors = [error(index) for index in range(len(rows))]
        oracle = int(np.argmin(errors))
        streak = streak + 1 if gamma >= gamma_threshold else 0
        declared_fix = streak >= fix_streak
        epoch_rows.append(
            {
                "epoch": epoch,
                "tow": float(rows[0]["tow"]),
                "n_basins": len(rows),
                "triggered": int(triggered),
                "base_error_m": errors[base],
                "road_error_m": errors[selected],
                "nearest_road_error_m": errors[nearest],
                "temporal_road_error_m": errors[temporal],
                "oracle_error_m": min(errors),
                "oracle_road_distance_m": float(rows[oracle]["road_distance_m"]),
                "base_road_distance_m": float(rows[base]["road_distance_m"]),
                "selected_road_distance_m": float(rows[selected]["road_distance_m"]),
                "road_gamma": gamma,
                "road_fix_streak": streak,
                "road_declared_fix": int(declared_fix),
                "road_false_fix": int(declared_fix and errors[selected] >= 0.5),
            }
        )

    n_epochs = len(epoch_rows)
    fixed = [row for row in epoch_rows if row["road_declared_fix"]]
    summary = {
        "n_epochs": n_epochs,
        "bbox": {"north": north, "south": south, "east": east, "west": west},
        "epsg": epsg,
        "n_road_geometries": n_road_geometries,
        "sigma_m": sigma_m,
        "trigger_distance_m": trigger_distance_m,
        "corridor_half_width_m": corridor_half_width_m,
        "gamma_threshold": gamma_threshold,
        "fix_streak": fix_streak,
        "triggered_epochs": sum(int(row["triggered"]) for row in epoch_rows),
        "base_sub50cm_epochs": sum(row["base_error_m"] < 0.5 for row in epoch_rows),
        "road_sub50cm_epochs": sum(row["road_error_m"] < 0.5 for row in epoch_rows),
        "nearest_road_sub50cm_epochs": sum(
            row["nearest_road_error_m"] < 0.5 for row in epoch_rows
        ),
        "temporal_road_sub50cm_epochs": sum(
            row["temporal_road_error_m"] < 0.5 for row in epoch_rows
        ),
        "temporal_road_sigma_m": temporal_road_sigma_m,
        "temporal_emission_weight": temporal_emission_weight,
        "oracle_sub50cm_epochs": sum(row["oracle_error_m"] < 0.5 for row in epoch_rows),
        "declared_fix_epochs": len(fixed),
        "fix_rate_pct": 100.0 * len(fixed) / max(n_epochs, 1),
        "false_fix_epochs": sum(int(row["road_false_fix"]) for row in fixed),
        "false_fix_pct": 100.0
        * sum(int(row["road_false_fix"]) for row in fixed)
        / max(len(fixed), 1),
    }
    return summary, epoch_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("basin_trace", type=Path)
    parser.add_argument("reference_csv", type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int)
    parser.add_argument("--bbox-margin-deg", type=float, default=0.002)
    parser.add_argument("--epsg", type=int, default=32654)
    parser.add_argument("--sigma-m", type=float, default=1.0)
    parser.add_argument("--trigger-distance-m", type=float, default=2.5)
    parser.add_argument("--corridor-half-width-m", type=float, default=0.0)
    parser.add_argument("--gamma-threshold", type=float, default=0.99)
    parser.add_argument("--fix-streak", type=int, default=3)
    parser.add_argument("--temporal-road-sigma-m", type=float, default=0.2)
    parser.add_argument("--temporal-emission-weight", type=float, default=0.01)
    parser.add_argument("--out-summary", type=Path, required=True)
    parser.add_argument("--out-epochs", type=Path, required=True)
    parser.add_argument("--out-basin-road", type=Path)
    args = parser.parse_args()
    if args.sigma_m <= 0.0:
        parser.error("--sigma-m must be positive")
    if args.corridor_half_width_m < 0.0:
        parser.error("--corridor-half-width-m must be non-negative")
    if args.temporal_road_sigma_m <= 0.0:
        parser.error("--temporal-road-sigma-m must be positive")

    basin_rows = [
        row
        for row in _read_csv(args.basin_trace)
        if int(row["epoch"]) >= int(args.start)
        and (args.end is None or int(row["epoch"]) < int(args.end))
    ]
    if not basin_rows:
        parser.error("requested epoch range contains no basin rows")
    basin_road_rows: list[dict[str, Any]] = []
    summary, rows = analyze(
        basin_rows,
        _reference_by_tow(args.reference_csv),
        bbox_margin_deg=float(args.bbox_margin_deg),
        epsg=int(args.epsg),
        sigma_m=float(args.sigma_m),
        trigger_distance_m=float(args.trigger_distance_m),
        corridor_half_width_m=float(args.corridor_half_width_m),
        gamma_threshold=float(args.gamma_threshold),
        fix_streak=int(args.fix_streak),
        temporal_road_sigma_m=float(args.temporal_road_sigma_m),
        temporal_emission_weight=float(args.temporal_emission_weight),
        basin_road_output=basin_road_rows,
    )
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    args.out_epochs.parent.mkdir(parents=True, exist_ok=True)
    with args.out_epochs.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if args.out_basin_road is not None:
        args.out_basin_road.parent.mkdir(parents=True, exist_ok=True)
        with args.out_basin_road.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(basin_road_rows[0]))
            writer.writeheader()
            writer.writerows(basin_road_rows)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
