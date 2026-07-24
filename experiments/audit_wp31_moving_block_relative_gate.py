#!/usr/bin/env python3
"""Apply the frozen truth-free relative gate to a moving-block artifact."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from pyproj import Transformer
from shapely import points
from shapely.geometry import LineString
from shapely.strtree import STRtree


def read_route(path: Path, start: int, end: int) -> dict[int, np.ndarray]:
    with path.open(newline="", encoding="utf-8-sig") as fh:
        return {
            int(row["epoch"]): np.asarray([float(row["ecef_x"]), float(row["ecef_y"]), float(row["ecef_z"])])
            for row in csv.DictReader(fh) if start <= int(row["epoch"]) < end
        }


def passing_relative_gate(
    rows: list[dict[str, Any]],
    *,
    baseline_ddpr_rms_m: float,
    max_carrier_rms_cycles: float = 0.20,
    max_ddpr_ratio: float = 0.65,
    max_road_p95_m: float = 1.0,
    max_block_spread_m: float = 0.10,
) -> list[int]:
    return [
        int(row["seed_id"]) for row in rows
        if float(row["carrier_rms_cycles"]) <= max_carrier_rms_cycles
        and float(row["ddpr_rms_m"]) / baseline_ddpr_rms_m <= max_ddpr_ratio
        and float(row["road_p95_m"]) <= max_road_p95_m
        and float(row["block_spread_m"]) <= max_block_spread_m
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path); parser.add_argument("trajectory", type=Path)
    parser.add_argument("osm_cache", type=Path); parser.add_argument("--scope", choices=("development", "holdout"), required=True)
    parser.add_argument("--output", type=Path, required=True); args = parser.parse_args()
    source = json.loads(args.artifact.read_text(encoding="utf-8")); start, end = source["segment"]
    route = read_route(args.trajectory, int(start), int(end))
    cache = json.loads(args.osm_cache.read_text(encoding="utf-8")); epsg = int(cache["epsg"])
    road = STRtree([LineString(row) for row in cache["projected_road_lines"]])
    transformer = Transformer.from_crs("EPSG:4978", f"EPSG:{epsg}", always_xy=True)
    rows = []
    for hypothesis in source["hypotheses"]:
        offset = np.asarray(hypothesis["offset_ecef_m"])
        positions = np.asarray([route[epoch] + offset for epoch in sorted(route)])
        x, y, _z = transformer.transform(positions[:, 0], positions[:, 1], positions[:, 2])
        _indices, distances = road.query_nearest(points(x, y), return_distance=True, all_matches=False)
        rows.append({
            "seed_id": int(hypothesis["seed_id"]),
            "carrier_rms_cycles": float(hypothesis["carrier_rms_cycles"]),
            "ddpr_rms_m": float(hypothesis["ddpr_rms_m"]),
            "block_spread_m": float(hypothesis["block_spread_m"]),
            "road_p95_m": float(np.percentile(distances, 95.0)),
            "audit_median_error_m": float(hypothesis["audit_median_error_m"]),
            "audit_sub50cm_epochs": int(hypothesis["audit_sub50cm_epochs"]),
        })
    baseline = next((row["ddpr_rms_m"] for row in rows if row["seed_id"] == 0), None)
    passing = [] if baseline is None else passing_relative_gate(rows, baseline_ddpr_rms_m=baseline)
    by_id = {row["seed_id"]: row for row in rows}
    result = {
        "schema": "wp31_moving_block_relative_gate_audit_v1",
        "scope": args.scope, "production_input_truth": False,
        "truth_usage": "post_gate_audit_only", "segment": source["segment"],
        "frozen_gate": {"max_carrier_rms_cycles": 0.20, "max_ddpr_ratio": 0.65, "max_road_p95_m": 1.0, "max_block_spread_m": 0.10},
        "baseline_ddpr_rms_m": baseline, "passing_seed_ids": passing,
        "unique_pass": len(passing) == 1,
        "passing_post_gate_audit": [
            {"seed_id": seed, "audit_median_error_m": by_id[seed]["audit_median_error_m"], "audit_sub50cm_epochs": by_id[seed]["audit_sub50cm_epochs"]}
            for seed in passing
        ],
        "hypotheses": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("scope", "segment", "baseline_ddpr_rms_m", "passing_seed_ids", "unique_pass", "passing_post_gate_audit")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
