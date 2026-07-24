#!/usr/bin/env python3
"""Select one affine carrier basis with independent road, carrier, and CP/PR ranks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from pyproj import Transformer
from shapely.geometry import LineString
from shapely.strtree import STRtree

from build_wp31_osm_particle_route_bridge import _road_distances
from select_wp76_affine_multibasis_road_carrier import (
    _dense_ranks,
    _eligible_modes,
    affine_baseline_route,
    profile_scales,
)


def select_single_basis(
    source: dict[str, Any],
    route_ecef: np.ndarray,
    road: Any,
    transformer: Transformer,
    *,
    scales: np.ndarray,
    road_lower_m: float,
    road_upper_m: float,
    min_checked_pairs: int = 40,
    max_bad_pair_fraction: float = 0.05,
    max_block_spread_m: float = 0.5,
    within_basis_dedup_m: float = 0.08,
    max_family_rank_fraction: float = 0.2,
    min_runner_margin: float = 0.2,
) -> dict[str, Any]:
    if bool(source.get("production_input_truth", True)):
        raise ValueError("single-basis selector source is not production-safe")
    modes = _eligible_modes(
        source,
        dedup_m=within_basis_dedup_m,
        min_checked_pairs=min_checked_pairs,
        max_bad_pair_fraction=max_bad_pair_fraction,
        max_block_spread_m=max_block_spread_m,
    )
    ranked = []
    for mode in modes:
        profile = np.asarray(mode["offset_ecef_m"], dtype=np.float64)
        shifted = route_ecef + scales[:, None] * profile
        x, y, _height = transformer.transform(
            shifted[:, 0], shifted[:, 1], shifted[:, 2]
        )
        distances = _road_distances(road, np.asarray(x), np.asarray(y))
        outside = np.maximum(road_lower_m - distances, 0.0) + np.maximum(
            distances - road_upper_m, 0.0
        )
        ranked.append(
            {
                "candidate_id": mode["candidate_id"],
                "offset_ecef_m": mode["offset_ecef_m"],
                "block_offsets_ecef_m": mode["block_offsets_ecef_m"],
                "checked_pairs": mode["checked_pairs"],
                "bad_pair_fraction": mode["bad_pair_fraction"],
                "block_spread_m": mode["block_spread_m"],
                "cppr_rank_sum": mode["rank_sum"],
                "carrier_rms_cycles": mode["carrier_rms_cycles"],
                "road_band_violation_m2": float(np.mean(np.square(outside))),
                "road_median_m": float(np.median(distances)),
                "road_p95_m": float(np.percentile(distances, 95.0)),
            }
        )
    if len(ranked) < 2:
        return {
            "accepted": False,
            "reason": "fewer_than_two_absolute_single_basis_modes",
            "selected_profile": None,
            "mode_count": len(ranked),
            "modes": ranked,
        }
    family_values = {
        "road_band": [row["road_band_violation_m2"] for row in ranked],
        "carrier_rms": [row["carrier_rms_cycles"] for row in ranked],
        "cppr": [row["cppr_rank_sum"] for row in ranked],
    }
    family_ranks = {
        name: _dense_ranks(values) for name, values in family_values.items()
    }
    for index, row in enumerate(ranked):
        row["family_ranks"] = {
            name: ranks[index] for name, ranks in family_ranks.items()
        }
        row["rank_sum"] = int(sum(row["family_ranks"].values()))
    ranked.sort(key=lambda row: (row["rank_sum"], row["candidate_id"]))
    winner, runner = ranked[:2]
    runner_margin = float(
        (runner["rank_sum"] - winner["rank_sum"]) / max(winner["rank_sum"], 1)
    )
    family_limit = int(math.ceil(len(ranked) * max_family_rank_fraction))
    family_pass = max(winner["family_ranks"].values()) <= family_limit
    margin_pass = runner_margin >= min_runner_margin
    accepted = family_pass and margin_pass
    return {
        "accepted": accepted,
        "reason": (
            "unique_singlebasis_road_carrier_cppr_mode"
            if accepted
            else "singlebasis_family_or_margin_gate_failed"
        ),
        "selected_profile": winner if accepted else None,
        "winner": winner,
        "runner": runner,
        "runner_margin": runner_margin,
        "family_rank_limit": family_limit,
        "family_rank_pass": family_pass,
        "runner_margin_pass": margin_pass,
        "mode_count": len(ranked),
        "modes": ranked,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--osm-cache", type=Path, required=True)
    parser.add_argument("--road-lower-m", type=float, required=True)
    parser.add_argument("--road-upper-m", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source_bytes = args.source.read_bytes()
    source = json.loads(source_bytes)
    start, end = (int(value) for value in source["segment"])
    with args.trajectory.open(newline="", encoding="utf-8-sig") as fh:
        rows = [row for row in csv.DictReader(fh) if start <= int(row["epoch"]) < end]
    if len(rows) != end - start:
        raise ValueError("trajectory does not fully cover single-basis segment")
    route = np.asarray(
        [[float(row[axis]) for axis in ("ecef_x", "ecef_y", "ecef_z")] for row in rows]
    )
    model = source["offset_model"]
    scales = profile_scales(start, end, model)
    route = affine_baseline_route(route, scales, model)
    cache_bytes = args.osm_cache.read_bytes()
    cache = json.loads(cache_bytes)
    road = STRtree([LineString(row) for row in cache["projected_road_lines"]])
    transformer = Transformer.from_crs(
        "EPSG:4978", f"EPSG:{int(cache['epsg'])}", always_xy=True
    )
    result = select_single_basis(
        source,
        route,
        road,
        transformer,
        scales=scales,
        road_lower_m=args.road_lower_m,
        road_upper_m=args.road_upper_m,
    )
    result.update(
        {
            "schema": "wp87_singlebasis_road_carrier_v1",
            "production_input_truth": False,
            "truth_usage": "none",
            "production_promoted": False,
            "segment": [start, end],
            "offset_model": model,
            "carrier_basis": {
                "families": source.get("carrier_families", []),
                "reference_rank": source.get("carrier_reference_rank"),
            },
            "selector_gates": {
                "min_checked_pairs": 40,
                "max_bad_pair_fraction": 0.05,
                "max_block_spread_m": 0.5,
                "within_basis_dedup_m": 0.08,
                "max_family_rank_fraction": 0.2,
                "min_runner_margin": 0.2,
            },
            "input_sha256": {
                "source": hashlib.sha256(source_bytes).hexdigest(),
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
