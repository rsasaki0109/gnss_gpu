#!/usr/bin/env python3
"""Select an affine mode by three-basis agreement, road veto, and carrier fit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
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
from select_wp54_cppr_rank import select as cppr_rank  # noqa: E402


def _dense_ranks(values: list[float]) -> list[int]:
    ordered = {value: rank for rank, value in enumerate(sorted(set(values)), 1)}
    return [ordered[value] for value in values]


def affine_baseline_route(
    route_ecef: np.ndarray, scales: np.ndarray, model: dict[str, Any]
) -> np.ndarray:
    """Add the fixed part of an affine correction before ranking profiles."""

    mode = model.get("mode")
    if mode in {"constant", "right_boundary_affine_zero"}:
        return np.asarray(route_ecef, dtype=np.float64)
    if mode != "right_boundary_affine_fixed":
        raise ValueError("affine selector source model is unsupported")
    boundary_offset = np.asarray(model.get("boundary_offset_ecef_m"), dtype=np.float64)
    if boundary_offset.shape != (3,) or not np.all(np.isfinite(boundary_offset)):
        raise ValueError("fixed affine selector boundary offset is invalid")
    return (
        np.asarray(route_ecef, dtype=np.float64)
        + (1.0 - np.asarray(scales, dtype=np.float64))[:, None] * boundary_offset
    )


def profile_scales(
    start: int, end: int, model: dict[str, Any]
) -> np.ndarray:
    """Return per-epoch scale for constant or right-boundary affine profiles."""

    if model.get("mode") == "constant":
        return np.ones(end - start, dtype=np.float64)
    boundary = int(model["boundary_epoch"])
    return np.asarray(
        [(boundary - epoch) / (boundary - start) for epoch in range(start, end)],
        dtype=np.float64,
    )


def _eligible_modes(
    source: dict[str, Any],
    *,
    dedup_m: float,
    min_checked_pairs: int,
    max_bad_pair_fraction: float,
    max_block_spread_m: float,
) -> list[dict[str, Any]]:
    ranked = cppr_rank(source)
    candidates = [
        row
        for row in ranked["candidates"]
        if row["checked_pairs"] >= min_checked_pairs
        and row["bad_pair_fraction"] <= max_bad_pair_fraction
        and row["block_spread_m"] <= max_block_spread_m
    ]
    candidates.sort(key=lambda row: (row["rank_sum"], row["candidate_id"]))
    modes = []
    for candidate in candidates:
        if all(
            np.linalg.norm(
                np.asarray(candidate["offset_ecef_m"])
                - np.asarray(prior["offset_ecef_m"])
            )
            > dedup_m
            for prior in modes
        ):
            modes.append(candidate)
    return modes


def select_clusters(
    sources: list[dict[str, Any]],
    route_ecef: np.ndarray,
    road: Any,
    transformer: Transformer,
    *,
    scales: np.ndarray,
    road_lower_m: float,
    road_upper_m: float,
    within_basis_dedup_m: float = 0.08,
    max_cluster_diameter_m: float = 0.12,
    max_family_rank_fraction: float = 0.2,
    min_runner_margin: float = 0.2,
    min_checked_pairs: int = 40,
    max_bad_pair_fraction: float = 0.05,
    max_block_spread_m: float = 0.5,
) -> dict[str, Any]:
    if len(sources) != 3:
        raise ValueError("affine selector requires exactly three carrier bases")
    if any(bool(source.get("production_input_truth", True)) for source in sources):
        raise ValueError("affine selector source is not production-safe")
    basis_descriptors = [
        (
            tuple(str(value) for value in source.get("carrier_families", [])),
            int(source.get("carrier_reference_rank", -1)),
        )
        for source in sources
    ]
    if (
        all(descriptor[0] for descriptor in basis_descriptors)
        and len(set(basis_descriptors)) != 3
    ):
        raise ValueError("affine selector requires three distinct carrier bases")
    segments = {tuple(int(v) for v in source.get("segment", [])) for source in sources}
    models = [source.get("offset_model", {}) for source in sources]
    if len(segments) != 1 or len(next(iter(segments))) != 2:
        raise ValueError("affine selector sources do not share one segment")
    modes = {model.get("mode") for model in models}
    if len(modes) != 1 or next(iter(modes)) not in {
        "constant",
        "right_boundary_affine_zero",
        "right_boundary_affine_fixed",
    }:
        raise ValueError("affine selector source model is unsupported")
    if next(iter(modes)) != "constant":
        boundaries = {int(model["boundary_epoch"]) for model in models}
        if len(boundaries) != 1:
            raise ValueError("affine selector sources do not share one boundary")
    if next(iter(modes)) == "right_boundary_affine_fixed":
        fixed_offsets = {
            tuple(float(value) for value in model.get("boundary_offset_ecef_m", []))
            for model in models
        }
        if len(fixed_offsets) != 1 or len(next(iter(fixed_offsets))) != 3:
            raise ValueError("affine selector sources do not share one fixed boundary")

    mode_sets = [
        _eligible_modes(
            source,
            dedup_m=within_basis_dedup_m,
            min_checked_pairs=min_checked_pairs,
            max_bad_pair_fraction=max_bad_pair_fraction,
            max_block_spread_m=max_block_spread_m,
        )
        for source in sources
    ]
    clusters = []
    for members in itertools.product(*mode_sets):
        distances = [
            float(
                np.linalg.norm(
                    np.asarray(left["offset_ecef_m"])
                    - np.asarray(right["offset_ecef_m"])
                )
            )
            for left, right in itertools.combinations(members, 2)
        ]
        diameter = max(distances)
        if diameter > max_cluster_diameter_m:
            continue
        profile = np.median(
            np.asarray([member["offset_ecef_m"] for member in members]), axis=0
        )
        block_profiles = np.median(
            np.asarray([member["block_offsets_ecef_m"] for member in members]),
            axis=0,
        )
        shifted = route_ecef + scales[:, None] * profile
        x, y, _height = transformer.transform(
            shifted[:, 0], shifted[:, 1], shifted[:, 2]
        )
        road_distances = _road_distances(road, np.asarray(x), np.asarray(y))
        outside = np.maximum(road_lower_m - road_distances, 0.0) + np.maximum(
            road_distances - road_upper_m, 0.0
        )
        clusters.append(
            {
                "member_candidate_ids": [member["candidate_id"] for member in members],
                "member_cppr_rank_sums": [member["rank_sum"] for member in members],
                "cppr_rank_sum": int(sum(member["rank_sum"] for member in members)),
                "cluster_diameter_m": diameter,
                "offset_ecef_m": profile.tolist(),
                "block_offsets_ecef_m": block_profiles.tolist(),
                "road_band_violation_m2": float(np.mean(np.square(outside))),
                "road_median_m": float(np.median(road_distances)),
                "road_p95_m": float(np.percentile(road_distances, 95.0)),
                "carrier_rms_sum_cycles": float(
                    sum(member["carrier_rms_cycles"] for member in members)
                ),
            }
        )
    if len(clusters) < 2:
        return {
            "accepted": False,
            "reason": "fewer_than_two_three_basis_clusters",
            "selected_profile": None,
            "cluster_count": len(clusters),
            "basis_mode_counts": [len(rows) for rows in mode_sets],
            "clusters": clusters,
        }

    road_ranks = _dense_ranks([row["road_band_violation_m2"] for row in clusters])
    carrier_ranks = _dense_ranks([row["carrier_rms_sum_cycles"] for row in clusters])
    for row, road_rank, carrier_rank in zip(clusters, road_ranks, carrier_ranks):
        row["family_ranks"] = {"road_band": road_rank, "carrier_rms": carrier_rank}
        row["rank_sum"] = int(road_rank + carrier_rank)
    clusters.sort(
        key=lambda row: (
            row["rank_sum"],
            row["cppr_rank_sum"],
            row["member_candidate_ids"],
        )
    )
    winner, runner = clusters[:2]
    runner_margin = float(
        (runner["rank_sum"] - winner["rank_sum"]) / max(winner["rank_sum"], 1)
    )
    family_limit = int(math.ceil(len(clusters) * max_family_rank_fraction))
    family_pass = max(winner["family_ranks"].values()) <= family_limit
    margin_pass = runner_margin >= min_runner_margin
    accepted = family_pass and margin_pass
    return {
        "accepted": accepted,
        "reason": (
            "unique_affine_multibasis_road_carrier_cluster"
            if accepted
            else "affine_multibasis_cluster_gate_failed"
        ),
        "selected_profile": winner if accepted else None,
        "winner": winner,
        "runner": runner,
        "runner_margin": runner_margin,
        "family_rank_limit": family_limit,
        "family_rank_pass": family_pass,
        "runner_margin_pass": margin_pass,
        "cluster_count": len(clusters),
        "basis_mode_counts": [len(rows) for rows in mode_sets],
        "clusters": clusters,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", type=Path, nargs=3)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--osm-cache", type=Path, required=True)
    parser.add_argument("--road-lower-m", type=float, required=True)
    parser.add_argument("--road-upper-m", type=float, required=True)
    parser.add_argument("--min-checked-pairs", type=int, default=40)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.min_checked_pairs < 24:
        raise ValueError("affine selector requires at least 24 checked CP/PR pairs")

    source_bytes = [path.read_bytes() for path in args.sources]
    sources = [json.loads(value) for value in source_bytes]
    start, end = (int(value) for value in sources[0]["segment"])
    with args.trajectory.open(newline="", encoding="utf-8-sig") as fh:
        rows = [row for row in csv.DictReader(fh) if start <= int(row["epoch"]) < end]
    if len(rows) != end - start:
        raise ValueError("trajectory does not fully cover affine selector segment")
    route_ecef = np.asarray(
        [[float(row[axis]) for axis in ("ecef_x", "ecef_y", "ecef_z")] for row in rows]
    )
    cache_bytes = args.osm_cache.read_bytes()
    cache = json.loads(cache_bytes)
    epsg = int(cache["epsg"])
    road = STRtree([LineString(row) for row in cache["projected_road_lines"]])
    transformer = Transformer.from_crs("EPSG:4978", f"EPSG:{epsg}", always_xy=True)
    scales = profile_scales(start, end, sources[0]["offset_model"])
    route_ecef = affine_baseline_route(route_ecef, scales, sources[0]["offset_model"])
    result = select_clusters(
        sources,
        route_ecef,
        road,
        transformer,
        scales=scales,
        road_lower_m=args.road_lower_m,
        road_upper_m=args.road_upper_m,
        min_checked_pairs=args.min_checked_pairs,
    )
    result.update(
        {
            "schema": "wp76_affine_multibasis_road_carrier_v1",
            "production_input_truth": False,
            "truth_usage": "none",
            "production_promoted": False,
            "segment": [start, end],
            "offset_model": sources[0]["offset_model"],
            "selector_gates": {
                "min_checked_pairs": args.min_checked_pairs,
                "max_bad_pair_fraction": 0.05,
                "max_block_spread_m": 0.5,
                "within_basis_dedup_m": 0.08,
                "max_cluster_diameter_m": 0.12,
                "max_family_rank_fraction": 0.2,
                "min_runner_margin": 0.2,
            },
            "input_sha256": {
                "sources": [
                    hashlib.sha256(value).hexdigest() for value in source_bytes
                ],
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
