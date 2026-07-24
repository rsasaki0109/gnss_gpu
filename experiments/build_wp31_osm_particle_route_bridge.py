#!/usr/bin/env python3
"""Build a truth-free OSM-constrained particle route across one long outage."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from pyproj import Transformer
from shapely import distance, points
from shapely.geometry import LineString
from shapely.strtree import STRtree

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "python"))
sys.path.insert(0, str(_ROOT / "experiments"))

from analyze_phase67_osm_road_centerline_feasibility import (  # noqa: E402
    _ecef_to_llh,
    _road_union_from_osm,
)
from build_wp29_imu_heading_route_seed_trace import integrate_gyro_intervals  # noqa: E402
from gnss_gpu.io.ppc import PPCDatasetLoader  # noqa: E402
from run_wp29_tdcp_anchor_smoother import (  # noqa: E402
    _load_fusion_static_override,
    _load_static_position_override,
)


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def _ecef(row: dict[str, str], prefix: str = "") -> np.ndarray:
    return np.asarray([float(row[f"{prefix}{axis}_m"]) for axis in ("x", "y", "z")])


def road_band_log_likelihood(
    distances_m: np.ndarray, *, lower_m: float, upper_m: float, sigma_m: float
) -> np.ndarray:
    outside = np.maximum(lower_m - distances_m, 0.0) + np.maximum(distances_m - upper_m, 0.0)
    return -0.5 * np.square(outside / sigma_m)


def systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / np.sum(weights)
    positions = (rng.random() + np.arange(len(weights))) / len(weights)
    return np.searchsorted(np.cumsum(weights), positions, side="right")


def _road_cache(
    path: Path,
    *,
    boundary_ecef: np.ndarray,
    epsg: int,
    margin_deg: float,
) -> tuple[STRtree, dict[str, Any]]:
    llh = [_ecef_to_llh(row) for row in boundary_ecef]
    bbox = {
        "north": max(row[0] for row in llh) + margin_deg,
        "south": min(row[0] for row in llh) - margin_deg,
        "east": max(row[1] for row in llh) + margin_deg,
        "west": min(row[1] for row in llh) - margin_deg,
    }
    source_hash = hashlib.sha256(np.asarray(boundary_ecef, dtype="<f8").tobytes()).hexdigest()
    if path.exists():
        cache = json.loads(path.read_text(encoding="utf-8"))
        if cache.get("boundary_ecef_sha256") != source_hash or int(cache.get("epsg", -1)) != epsg:
            raise RuntimeError("OSM route cache provenance mismatch")
        lines = [LineString(row) for row in cache["projected_road_lines"]]
        return STRtree(lines), cache
    road, _transformer, count = _road_union_from_osm(epsg=epsg, **bbox)
    parts = list(road.geoms) if hasattr(road, "geoms") else [road]
    lines = [list(map(list, geom.coords)) for geom in parts if hasattr(geom, "coords")]
    cache = {
        "schema": "wp31_osm_route_cache_v1",
        "epsg": epsg,
        "bbox": bbox,
        "bbox_margin_deg": margin_deg,
        "boundary_ecef_sha256": source_hash,
        "road_geometry_count": count,
        "projected_road_lines": lines,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2) + "\n", encoding="utf-8")
    return STRtree([LineString(row) for row in lines]), cache


def _road_distances(road_geometry: Any, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    query_points = points(x, y)
    if isinstance(road_geometry, STRtree):
        _indices, distances = road_geometry.query_nearest(
            query_points, return_distance=True, all_matches=False
        )
        return np.asarray(distances, dtype=np.float64)
    return np.asarray(distance(road_geometry, query_points), dtype=np.float64)


def particle_route(
    *,
    start_xy: np.ndarray,
    end_xy: np.ndarray,
    step_lengths_m: np.ndarray,
    gyro_increments_rad: np.ndarray,
    dt_s: np.ndarray,
    initial_heading_rad: float,
    road_geometry: Any,
    particles: int,
    random_seed: int,
    road_lower_m: float,
    road_upper_m: float,
    road_sigma_m: float,
    heading_sigma_deg: float = 25.0,
    gyro_bias_sigma_dps: float = 0.8,
    turn_noise_deg: float = 0.5,
    scale_lower: float = 0.8,
    scale_upper: float = 1.2,
    trusted_steps_xy: np.ndarray | None = None,
    trusted_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return one posterior lineage and truth-free diagnostics."""

    rng = np.random.default_rng(random_seed)
    n_steps = len(step_lengths_m)
    if n_steps < 2 or len(gyro_increments_rad) != n_steps or len(dt_s) != n_steps:
        raise ValueError("particle route step arrays are inconsistent")
    if trusted_steps_xy is None:
        trusted_steps_xy = np.zeros((n_steps, 2), dtype=np.float64)
    if trusted_mask is None:
        trusted_mask = np.zeros(n_steps, dtype=bool)
    trusted_steps_xy = np.asarray(trusted_steps_xy, dtype=np.float64)
    trusted_mask = np.asarray(trusted_mask, dtype=bool)
    if trusted_steps_xy.shape != (n_steps, 2) or trusted_mask.shape != (n_steps,):
        raise ValueError("trusted TDCP step inputs are inconsistent")
    x = np.full(particles, float(start_xy[0]))
    y = np.full(particles, float(start_xy[1]))
    heading = initial_heading_rad + rng.normal(0.0, np.deg2rad(heading_sigma_deg), particles)
    bias = rng.normal(0.0, np.deg2rad(gyro_bias_sigma_dps), particles)
    scale = rng.uniform(scale_lower, scale_upper, particles)
    logw = np.zeros(particles)
    history_x = np.empty((n_steps, particles), dtype=np.float32)
    history_y = np.empty((n_steps, particles), dtype=np.float32)
    parents = np.empty((n_steps, particles), dtype=np.int32)
    parent_map = np.arange(particles, dtype=np.int32)
    resamples = 0
    for step in range(n_steps):
        if trusted_mask[step]:
            vector = trusted_steps_xy[step]
            x += vector[0]
            y += vector[1]
            if np.linalg.norm(vector) > 0.05:
                heading[:] = math.atan2(vector[0], vector[1])
        else:
            heading += gyro_increments_rad[step] - bias * dt_s[step]
            heading += rng.normal(0.0, np.deg2rad(turn_noise_deg), particles)
            length = step_lengths_m[step] * scale
            x += length * np.sin(heading)
            y += length * np.cos(heading)
        road_distance = _road_distances(road_geometry, x, y)
        logw += road_band_log_likelihood(
            road_distance, lower_m=road_lower_m, upper_m=road_upper_m, sigma_m=road_sigma_m
        )
        remaining = float(np.sum(step_lengths_m[step + 1 :]) * scale_upper)
        target_distance = np.hypot(x - end_xy[0], y - end_xy[1])
        impossible = np.maximum(target_distance - remaining - 10.0, 0.0)
        logw -= 0.5 * np.square(impossible / 5.0)
        history_x[step] = x
        history_y[step] = y
        parents[step] = parent_map
        shifted = logw - np.max(logw)
        weights = np.exp(np.clip(shifted, -700.0, 0.0))
        weights /= np.sum(weights)
        ess = 1.0 / np.sum(np.square(weights))
        if step + 1 < n_steps and ess < 0.5 * particles:
            chosen = systematic_resample(weights, rng)
            x, y, heading, bias, scale = x[chosen], y[chosen], heading[chosen], bias[chosen], scale[chosen]
            parent_map = chosen.astype(np.int32)
            logw = np.zeros(particles)
            resamples += 1
        else:
            parent_map = np.arange(particles, dtype=np.int32)
    endpoint_distance = np.hypot(x - end_xy[0], y - end_xy[1])
    terminal_score = logw - 0.5 * np.square(endpoint_distance / 5.0)
    ranked = np.argsort(terminal_score)[::-1]
    best = int(ranked[0])
    runner = next(
        (
            int(candidate)
            for candidate in ranked[1:]
            if math.hypot(x[int(candidate)] - x[best], y[int(candidate)] - y[best]) >= 1.0
        ),
        None,
    )
    runner_score = float("-inf") if runner is None else float(terminal_score[runner])
    lineage = np.empty((n_steps, 2), dtype=np.float64)
    index = best
    for step in range(n_steps - 1, -1, -1):
        lineage[step] = [history_x[step, index], history_y[step, index]]
        index = int(parents[step, index])
    preclose_error = float(np.linalg.norm(lineage[-1] - end_xy))
    closure_lengths = np.where(trusted_mask, 0.0, step_lengths_m)
    if float(np.sum(closure_lengths)) <= 0.0:
        closure_lengths = step_lengths_m
    progress = np.r_[0.0, np.cumsum(closure_lengths)]
    progress /= progress[-1]
    route = np.vstack([start_xy, lineage])
    route += progress[:, None] * (end_xy - route[-1])
    closed_distances = _road_distances(road_geometry, route[:, 0], route[:, 1])
    route_length = float(np.sum(np.linalg.norm(np.diff(route, axis=0), axis=1)))
    raw_length = float(np.sum(step_lengths_m))
    return route, {
        "particle_count": particles,
        "random_seed": random_seed,
        "resample_count": resamples,
        "trusted_tdcp_steps": int(np.count_nonzero(trusted_mask)),
        "particle_propagated_steps": int(n_steps - np.count_nonzero(trusted_mask)),
        "terminal_score": float(terminal_score[best]),
        "terminal_distinct_runner_found": runner is not None,
        "terminal_runner_score": runner_score,
        "terminal_score_gap": float("inf") if runner is None else float(terminal_score[best] - terminal_score[runner]),
        "preclosure_endpoint_error_m": preclose_error,
        "postclosure_endpoint_error_m": float(np.linalg.norm(route[-1] - end_xy)),
        "road_distance_median_m": float(np.median(closed_distances)),
        "road_distance_p95_m": float(np.percentile(closed_distances, 95.0)),
        "road_distance_max_m": float(np.max(closed_distances)),
        "route_length_m": route_length,
        "raw_doppler_length_m": raw_length,
        "route_length_scale": route_length / raw_length,
        "endpoint_closure_mode": "uncertain_steps_only" if np.any(trusted_mask) else "all_steps",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_displacements", type=Path)
    parser.add_argument("hybrid_displacements", type=Path)
    parser.add_argument("imu_csv", type=Path)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--left-anchor", type=Path)
    parser.add_argument("--left-static-anchor", type=Path, nargs=2)
    parser.add_argument("--right-anchor", type=Path)
    parser.add_argument("--right-static-anchor", type=Path, nargs=2)
    parser.add_argument("--gap-start", type=int, required=True)
    parser.add_argument("--gap-end", type=int, required=True)
    parser.add_argument("--epsg", type=int, default=32654)
    parser.add_argument("--bbox-margin-deg", type=float, default=0.02)
    parser.add_argument("--particles", type=int, default=4096)
    parser.add_argument("--random-seed", type=int, default=3103403)
    parser.add_argument("--road-lower-m", type=float, default=0.21238646519190227)
    parser.add_argument("--road-upper-m", type=float, default=5.10538739585983)
    parser.add_argument("--road-sigma-m", type=float, default=2.0)
    parser.add_argument("--use-reliable-tdcp-steps", action="store_true")
    parser.add_argument("--osm-cache", type=Path, required=True)
    parser.add_argument("--output-route", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    args = parser.parse_args()

    raw, hybrid = _read(args.raw_displacements), _read(args.hybrid_displacements)
    if len(raw) != len(hybrid):
        parser.error("displacement row counts differ")
    if (args.left_anchor is None) == (args.left_static_anchor is None):
        parser.error("provide exactly one of --left-anchor or --left-static-anchor")
    if (args.right_anchor is None) == (args.right_static_anchor is None):
        parser.error("provide exactly one of --right-anchor or --right-static-anchor")
    left = (
        _load_static_position_override(args.left_anchor)
        if args.left_anchor is not None
        else _load_fusion_static_override(*args.left_static_anchor)
    )
    right = (
        _load_static_position_override(args.right_anchor)
        if args.right_anchor is not None
        else _load_fusion_static_override(*args.right_static_anchor)
    )
    gap_start, gap_end = int(args.gap_start), int(args.gap_end)
    hybrid_steps = np.asarray([_ecef(row, "d") for row in hybrid])
    start_ecef = left[2] + np.sum(hybrid_steps[left[1] : gap_start], axis=0)
    end_ecef = right[2] - np.sum(hybrid_steps[gap_end : right[0] + 1], axis=0)
    ecef_to_map = Transformer.from_crs("EPSG:4978", f"EPSG:{args.epsg}", always_xy=True)
    map_to_ecef = Transformer.from_crs(f"EPSG:{args.epsg}", "EPSG:4978", always_xy=True)
    sx, sy, _ = ecef_to_map.transform(*start_ecef)
    ex, ey, _ = ecef_to_map.transform(*end_ecef)
    road, cache = _road_cache(args.osm_cache, boundary_ecef=np.vstack([start_ecef, end_ecef]), epsg=args.epsg, margin_deg=args.bbox_margin_deg)
    times = np.asarray([float(row["tow"]) for row in raw])
    imu = _read(args.imu_csv)
    imu_times = np.asarray([float(row["GPS TOW (s)"]) for row in imu])
    gyro = np.deg2rad(np.asarray([float(row["  Ang Rate Z (deg/s)"]) for row in imu]))
    gyro_all = integrate_gyro_intervals(times, imu_times, gyro)
    step_slice = slice(gap_start - 1, gap_end - 1)
    doppler = np.asarray([_ecef(row, "doppler_d") for row in raw])
    raw_lengths = np.linalg.norm(doppler[gap_start:gap_end], axis=1)
    hybrid_lengths = np.linalg.norm(hybrid_steps[gap_start:gap_end], axis=1)
    doppler_speed_fallback = ~np.isfinite(raw_lengths)
    lengths = np.where(doppler_speed_fallback, hybrid_lengths, raw_lengths)
    if not np.isfinite(lengths).all():
        raise RuntimeError("OSM route speed evidence remains nonfinite after fallback")
    initial = hybrid_steps[gap_start - 1]
    x0, y0, _ = ecef_to_map.transform(*(start_ecef - initial))
    initial_heading = math.atan2(sx - x0, sy - y0)
    trusted_mask = np.asarray(
        [row.get("source") == "tdcp" for row in hybrid[gap_start:gap_end]], dtype=bool
    ) if args.use_reliable_tdcp_steps else np.zeros(gap_end - gap_start, dtype=bool)
    step_ecef = hybrid_steps[gap_start:gap_end]
    base = np.repeat(left[2].reshape(1, 3), len(step_ecef), axis=0)
    bx, by, _bz = ecef_to_map.transform(base[:, 0], base[:, 1], base[:, 2])
    tx, ty, _tz = ecef_to_map.transform(
        base[:, 0] + step_ecef[:, 0], base[:, 1] + step_ecef[:, 1], base[:, 2] + step_ecef[:, 2]
    )
    trusted_steps_xy = np.column_stack([np.asarray(tx) - np.asarray(bx), np.asarray(ty) - np.asarray(by)])
    route_xy, diagnostics = particle_route(
        start_xy=np.asarray([sx, sy]), end_xy=np.asarray([ex, ey]), step_lengths_m=lengths,
        gyro_increments_rad=gyro_all[step_slice], dt_s=np.diff(times)[step_slice], initial_heading_rad=initial_heading,
        road_geometry=road, particles=args.particles, random_seed=args.random_seed,
        road_lower_m=args.road_lower_m, road_upper_m=args.road_upper_m, road_sigma_m=args.road_sigma_m,
        trusted_steps_xy=trusted_steps_xy, trusted_mask=trusted_mask,
    )
    diagnostics["doppler_speed_fallback_steps"] = int(
        np.count_nonzero(doppler_speed_fallback)
    )
    start_height = _ecef_to_llh(start_ecef)[2]
    end_height = _ecef_to_llh(end_ecef)[2]
    heights = np.linspace(start_height, end_height, len(route_xy))
    route_ecef = np.asarray([map_to_ecef.transform(x, y, height) for (x, y), height in zip(route_xy, heights)])
    production_selected = (
        diagnostics["preclosure_endpoint_error_m"] <= 25.0
        and diagnostics["road_distance_p95_m"] <= args.road_upper_m
        and 0.8 <= diagnostics["route_length_scale"] <= 1.2
        and diagnostics["terminal_distinct_runner_found"]
        and diagnostics["terminal_score_gap"] >= 2.0
    )
    truth_times, truth_positions = PPCDatasetLoader(args.data_dir).load_ground_truth()
    audit_truth = np.asarray([truth_positions[int(np.argmin(np.abs(truth_times - times[e])))] for e in range(gap_start - 1, gap_end)])
    audit_errors = np.linalg.norm(route_ecef - audit_truth, axis=1)
    rows = [{"epoch": gap_start - 1 + i, "tow": float(times[gap_start - 1 + i]), "ecef_x": float(p[0]), "ecef_y": float(p[1]), "ecef_z": float(p[2]), "audit_error_m": float(audit_errors[i]), "audit_sub50cm": int(audit_errors[i] < 0.5)} for i, p in enumerate(route_ecef)]
    args.output_route.parent.mkdir(parents=True, exist_ok=True)
    with args.output_route.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "schema": "wp31_osm_particle_route_bridge_v1", "production_input_truth": False,
        "segment": [gap_start - 1, gap_end], "left_anchor_reason": left[4], "right_anchor_reason": right[4],
        "osm_cache_sha256": hashlib.sha256(args.osm_cache.read_bytes()).hexdigest(),
        "road_calibration_bounds_m": [args.road_lower_m, args.road_upper_m], **diagnostics,
        "production_selected": production_selected,
        "production_reason": "osm_particle_route_all_gates" if production_selected else "osm_particle_route_rejected",
        "audit_sub50cm_epochs": int(np.count_nonzero(audit_errors < 0.5)),
        "audit_sub50cm_pct": float(100.0 * np.mean(audit_errors < 0.5)),
        "audit_median_error_m": float(np.median(audit_errors)), "audit_p95_error_m": float(np.percentile(audit_errors, 95.0)),
        "osm_road_geometry_count": int(cache["road_geometry_count"]),
    }
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
