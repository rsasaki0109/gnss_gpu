#!/usr/bin/env python3
"""Build a truth-free endpoint-closed OSM graph proposal for one outage."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
from pyproj import Transformer
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points, unary_union

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "python"))
sys.path.insert(0, str(_ROOT / "experiments"))

from analyze_phase67_osm_road_centerline_feasibility import _ecef_to_llh  # noqa: E402
from gnss_gpu.io.ppc import PPCDatasetLoader  # noqa: E402
from run_wp29_tdcp_anchor_smoother import _load_static_position_override  # noqa: E402


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def _ecef(row: dict[str, str], prefix: str = "") -> np.ndarray:
    return np.asarray([float(row[f"{prefix}{axis}_m"]) for axis in ("x", "y", "z")])


def build_road_graph(lines: list[list[list[float]]], *, rounding_m: float = 0.01) -> nx.Graph:
    """Convert cached OSM edge geometries to a weighted undirected graph."""

    if rounding_m <= 0:
        raise ValueError("rounding_m must be positive")
    graph = nx.Graph()

    def key(point: list[float]) -> tuple[int, int]:
        return tuple(int(round(float(value) / rounding_m)) for value in point[:2])

    for line in lines:
        for first, second in zip(line, line[1:]):
            u, v = key(first), key(second)
            xy_u = np.asarray(first[:2], dtype=np.float64)
            xy_v = np.asarray(second[:2], dtype=np.float64)
            length = float(np.linalg.norm(xy_v - xy_u))
            if u == v or not math.isfinite(length) or length <= 0:
                continue
            graph.add_node(u, xy=xy_u)
            graph.add_node(v, xy=xy_v)
            if not graph.has_edge(u, v) or length < float(graph[u][v]["weight"]):
                graph.add_edge(u, v, weight=length)
    return graph


def _nearest_node(graph: nx.Graph, xy: np.ndarray) -> tuple[tuple[int, int], float]:
    nodes = list(graph.nodes)
    coords = np.asarray([graph.nodes[node]["xy"] for node in nodes])
    distances = np.linalg.norm(coords - np.asarray(xy), axis=1)
    index = int(np.argmin(distances))
    return nodes[index], float(distances[index])


def shortest_road_paths(
    graph: nx.Graph, start_xy: np.ndarray, end_xy: np.ndarray, *, count: int = 2
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Return distinct shortest node paths and topology-only diagnostics."""

    start, start_snap = _nearest_node(graph, start_xy)
    end, end_snap = _nearest_node(graph, end_xy)
    generator = nx.shortest_simple_paths(graph, start, end, weight="weight")
    paths: list[np.ndarray] = []
    lengths: list[float] = []
    for nodes in generator:
        xy = np.asarray([graph.nodes[node]["xy"] for node in nodes])
        paths.append(xy)
        lengths.append(float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1))))
        if len(paths) >= count:
            break
    gap = float("inf") if len(lengths) < 2 else lengths[1] - lengths[0]
    return paths, {
        "start_node_snap_m": start_snap,
        "end_node_snap_m": end_snap,
        "path_lengths_m": lengths,
        "second_path_length_gap_m": gap,
    }


def resample_offset_path(
    centerline_xy: np.ndarray,
    start_xy: np.ndarray,
    end_xy: np.ndarray,
    step_lengths_m: np.ndarray,
) -> np.ndarray:
    """Resample a centerline while interpolating the endpoint road offsets."""

    line = LineString(centerline_xy)
    start_projection = nearest_points(line, Point(start_xy))[0]
    end_projection = nearest_points(line, Point(end_xy))[0]
    start_offset = np.asarray(start_xy) - np.asarray(start_projection.coords[0])
    end_offset = np.asarray(end_xy) - np.asarray(end_projection.coords[0])
    progress = np.r_[0.0, np.cumsum(np.asarray(step_lengths_m, dtype=np.float64))]
    if progress[-1] <= 0:
        progress = np.linspace(0.0, 1.0, len(progress))
    else:
        progress /= progress[-1]
    along = np.asarray([line.interpolate(float(value), normalized=True).coords[0] for value in progress])
    result = along + start_offset + progress[:, None] * (end_offset - start_offset)
    result[0], result[-1] = start_xy, end_xy
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_displacements", type=Path)
    parser.add_argument("hybrid_displacements", type=Path)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--left-anchor", type=Path, required=True)
    parser.add_argument("--right-anchor", type=Path, required=True)
    parser.add_argument("--gap-start", type=int, required=True)
    parser.add_argument("--gap-end", type=int, required=True)
    parser.add_argument("--epsg", type=int, default=32654)
    parser.add_argument("--osm-cache", type=Path, required=True)
    parser.add_argument("--max-node-snap-m", type=float, default=15.0)
    parser.add_argument("--max-length-scale-error", type=float, default=0.2)
    parser.add_argument("--min-second-path-gap-m", type=float, default=10.0)
    parser.add_argument("--max-road-p95-m", type=float, default=5.10538739585983)
    parser.add_argument("--output-route", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    args = parser.parse_args()

    raw, hybrid = _read_csv(args.raw_displacements), _read_csv(args.hybrid_displacements)
    if len(raw) != len(hybrid):
        parser.error("displacement row counts differ")
    left = _load_static_position_override(args.left_anchor)
    right = _load_static_position_override(args.right_anchor)
    gap_start, gap_end = args.gap_start, args.gap_end
    if not (left[1] <= gap_start < gap_end <= right[0]):
        parser.error("gap is not bracketed by the accepted anchors")

    hybrid_steps = np.asarray([_ecef(row, "d") for row in hybrid])
    start_ecef = left[2] + np.sum(hybrid_steps[left[1] : gap_start], axis=0)
    end_ecef = right[2] - np.sum(hybrid_steps[gap_end : right[0] + 1], axis=0)
    ecef_to_map = Transformer.from_crs("EPSG:4978", f"EPSG:{args.epsg}", always_xy=True)
    map_to_ecef = Transformer.from_crs(f"EPSG:{args.epsg}", "EPSG:4978", always_xy=True)
    sx, sy, _ = ecef_to_map.transform(*start_ecef)
    ex, ey, _ = ecef_to_map.transform(*end_ecef)
    start_xy, end_xy = np.asarray([sx, sy]), np.asarray([ex, ey])

    cache = json.loads(args.osm_cache.read_text(encoding="utf-8"))
    if int(cache.get("epsg", -1)) != args.epsg:
        parser.error("OSM cache EPSG mismatch")
    graph = build_road_graph(cache["projected_road_lines"])
    paths, diagnostics = shortest_road_paths(graph, start_xy, end_xy)
    if not paths:
        raise RuntimeError("no connected OSM road path between gap boundaries")
    doppler = np.asarray([_ecef(row, "doppler_d") for row in raw])
    step_lengths = np.linalg.norm(doppler[gap_start:gap_end], axis=1)
    route_xy = resample_offset_path(paths[0], start_xy, end_xy, step_lengths)
    centerline_length = float(diagnostics["path_lengths_m"][0])
    doppler_length = float(np.sum(step_lengths))
    length_scale = centerline_length / doppler_length
    road = unary_union([LineString(line) for line in cache["projected_road_lines"]])
    road_distances = np.asarray([road.distance(Point(xy)) for xy in route_xy])

    start_height, end_height = _ecef_to_llh(start_ecef)[2], _ecef_to_llh(end_ecef)[2]
    heights = np.linspace(start_height, end_height, len(route_xy))
    route_ecef = np.asarray(
        [map_to_ecef.transform(x, y, h) for (x, y), h in zip(route_xy, heights)]
    )
    road_p95 = float(np.percentile(road_distances, 95.0))
    production_selected = (
        diagnostics["start_node_snap_m"] <= args.max_node_snap_m
        and diagnostics["end_node_snap_m"] <= args.max_node_snap_m
        and abs(length_scale - 1.0) <= args.max_length_scale_error
        and diagnostics["second_path_length_gap_m"] >= args.min_second_path_gap_m
        and road_p95 <= args.max_road_p95_m
    )

    times = np.asarray([float(row["tow"]) for row in raw])
    truth_times, truth_positions = PPCDatasetLoader(args.data_dir).load_ground_truth()
    truth = np.asarray(
        [truth_positions[int(np.argmin(np.abs(truth_times - times[epoch])))] for epoch in range(gap_start, gap_end + 1)]
    )
    audit_errors = np.linalg.norm(route_ecef - truth, axis=1)
    output_rows = [
        {
            "epoch": gap_start + index,
            "tow": float(times[gap_start + index]),
            "ecef_x": float(position[0]),
            "ecef_y": float(position[1]),
            "ecef_z": float(position[2]),
            "audit_error_m": float(audit_errors[index]),
            "audit_sub50cm": int(audit_errors[index] < 0.5),
        }
        for index, position in enumerate(route_ecef)
    ]
    args.output_route.parent.mkdir(parents=True, exist_ok=True)
    with args.output_route.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    summary = {
        "schema": "wp31_osm_graph_gap_proposal_v1",
        "production_input_truth": False,
        "segment": [gap_start, gap_end],
        "left_anchor_reason": left[4],
        "right_anchor_reason": right[4],
        "osm_cache_sha256": hashlib.sha256(args.osm_cache.read_bytes()).hexdigest(),
        "graph_nodes": graph.number_of_nodes(),
        "graph_edges": graph.number_of_edges(),
        **diagnostics,
        "doppler_length_m": doppler_length,
        "centerline_length_m": centerline_length,
        "length_scale": length_scale,
        "road_distance_p95_m": road_p95,
        "production_selected": production_selected,
        "production_reason": "osm_graph_gap_all_gates" if production_selected else "osm_graph_gap_rejected",
        "audit_sub50cm_epochs": int(np.count_nonzero(audit_errors < 0.5)),
        "audit_sub50cm_pct": float(100.0 * np.mean(audit_errors < 0.5)),
        "audit_start_boundary_error_m": float(audit_errors[0]),
        "audit_end_boundary_error_m": float(audit_errors[-1]),
        "audit_median_error_m": float(np.median(audit_errors)),
        "audit_p95_error_m": float(np.percentile(audit_errors, 95.0)),
    }
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
