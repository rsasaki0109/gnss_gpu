#!/usr/bin/env python3
"""Run a truth-free road-edge particle filter between two accepted anchors."""

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
from shapely.strtree import STRtree

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "python"))
sys.path.insert(0, str(_ROOT / "experiments"))

from analyze_phase67_osm_road_centerline_feasibility import _ecef_to_llh  # noqa: E402
from build_wp29_imu_heading_route_seed_trace import integrate_gyro_intervals  # noqa: E402
from build_wp31_osm_graph_gap_proposal import build_road_graph  # noqa: E402
from build_wp31_osm_particle_route_bridge import systematic_resample  # noqa: E402
from gnss_gpu.io.ppc import PPCDatasetLoader  # noqa: E402
from run_wp29_tdcp_anchor_smoother import _load_static_position_override  # noqa: E402


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def _ecef(row: dict[str, str], prefix: str = "") -> np.ndarray:
    return np.asarray([float(row[f"{prefix}{axis}_m"]) for axis in ("x", "y", "z")])


def wrap_angle(value: float | np.ndarray) -> float | np.ndarray:
    return (value + np.pi) % (2.0 * np.pi) - np.pi


def edge_bearing(first_xy: np.ndarray, second_xy: np.ndarray) -> float:
    delta = np.asarray(second_xy) - np.asarray(first_xy)
    return math.atan2(float(delta[0]), float(delta[1]))


def nearest_edge_state(
    graph: nx.Graph, xy: np.ndarray
) -> tuple[tuple[int, int], tuple[int, int], float, np.ndarray, float]:
    edges = list(graph.edges)
    lines = [LineString([graph.nodes[u]["xy"], graph.nodes[v]["xy"]]) for u, v in edges]
    tree = STRtree(lines)
    index = int(tree.nearest(Point(xy)))
    u, v = edges[index]
    line = lines[index]
    along = float(line.project(Point(xy)))
    projected = np.asarray(line.interpolate(along).coords[0])
    return u, v, along, projected, float(np.linalg.norm(projected - xy))


def nearby_edge_states(
    graph: nx.Graph,
    xy: np.ndarray,
    *,
    max_distance_m: float,
    limit: int,
) -> list[tuple[tuple[int, int], tuple[int, int], float, np.ndarray, float]]:
    """Return distinct start-edge hypotheses inside the calibrated road band."""

    point = Point(xy)
    candidates = []
    for u, v in graph.edges:
        line = LineString([graph.nodes[u]["xy"], graph.nodes[v]["xy"]])
        distance = float(line.distance(point))
        if distance <= max_distance_m:
            along = float(line.project(point))
            projected = np.asarray(line.interpolate(along).coords[0])
            candidates.append((u, v, along, projected, distance))
    candidates.sort(key=lambda row: row[4])
    return candidates[:limit]


def advance_particle(
    graph: nx.Graph,
    u: tuple[int, int],
    v: tuple[int, int],
    along_m: float,
    travel_m: float,
    desired_heading: float,
    rng: np.random.Generator,
    branch_sigma_rad: float,
) -> tuple[tuple[int, int], tuple[int, int], float, float, float, int]:
    """Advance one directed edge state, sampling gyro-consistent branches."""

    branch_log_likelihood = 0.0
    branch_count = 0
    while travel_m > 0:
        edge_length = float(graph[u][v]["weight"])
        available = max(edge_length - along_m, 0.0)
        if travel_m <= available:
            along_m += travel_m
            travel_m = 0.0
            break
        travel_m -= available
        previous, node = u, v
        neighbors = list(graph.neighbors(node))
        forward = [candidate for candidate in neighbors if candidate != previous]
        choices = forward if forward else neighbors
        if not choices:
            return u, v, edge_length, desired_heading, branch_log_likelihood - 100.0, branch_count
        bearings = np.asarray(
            [edge_bearing(graph.nodes[node]["xy"], graph.nodes[candidate]["xy"]) for candidate in choices]
        )
        errors = np.asarray(wrap_angle(bearings - desired_heading))
        scores = -0.5 * np.square(errors / branch_sigma_rad)
        weights = np.exp(scores - np.max(scores))
        weights /= np.sum(weights)
        selected = int(rng.choice(len(choices), p=weights))
        branch_log_likelihood += float(scores[selected] - math.log(np.sum(np.exp(scores))))
        u, v, along_m = node, choices[selected], 0.0
        desired_heading = float(bearings[selected])
        branch_count += 1
    return u, v, along_m, desired_heading, branch_log_likelihood, branch_count


def state_xy(graph: nx.Graph, u: tuple[int, int], v: tuple[int, int], along_m: float) -> np.ndarray:
    first = np.asarray(graph.nodes[u]["xy"])
    second = np.asarray(graph.nodes[v]["xy"])
    length = float(graph[u][v]["weight"])
    return first + np.clip(along_m / length, 0.0, 1.0) * (second - first)


def endpoint_node_distances(
    graph: nx.Graph,
    end_u: tuple[int, int],
    end_v: tuple[int, int],
    end_along_m: float,
) -> dict[tuple[int, int], float]:
    """Shortest road distance from every node to an interior edge endpoint."""

    edge_length = float(graph[end_u][end_v]["weight"])
    from_u = nx.single_source_dijkstra_path_length(graph, end_u, weight="weight")
    from_v = nx.single_source_dijkstra_path_length(graph, end_v, weight="weight")
    return {
        node: min(
            float(from_u.get(node, float("inf"))) + end_along_m,
            float(from_v.get(node, float("inf"))) + edge_length - end_along_m,
        )
        for node in graph.nodes
    }


def state_endpoint_distance(
    graph: nx.Graph,
    node_distances: dict[tuple[int, int], float],
    u: tuple[int, int],
    v: tuple[int, int],
    along_m: float,
) -> float:
    edge_length = float(graph[u][v]["weight"])
    return min(
        along_m + node_distances[u],
        edge_length - along_m + node_distances[v],
    )


def graph_particle_route(
    *,
    graph: nx.Graph,
    start_xy: np.ndarray,
    end_xy: np.ndarray,
    step_lengths_m: np.ndarray,
    gyro_increments_rad: np.ndarray,
    dt_s: np.ndarray,
    tdcp_headings_rad: np.ndarray,
    tdcp_mask: np.ndarray,
    particles: int,
    random_seed: int,
    branch_sigma_deg: float,
    tdcp_heading_sigma_deg: float,
    endpoint_schedule_stride: int,
    endpoint_schedule_sigma_m: float,
    scale_lower: float,
    scale_upper: float,
    start_edge_limit: int,
    gyro_bias_mean_dps: float,
    gyro_bias_sigma_dps: float,
    forward_reference_xy: np.ndarray,
    backward_reference_xy: np.ndarray,
    reference_sigma_m: float,
    reference_stride: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(random_seed)
    n_steps = len(step_lengths_m)
    start_states = nearby_edge_states(
        graph, start_xy, max_distance_m=5.10538739585983, limit=start_edge_limit
    )
    if not start_states:
        start_states = [nearest_edge_state(graph, start_xy)]
    start_road_distance = start_states[0][4]
    end_u, end_v, end_along, end_projection, end_road_distance = nearest_edge_state(graph, end_xy)
    node_endpoint_distances = endpoint_node_distances(graph, end_u, end_v, end_along)
    us = np.empty(particles, dtype=object)
    vs = np.empty(particles, dtype=object)
    along = np.empty(particles)
    headings = np.empty(particles)
    initial_projections = np.empty((particles, 2))
    assignments = np.arange(particles) % (2 * len(start_states))
    rng.shuffle(assignments)
    for index, assignment in enumerate(assignments):
        start_u, start_v, start_along, start_projection, _distance = start_states[int(assignment) // 2]
        reverse = int(assignment) % 2
        edge_length = float(graph[start_u][start_v]["weight"])
        if reverse:
            us[index], vs[index], along[index] = start_v, start_u, edge_length - start_along
        else:
            us[index], vs[index], along[index] = start_u, start_v, start_along
        headings[index] = edge_bearing(graph.nodes[us[index]]["xy"], graph.nodes[vs[index]]["xy"])
        initial_projections[index] = start_projection
    biases = rng.normal(
        np.deg2rad(gyro_bias_mean_dps), np.deg2rad(gyro_bias_sigma_dps), particles
    )
    scales = rng.uniform(scale_lower, scale_upper, particles)
    logw = np.zeros(particles)
    history = np.empty((n_steps + 1, particles, 2), dtype=np.float32)
    parents = np.empty((n_steps + 1, particles), dtype=np.int32)
    history[0] = initial_projections
    parents[0] = np.arange(particles)
    parent_map = np.arange(particles, dtype=np.int32)
    branch_sigma = np.deg2rad(branch_sigma_deg)
    tdcp_sigma = np.deg2rad(tdcp_heading_sigma_deg)
    branch_total = 0
    resamples = 0
    start_shortest_distances = np.asarray(
        [state_endpoint_distance(graph, node_endpoint_distances, us[i], vs[i], along[i]) for i in range(particles)]
    )
    cumulative_lengths = np.r_[0.0, np.cumsum(step_lengths_m)]
    total_length = float(cumulative_lengths[-1])
    for step in range(n_steps):
        if tdcp_mask[step]:
            # This observation describes the current displacement interval,
            # so it must drive any branch crossed during this advance rather
            # than being applied one epoch late.
            headings[:] = tdcp_headings_rad[step]
        else:
            headings += gyro_increments_rad[step] - biases * dt_s[step]
        for particle in range(particles):
            result = advance_particle(
                graph, us[particle], vs[particle], along[particle],
                float(step_lengths_m[step] * scales[particle]), float(headings[particle]),
                rng, branch_sigma,
            )
            us[particle], vs[particle], along[particle], headings[particle], branch_logw, branches = result
            # Branches are already sampled from the gyro proposal.  Adding
            # that proposal probability again would double-count it and favor
            # routes merely because they cross fewer intersections.
            _ = branch_logw
            branch_total += branches
            history[step + 1, particle] = state_xy(graph, us[particle], vs[particle], along[particle])
        if tdcp_mask[step]:
            errors = np.asarray(wrap_angle(headings - tdcp_headings_rad[step]))
            logw -= 0.5 * np.square(errors / tdcp_sigma)
        # Backward graph information removes branches that cannot reach the
        # accepted endpoint with the remaining maximum travel distance.  It
        # does not reward the shortest route, so loops remain representable.
        remaining = float(np.sum(step_lengths_m[step + 1 :]) * 1.1)
        graph_distances = np.asarray(
            [state_endpoint_distance(graph, node_endpoint_distances, us[i], vs[i], along[i]) for i in range(particles)]
        )
        impossible = np.maximum(graph_distances - remaining - 15.0, 0.0)
        logw -= 0.5 * np.square(impossible / 5.0)
        if reference_stride > 0 and (
            (step + 1) % reference_stride == 0 or step + 1 == n_steps
        ):
            xy = np.asarray(history[step + 1], dtype=np.float64)
            forward_distance = np.linalg.norm(xy - forward_reference_xy[step + 1], axis=1)
            backward_distance = np.linalg.norm(xy - backward_reference_xy[step + 1], axis=1)
            progress_fraction = (step + 1) / n_steps
            log_forward = math.log(max(1.0 - progress_fraction, 1e-6)) - np.log1p(
                np.square(forward_distance / reference_sigma_m)
            )
            log_backward = math.log(max(progress_fraction, 1e-6)) - np.log1p(
                np.square(backward_distance / reference_sigma_m)
            )
            logw += np.logaddexp(log_forward, log_backward)
        if endpoint_schedule_stride > 0 and (
            (step + 1) % endpoint_schedule_stride == 0 or step + 1 == n_steps
        ):
            remaining_fraction = max(0.0, 1.0 - cumulative_lengths[step + 1] / total_length)
            scheduled_distance = start_shortest_distances * remaining_fraction
            schedule_sigma = endpoint_schedule_sigma_m * max(0.25, math.sqrt(remaining_fraction))
            logw -= 0.5 * np.square((graph_distances - scheduled_distance) / schedule_sigma)
        parents[step + 1] = parent_map
        shifted = logw - np.max(logw)
        weights = np.exp(np.clip(shifted, -700.0, 0.0))
        weights /= np.sum(weights)
        ess = 1.0 / np.sum(np.square(weights))
        if step + 1 < n_steps and ess < 0.5 * particles:
            chosen = systematic_resample(weights, rng)
            us, vs, along = us[chosen], vs[chosen], along[chosen]
            headings, biases, scales = headings[chosen], biases[chosen], scales[chosen]
            start_shortest_distances = start_shortest_distances[chosen]
            parent_map = chosen.astype(np.int32)
            logw = np.zeros(particles)
            resamples += 1
        else:
            parent_map = np.arange(particles, dtype=np.int32)
    terminal_distances = np.asarray(
        [state_endpoint_distance(graph, node_endpoint_distances, us[i], vs[i], along[i]) for i in range(particles)]
    )
    terminal_scores = logw - 0.5 * np.square(terminal_distances / 5.0)
    ranked = np.argsort(terminal_scores)[::-1]
    best = int(ranked[0])
    runner = next(
        (int(i) for i in ranked[1:] if np.linalg.norm(history[-1, i] - history[-1, best]) >= 1.0),
        None,
    )
    lineage = np.empty((n_steps + 1, 2))
    index = best
    for step in range(n_steps, -1, -1):
        lineage[step] = history[step, index]
        index = int(parents[step, index])
    progress = np.r_[0.0, np.cumsum(step_lengths_m)]
    progress /= progress[-1]
    start_offset = start_xy - lineage[0]
    end_offset = end_xy - end_projection
    route = lineage + start_offset + progress[:, None] * (end_offset - start_offset)
    runner_score = float("-inf") if runner is None else float(terminal_scores[runner])
    return route, {
        "particle_count": particles,
        "random_seed": random_seed,
        "resample_count": resamples,
        "sampled_graph_branches": branch_total,
        "start_road_projection_m": start_road_distance,
        "end_road_projection_m": end_road_distance,
        "terminal_endpoint_error_m": float(terminal_distances[best]),
        "terminal_euclidean_error_m": float(np.linalg.norm(history[-1, best] - end_projection)),
        "terminal_distinct_runner_found": runner is not None,
        "terminal_score_gap": float("inf") if runner is None else float(terminal_scores[best] - runner_score),
        "selected_speed_scale": float(scales[best]),
        "selected_gyro_bias_dps": float(np.rad2deg(biases[best])),
        "start_edge_candidates": len(start_states),
        "selected_start_to_end_shortest_graph_m": float(start_shortest_distances[best]),
        "endpoint_schedule_stride": endpoint_schedule_stride,
        "endpoint_schedule_sigma_m": endpoint_schedule_sigma_m,
        "gyro_bias_prior_mean_dps": gyro_bias_mean_dps,
        "gyro_bias_prior_sigma_dps": gyro_bias_sigma_dps,
        "reference_mode": "bidirectional_cauchy_mixture",
        "reference_sigma_m": reference_sigma_m,
        "reference_stride": reference_stride,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_displacements", type=Path)
    parser.add_argument("hybrid_displacements", type=Path)
    parser.add_argument("imu_csv", type=Path)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--left-anchor", type=Path, required=True)
    parser.add_argument("--right-anchor", type=Path, required=True)
    parser.add_argument("--start-epoch", type=int, required=True)
    parser.add_argument("--end-epoch", type=int, required=True)
    parser.add_argument("--epsg", type=int, default=32654)
    parser.add_argument("--osm-cache", type=Path, required=True)
    parser.add_argument("--particles", type=int, default=2048)
    parser.add_argument("--random-seed", type=int, default=3103403)
    parser.add_argument("--branch-sigma-deg", type=float, default=30.0)
    parser.add_argument("--tdcp-heading-sigma-deg", type=float, default=12.0)
    parser.add_argument("--gyro-sign", type=int, choices=(-1, 1), default=-1)
    parser.add_argument("--endpoint-schedule-stride", type=int, default=0)
    parser.add_argument("--endpoint-schedule-sigma-m", type=float, default=150.0)
    parser.add_argument("--scale-lower", type=float, default=0.98)
    parser.add_argument("--scale-upper", type=float, default=1.02)
    parser.add_argument("--start-edge-limit", type=int, default=32)
    parser.add_argument("--gyro-bias-mean-dps", type=float, default=0.0)
    parser.add_argument("--gyro-bias-sigma-dps", type=float, default=0.1)
    parser.add_argument("--reference-sigma-m", type=float, default=25.0)
    parser.add_argument("--reference-stride", type=int, default=5)
    parser.add_argument("--reverse-filter", action="store_true")
    parser.add_argument("--output-route", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    args = parser.parse_args()

    raw, hybrid = _read(args.raw_displacements), _read(args.hybrid_displacements)
    left, right = _load_static_position_override(args.left_anchor), _load_static_position_override(args.right_anchor)
    start, end = args.start_epoch, args.end_epoch
    if start != left[1] or end != right[0]:
        parser.error("the PF interval must use the adjacent accepted-anchor boundaries")
    steps_ecef = np.asarray([_ecef(row, "d") for row in hybrid])
    start_ecef, end_ecef = left[2], right[2]
    ecef_to_map = Transformer.from_crs("EPSG:4978", f"EPSG:{args.epsg}", always_xy=True)
    map_to_ecef = Transformer.from_crs(f"EPSG:{args.epsg}", "EPSG:4978", always_xy=True)
    sx, sy, _ = ecef_to_map.transform(*start_ecef)
    ex, ey, _ = ecef_to_map.transform(*end_ecef)
    cache = json.loads(args.osm_cache.read_text(encoding="utf-8"))
    graph = build_road_graph(cache["projected_road_lines"])
    raw_doppler = np.asarray([_ecef(row, "doppler_d") for row in raw])
    step_lengths = np.linalg.norm(raw_doppler[start:end], axis=1)
    times = np.asarray([float(row["tow"]) for row in raw])
    imu = _read(args.imu_csv)
    imu_times = np.asarray([float(row["GPS TOW (s)"]) for row in imu])
    gyro = np.deg2rad(np.asarray([float(row["  Ang Rate Z (deg/s)"]) for row in imu]))
    gyro_all = integrate_gyro_intervals(times, imu_times, gyro)
    step_slice = slice(start - 1, end - 1)
    base = np.repeat(start_ecef.reshape(1, 3), end - start, axis=0)
    bx, by, _ = ecef_to_map.transform(base[:, 0], base[:, 1], base[:, 2])
    tx, ty, _ = ecef_to_map.transform(
        base[:, 0] + steps_ecef[start:end, 0],
        base[:, 1] + steps_ecef[start:end, 1],
        base[:, 2] + steps_ecef[start:end, 2],
    )
    step_xy = np.column_stack([np.asarray(tx) - np.asarray(bx), np.asarray(ty) - np.asarray(by)])
    tdcp_mask = np.asarray([row.get("source") == "tdcp" for row in hybrid[start:end]])
    tdcp_headings = np.arctan2(step_xy[:, 0], step_xy[:, 1])
    interval_steps = steps_ecef[start:end]
    forward_ecef = np.vstack([start_ecef, start_ecef + np.cumsum(interval_steps, axis=0)])
    reverse_suffix = np.vstack([np.cumsum(interval_steps[::-1], axis=0)[::-1], np.zeros(3)])
    backward_ecef = end_ecef - reverse_suffix
    fx, fy, _ = ecef_to_map.transform(forward_ecef[:, 0], forward_ecef[:, 1], forward_ecef[:, 2])
    rx, ry, _ = ecef_to_map.transform(backward_ecef[:, 0], backward_ecef[:, 1], backward_ecef[:, 2])
    forward_reference = np.column_stack([fx, fy])
    backward_reference = np.column_stack([rx, ry])
    signed_gyro = args.gyro_sign * gyro_all[step_slice]
    dt = np.diff(times)[step_slice]
    pf_start, pf_end = np.asarray([sx, sy]), np.asarray([ex, ey])
    if args.reverse_filter:
        pf_start, pf_end = pf_end, pf_start
        step_lengths = step_lengths[::-1]
        signed_gyro = -signed_gyro[::-1]
        dt = dt[::-1]
        tdcp_headings = np.asarray(wrap_angle(tdcp_headings[::-1] + np.pi))
        tdcp_mask = tdcp_mask[::-1]
        forward_reference, backward_reference = backward_reference[::-1], forward_reference[::-1]
    route_xy, diagnostics = graph_particle_route(
        graph=graph, start_xy=pf_start, end_xy=pf_end,
        step_lengths_m=step_lengths, gyro_increments_rad=signed_gyro,
        dt_s=dt, tdcp_headings_rad=tdcp_headings,
        tdcp_mask=tdcp_mask, particles=args.particles, random_seed=args.random_seed,
        branch_sigma_deg=args.branch_sigma_deg, tdcp_heading_sigma_deg=args.tdcp_heading_sigma_deg,
        endpoint_schedule_stride=args.endpoint_schedule_stride,
        endpoint_schedule_sigma_m=args.endpoint_schedule_sigma_m,
        scale_lower=args.scale_lower, scale_upper=args.scale_upper,
        start_edge_limit=args.start_edge_limit,
        gyro_bias_mean_dps=args.gyro_bias_mean_dps,
        gyro_bias_sigma_dps=args.gyro_bias_sigma_dps,
        forward_reference_xy=forward_reference,
        backward_reference_xy=backward_reference,
        reference_sigma_m=args.reference_sigma_m,
        reference_stride=args.reference_stride,
    )
    if args.reverse_filter:
        route_xy = route_xy[::-1]
    heights = np.linspace(_ecef_to_llh(start_ecef)[2], _ecef_to_llh(end_ecef)[2], len(route_xy))
    route_ecef = np.asarray([map_to_ecef.transform(x, y, h) for (x, y), h in zip(route_xy, heights)])
    road = unary_union([LineString(line) for line in cache["projected_road_lines"]])
    road_distances = np.asarray([road.distance(Point(xy)) for xy in route_xy])
    truth_times, truth_positions = PPCDatasetLoader(args.data_dir).load_ground_truth()
    truth = np.asarray([truth_positions[int(np.argmin(np.abs(truth_times - times[e])))] for e in range(start, end + 1)])
    errors = np.linalg.norm(route_ecef - truth, axis=1)
    production_selected = (
        diagnostics["start_road_projection_m"] <= 5.10538739585983
        and diagnostics["end_road_projection_m"] <= 5.10538739585983
        and diagnostics["terminal_endpoint_error_m"] <= 10.0
        and diagnostics["terminal_distinct_runner_found"]
        and diagnostics["terminal_score_gap"] >= 2.0
        and float(np.percentile(road_distances, 95.0)) <= 5.10538739585983
    )
    rows = [
        {"epoch": start + i, "tow": float(times[start + i]), "ecef_x": float(p[0]),
         "ecef_y": float(p[1]), "ecef_z": float(p[2]), "audit_error_m": float(errors[i]),
         "audit_sub50cm": int(errors[i] < 0.5)}
        for i, p in enumerate(route_ecef)
    ]
    args.output_route.parent.mkdir(parents=True, exist_ok=True)
    with args.output_route.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary = {
        "schema": "wp31_osm_graph_particle_route_v1", "production_input_truth": False,
        "segment": [start, end], "left_anchor_reason": left[4], "right_anchor_reason": right[4],
        "osm_cache_sha256": hashlib.sha256(args.osm_cache.read_bytes()).hexdigest(),
        "graph_nodes": graph.number_of_nodes(), "graph_edges": graph.number_of_edges(), **diagnostics,
        "filter_direction": "reverse" if args.reverse_filter else "forward",
        "gyro_sign": args.gyro_sign,
        "reliable_tdcp_steps": int(np.count_nonzero(tdcp_mask)),
        "road_distance_p95_m": float(np.percentile(road_distances, 95.0)),
        "production_selected": production_selected,
        "production_reason": "osm_graph_particle_all_gates" if production_selected else "osm_graph_particle_rejected",
        "audit_sub50cm_epochs": int(np.count_nonzero(errors < 0.5)),
        "audit_sub50cm_pct": float(100.0 * np.mean(errors < 0.5)),
        "audit_median_error_m": float(np.median(errors)),
        "audit_p95_error_m": float(np.percentile(errors, 95.0)),
    }
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
