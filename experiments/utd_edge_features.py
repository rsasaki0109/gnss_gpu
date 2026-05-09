"""Geometry helpers for PPC UTD diffraction candidate features.

This module deliberately stops short of a full Uniform Theory of
Diffraction implementation.  It extracts building edges that can act as
knife-edge / wedge diffraction candidates, then scores how close those
edges are to each receiver-to-satellite ray.  The output is a deployable
feature family that can be tested before implementing GPU UTD
coefficients.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DiffractionEdgeSet:
    """Candidate building edges represented in ECEF coordinates."""

    start: np.ndarray
    end: np.ndarray
    midpoint: np.ndarray
    length_m: np.ndarray
    dihedral_deg: np.ndarray
    is_boundary: np.ndarray

    @property
    def size(self) -> int:
        return int(self.midpoint.shape[0])

    def subset(self, mask: np.ndarray) -> "DiffractionEdgeSet":
        mask = np.asarray(mask)
        return DiffractionEdgeSet(
            start=self.start[mask],
            end=self.end[mask],
            midpoint=self.midpoint[mask],
            length_m=self.length_m[mask],
            dihedral_deg=self.dihedral_deg[mask],
            is_boundary=self.is_boundary[mask],
        )


def _triangle_normals(triangles: np.ndarray) -> np.ndarray:
    v0 = triangles[:, 0]
    v1 = triangles[:, 1]
    v2 = triangles[:, 2]
    normals = np.cross(v1 - v0, v2 - v0)
    norm = np.linalg.norm(normals, axis=1)
    out = np.zeros_like(normals)
    valid = norm > 1e-9
    out[valid] = normals[valid] / norm[valid, None]
    return out


def _edge_key(a: np.ndarray, b: np.ndarray, quantization_m: float) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    qa = tuple(np.rint(a / quantization_m).astype(np.int64).tolist())
    qb = tuple(np.rint(b / quantization_m).astype(np.int64).tolist())
    return (qa, qb) if qa <= qb else (qb, qa)


def extract_diffraction_edges(
    triangles: np.ndarray,
    *,
    route_ecef: np.ndarray | None = None,
    route_margin_m: float = 250.0,
    quantization_m: float = 0.05,
    min_edge_length_m: float = 2.0,
    min_dihedral_deg: float = 20.0,
    include_boundary_edges: bool = True,
    voxel_size_m: float = 0.0,
    max_edges: int = 0,
) -> DiffractionEdgeSet:
    """Extract candidate UTD diffraction edges from a triangle mesh.

    Non-coplanar shared edges are kept when their dihedral angle exceeds
    ``min_dihedral_deg``.  Boundary edges are optionally kept because
    PLATEAU surfaces are often not topologically welded across adjacent
    polygons, so many physical roof / wall edges appear as boundaries.
    """
    tri = np.asarray(triangles, dtype=np.float64)
    if tri.ndim != 3 or tri.shape[1:] != (3, 3):
        raise ValueError("triangles must have shape [N, 3, 3]")
    if quantization_m <= 0:
        raise ValueError("quantization_m must be positive")

    route_lo = route_hi = None
    if route_ecef is not None:
        route = np.asarray(route_ecef, dtype=np.float64).reshape(-1, 3)
        route_lo = route.min(axis=0) - route_margin_m
        route_hi = route.max(axis=0) + route_margin_m

    normals = _triangle_normals(tri)
    edge_map: dict[tuple[tuple[int, int, int], tuple[int, int, int]], dict[str, object]] = {}
    edge_indices = ((0, 1), (1, 2), (2, 0))

    for tri_idx, corners in enumerate(tri):
        normal = normals[tri_idx]
        if not np.any(normal):
            continue
        for i0, i1 in edge_indices:
            a = corners[i0]
            b = corners[i1]
            mid = 0.5 * (a + b)
            if route_lo is not None and (np.any(mid < route_lo) or np.any(mid > route_hi)):
                continue
            length = float(np.linalg.norm(b - a))
            if length < min_edge_length_m:
                continue
            key = _edge_key(a, b, quantization_m)
            item = edge_map.get(key)
            if item is None:
                edge_map[key] = {
                    "start": a.copy(),
                    "end": b.copy(),
                    "length": length,
                    "normals": [normal.copy()],
                }
            else:
                item["normals"].append(normal.copy())  # type: ignore[index, union-attr]
                if length > float(item["length"]):
                    item["start"] = a.copy()
                    item["end"] = b.copy()
                    item["length"] = length

    starts: list[np.ndarray] = []
    ends: list[np.ndarray] = []
    lengths: list[float] = []
    dihedrals: list[float] = []
    boundaries: list[bool] = []

    for item in edge_map.values():
        normals_for_edge = np.asarray(item["normals"], dtype=np.float64)
        is_boundary = normals_for_edge.shape[0] < 2
        if is_boundary:
            if not include_boundary_edges:
                continue
            dihedral = np.nan
        else:
            dots = np.clip(np.abs(normals_for_edge @ normals_for_edge.T), 0.0, 1.0)
            upper = dots[np.triu_indices_from(dots, k=1)]
            dihedral = float(np.degrees(np.arccos(np.min(upper)))) if upper.size else 0.0
            if dihedral < min_dihedral_deg:
                continue
        starts.append(np.asarray(item["start"], dtype=np.float64))
        ends.append(np.asarray(item["end"], dtype=np.float64))
        lengths.append(float(item["length"]))
        dihedrals.append(dihedral)
        boundaries.append(is_boundary)

    if not starts:
        empty = np.empty((0, 3), dtype=np.float64)
        return DiffractionEdgeSet(
            start=empty,
            end=empty,
            midpoint=empty,
            length_m=np.empty(0, dtype=np.float64),
            dihedral_deg=np.empty(0, dtype=np.float64),
            is_boundary=np.empty(0, dtype=bool),
        )

    start = np.vstack(starts)
    end = np.vstack(ends)
    length_m = np.asarray(lengths, dtype=np.float64)
    edges = DiffractionEdgeSet(
        start=start,
        end=end,
        midpoint=0.5 * (start + end),
        length_m=length_m,
        dihedral_deg=np.asarray(dihedrals, dtype=np.float64),
        is_boundary=np.asarray(boundaries, dtype=bool),
    )
    if voxel_size_m > 0.0:
        edges = thin_edges_by_midpoint_voxel(edges, voxel_size_m)
    if max_edges > 0 and edges.size > max_edges:
        order = np.argsort(edges.length_m)[::-1][:max_edges]
        edges = edges.subset(order)
    return edges


def thin_edges_by_midpoint_voxel(edges: DiffractionEdgeSet, voxel_size_m: float) -> DiffractionEdgeSet:
    """Keep the longest edge in each midpoint voxel."""
    if voxel_size_m <= 0:
        raise ValueError("voxel_size_m must be positive")
    if edges.size == 0:
        return edges
    voxels = np.floor(edges.midpoint / voxel_size_m).astype(np.int64)
    best: dict[tuple[int, int, int], int] = {}
    for idx, voxel in enumerate(voxels):
        key = tuple(voxel.tolist())
        prev = best.get(key)
        if prev is None or edges.length_m[idx] > edges.length_m[prev]:
            best[key] = idx
    keep = np.array(sorted(best.values()), dtype=np.int64)
    return edges.subset(keep)


def _ray_segment_closest(
    rx: np.ndarray,
    direction: np.ndarray,
    edge_start: np.ndarray,
    edge_end: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Closest point from a receiver ray to every edge segment.

    Returns ``(point_on_edge, distance_to_ray_m, ray_distance_m)``.
    """
    edge_vec = edge_end - edge_start
    w = rx[None, :] - edge_start
    b = edge_vec @ direction
    c = np.einsum("ij,ij->i", edge_vec, edge_vec)
    d = w @ direction
    e = np.einsum("ij,ij->i", edge_vec, w)
    denom = c - b * b

    t_edge = np.zeros(edge_start.shape[0], dtype=np.float64)
    valid = denom > 1e-9
    t_edge[valid] = (e[valid] - b[valid] * d[valid]) / denom[valid]
    t_edge = np.clip(t_edge, 0.0, 1.0)
    point = edge_start + t_edge[:, None] * edge_vec

    rel = point - rx[None, :]
    ray_distance = rel @ direction
    closest_on_ray = rx[None, :] + ray_distance[:, None] * direction
    distance_to_ray = np.linalg.norm(point - closest_on_ray, axis=1)
    return point, distance_to_ray, ray_distance


def per_sat_utd_candidates(
    rx_ecef: np.ndarray,
    sat_ecef: np.ndarray,
    edges: DiffractionEdgeSet,
    *,
    max_edge_range_m: float = 250.0,
    max_ray_edge_distance_m: float = 25.0,
    max_excess_path_m: float = 80.0,
    wavelength_m: float = 0.1902936728,
    score_excess_scale_m: float = 20.0,
    score_distance_scale_m: float = 10.0,
) -> dict[str, np.ndarray]:
    """Score UTD diffraction candidates for each receiver-satellite ray."""
    rx = np.asarray(rx_ecef, dtype=np.float64).reshape(3)
    sats = np.asarray(sat_ecef, dtype=np.float64).reshape(-1, 3)
    n_sat = sats.shape[0]
    count = np.zeros(n_sat, dtype=np.int64)
    min_excess = np.full(n_sat, np.inf, dtype=np.float64)
    min_distance = np.full(n_sat, np.inf, dtype=np.float64)
    min_fresnel_v = np.full(n_sat, np.inf, dtype=np.float64)
    score = np.zeros(n_sat, dtype=np.float64)

    if n_sat == 0 or edges.size == 0:
        return {
            "candidate_count": count,
            "min_excess_path_m": min_excess,
            "min_edge_distance_m": min_distance,
            "min_fresnel_v": min_fresnel_v,
            "score": score,
        }

    local_delta = edges.midpoint - rx[None, :]
    local_mask = np.einsum("ij,ij->i", local_delta, local_delta) <= max_edge_range_m * max_edge_range_m
    if not np.any(local_mask):
        return {
            "candidate_count": count,
            "min_excess_path_m": min_excess,
            "min_edge_distance_m": min_distance,
            "min_fresnel_v": min_fresnel_v,
            "score": score,
        }

    start = edges.start[local_mask]
    end = edges.end[local_mask]
    for sat_idx, sat in enumerate(sats):
        sat_vec = sat - rx
        direct_dist = float(np.linalg.norm(sat_vec))
        if direct_dist <= 1e-6:
            continue
        direction = sat_vec / direct_dist
        point, ray_distance, along_ray = _ray_segment_closest(rx, direction, start, end)
        in_front = (along_ray > 0.0) & (along_ray < min(max_edge_range_m, direct_dist))
        close_to_ray = ray_distance <= max_ray_edge_distance_m
        if not np.any(in_front & close_to_ray):
            continue
        p = point[in_front & close_to_ray]
        dist_ray = ray_distance[in_front & close_to_ray]
        d1 = np.linalg.norm(p - rx[None, :], axis=1)
        d2 = np.linalg.norm(sat[None, :] - p, axis=1)
        excess = d1 + d2 - direct_dist
        candidate = excess <= max_excess_path_m
        if not np.any(candidate):
            continue
        excess = excess[candidate]
        dist_ray = dist_ray[candidate]
        d1 = d1[candidate]
        d2 = d2[candidate]
        fresnel_v = dist_ray * np.sqrt(2.0 * (1.0 / np.maximum(d1, 1e-6) + 1.0 / np.maximum(d2, 1e-6)) / wavelength_m)
        candidate_score = np.exp(-excess / score_excess_scale_m) * np.exp(-dist_ray / score_distance_scale_m)
        count[sat_idx] = int(excess.size)
        min_excess[sat_idx] = float(excess.min())
        min_distance[sat_idx] = float(dist_ray.min())
        min_fresnel_v[sat_idx] = float(fresnel_v.min())
        score[sat_idx] = float(candidate_score.sum())

    return {
        "candidate_count": count,
        "min_excess_path_m": min_excess,
        "min_edge_distance_m": min_distance,
        "min_fresnel_v": min_fresnel_v,
        "score": score,
    }


def epoch_utd_summary(
    rx_ecef: np.ndarray,
    sat_ecef: np.ndarray,
    is_los: np.ndarray,
    edges: DiffractionEdgeSet,
    **candidate_kwargs: float,
) -> dict[str, float | int]:
    """Return aggregate UTD candidate features for one epoch."""
    sats = np.asarray(sat_ecef, dtype=np.float64).reshape(-1, 3)
    los = np.asarray(is_los, dtype=bool).reshape(-1)
    per_sat = per_sat_utd_candidates(rx_ecef, sats, edges, **candidate_kwargs)
    counts = per_sat["candidate_count"]
    has_candidate = counts > 0
    nlos = ~los[: counts.size]
    finite_excess = per_sat["min_excess_path_m"][np.isfinite(per_sat["min_excess_path_m"])]
    finite_distance = per_sat["min_edge_distance_m"][np.isfinite(per_sat["min_edge_distance_m"])]
    finite_fresnel = per_sat["min_fresnel_v"][np.isfinite(per_sat["min_fresnel_v"])]
    score = per_sat["score"]
    return {
        "sat_count": int(sats.shape[0]),
        "los_count": int(np.count_nonzero(los)),
        "nlos_count": int(np.count_nonzero(~los)),
        "utd_candidate_sat_count": int(np.count_nonzero(has_candidate)),
        "utd_candidate_nlos_sat_count": int(np.count_nonzero(has_candidate & nlos)),
        "utd_candidate_count_total": int(counts.sum()),
        "utd_candidate_count_nlos": int(counts[nlos].sum()),
        "utd_min_excess_path_m": float(finite_excess.min()) if finite_excess.size else 0.0,
        "utd_min_edge_distance_m": float(finite_distance.min()) if finite_distance.size else 0.0,
        "utd_min_fresnel_v": float(finite_fresnel.min()) if finite_fresnel.size else 0.0,
        "utd_score_sum": float(score.sum()),
        "utd_score_nlos_sum": float(score[nlos].sum()),
    }
