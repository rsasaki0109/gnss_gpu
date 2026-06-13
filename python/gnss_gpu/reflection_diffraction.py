from __future__ import annotations

from dataclasses import dataclass

import numpy as np

GPS_L1_FREQ = 1575.42e6
SPEED_OF_LIGHT = 299792458.0
GPS_L1_WAVELENGTH = SPEED_OF_LIGHT / GPS_L1_FREQ

area_eps = 1e-12
param_eps = 1e-12
bary_tol = 1e-9
delay_eps = 1e-6


@dataclass
class ReflectionDiffractionPath:
    excess_delay: float
    order: str
    reflection_point: np.ndarray
    diffraction_point: np.ndarray
    triangle_id: int
    edge_id: int
    incidence_angle: float
    fresnel_v: float
    attenuation_db: float
    amplitude: float


def _as_satellites(sat_ecef) -> np.ndarray:
    sats = np.asarray(sat_ecef, dtype=float)
    if sats.size == 0:
        return np.empty((0, 3), dtype=float)
    if sats.shape == (3,):
        return sats.reshape(1, 3)
    if sats.ndim != 2 or sats.shape[1] != 3:
        raise ValueError("sat_ecef must have shape (3,) or (n_sat, 3)")
    return sats


def _edge_size(edges) -> int:
    size = getattr(edges, "size", None)
    if callable(size):
        return int(size())
    if size is not None:
        return int(size)
    return int(np.asarray(edges.start).shape[0])


def _coerce_edges(edges):
    if edges is None:
        return None, None, None, 0

    start = np.asarray(edges.start, dtype=float)
    end = np.asarray(edges.end, dtype=float)

    if start.size == 0 or end.size == 0:
        return start.reshape(0, 3), end.reshape(0, 3), np.empty((0, 3), dtype=float), 0

    if start.ndim != 2 or start.shape[1] != 3 or end.ndim != 2 or end.shape[1] != 3:
        raise ValueError("edges.start and edges.end must have shape (n_edge, 3)")

    n_edge = min(_edge_size(edges), start.shape[0], end.shape[0])
    start = start[:n_edge]
    end = end[:n_edge]

    midpoint_attr = getattr(edges, "midpoint", None)
    if midpoint_attr is None:
        midpoint = 0.5 * (start + end)
    else:
        midpoint = np.asarray(midpoint_attr, dtype=float)
        if midpoint.ndim != 2 or midpoint.shape[1] != 3:
            raise ValueError("edges.midpoint must have shape (n_edge, 3)")
        midpoint = midpoint[:n_edge]

    return start, end, midpoint, n_edge


def _mirror(point: np.ndarray, plane_point: np.ndarray, unit_normal: np.ndarray) -> np.ndarray:
    return point - 2.0 * np.dot(point - plane_point, unit_normal) * unit_normal


def _unit_or_none(vec: np.ndarray, eps: float = 1e-12):
    norm = float(np.linalg.norm(vec))
    if norm <= eps:
        return None, norm
    return vec / norm, norm


def _point_in_triangle(
    point: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    tol: float = bary_tol,
) -> bool:
    v0 = p2 - p0
    v1 = p1 - p0
    v2 = point - p0

    dot00 = float(np.dot(v0, v0))
    dot01 = float(np.dot(v0, v1))
    dot02 = float(np.dot(v0, v2))
    dot11 = float(np.dot(v1, v1))
    dot12 = float(np.dot(v1, v2))

    denom = dot00 * dot11 - dot01 * dot01
    if abs(denom) <= area_eps:
        return False

    inv_denom = 1.0 / denom
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom

    return u >= -tol and v >= -tol and (u + v) <= 1.0 + tol


def _diffraction_geometry(
    source: np.ndarray,
    target: np.ndarray,
    edge_start: np.ndarray,
    edge_end: np.ndarray,
):
    ray = target - source
    direction, ray_len = _unit_or_none(ray)
    if direction is None:
        return None

    edge_vec = edge_end - edge_start
    w = source - edge_start

    b = float(np.dot(edge_vec, direction))
    c = float(np.dot(edge_vec, edge_vec))
    d = float(np.dot(w, direction))
    e = float(np.dot(edge_vec, w))

    denom = c - b * b
    if denom <= 1e-9:
        return None

    t_edge = (e - b * d) / denom
    t_edge = float(np.clip(t_edge, 0.0, 1.0))

    point = edge_start + t_edge * edge_vec
    rel = point - source
    along_ray = float(np.dot(rel, direction))
    closest_on_ray = source + along_ray * direction
    h = float(np.linalg.norm(point - closest_on_ray))
    d1 = float(np.linalg.norm(point - source))
    d2 = float(np.linalg.norm(target - point))

    return point, h, along_ray, d1, d2, ray_len


def fresnel_v(h: float, d1: float, d2: float, wavelength: float) -> float:
    h = max(float(h), 0.0)
    d1 = max(float(d1), 1e-6)
    d2 = max(float(d2), 1e-6)
    return h * float(np.sqrt((2.0 / wavelength) * (d1 + d2) / (d1 * d2)))


def knife_edge_loss_db(v: float) -> float:
    x = float(v) - 0.1
    tiny = np.finfo(float).tiny
    bracket = max(float(np.sqrt(x * x + 1.0) + x), tiny)
    approx = 6.9 + 20.0 * float(np.log10(bracket))
    return 0.0 if v <= -0.78 else approx


def knife_edge_amplitude(v: float) -> float:
    loss = knife_edge_loss_db(v)
    return float(np.clip(10.0 ** (-loss / 20.0), 0.0, 1.0))


def _build_rd_path(
    rx: np.ndarray,
    sat: np.ndarray,
    tri: np.ndarray,
    normal: np.ndarray,
    edge_start: np.ndarray,
    edge_end: np.ndarray,
    triangle_id: int,
    edge_id: int,
    direct_dist: float,
    max_edge_range_m: float,
    max_ray_edge_distance_m: float,
    max_excess_path_m: float,
    wavelength_m: float,
):
    p0, p1, p2 = tri
    source = _mirror(rx, p0, normal)
    target = sat

    geom = _diffraction_geometry(source, target, edge_start, edge_end)
    if geom is None:
        return None

    diffraction_point, h, along_ray, d1, d2, ray_len = geom
    if along_ray <= 0.0 or along_ray >= min(max_edge_range_m, ray_len):
        return None
    if h > max_ray_edge_distance_m:
        return None

    seg = diffraction_point - source
    denom_r = float(np.dot(seg, normal))
    if abs(denom_r) <= 1e-9:
        return None

    t_refl = float(np.dot(p0 - source, normal) / denom_r)
    if not (param_eps < t_refl < 1.0 - param_eps):
        return None

    reflection_point = source + t_refl * seg
    if not _point_in_triangle(reflection_point, p0, p1, p2):
        return None

    excess = d1 + d2 - direct_dist
    if excess <= delay_eps or excess > max_excess_path_m:
        return None

    direction, _ = _unit_or_none(seg)
    if direction is None:
        return None

    incidence_angle = float(np.arccos(np.clip(abs(float(np.dot(direction, normal))), 0.0, 1.0)))
    v = fresnel_v(h, d1, d2, wavelength_m)
    loss = knife_edge_loss_db(v)
    amplitude = knife_edge_amplitude(v)

    return ReflectionDiffractionPath(
        excess_delay=float(excess),
        order="RD",
        reflection_point=reflection_point,
        diffraction_point=diffraction_point,
        triangle_id=int(triangle_id),
        edge_id=int(edge_id),
        incidence_angle=incidence_angle,
        fresnel_v=float(v),
        attenuation_db=float(loss),
        amplitude=amplitude,
    )


def _build_dr_path(
    rx: np.ndarray,
    sat: np.ndarray,
    tri: np.ndarray,
    normal: np.ndarray,
    edge_start: np.ndarray,
    edge_end: np.ndarray,
    triangle_id: int,
    edge_id: int,
    direct_dist: float,
    max_edge_range_m: float,
    max_ray_edge_distance_m: float,
    max_excess_path_m: float,
    wavelength_m: float,
):
    p0, p1, p2 = tri
    source = rx
    target = _mirror(sat, p0, normal)

    geom = _diffraction_geometry(source, target, edge_start, edge_end)
    if geom is None:
        return None

    diffraction_point, h, along_ray, d1, d2, ray_len = geom
    if along_ray <= 0.0 or along_ray >= min(max_edge_range_m, ray_len):
        return None
    if h > max_ray_edge_distance_m:
        return None

    seg = target - diffraction_point
    denom_r = float(np.dot(seg, normal))
    if abs(denom_r) <= 1e-9:
        return None

    t_refl = float(np.dot(p0 - diffraction_point, normal) / denom_r)
    if not (param_eps < t_refl < 1.0 - param_eps):
        return None

    reflection_point = diffraction_point + t_refl * seg
    if not _point_in_triangle(reflection_point, p0, p1, p2):
        return None

    excess = d1 + d2 - direct_dist
    if excess <= delay_eps or excess > max_excess_path_m:
        return None

    direction, _ = _unit_or_none(seg)
    if direction is None:
        return None

    incidence_angle = float(np.arccos(np.clip(abs(float(np.dot(direction, normal))), 0.0, 1.0)))
    v = fresnel_v(h, d1, d2, wavelength_m)
    loss = knife_edge_loss_db(v)
    amplitude = knife_edge_amplitude(v)

    return ReflectionDiffractionPath(
        excess_delay=float(excess),
        order="DR",
        reflection_point=reflection_point,
        diffraction_point=diffraction_point,
        triangle_id=int(triangle_id),
        edge_id=int(edge_id),
        incidence_angle=incidence_angle,
        fresnel_v=float(v),
        attenuation_db=float(loss),
        amplitude=amplitude,
    )


def compute_reflection_diffraction_paths(
    triangles,
    edges,
    rx_ecef,
    sat_ecef,
    *,
    max_paths: int = 4,
    max_edge_range_m: float = 250.0,
    max_ray_edge_distance_m: float = 25.0,
    max_excess_path_m: float = 120.0,
    max_triangles: int | None = None,
    wavelength_m: float = GPS_L1_WAVELENGTH,
    orders=("RD", "DR"),
) -> list[list[ReflectionDiffractionPath]]:
    sats = _as_satellites(sat_ecef)
    if sats.shape[0] == 0:
        return []

    if max_paths <= 0 or triangles is None or edges is None:
        return [[] for _ in range(sats.shape[0])]

    tris = np.asarray(triangles, dtype=float)
    if tris.size == 0:
        return [[] for _ in range(sats.shape[0])]
    if tris.ndim != 3 or tris.shape[1:] != (3, 3):
        raise ValueError("triangles must have shape (n_tri, 3, 3)")

    if max_triangles is not None:
        tris = tris[: max(0, int(max_triangles))]
    if tris.shape[0] == 0:
        return [[] for _ in range(sats.shape[0])]

    edge_start, edge_end, edge_midpoint, n_edge = _coerce_edges(edges)
    if n_edge == 0:
        return [[] for _ in range(sats.shape[0])]

    rx = np.asarray(rx_ecef, dtype=float)
    if rx.shape != (3,):
        raise ValueError("rx_ecef must have shape (3,)")

    order_set = {str(order).upper() for order in orders}
    use_rd = "RD" in order_set
    use_dr = "DR" in order_set
    if not use_rd and not use_dr:
        return [[] for _ in range(sats.shape[0])]

    edge_range = np.linalg.norm(edge_midpoint - rx, axis=1)
    candidate_edges = np.nonzero(edge_range <= max_edge_range_m)[0]
    if candidate_edges.size == 0:
        return [[] for _ in range(sats.shape[0])]

    results: list[list[ReflectionDiffractionPath]] = []

    for sat in sats:
        direct_dist = float(np.linalg.norm(sat - rx))
        sat_paths: list[ReflectionDiffractionPath] = []

        for triangle_id, tri in enumerate(tris):
            p0, p1, p2 = tri
            normal_vec = np.cross(p1 - p0, p2 - p0)
            normal_len = float(np.linalg.norm(normal_vec))
            if normal_len <= area_eps:
                continue
            normal = normal_vec / normal_len

            for edge_id in candidate_edges:
                start = edge_start[edge_id]
                end = edge_end[edge_id]

                if use_rd:
                    path = _build_rd_path(
                        rx,
                        sat,
                        tri,
                        normal,
                        start,
                        end,
                        triangle_id,
                        int(edge_id),
                        direct_dist,
                        max_edge_range_m,
                        max_ray_edge_distance_m,
                        max_excess_path_m,
                        wavelength_m,
                    )
                    if path is not None:
                        sat_paths.append(path)

                if use_dr:
                    path = _build_dr_path(
                        rx,
                        sat,
                        tri,
                        normal,
                        start,
                        end,
                        triangle_id,
                        int(edge_id),
                        direct_dist,
                        max_edge_range_m,
                        max_ray_edge_distance_m,
                        max_excess_path_m,
                        wavelength_m,
                    )
                    if path is not None:
                        sat_paths.append(path)

        sat_paths.sort(key=lambda p: (-p.amplitude, p.excess_delay, p.triangle_id, p.edge_id, p.order))
        results.append(sat_paths[:max_paths])

    return results


__all__ = [
    "ReflectionDiffractionPath",
    "compute_reflection_diffraction_paths",
    "GPS_L1_WAVELENGTH",
]
