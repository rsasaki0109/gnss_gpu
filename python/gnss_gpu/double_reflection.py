from __future__ import annotations

from dataclasses import dataclass

import numpy as np

plane_eps = 1e-6
area_eps = 1e-12
param_eps = 1e-12
bary_tol = 1e-9
delay_eps = 1e-6
intersect_eps = 1e-6


@dataclass
class DoubleReflectionPath:
    excess_delay: float
    points: list
    triangle_ids: tuple
    normals: list
    incidence_angles: tuple


def _norm(vector):
    return float(np.linalg.norm(vector))


def _coerce_triangles(triangles):
    tri_array = np.asarray(triangles, dtype=float)
    if tri_array.size == 0:
        return np.empty((0, 3, 3), dtype=float)
    if tri_array.ndim != 3 or tri_array.shape[1:] != (3, 3):
        raise ValueError("triangles must have shape (N, 3, 3)")
    return tri_array


def _coerce_rx(rx_ecef):
    rx = np.asarray(rx_ecef, dtype=float)
    if rx.shape == (3,):
        return rx
    if rx.size == 3:
        return rx.reshape(3)
    raise ValueError("rx_ecef must have shape (3,)")


def _coerce_satellites(sat_ecef):
    sat_array = np.asarray(sat_ecef, dtype=float)
    if sat_array.size == 0:
        return np.empty((0, 3), dtype=float)
    if sat_array.shape == (3,):
        return sat_array.reshape(1, 3)
    if sat_array.ndim == 2 and sat_array.shape[1] == 3:
        return sat_array
    raise ValueError("sat_ecef must have shape (3,) or (n_sat, 3)")


def _triangle_normal(triangle):
    p0, p1, p2 = triangle
    normal = np.cross(p1 - p0, p2 - p0)
    normal_len = _norm(normal)
    if normal_len <= area_eps:
        return None
    return normal / normal_len


def _cross3_float64(a, b):
    """Original component multiply/subtract order for bounded FP64 vectors."""
    return np.array((
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ), dtype=np.float64)


def _point_in_triangle(point, triangle, geometry=None):
    if geometry is None:
        a, b, c = triangle
        v0 = b - a
        v1 = c - a
    else:
        a, v0, v1 = geometry
    v2 = point - a

    d00 = v0 @ v0
    d01 = v0 @ v1
    d02 = v0 @ v2
    d11 = v1 @ v1
    d12 = v1 @ v2

    denom = d00 * d11 - d01 * d01
    if abs(denom) <= area_eps:
        return False

    inv = 1.0 / denom
    u = (d11 * d02 - d01 * d12) * inv
    v = (d00 * d12 - d01 * d02) * inv
    return bool(u >= -bary_tol and v >= -bary_tol and (u + v) <= 1.0 + bary_tol)


def _segment_intersects_triangle(start, end, triangle, geometry=None, segment=None):
    if segment is None:
        direction = end - start
        seg_len = _norm(direction)
    else:
        direction, seg_len = segment
    if seg_len <= intersect_eps:
        return False

    if geometry is None:
        v0, v1, v2 = triangle
        edge1 = v1 - v0
        edge2 = v2 - v0
        normal_len = _norm(np.cross(edge1, edge2))
        cross = np.cross
    else:
        v0, edge1, edge2, normal_len = geometry
        cross = _cross3_float64
    if normal_len <= area_eps:
        return False

    pvec = cross(direction, edge2)
    det = edge1 @ pvec
    if abs(det) <= area_eps:
        return False

    inv_det = 1.0 / det
    tvec = start - v0
    u = (tvec @ pvec) * inv_det
    if u < -bary_tol or u > 1.0 + bary_tol:
        return False

    qvec = cross(tvec, edge1)
    v = (direction @ qvec) * inv_det
    if v < -bary_tol or (u + v) > 1.0 + bary_tol:
        return False

    t = (edge2 @ qvec) * inv_det
    endpoint_tol = intersect_eps / seg_len
    return bool(endpoint_tol < t < 1.0 - endpoint_tol)


def _segment_blocked(start, end, tri_array, excluded_ids, invariants=None):
    segment = None
    for oid, obstacle in enumerate(tri_array):
        if oid in excluded_ids:
            continue
        if invariants is None:
            blocked = _segment_intersects_triangle(start, end, obstacle)
        else:
            if segment is None:
                direction = end - start
                segment = (direction, _norm(direction))
            # Preserve the original short-segment rejection before geometry.
            geometry = None
            if segment[1] > intersect_eps:
                geometry = invariants.obstacle_geometry(oid)
            blocked = _segment_intersects_triangle(start, end, obstacle, geometry, segment)
        if blocked:
            return True
    return False


def _ordered_pairs(n_tri, max_pairs):
    if max_pairs is None:
        pair_limit = None
    else:
        pair_limit = int(max_pairs)
        if pair_limit <= 0:
            return []

    pairs = []
    count = 0
    for tri_a_id in range(n_tri):
        for tri_b_id in range(n_tri):
            if tri_a_id == tri_b_id:
                continue
            if pair_limit is not None and count >= pair_limit:
                return pairs
            pairs.append((tri_a_id, tri_b_id))
            count += 1
    return pairs


def _unit(vector):
    length = _norm(vector)
    if length <= intersect_eps:
        return None
    return vector / length


class _ImageInvariants:
    """Bounded call-local arrays; every original ordered pair is still visited."""

    __slots__ = ("triangles", "vertices", "receiver_distances", "receiver_images",
                 "pair_images", "satellite_distances", "receiver_numerators",
                 "pair_numerators", "edges", "obstacles")

    def __init__(self, triangles, pair_count):
        self.triangles = list(triangles)
        self.vertices = [triangle[0] for triangle in self.triangles]
        self.receiver_distances = [None] * len(triangles)
        self.receiver_images = [None] * len(triangles)
        self.pair_images = [None] * pair_count
        self.satellite_distances = [None] * len(triangles)
        self.receiver_numerators = [None] * len(triangles)
        self.pair_numerators = [None] * pair_count
        self.edges = [None] * len(triangles)
        self.obstacles = [None] * len(triangles)

    def triangle_edges(self, index):
        cached = self.edges[index]
        if cached is None:
            a, b, c = self.triangles[index]
            cached = (a, b - a, c - a)
            self.edges[index] = cached
        return cached

    def obstacle_geometry(self, index):
        cached = self.obstacles[index]
        if cached is None:
            a, edge1, edge2 = self.triangle_edges(index)
            cached = (a, edge1, edge2, _norm(np.cross(edge1, edge2)))
            self.obstacles[index] = cached  # pyright: ignore[reportArgumentType, reportCallIssue]
        return cached


def _compute_single_path(tri_array, normals, rx, sat, direct_len, tri_a_id, tri_b_id,
                         invariants=None, pair_index=None):
    if invariants is None:
        tri_a = tri_array[tri_a_id]
        tri_b = tri_array[tri_b_id]
    else:
        tri_a = invariants.triangles[tri_a_id]
        tri_b = invariants.triangles[tri_b_id]
    na = normals[tri_a_id]
    nb = normals[tri_b_id]
    if na is None or nb is None:
        return None

    if invariants is None:
        a0 = tri_a[0]
        b0 = tri_b[0]
        rx_plane_distance = float(np.dot(rx - a0, na))
        sat_plane_distance = float(np.dot(sat - b0, nb))
    else:
        a0 = invariants.vertices[tri_a_id]
        b0 = invariants.vertices[tri_b_id]
        rx_plane_distance = invariants.receiver_distances[tri_a_id]
        if rx_plane_distance is None:
            rx_plane_distance = float(np.dot(rx - a0, na))
            invariants.receiver_distances[tri_a_id] = rx_plane_distance
        sat_plane_distance = invariants.satellite_distances[tri_b_id]
        if sat_plane_distance is None:
            sat_plane_distance = float(np.dot(sat - b0, nb))
            invariants.satellite_distances[tri_b_id] = sat_plane_distance
    if abs(rx_plane_distance) <= plane_eps or abs(sat_plane_distance) <= plane_eps:
        return None

    if invariants is None:
        i1 = rx - 2.0 * rx_plane_distance * na
        i12 = i1 - 2.0 * float(np.dot(i1 - b0, nb)) * nb
    else:
        i1 = invariants.receiver_images[tri_a_id]
        if i1 is None:
            i1 = rx - 2.0 * rx_plane_distance * na
            invariants.receiver_images[tri_a_id] = i1
        i12 = invariants.pair_images[pair_index]
        if i12 is None:
            i12 = i1 - 2.0 * float(np.dot(i1 - b0, nb)) * nb
            invariants.pair_images[pair_index] = i12

    if invariants is None:
        denom2 = float(np.dot(sat - i12, nb))
    else:
        sat_delta = sat - i12
        denom2 = float(np.dot(sat_delta, nb))
    if abs(denom2) <= plane_eps:
        return None
    if invariants is None:
        t2 = float(np.dot(b0 - i12, nb) / denom2)
    else:
        numerator2 = invariants.pair_numerators[pair_index]
        if numerator2 is None:
            numerator2 = np.dot(b0 - i12, nb)
            invariants.pair_numerators[pair_index] = numerator2
        t2 = float(numerator2 / denom2)
    if not np.isfinite(t2) or t2 <= param_eps or t2 >= 1.0 - param_eps:
        return None
    if invariants is None:
        p2 = i12 + t2 * (sat - i12)
        denom1 = float(np.dot(p2 - i1, na))
    else:
        p2 = i12 + t2 * sat_delta
        first_delta = p2 - i1
        denom1 = float(np.dot(first_delta, na))
    if abs(denom1) <= plane_eps:
        return None
    if invariants is None:
        t1 = float(np.dot(a0 - i1, na) / denom1)
    else:
        numerator1 = invariants.receiver_numerators[tri_a_id]
        if numerator1 is None:
            numerator1 = np.dot(a0 - i1, na)
            invariants.receiver_numerators[tri_a_id] = numerator1
        t1 = float(numerator1 / denom1)
    if not np.isfinite(t1) or t1 <= param_eps or t1 >= 1.0 - param_eps:
        return None
    if invariants is None:
        p1 = i1 + t1 * (p2 - i1)
    else:
        p1 = i1 + t1 * first_delta

    if invariants is None:
        if not _point_in_triangle(p1, tri_a):
            return None
        if not _point_in_triangle(p2, tri_b):
            return None
    else:
        if not _point_in_triangle(p1, tri_a, invariants.triangle_edges(tri_a_id)):
            return None
        if not _point_in_triangle(p2, tri_b, invariants.triangle_edges(tri_b_id)):
            return None

    excluded_ids = {tri_a_id, tri_b_id}
    if invariants is None:
        if _segment_blocked(rx, p1, tri_array, excluded_ids):
            return None
        if _segment_blocked(p1, p2, tri_array, excluded_ids):
            return None
        if _segment_blocked(p2, sat, tri_array, excluded_ids):
            return None
    else:
        if _segment_blocked(rx, p1, tri_array, excluded_ids, invariants):
            return None
        if _segment_blocked(p1, p2, tri_array, excluded_ids, invariants):
            return None
        if _segment_blocked(p2, sat, tri_array, excluded_ids, invariants):
            return None

    seg1 = _norm(p1 - rx)
    seg2 = _norm(p2 - p1)
    seg3 = _norm(sat - p2)
    path_len = seg1 + seg2 + seg3
    excess_delay = float(path_len - direct_len)
    if not np.isfinite(excess_delay) or excess_delay <= delay_eps:
        return None

    u1 = _unit(p1 - rx)
    u2 = _unit(p2 - p1)
    if u1 is None or u2 is None:
        return None

    inc1 = float(np.arccos(np.clip(abs(float(np.dot(u1, na))), 0.0, 1.0)))
    inc2 = float(np.arccos(np.clip(abs(float(np.dot(u2, nb))), 0.0, 1.0)))

    return DoubleReflectionPath(
        excess_delay=excess_delay,
        points=[p1.copy(), p2.copy()],
        triangle_ids=(int(tri_a_id), int(tri_b_id)),
        normals=[na.copy(), nb.copy()],
        incidence_angles=(inc1, inc2),
    )


_IMAGE_HELPERS = (_compute_single_path, _triangle_normal, _norm, _ordered_pairs,
                  _point_in_triangle, _segment_blocked, _segment_intersects_triangle, _cross3_float64)


def compute_double_reflection_paths(
    triangles,
    rx_ecef,
    sat_ecef,
    max_paths=4,
    max_pairs=None,
):
    tri_array = _coerce_triangles(triangles)
    rx = _coerce_rx(rx_ecef)
    sat_array = _coerce_satellites(sat_ecef)

    n_sat = sat_array.shape[0]
    if n_sat == 0:
        return []

    if max_paths is None:
        path_limit = None
    else:
        path_limit = int(max_paths)
        if path_limit <= 0:
            return [[] for _ in range(n_sat)]

    if tri_array.shape[0] < 2:
        return [[] for _ in range(n_sat)]

    pairs = _ordered_pairs(tri_array.shape[0], max_pairs)
    if not pairs:
        return [[] for _ in range(n_sat)]

    normals = [_triangle_normal(triangle) for triangle in tri_array]

    # Keep exceptional inputs/policies and replaced helper hooks in their
    # original evaluation order. Finite bounded vectors keep mirror arithmetic
    # safe; prepare each original expression only on its first actual use.
    invariants = None
    policy = np.geterr()
    if (n_sat >= 2 and 8 <= len(tri_array) <= 256
            and 64 <= len(pairs) <= 32768 and n_sat * len(pairs) >= 1024
            and all(current is original for current, original in zip(
                (_compute_single_path, _triangle_normal, _norm, _ordered_pairs,
                 _point_in_triangle, _segment_blocked, _segment_intersects_triangle, _cross3_float64), _IMAGE_HELPERS))
            and area_eps == 1e-12 and plane_eps == 1e-6
            and type(tri_array) is np.ndarray and tri_array.dtype == np.float64
            and type(rx) is np.ndarray and type(sat_array) is np.ndarray
            and policy["under"] == "ignore"
            and all(mode in ("warn", "ignore") for mode in policy.values())
            and np.all(np.isfinite(tri_array)) and np.all(np.isfinite(rx))
            and np.all(np.isfinite(sat_array))
            and np.max(np.abs(tri_array)) <= 1e50 and np.max(np.abs(rx)) <= 1e50
            and np.max(np.abs(sat_array)) <= 1e50):
        invariants = _ImageInvariants(tri_array, len(pairs))

    results = []
    for sat in sat_array:
        direct_len = _norm(sat - rx)
        if invariants is not None:
            invariants.satellite_distances = [None] * len(tri_array)
        paths = []
        for pair_index, (tri_a_id, tri_b_id) in enumerate(pairs):
            if invariants is None:
                path = _compute_single_path(
                    tri_array, normals, rx, sat, direct_len, tri_a_id, tri_b_id)
            else:
                path = _compute_single_path(
                    tri_array, normals, rx, sat, direct_len, tri_a_id, tri_b_id,
                    invariants, pair_index)
            if path is not None:
                paths.append(path)

        paths.sort(key=lambda path: path.excess_delay)
        if path_limit is not None:
            paths = paths[:path_limit]
        results.append(paths)

    return results


__all__ = ["DoubleReflectionPath", "compute_double_reflection_paths"]
