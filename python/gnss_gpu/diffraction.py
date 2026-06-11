from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np


GPS_L1_FREQ = 1575.42e6
SPEED_OF_LIGHT = 299792458.0
GPS_L1_WAVELENGTH = SPEED_OF_LIGHT / GPS_L1_FREQ

_FRESNEL_STEP = 1.0e-3
_FRESNEL_ASYMPTOTIC_START = 8.0


def _maybe_scalar(value):
    arr = np.asarray(value)
    if arr.ndim == 0:
        return arr.item()
    return value


def fresnel_integral(v) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(v, dtype=float)
    flat = values.ravel()
    sign = np.sign(flat)
    abs_v = np.abs(flat)

    c_flat = np.zeros_like(flat, dtype=float)
    s_flat = np.zeros_like(flat, dtype=float)

    numeric = abs_v <= _FRESNEL_ASYMPTOTIC_START
    if np.any(numeric):
        max_v = float(np.max(abs_v[numeric]))
        if max_v > 0.0:
            n = max(16, int(np.ceil(max_v / _FRESNEL_STEP)))
            x = np.linspace(0.0, max_v, n + 1)
            dx = x[1] - x[0]
            phase = 0.5 * np.pi * x * x

            cos_y = np.cos(phase)
            sin_y = np.sin(phase)

            c_cum = np.zeros_like(x)
            s_cum = np.zeros_like(x)
            c_cum[1:] = np.cumsum(0.5 * dx * (cos_y[:-1] + cos_y[1:]))
            s_cum[1:] = np.cumsum(0.5 * dx * (sin_y[:-1] + sin_y[1:]))

            c_abs = np.interp(abs_v[numeric], x, c_cum)
            s_abs = np.interp(abs_v[numeric], x, s_cum)
            c_flat[numeric] = sign[numeric] * c_abs
            s_flat[numeric] = sign[numeric] * s_abs

    asymptotic = abs_v > _FRESNEL_ASYMPTOTIC_START
    if np.any(asymptotic):
        x = abs_v[asymptotic]
        phase = 0.5 * np.pi * x * x
        c_abs = 0.5 + np.sin(phase) / (np.pi * x)
        s_abs = 0.5 - np.cos(phase) / (np.pi * x)
        c_flat[asymptotic] = sign[asymptotic] * c_abs
        s_flat[asymptotic] = sign[asymptotic] * s_abs

    c = c_flat.reshape(values.shape)
    s = s_flat.reshape(values.shape)
    return _maybe_scalar(c), _maybe_scalar(s)


def knife_edge_loss_db(v) -> Union[float, np.ndarray]:
    values = np.asarray(v, dtype=float)
    x = values - 0.1
    bracket = np.sqrt(x * x + 1.0) + x
    bracket = np.maximum(bracket, np.finfo(float).tiny)
    approx = 6.9 + 20.0 * np.log10(bracket)
    loss = np.where(values <= -0.78, 0.0, approx)
    return _maybe_scalar(loss)


def knife_edge_amplitude(v) -> Union[float, np.ndarray]:
    loss = np.asarray(knife_edge_loss_db(v), dtype=float)
    amplitude = np.clip(10.0 ** (-loss / 20.0), 0.0, 1.0)
    return _maybe_scalar(amplitude)


def diffraction_coefficient_complex(v):
    c, s = fresnel_integral(v)
    coeff = (1.0 + 1.0j) * 0.5 * ((0.5 - c) - 1.0j * (0.5 - s))
    return _maybe_scalar(coeff)


def fresnel_v(
    clearance_m,
    d1_m,
    d2_m,
    wavelength_m: float = GPS_L1_WAVELENGTH,
) -> Union[float, np.ndarray]:
    h = np.maximum(np.asarray(clearance_m, dtype=float), 0.0)
    d1 = np.maximum(np.asarray(d1_m, dtype=float), 1.0e-6)
    d2 = np.maximum(np.asarray(d2_m, dtype=float), 1.0e-6)
    wavelength = max(float(wavelength_m), 1.0e-12)
    v = h * np.sqrt((2.0 / wavelength) * (d1 + d2) / (d1 * d2))
    return _maybe_scalar(v)


def _face_blocks_direct(rx, sat, edge_point, edge_dir, face_dir):
    """True if the semi-infinite wall (edge_dir, face_dir half-plane) occludes the
    straight rx->sat segment, i.e. the receiver is in the geometric shadow of that
    face. The wall is the half-plane {edge_point + a*edge_dir + b*face_dir : b>=0}.
    """
    n = np.cross(edge_dir, face_dir)
    nl = float(np.linalg.norm(n))
    if nl < 1.0e-12:
        return False
    n = n / nl
    dr = float(np.dot(rx - edge_point, n))
    ds = float(np.dot(sat - edge_point, n))
    if dr * ds >= 0.0:  # rx and sat on the same side -> segment never crosses
        return False
    t = dr / (dr - ds)
    crossing = rx + t * (sat - rx)
    return float(np.dot(crossing - edge_point, face_dir)) >= 0.0


def edge_lit_shadow_sign(rx, sat, edge_point, edge_start, edge_end,
                         face_dir_a, face_dir_b=None):
    """Sign of the knife-edge clearance: +1 if the receiver is in the geometric
    shadow of the edge's wall(s) for this satellite (occluded direct ray), -1 if
    it is lit (clear line of sight past the edge).

    This is the lit/shadow distinction the unsigned ``fresnel_v`` (which clamps
    the clearance to >=0) cannot represent: feeding ``sign * fresnel_v`` into
    ``knife_edge_loss_db`` recovers ~0 loss in the lit region (v <= -0.78) while
    keeping the increasing shadow loss for v > 0.
    """
    rx = np.asarray(rx, dtype=float).reshape(3)
    sat = np.asarray(sat, dtype=float).reshape(3)
    edge_point = np.asarray(edge_point, dtype=float).reshape(3)
    edge_dir = np.asarray(edge_end, dtype=float).reshape(3) - np.asarray(
        edge_start, dtype=float).reshape(3)
    edge_len = float(np.linalg.norm(edge_dir))
    if edge_len < 1.0e-12:
        return 1.0
    edge_dir = edge_dir / edge_len

    faces = [np.asarray(face_dir_a, dtype=float).reshape(3)]
    if face_dir_b is not None:
        faces.append(np.asarray(face_dir_b, dtype=float).reshape(3))
    for face in faces:
        if _face_blocks_direct(rx, sat, edge_point, edge_dir, face):
            return 1.0
    return -1.0


@dataclass
class DiffractionPath:
    excess_delay: float
    diffraction_point: np.ndarray
    edge_id: int
    fresnel_v: float
    attenuation_db: float
    amplitude: float


def _ray_segment_closest(rx, direction, edge_start, edge_end):
    edge_vec = edge_end - edge_start
    w = rx[None, :] - edge_start
    b = edge_vec @ direction
    c = np.einsum("ij,ij->i", edge_vec, edge_vec)
    d = w @ direction
    e = np.einsum("ij,ij->i", edge_vec, w)
    denom = c - b * b

    t_edge = np.zeros(edge_start.shape[0], dtype=float)
    valid = denom > 1.0e-9
    t_edge[valid] = (e[valid] - b[valid] * d[valid]) / denom[valid]
    t_edge = np.clip(t_edge, 0.0, 1.0)

    point = edge_start + t_edge[:, None] * edge_vec
    rel = point - rx[None, :]
    ray_distance = rel @ direction
    closest_on_ray = rx[None, :] + ray_distance[:, None] * direction
    distance_to_ray = np.linalg.norm(point - closest_on_ray, axis=1)
    return point, distance_to_ray, ray_distance


def _edge_size(edges) -> int:
    size = getattr(edges, "size", None)
    if size is not None:
        return int(size() if callable(size) else size)

    start = np.asarray(getattr(edges, "start"), dtype=float)
    if start.size == 0:
        return 0
    return int(start.reshape((-1, 3)).shape[0])


def _satellite_array(sat_ecef) -> np.ndarray:
    sats = np.asarray(sat_ecef, dtype=float)
    if sats.size == 0:
        return np.empty((0, 3), dtype=float)
    if sats.ndim == 1:
        if sats.shape[0] != 3:
            raise ValueError("sat_ecef must have shape [3] or [N, 3]")
        return sats.reshape(1, 3)
    if sats.ndim != 2 or sats.shape[1] != 3:
        raise ValueError("sat_ecef must have shape [3] or [N, 3]")
    return sats


def compute_diffraction_paths(
    rx_ecef,
    sat_ecef,
    edges,
    *,
    max_paths: int = 4,
    max_edge_range_m: float = 250.0,
    max_ray_edge_distance_m: float = 25.0,
    max_excess_path_m: float = 80.0,
    wavelength_m: float = GPS_L1_WAVELENGTH,
    lit_shadow: bool = False,
    require_shadow: bool = False,
) -> list[list[DiffractionPath]]:
    """Knife-edge (ITU-R P.526) diffraction candidates per satellite.

    With ``lit_shadow=True`` and an edge set carrying wedge face directions
    (``face_dir_a``/``face_dir_b``), the Fresnel parameter is signed by whether
    the receiver is in the geometric shadow of the edge (positive, attenuating)
    or lit (negative, ~no loss). The default keeps the legacy unsigned behaviour
    (every candidate treated as shadow-side) for backward compatibility.

    ``require_shadow=True`` (implies ``lit_shadow``) additionally discards
    candidate edges whose wall does not occlude the direct ray -- i.e. keeps only
    edges for which the receiver is genuinely in shadow. This is the physically
    correct selection for NLOS diffraction: without it the nearest-to-ray edges
    are frequently lit edges of unrelated buildings, not the silhouette edge that
    actually shadows the satellite.
    """
    rx = np.asarray(rx_ecef, dtype=float).reshape(3)
    sats = _satellite_array(sat_ecef)
    n_sat = int(sats.shape[0])

    if n_sat == 0:
        return []

    edge_count = _edge_size(edges)
    if edge_count == 0 or max_paths <= 0:
        return [[] for _ in range(n_sat)]

    start = np.asarray(edges.start, dtype=float).reshape((-1, 3))
    end = np.asarray(edges.end, dtype=float).reshape((-1, 3))
    face_a_attr = getattr(edges, "face_dir_a", None)
    face_b_attr = getattr(edges, "face_dir_b", None)
    use_sign = (lit_shadow or require_shadow) and face_a_attr is not None
    face_a = (np.asarray(face_a_attr, dtype=float).reshape((-1, 3))
              if use_sign else None)
    face_b = (np.asarray(face_b_attr, dtype=float).reshape((-1, 3))
              if (use_sign and face_b_attr is not None) else None)
    midpoint_attr = getattr(edges, "midpoint", None)
    if midpoint_attr is None:
        midpoint = 0.5 * (start + end)
    else:
        midpoint = np.asarray(midpoint_attr, dtype=float).reshape((-1, 3))

    edge_count = min(edge_count, start.shape[0], end.shape[0], midpoint.shape[0])
    start = start[:edge_count]
    end = end[:edge_count]
    midpoint = midpoint[:edge_count]

    midpoint_range = np.linalg.norm(midpoint - rx[None, :], axis=1)
    pre_mask = midpoint_range <= float(max_edge_range_m)

    result: list[list[DiffractionPath]] = []
    for sat in sats:
        direct_vec = sat - rx
        direct_dist = float(np.linalg.norm(direct_vec))
        if direct_dist <= 1.0e-9 or not np.any(pre_mask):
            result.append([])
            continue

        direction = direct_vec / direct_dist
        edge_indices = np.flatnonzero(pre_mask)
        points, h, along_ray = _ray_segment_closest(
            rx,
            direction,
            start[edge_indices],
            end[edge_indices],
        )

        range_limit = min(float(max_edge_range_m), direct_dist)
        valid = (
            (along_ray > 0.0)
            & (along_ray < range_limit)
            & (h <= float(max_ray_edge_distance_m))
        )

        paths: list[DiffractionPath] = []
        for local_idx in np.flatnonzero(valid):
            point = points[local_idx]
            d1 = float(np.linalg.norm(point - rx))
            d2 = float(np.linalg.norm(sat - point))
            excess = max(0.0, d1 + d2 - direct_dist)
            if excess > float(max_excess_path_m):
                continue

            v = float(fresnel_v(h[local_idx], d1, d2, wavelength_m))
            if use_sign:
                edge_id = int(edge_indices[local_idx])
                fb = None if face_b is None else face_b[edge_id]
                sign = edge_lit_shadow_sign(
                    rx, sat, point, start[edge_id], end[edge_id],
                    face_a[edge_id], fb)
                if require_shadow and sign < 0.0:
                    continue  # lit edge: not the silhouette that shadows this sat
                v *= sign
            attenuation_db = float(knife_edge_loss_db(v))
            amplitude = float(knife_edge_amplitude(v))

            paths.append(
                DiffractionPath(
                    excess_delay=excess,
                    diffraction_point=np.asarray(point, dtype=float).copy(),
                    edge_id=int(edge_indices[local_idx]),
                    fresnel_v=v,
                    attenuation_db=attenuation_db,
                    amplitude=amplitude,
                )
            )

        paths.sort(key=lambda path: path.amplitude, reverse=True)
        result.append(paths[:max_paths])

    return result


def compute_diffraction_paths_gpu(
    rx_ecef,
    sat_ecef,
    edges,
    *,
    max_paths: int = 4,
    max_edge_range_m: float = 250.0,
    max_ray_edge_distance_m: float = 25.0,
    max_excess_path_m: float = 80.0,
    wavelength_m: float = GPS_L1_WAVELENGTH,
) -> list[list[DiffractionPath]]:
    """GPU-accelerated equivalent of :func:`compute_diffraction_paths`.

    Offloads the O(n_sat * n_edge) knife-edge geometry to a CUDA kernel, then
    gathers/sorts the surviving candidates on the host. Returns the same
    structure as the pure-Python version (per-satellite lists of
    :class:`DiffractionPath`, strongest first, capped at ``max_paths``).

    Requires the compiled ``_gnss_gpu_diffraction`` extension.
    """
    from gnss_gpu._gnss_gpu_diffraction import diffraction_candidates

    rx = np.asarray(rx_ecef, dtype=float).reshape(3)
    sats = _satellite_array(sat_ecef)
    n_sat = int(sats.shape[0])
    if n_sat == 0:
        return []

    n_edge = _edge_size(edges)
    if n_edge == 0 or max_paths <= 0:
        return [[] for _ in range(n_sat)]

    start = np.ascontiguousarray(np.asarray(edges.start, dtype=float).reshape(-1, 3))
    end = np.ascontiguousarray(np.asarray(edges.end, dtype=float).reshape(-1, 3))
    midpoint_attr = getattr(edges, "midpoint", None)
    if midpoint_attr is None:
        midpoint = 0.5 * (start + end)
    else:
        midpoint = np.ascontiguousarray(
            np.asarray(midpoint_attr, dtype=float).reshape(-1, 3))

    n_edge = min(n_edge, start.shape[0], end.shape[0], midpoint.shape[0])
    start = start[:n_edge]
    end = end[:n_edge]
    midpoint = midpoint[:n_edge]

    valid, excess, amplitude, fresnel_v_arr, atten_db, point = diffraction_candidates(
        np.ascontiguousarray(rx),
        np.ascontiguousarray(sats.reshape(-1)),
        start.reshape(-1), end.reshape(-1), midpoint.reshape(-1),
        n_sat, n_edge,
        float(max_edge_range_m), float(max_ray_edge_distance_m),
        float(max_excess_path_m), float(wavelength_m),
    )

    valid = np.asarray(valid).reshape(n_sat, n_edge).astype(bool)
    excess = np.asarray(excess).reshape(n_sat, n_edge)
    amplitude = np.asarray(amplitude).reshape(n_sat, n_edge)
    fresnel_v_arr = np.asarray(fresnel_v_arr).reshape(n_sat, n_edge)
    atten_db = np.asarray(atten_db).reshape(n_sat, n_edge)
    point = np.asarray(point).reshape(n_sat, n_edge, 3)

    result: list[list[DiffractionPath]] = []
    for s in range(n_sat):
        e_idx = np.flatnonzero(valid[s])
        if e_idx.size == 0:
            result.append([])
            continue
        order = e_idx[np.argsort(-amplitude[s, e_idx], kind="stable")][:max_paths]
        result.append([
            DiffractionPath(
                excess_delay=float(excess[s, e]),
                diffraction_point=point[s, e].copy(),
                edge_id=int(e),
                fresnel_v=float(fresnel_v_arr[s, e]),
                attenuation_db=float(atten_db[s, e]),
                amplitude=float(amplitude[s, e]),
            )
            for e in order
        ])
    return result


__all__ = [
    "GPS_L1_FREQ",
    "SPEED_OF_LIGHT",
    "GPS_L1_WAVELENGTH",
    "DiffractionPath",
    "fresnel_integral",
    "knife_edge_loss_db",
    "knife_edge_amplitude",
    "diffraction_coefficient_complex",
    "fresnel_v",
    "edge_lit_shadow_sign",
    "compute_diffraction_paths",
    "compute_diffraction_paths_gpu",
]
