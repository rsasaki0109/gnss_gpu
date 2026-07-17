from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from gnss_gpu.diffraction import fresnel_integral


GPS_L1_FREQ = 1575.42e6
SPEED_OF_LIGHT = 299792458.0
GPS_L1_WAVELENGTH = SPEED_OF_LIGHT / GPS_L1_FREQ

__all__ = [
    "UTDDiffractionPath",
    "utd_coefficient",
    "fresnel_transition",
    "compute_utd_diffraction_paths",
    "compute_utd_diffraction_paths_gpu",
    "GPS_L1_WAVELENGTH",
]

_UTD_MODE_CODES = {"absorbing": 0, "soft": 1, "hard": 2}


@dataclass
class UTDDiffractionPath:
    excess_delay: float
    diffraction_point: np.ndarray
    edge_id: int
    beta0: float
    phi: float
    phi_p: float
    wedge_n: float
    amplitude: float
    attenuation_db: float


def _scalar_float(value) -> float:
    arr = np.asarray(value)
    if arr.shape == ():
        return float(arr)
    return float(arr.reshape(-1)[0])


def fresnel_transition(x: float) -> complex:
    x = float(x)
    if x == 0.0:
        return 0.0 + 0.0j
    if x < 0.0:
        if x > -1e-15:
            return 0.0 + 0.0j
        raise ValueError("fresnel_transition requires x >= 0")

    u0 = math.sqrt(2.0 * x / math.pi)
    C, S = fresnel_integral(u0)
    C = _scalar_float(C)
    S = _scalar_float(S)

    integral_tail = math.sqrt(math.pi / 2.0) * (
        (0.5 - C) - 1j * (0.5 - S)
    )
    return complex(2j * math.sqrt(x) * np.exp(1j * x) * integral_tail)


def _cot_times_F(arg: float, klA: float, k: float, L: float, n: float) -> complex:
    sin_arg = math.sin(arg)
    if abs(sin_arg) < 1e-7:
        m = int(round(arg / math.pi))
        eps = (arg - m * math.pi) * n
        sgn = 1.0 if eps >= 0.0 else -1.0
        phase = complex(np.exp(1j * math.pi / 4.0))
        root = math.sqrt(max(0.0, 2.0 * math.pi * k * L))
        return n * phase * (root * sgn - 2.0 * k * L * eps * phase)

    return (math.cos(arg) / sin_arg) * fresnel_transition(max(0.0, float(klA)))


def utd_coefficient(
    phi: float,
    phi_p: float,
    beta0: float,
    n: float,
    k: float,
    L: float,
    mode: str = "absorbing",
) -> complex:
    mode_key = str(mode).strip().lower()
    if mode_key not in {"soft", "hard", "absorbing"}:
        raise ValueError("mode must be 'soft', 'hard', or 'absorbing'")

    phi = float(phi)
    phi_p = float(phi_p)
    beta0 = float(beta0)
    n = float(n)
    k = float(k)
    L = float(L)

    if n <= 0.0:
        raise ValueError("n must be positive")
    if k <= 0.0:
        raise ValueError("k must be positive")
    if L < 0.0:
        raise ValueError("L must be non-negative")

    sin_beta0 = math.sin(beta0)
    if abs(sin_beta0) < 1e-15:
        return 0.0 + 0.0j

    prefac = -np.exp(-1j * math.pi / 4.0) / (
        2.0 * n * math.sqrt(2.0 * math.pi * k) * sin_beta0
    )
    kL = k * L

    def term_plus(beta: float) -> complex:
        Nplus = int(round((math.pi + beta) / (2.0 * math.pi * n)))
        aplus = 2.0 * math.cos((2.0 * math.pi * n * Nplus - beta) / 2.0) ** 2
        argp = (math.pi + beta) / (2.0 * n)
        return _cot_times_F(argp, kL * aplus, k, L, n)

    def term_minus(beta: float) -> complex:
        Nminus = int(round((-math.pi + beta) / (2.0 * math.pi * n)))
        aminus = 2.0 * math.cos((2.0 * math.pi * n * Nminus - beta) / 2.0) ** 2
        argm = (math.pi - beta) / (2.0 * n)
        return _cot_times_F(argm, kL * aminus, k, L, n)

    beta_minus = phi - phi_p
    beta_plus = phi + phi_p

    D1 = term_plus(beta_minus) + term_minus(beta_minus)
    D2 = term_plus(beta_plus) + term_minus(beta_plus)

    if mode_key == "soft":
        return complex(prefac * (D1 - D2))
    if mode_key == "hard":
        return complex(prefac * (D1 + D2))
    return complex(prefac * D1)


def _as_point_array(value):
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None

    if arr.size == 0:
        return np.empty((0, 3), dtype=float)
    if arr.ndim == 1:
        if arr.size % 3 != 0:
            return None
        return arr.reshape(-1, 3)
    if arr.ndim == 2 and arr.shape[1] == 3:
        return arr
    if arr.size % 3 == 0:
        return arr.reshape(-1, 3)
    return None


def _as_satellite_array(value) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.size == 0:
        return np.empty((0, 3), dtype=float)
    if arr.ndim == 1:
        if arr.size % 3 != 0:
            raise ValueError("sat_ecef must have shape (3,) or (n_sat, 3)")
        return arr.reshape(-1, 3)
    if arr.ndim == 2 and arr.shape[1] == 3:
        return arr
    if arr.size % 3 == 0:
        return arr.reshape(-1, 3)
    raise ValueError("sat_ecef must have shape (3,) or (n_sat, 3)")


def _as_wedge_array(value) -> np.ndarray:
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return np.empty(0, dtype=float)
    if arr.size == 0:
        return np.empty(0, dtype=float)
    return arr.reshape(-1)


def _edge_count(edges, start: np.ndarray, end: np.ndarray) -> int:
    count = min(start.shape[0], end.shape[0])
    size_attr = getattr(edges, "size", None)
    if size_attr is not None:
        try:
            size_value = size_attr() if callable(size_attr) else size_attr
            count = min(count, max(0, int(size_value)))
        except (TypeError, ValueError):
            pass
    return count


def _norm(vec: np.ndarray) -> float:
    return float(np.linalg.norm(vec))


def _is_finite_vec(vec: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(vec)))


def _clip_scalar(value: float, low: float, high: float) -> float:
    return min(max(float(value), low), high)


def _wedge_n_at(values: np.ndarray, index: int) -> float:
    if values.size == 0:
        return 2.0
    if values.size == 1:
        val = float(values[0])
    elif index < values.size:
        val = float(values[index])
    else:
        return 2.0

    if not math.isfinite(val) or val <= 0.0:
        return 2.0
    return val


def _exterior_angle(
    vec: np.ndarray,
    ehat: np.ndarray,
    uhat: np.ndarray,
    vhat: np.ndarray,
):
    p = vec - float(np.dot(vec, ehat)) * ehat
    plen = _norm(p)
    if plen < 1e-12:
        return None

    p = p / plen
    ang = math.atan2(float(np.dot(p, vhat)), float(np.dot(p, uhat)))
    if not math.isfinite(ang):
        return None
    if ang < 0.0:
        ang += 2.0 * math.pi
    return ang


def compute_utd_diffraction_paths(
    rx_ecef,
    sat_ecef,
    edges,
    *,
    max_paths=4,
    max_edge_range_m=250.0,
    max_ray_edge_distance_m=25.0,
    max_excess_path_m=80.0,
    wavelength_m=GPS_L1_WAVELENGTH,
    mode="absorbing",
) -> list[list[UTDDiffractionPath]]:
    rx = np.asarray(rx_ecef, dtype=float).reshape(3)
    sats = _as_satellite_array(sat_ecef)
    if sats.shape[0] == 0:
        return []

    empty = [[] for _ in range(sats.shape[0])]

    if int(max_paths) <= 0:
        return empty
    if edges is None:
        return empty
    if not all(
        hasattr(edges, name)
        for name in ("start", "end", "face_dir_a", "face_dir_b", "wedge_n")
    ):
        return empty

    wavelength_m = float(wavelength_m)
    if not math.isfinite(wavelength_m) or wavelength_m <= 0.0:
        raise ValueError("wavelength_m must be positive")

    start = _as_point_array(getattr(edges, "start"))
    end = _as_point_array(getattr(edges, "end"))
    face_dir_a = _as_point_array(getattr(edges, "face_dir_a"))
    face_dir_b = _as_point_array(getattr(edges, "face_dir_b"))
    if start is None or end is None or face_dir_a is None or face_dir_b is None:
        return empty

    n_edges = _edge_count(edges, start, end)
    n_edges = min(n_edges, face_dir_a.shape[0], face_dir_b.shape[0])
    if n_edges <= 0:
        return empty

    midpoint_value = getattr(edges, "midpoint", None)
    if callable(midpoint_value):
        try:
            midpoint_value = midpoint_value()
        except TypeError:
            midpoint_value = None
    midpoint = _as_point_array(midpoint_value) if midpoint_value is not None else None

    wedge_n = _as_wedge_array(getattr(edges, "wedge_n"))
    max_paths_i = int(max_paths)
    k = 2.0 * math.pi / wavelength_m
    results: list[list[UTDDiffractionPath]] = []

    for sat in sats:
        if not _is_finite_vec(sat):
            results.append([])
            continue

        ray = sat - rx
        direct_dist = _norm(ray)
        if not math.isfinite(direct_dist) or direct_dist <= 0.0:
            results.append([])
            continue

        direction = ray / direct_dist
        paths: list[UTDDiffractionPath] = []

        for edge_id in range(n_edges):
            p0 = start[edge_id]
            p1 = end[edge_id]
            if not (_is_finite_vec(p0) and _is_finite_vec(p1)):
                continue

            edge_mid = (
                midpoint[edge_id]
                if midpoint is not None and edge_id < midpoint.shape[0]
                else 0.5 * (p0 + p1)
            )
            if not _is_finite_vec(edge_mid):
                continue
            if _norm(edge_mid - rx) > max_edge_range_m:
                continue

            edge_vec = p1 - p0
            c = float(np.dot(edge_vec, edge_vec))
            if not math.isfinite(c) or c <= 1e-18:
                continue

            w = rx - p0
            b = float(np.dot(edge_vec, direction))
            d = float(np.dot(w, direction))
            e = float(np.dot(edge_vec, w))
            denom = c - b * b
            if not math.isfinite(denom) or denom <= 1e-9:
                continue

            t = _clip_scalar((e - b * d) / denom, 0.0, 1.0)
            Q = p0 + t * edge_vec

            along_ray = float(np.dot(Q - rx, direction))
            along_limit = min(float(max_edge_range_m), direct_dist)
            if along_ray <= 0.0 or along_ray >= along_limit:
                continue

            closest_on_ray = rx + along_ray * direction
            h = _norm(Q - closest_on_ray)
            if not math.isfinite(h) or h > max_ray_edge_distance_m:
                continue

            d1 = _norm(Q - rx)
            d2 = _norm(sat - Q)
            if d1 <= 0.0 or d2 <= 0.0:
                continue

            excess = d1 + d2 - direct_dist
            if excess <= 1e-6 or excess > max_excess_path_m:
                continue

            edge_len = math.sqrt(c)
            ehat = edge_vec / edge_len
            edge_dot = abs(float(np.dot(direction, ehat)))
            if not math.isfinite(edge_dot):
                continue

            beta0 = math.acos(_clip_scalar(edge_dot, 0.0, 1.0))
            sin_beta0 = math.sin(beta0)
            if sin_beta0 < 1e-6:
                continue

            fa = face_dir_a[edge_id]
            fb = face_dir_b[edge_id]
            if not (_is_finite_vec(fa) and _is_finite_vec(fb)):
                continue

            uhat = fa - float(np.dot(fa, ehat)) * ehat
            ulen = _norm(uhat)
            if ulen < 1e-12:
                continue
            uhat = uhat / ulen

            v0 = np.cross(ehat, uhat)
            v0_len = _norm(v0)
            if v0_len < 1e-12:
                continue
            v0 = v0 / v0_len

            fb_proj = fb - float(np.dot(fb, ehat)) * ehat
            fb_len = _norm(fb_proj)
            if fb_len > 1e-9:
                fb_proj = fb_proj / fb_len
                sign_value = float(np.dot(fb_proj, v0))
                s = 1.0 if sign_value >= 0.0 else -1.0
            else:
                s = 1.0

            vhat = -s * v0
            phi_p_raw = _exterior_angle(sat - Q, ehat, uhat, vhat)
            phi_raw = _exterior_angle(rx - Q, ehat, uhat, vhat)
            if phi_p_raw is None or phi_raw is None:
                continue

            nn = _wedge_n_at(wedge_n, edge_id)
            upper = nn * math.pi - 1e-6
            if upper <= 1e-6:
                continue

            phi_p = _clip_scalar(phi_p_raw, 1e-6, upper)
            phi = _clip_scalar(phi_raw, 1e-6, upper)
            L = (d1 * d2 * sin_beta0 * sin_beta0) / (d1 + d2)

            D = utd_coefficient(phi, phi_p, beta0, nn, k, L, mode=mode)
            raw_amplitude = abs(D) * math.sqrt(1.0 / d1 + 1.0 / d2)
            if not math.isfinite(raw_amplitude):
                continue

            amplitude = float(min(max(raw_amplitude, 0.0), 1.0))
            attenuation_db = max(
                0.0,
                -20.0 * math.log10(max(amplitude, 1e-12)),
            )

            paths.append(
                UTDDiffractionPath(
                    excess_delay=float(excess),
                    diffraction_point=np.asarray(Q, dtype=float),
                    edge_id=int(edge_id),
                    beta0=float(beta0),
                    phi=float(phi),
                    phi_p=float(phi_p),
                    wedge_n=float(nn),
                    amplitude=amplitude,
                    attenuation_db=float(attenuation_db),
                )
            )

        paths.sort(key=lambda path: path.amplitude, reverse=True)
        results.append(paths[:max_paths_i])

    return results


def compute_utd_diffraction_paths_gpu(
    rx_ecef,
    sat_ecef,
    edges,
    *,
    max_paths=4,
    max_edge_range_m=250.0,
    max_ray_edge_distance_m=25.0,
    max_excess_path_m=80.0,
    wavelength_m=GPS_L1_WAVELENGTH,
    mode="absorbing",
) -> list[list[UTDDiffractionPath]]:
    """GPU-accelerated equivalent of :func:`compute_utd_diffraction_paths`.

    Offloads the O(n_sat * n_edge) edge search and Kouyoumjian-Pathak UTD
    coefficient evaluation to a CUDA kernel, then gathers/sorts the
    surviving candidates on the host. Returns the same structure as the
    pure-Python version (per-satellite lists of :class:`UTDDiffractionPath`,
    strongest first, capped at ``max_paths``).

    Requires the compiled ``_gnss_gpu_diffraction`` extension.
    """
    rx = np.asarray(rx_ecef, dtype=float).reshape(3)
    if not np.all(np.isfinite(rx)):
        raise ValueError("rx_ecef must be finite")

    sats = _as_satellite_array(sat_ecef)
    n_sat = int(sats.shape[0])
    if n_sat == 0:
        return []

    empty = [[] for _ in range(n_sat)]

    if int(max_paths) <= 0:
        return empty
    if edges is None:
        return empty
    if not all(
        hasattr(edges, name)
        for name in ("start", "end", "face_dir_a", "face_dir_b", "wedge_n")
    ):
        return empty

    wavelength_m = float(wavelength_m)
    if not math.isfinite(wavelength_m) or wavelength_m <= 0.0:
        raise ValueError("wavelength_m must be positive")

    mode_key = str(mode).strip().lower()
    if mode_key not in _UTD_MODE_CODES:
        raise ValueError("mode must be 'soft', 'hard', or 'absorbing'")

    start = _as_point_array(getattr(edges, "start"))
    end = _as_point_array(getattr(edges, "end"))
    face_dir_a = _as_point_array(getattr(edges, "face_dir_a"))
    face_dir_b = _as_point_array(getattr(edges, "face_dir_b"))
    if start is None or end is None or face_dir_a is None or face_dir_b is None:
        return empty

    n_edges = _edge_count(edges, start, end)
    n_edges = min(n_edges, face_dir_a.shape[0], face_dir_b.shape[0])
    if n_edges <= 0:
        return empty

    midpoint_value = getattr(edges, "midpoint", None)
    if callable(midpoint_value):
        try:
            midpoint_value = midpoint_value()
        except TypeError:
            midpoint_value = None
    midpoint = _as_point_array(midpoint_value) if midpoint_value is not None else None
    if midpoint is None or midpoint.shape[0] < n_edges:
        midpoint = 0.5 * (start[:n_edges] + end[:n_edges])

    wedge_n = _as_wedge_array(getattr(edges, "wedge_n"))

    start = np.ascontiguousarray(start[:n_edges], dtype=float)
    end = np.ascontiguousarray(end[:n_edges], dtype=float)
    midpoint = np.ascontiguousarray(midpoint[:n_edges], dtype=float)
    face_dir_a = np.ascontiguousarray(face_dir_a[:n_edges], dtype=float)
    face_dir_b = np.ascontiguousarray(face_dir_b[:n_edges], dtype=float)
    wedge_n = np.ascontiguousarray(wedge_n, dtype=float)

    from gnss_gpu._gnss_gpu_diffraction import utd_diffraction_candidates

    (valid, excess, amplitude, beta0_arr, phi_arr, phi_p_arr, wedge_n_arr,
     atten_db, point) = utd_diffraction_candidates(
        np.ascontiguousarray(rx),
        np.ascontiguousarray(sats.reshape(-1)),
        start.reshape(-1), end.reshape(-1), midpoint.reshape(-1),
        face_dir_a.reshape(-1), face_dir_b.reshape(-1), wedge_n,
        n_sat, n_edges,
        float(max_edge_range_m), float(max_ray_edge_distance_m),
        float(max_excess_path_m), wavelength_m,
        _UTD_MODE_CODES[mode_key],
    )

    valid = np.asarray(valid).reshape(n_sat, n_edges).astype(bool)
    excess = np.asarray(excess).reshape(n_sat, n_edges)
    amplitude = np.asarray(amplitude).reshape(n_sat, n_edges)
    beta0_arr = np.asarray(beta0_arr).reshape(n_sat, n_edges)
    phi_arr = np.asarray(phi_arr).reshape(n_sat, n_edges)
    phi_p_arr = np.asarray(phi_p_arr).reshape(n_sat, n_edges)
    wedge_n_arr = np.asarray(wedge_n_arr).reshape(n_sat, n_edges)
    atten_db = np.asarray(atten_db).reshape(n_sat, n_edges)
    point = np.asarray(point).reshape(n_sat, n_edges, 3)

    max_paths_i = int(max_paths)
    results: list[list[UTDDiffractionPath]] = []
    for s in range(n_sat):
        e_idx = np.flatnonzero(valid[s])
        if e_idx.size == 0:
            results.append([])
            continue
        order = e_idx[np.argsort(-amplitude[s, e_idx], kind="stable")][:max_paths_i]
        results.append([
            UTDDiffractionPath(
                excess_delay=float(excess[s, e]),
                diffraction_point=point[s, e].copy(),
                edge_id=int(e),
                beta0=float(beta0_arr[s, e]),
                phi=float(phi_arr[s, e]),
                phi_p=float(phi_p_arr[s, e]),
                wedge_n=float(wedge_n_arr[s, e]),
                amplitude=float(amplitude[s, e]),
                attenuation_db=float(atten_db[s, e]),
            )
            for e in order
        ])
    return results
