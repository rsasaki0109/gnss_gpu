"""Height and phone-position offset helpers for GSDC2023 raw bridge."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from experiments.evaluate import ecef_to_lla, lla_to_ecef
from experiments.gsdc2023_imu import (
    ecef_to_enu_relative,
    enu_to_ecef_relative,
    estimate_rpy_from_velocity,
    wrap_to_180_deg,
)


HEIGHT_LOOP_DIST_M = 15.0
HEIGHT_LOOP_CUMDIST_M = 100.0
HEIGHT_ABSOLUTE_DIST_M = 15.0
HEIGHT_ABSOLUTE_SIGMA_M = 0.1


def mat_get_field(obj: object, *names: str) -> object | None:
    value = obj
    if isinstance(value, np.ndarray) and value.dtype == object and value.size == 1:
        value = value.item()
    for name in names:
        if isinstance(value, dict) and name in value:
            return value[name]
        if isinstance(value, np.ndarray) and value.dtype.names and name in value.dtype.names:
            field = value[name]
            return field.item() if isinstance(field, np.ndarray) and field.size == 1 else field
        if hasattr(value, name):
            return getattr(value, name)
    return None


def numeric_array_from_mat(value: object | None) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value)
    while arr.dtype == object and arr.size == 1:
        arr = np.asarray(arr.item())
    try:
        return np.asarray(arr, dtype=np.float64)
    except (TypeError, ValueError):
        return None


def as_n_by_3(value: object | None) -> np.ndarray | None:
    arr = numeric_array_from_mat(value)
    if arr is None:
        return None
    arr = np.squeeze(arr)
    if arr.ndim == 1 and arr.size == 3:
        return arr.reshape(1, 3)
    if arr.ndim != 2:
        return None
    if arr.shape[1] == 3:
        return np.asarray(arr, dtype=np.float64)
    if arr.shape[0] == 3:
        return np.asarray(arr.T, dtype=np.float64)
    return None


def as_1d_float(value: object | None) -> np.ndarray | None:
    arr = numeric_array_from_mat(value)
    if arr is None:
        return None
    return np.asarray(arr, dtype=np.float64).reshape(-1)


def load_ref_height_mat(course_dir: Path) -> object | None:
    for name in ("ref_hight.mat", "ref_height.mat"):
        path = course_dir / name
        if not path.is_file():
            continue
        try:
            from scipy.io import loadmat
        except ImportError:
            return None
        try:
            raw = loadmat(path, squeeze_me=True, struct_as_record=False)
        except Exception:  # noqa: BLE001
            return None
        return raw.get("posgt", raw)
    return None


def llh_to_ecef_array(llh: np.ndarray) -> np.ndarray | None:
    arr = np.asarray(llh, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 3:
        return None
    lat = arr[:, 0]
    lon = arr[:, 1]
    alt = arr[:, 2]
    if np.nanmax(np.abs(lat)) > np.pi or np.nanmax(np.abs(lon)) > 2.0 * np.pi:
        lat = np.deg2rad(lat)
        lon = np.deg2rad(lon)
    x, y, z = lla_to_ecef(lat, lon, alt)
    return np.column_stack([x, y, z]).astype(np.float64)


def load_absolute_height_reference_ecef(
    course_dir: Path,
    query_xyz_ecef: np.ndarray,
    *,
    dist_m: float = HEIGHT_ABSOLUTE_DIST_M,
) -> tuple[np.ndarray | None, int]:
    """Load MATLAB ``ref_hight.mat`` and map nearest reference up to each epoch."""

    query_xyz = np.asarray(query_xyz_ecef, dtype=np.float64).reshape(-1, 3)
    if query_xyz.size == 0:
        return None, 0
    finite_query = np.isfinite(query_xyz).all(axis=1)
    if not finite_query.any():
        return None, 0
    posgt = load_ref_height_mat(course_dir)
    if posgt is None:
        return None, 0

    origin_xyz = query_xyz[np.flatnonzero(finite_query)[0]]
    ref_xyz = as_n_by_3(
        mat_get_field(posgt, "xyz", "ecef", "posxyz", "pos", "xyz_ecef", "ecef_xyz"),
    )
    if ref_xyz is None:
        ref_llh = as_n_by_3(mat_get_field(posgt, "llh", "lla", "posllh", "llhDeg", "llh_rad"))
        if ref_llh is not None:
            ref_xyz = llh_to_ecef_array(ref_llh)
    ref_enu = as_n_by_3(mat_get_field(posgt, "enu", "posenu"))
    if ref_enu is None and ref_xyz is not None:
        ref_enu = ecef_to_enu_relative(ref_xyz, origin_xyz)
    ref_up = as_1d_float(mat_get_field(posgt, "up", "height", "hight", "alt", "altitude"))
    if ref_up is None and ref_enu is not None:
        ref_up = ref_enu[:, 2]

    if ref_enu is None or ref_up is None:
        return None, 0
    n_ref = min(ref_enu.shape[0], ref_up.size)
    if n_ref <= 0:
        return None, 0
    ref_enu = ref_enu[:n_ref]
    ref_up = ref_up[:n_ref]
    ref_ok = np.isfinite(ref_enu[:, :2]).all(axis=1) & np.isfinite(ref_up)
    if not ref_ok.any():
        return None, 0

    ref_xy = ref_enu[ref_ok, :2]
    ref_up = ref_up[ref_ok]
    query_enu = ecef_to_enu_relative(query_xyz, origin_xyz)
    out_enu = np.full_like(query_enu, np.nan, dtype=np.float64)
    count = 0
    for idx in np.flatnonzero(finite_query):
        if not np.isfinite(query_enu[idx, :2]).all():
            continue
        delta = ref_xy - query_enu[idx, :2]
        dist = np.linalg.norm(delta, axis=1)
        if dist.size == 0:
            continue
        ref_idx = int(np.nanargmin(dist))
        if not np.isfinite(dist[ref_idx]) or dist[ref_idx] >= float(dist_m):
            continue
        out_enu[idx, :2] = query_enu[idx, :2]
        out_enu[idx, 2] = float(ref_up[ref_idx])
        count += 1

    if count == 0:
        return None, 0
    return enu_to_ecef_relative(out_enu, origin_xyz), count


def phone_position_offset(phone: str) -> tuple[float, float] | None:
    phone_l = phone.lower()
    if "mi8" in phone_l:
        return 0.25, -0.35
    if "sm-g988" in phone_l:
        return 0.20, -0.05
    if "sm" in phone_l or "samsung" in phone_l:
        return 0.30, -0.25
    if phone_l == "pixel6pro":
        return -0.20, -0.15
    if "pixel7" in phone_l:
        return -0.10, -0.20
    if "pixel4" in phone_l:
        return -0.00, -0.15
    if "pixel5" in phone_l:
        return -0.10, -0.30
    return None


def apply_phone_position_offset(xyz_ecef: np.ndarray, phone: str) -> np.ndarray:
    xyz = np.asarray(xyz_ecef, dtype=np.float64).reshape(-1, 3)
    offset = phone_position_offset(phone)
    if xyz.size == 0 or offset is None:
        return xyz.copy()
    finite_rows = np.isfinite(xyz).all(axis=1)
    if not finite_rows.any():
        return xyz.copy()
    origin_xyz = xyz[np.flatnonzero(finite_rows)[0]]
    enu = ecef_to_enu_relative(xyz, origin_xyz)
    times_s = np.arange(enu.shape[0], dtype=np.float64)
    vel_enu = np.zeros_like(enu)
    for axis in range(3):
        vel_enu[:, axis] = np.gradient(enu[:, axis], times_s, edge_order=1) if enu.shape[0] > 1 else 0.0
    rpy = estimate_rpy_from_velocity(vel_enu)
    heading = wrap_to_180_deg(np.rad2deg(rpy[:, 2]) - 180.0)
    heading_rad = np.deg2rad(heading)
    offset_rl, offset_ud = offset
    offset_enu = np.column_stack(
        [
            np.cos(heading_rad) * offset_ud - np.sin(heading_rad) * offset_rl,
            np.sin(heading_rad) * offset_ud + np.cos(heading_rad) * offset_rl,
            np.zeros(heading_rad.size, dtype=np.float64),
        ],
    )
    return enu_to_ecef_relative(enu + offset_enu, origin_xyz)


def apply_phone_position_offset_state(state: np.ndarray, phone: str) -> np.ndarray:
    state_arr = np.asarray(state, dtype=np.float64)
    if state_arr.ndim != 2 or state_arr.shape[1] < 3:
        return state_arr.copy()
    out = state_arr.copy()
    out[:, :3] = apply_phone_position_offset(out[:, :3], phone)
    return out


def build_relative_height_groups(
    reference_xyz_ecef: np.ndarray,
    stop_mask: np.ndarray | None = None,
    *,
    loop_dist_m: float = HEIGHT_LOOP_DIST_M,
    loop_cumdist_m: float = HEIGHT_LOOP_CUMDIST_M,
) -> list[np.ndarray]:
    ref_xyz = np.asarray(reference_xyz_ecef, dtype=np.float64).reshape(-1, 3)
    if ref_xyz.shape[0] <= 1:
        return []
    finite_rows = np.isfinite(ref_xyz).all(axis=1)
    if not finite_rows.any():
        return []
    if stop_mask is not None:
        stop_mask_arr = np.asarray(stop_mask, dtype=bool).reshape(-1)
        if stop_mask_arr.size != ref_xyz.shape[0]:
            stop_mask_arr = None
    else:
        stop_mask_arr = None

    origin_xyz = ref_xyz[np.flatnonzero(finite_rows)[0]]
    ref_enu = ecef_to_enu_relative(ref_xyz, origin_xyz)
    horiz = ref_enu[:, :2]
    step = np.linalg.norm(np.diff(horiz, axis=0), axis=1)
    step[~np.isfinite(step)] = 0.0
    cumdist = np.concatenate([[0.0], np.cumsum(step)])

    parent = np.arange(ref_xyz.shape[0], dtype=np.int32)

    def find(idx: int) -> int:
        root = idx
        while parent[root] != root:
            root = int(parent[root])
        while parent[idx] != idx:
            nxt = int(parent[idx])
            parent[idx] = root
            idx = nxt
        return root

    def union(a: int, b: int) -> None:
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    for idx in range(ref_xyz.shape[0] - 1):
        if not finite_rows[idx]:
            continue
        if stop_mask_arr is not None and stop_mask_arr[idx]:
            continue
        delta = horiz[idx + 1 :] - horiz[idx]
        dist = np.linalg.norm(delta, axis=1)
        cumdistdiff = cumdist[idx + 1 :] - cumdist[idx]
        valid = np.isfinite(dist) & (dist < loop_dist_m) & (cumdistdiff > loop_cumdist_m)
        if stop_mask_arr is not None:
            valid &= ~stop_mask_arr[idx + 1 :]
        for other in np.flatnonzero(valid):
            union(idx, idx + 1 + int(other))

    groups: dict[int, list[int]] = {}
    for idx in np.flatnonzero(finite_rows):
        if stop_mask_arr is not None and stop_mask_arr[idx]:
            continue
        root = find(int(idx))
        groups.setdefault(root, []).append(int(idx))
    return [np.asarray(indices, dtype=np.int32) for indices in groups.values() if len(indices) > 1]


def enu_up_ecef_from_origin(origin_xyz: np.ndarray) -> np.ndarray:
    """Unit ENU up axis expressed in ECEF."""

    ox, oy, oz = float(origin_xyz[0]), float(origin_xyz[1]), float(origin_xyz[2])
    lat, lon, _ = ecef_to_lla(ox, oy, oz)
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)
    return np.array([cos_lat * cos_lon, cos_lat * sin_lon, sin_lat], dtype=np.float64)


def relative_height_star_edges_from_groups(groups: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Spanning edges (root -> other) per group for loop-aware relative-height factors."""

    edges_i: list[int] = []
    edges_j: list[int] = []
    for g in groups:
        g = np.asarray(g, dtype=np.int64).ravel()
        if g.size <= 1:
            continue
        root = int(g[0])
        for idx in g[1:]:
            edges_i.append(root)
            edges_j.append(int(idx))
    if not edges_i:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32)
    return np.asarray(edges_i, dtype=np.int32), np.asarray(edges_j, dtype=np.int32)


def relative_height_star_edges_for_reference(
    reference_xyz_ecef: np.ndarray,
    stop_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    groups = build_relative_height_groups(reference_xyz_ecef, stop_mask)
    return relative_height_star_edges_from_groups(groups)


def apply_relative_height_constraint(
    xyz_ecef: np.ndarray,
    reference_xyz_ecef: np.ndarray,
    stop_mask: np.ndarray | None = None,
) -> np.ndarray:
    xyz = np.asarray(xyz_ecef, dtype=np.float64).reshape(-1, 3)
    ref_xyz = np.asarray(reference_xyz_ecef, dtype=np.float64).reshape(-1, 3)
    if xyz.size == 0:
        return xyz.copy()
    groups = build_relative_height_groups(reference_xyz_ecef, stop_mask)
    if not groups:
        return xyz.copy()
    finite_rows = np.isfinite(xyz).all(axis=1)
    ref_finite_rows = np.isfinite(ref_xyz).all(axis=1)
    if not finite_rows.any() or not ref_finite_rows.any():
        return xyz.copy()
    origin_xyz = ref_xyz[np.flatnonzero(ref_finite_rows)[0]]
    enu = ecef_to_enu_relative(xyz, origin_xyz)
    for group in groups:
        group = group[np.isfinite(enu[group, 2])]
        if group.size <= 1:
            continue
        enu[group, 2] = float(np.nanmedian(enu[group, 2]))
    return enu_to_ecef_relative(enu, origin_xyz)
