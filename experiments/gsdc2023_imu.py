"""IMU preprocessing and preintegration helpers for GSDC2023.

This module owns device_imu parsing, IMU/GNSS time alignment, stop detection,
and the lightweight delta preintegration used by the raw bridge.  It avoids
importing ``gsdc2023_raw_bridge`` so the IMU path can be tested independently.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.evaluate import ecef_to_lla


DEVICE_IMU_COLUMNS = [
    "MessageType",
    "utcTimeMillis",
    "elapsedRealtimeNanos",
    "MeasurementX",
    "MeasurementY",
    "MeasurementZ",
    "BiasX",
    "BiasY",
    "BiasZ",
]
IMU_SYNC_MODE = "gyro"
IMU_SYNC_COEFFICIENT = 0.5
ACC_TIME_OFFSET_MS = -20.0
GYRO_TIME_OFFSET_MS = -20.0
STOP_WINDOW_SIZE = 500
STOP_ACC_STD_OFFSET = 0.08
STOP_GYRO_STD_OFFSET = 0.005
STOP_GYRO_MAX = 0.05
VELOCITY_SMOOTH_WINDOW = 20
VELOCITY_THRESHOLD_MPS = 0.5
IMU_GRAVITY_MPS2 = 9.80665
IMU_MOUNTING_ANGLE_RAD = np.deg2rad(np.array([-85.0, 178.0, -94.0], dtype=np.float64))
IMU_TAROZ_BODY_DELTA_FRAME = "taroz_body"
IMU_DELTA_FRAMES = ("body", "ecef", IMU_TAROZ_BODY_DELTA_FRAME)
IMU_SAMPLE_DT_MODES = ("bounded", "taroz")
IMU_ACCEL_BIAS_PRIOR_SIGMA_MPS2 = 10.0
IMU_ACCEL_BIAS_BETWEEN_SIGMA_MPS2 = 1.0

RawCsvReader = Callable[..., pd.DataFrame]


@dataclass(frozen=True)
class IMUMeasurements:
    times_ms: np.ndarray
    elapsed_ns: np.ndarray
    xyz: np.ndarray
    bias: np.ndarray


@dataclass(frozen=True)
class ProcessedIMU:
    times_ms: np.ndarray
    xyz: np.ndarray
    dt_s: np.ndarray
    norm_3d: np.ndarray
    norm_std: np.ndarray
    sync_coefficient: float
    bias: np.ndarray | None = None


@dataclass(frozen=True)
class IMUPreintegration:
    epoch_times_ms: np.ndarray
    delta_t_s: np.ndarray
    delta_v_body: np.ndarray
    delta_p_body: np.ndarray
    delta_angle_rad: np.ndarray
    sample_count: np.ndarray
    delta_frame: str = "body"
    acc_bias_mean_sensor: np.ndarray | None = None
    gyro_bias_mean_sensor: np.ndarray | None = None
    delta_p_bias_accel_jac: np.ndarray | None = None
    delta_v_bias_accel_jac: np.ndarray | None = None
    delta_p_bias_gyro_jac: np.ndarray | None = None
    delta_v_bias_gyro_jac: np.ndarray | None = None
    delta_angle_bias_gyro_jac: np.ndarray | None = None
    pva_accel_noise_cov: np.ndarray | None = None
    pva_gyro_noise_cov: np.ndarray | None = None
    pva_integration_noise_cov: np.ndarray | None = None
    gravity_ecef: np.ndarray | None = None
    preint_meas_cov: np.ndarray | None = None
    gravity_nav: np.ndarray | None = None


def _matrix_columns(prefix: str, dim: int) -> list[str]:
    return [f"{prefix}_{row}_{col}" for row in range(dim) for col in range(dim)]


def _matrix_stack_from_columns(frame: pd.DataFrame, prefix: str, dim: int) -> np.ndarray | None:
    cols = _matrix_columns(prefix, dim)
    if not set(cols).issubset(frame.columns):
        return None
    values = frame[cols].to_numpy(dtype=np.float64).reshape(frame.shape[0], dim, dim)
    if not np.isfinite(values).any():
        return None
    return values


def _vector_columns(prefix: str) -> list[str]:
    return [f"{prefix}_x", f"{prefix}_y", f"{prefix}_z"]


def _vector_stack_from_columns(frame: pd.DataFrame, prefix: str) -> np.ndarray:
    cols = _vector_columns(prefix)
    if not set(cols).issubset(frame.columns):
        return np.full((frame.shape[0], 3), np.nan, dtype=np.float64)
    return frame[cols].to_numpy(dtype=np.float64)


def load_taroz_imu_preintegration_csv(
    path: str | Path,
    *,
    epoch_times_ms: np.ndarray | None = None,
) -> IMUPreintegration:
    """Load Taroz-exported ``phone_data_imu_preintegration.csv`` as native preintegration.

    Newer exports may include native-positive bias Jacobians and GTSAM's full
    ``preintMeasCov``. Taroz's exported gravity is the local GTSAM navigation
    vector, so it is stored as ``gravity_nav`` rather than ``gravity_ecef``.
    Older exports still provide raw deltaR/P/V; in that case Jacobian and
    covariance fields remain unset and existing native fallbacks are used.
    """

    frame = pd.read_csv(Path(path))
    if frame.empty:
        raise ValueError(f"empty Taroz IMU preintegration CSV: {path}")
    required = {
        "utcTimeMillis",
        "nextUtcTimeMillis",
        "sample_count",
        "graph_dt_s",
        "preintegrated_dt_s",
        "delta_r_x",
        "delta_r_y",
        "delta_r_z",
        "delta_p_x",
        "delta_p_y",
        "delta_p_z",
        "delta_v_x",
        "delta_v_y",
        "delta_v_z",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Taroz IMU preintegration CSV missing columns {missing}: {path}")

    if epoch_times_ms is None:
        epoch_times = np.concatenate(
            (
                frame["utcTimeMillis"].to_numpy(dtype=np.float64),
                np.asarray([float(frame["nextUtcTimeMillis"].iloc[-1])], dtype=np.float64),
            )
        )
        target_idx = np.arange(frame.shape[0], dtype=np.int64)
    else:
        epoch_times = np.asarray(epoch_times_ms, dtype=np.float64).reshape(-1)
        if epoch_times.size < 2:
            raise ValueError("epoch_times_ms must contain at least two epochs")
        pair_to_idx = {
            (int(round(float(epoch_times[idx]))), int(round(float(epoch_times[idx + 1])))): idx
            for idx in range(epoch_times.size - 1)
        }
        target_idx = np.full(frame.shape[0], -1, dtype=np.int64)
        for row_idx, row in frame.iterrows():
            pair = (int(round(float(row["utcTimeMillis"]))), int(round(float(row["nextUtcTimeMillis"]))))
            idx = pair_to_idx.get(pair)
            if idx is None and "epoch_index" in frame.columns:
                epoch_idx = int(round(float(row["epoch_index"]))) - 1
                if 0 <= epoch_idx < epoch_times.size - 1:
                    idx = epoch_idx
            if idx is not None:
                target_idx[int(row_idx)] = int(idx)

    n_interval = int(epoch_times.size - 1)
    delta_t = np.full(n_interval, np.nan, dtype=np.float64)
    delta_p = np.full((n_interval, 3), np.nan, dtype=np.float64)
    delta_v = np.full((n_interval, 3), np.nan, dtype=np.float64)
    delta_angle = np.full((n_interval, 3), np.nan, dtype=np.float64)
    sample_count = np.zeros(n_interval, dtype=np.int32)
    gravity = np.full((n_interval, 3), np.nan, dtype=np.float64)

    source_delta_p = _vector_stack_from_columns(frame, "delta_p")
    source_delta_v = _vector_stack_from_columns(frame, "delta_v")
    source_delta_angle = _vector_stack_from_columns(frame, "delta_r")
    source_gravity = _vector_stack_from_columns(frame, "gravity")

    def aligned_matrix(prefix: str, dim: int) -> np.ndarray | None:
        source = _matrix_stack_from_columns(frame, prefix, dim)
        if source is None:
            return None
        out = np.full((n_interval, dim, dim), np.nan, dtype=np.float64)
        for src_idx, dst_idx in enumerate(target_idx):
            if 0 <= dst_idx < n_interval:
                out[dst_idx] = source[src_idx]
        if not np.isfinite(out).any():
            return None
        return out

    for src_idx, dst_idx in enumerate(target_idx):
        if not (0 <= dst_idx < n_interval):
            continue
        delta_t[dst_idx] = float(frame["preintegrated_dt_s"].iloc[src_idx])
        sample_count[dst_idx] = int(round(float(frame["sample_count"].iloc[src_idx])))
        delta_p[dst_idx] = source_delta_p[src_idx]
        delta_v[dst_idx] = source_delta_v[src_idx]
        delta_angle[dst_idx] = source_delta_angle[src_idx]
        gravity[dst_idx] = source_gravity[src_idx]

    return IMUPreintegration(
        epoch_times,
        delta_t,
        delta_v,
        delta_p,
        delta_angle,
        sample_count,
        IMU_TAROZ_BODY_DELTA_FRAME,
        delta_p_bias_accel_jac=aligned_matrix("delta_p_bias_accel_jac", 3),
        delta_v_bias_accel_jac=aligned_matrix("delta_v_bias_accel_jac", 3),
        delta_p_bias_gyro_jac=aligned_matrix("delta_p_bias_gyro_jac", 3),
        delta_v_bias_gyro_jac=aligned_matrix("delta_v_bias_gyro_jac", 3),
        delta_angle_bias_gyro_jac=aligned_matrix("delta_r_bias_gyro_jac", 3),
        preint_meas_cov=aligned_matrix("preint_meas_cov", 9),
        gravity_nav=gravity,
    )


def fill_nearest(values: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=np.float64).copy()
    if out.size == 0:
        return out
    finite = np.isfinite(out)
    if finite.all() or not finite.any():
        return out
    idx = np.arange(out.size)
    valid_idx = idx[finite]
    nearest = np.searchsorted(valid_idx, idx)
    nearest = np.clip(nearest, 0, valid_idx.size - 1)
    prev_idx = np.clip(nearest - 1, 0, valid_idx.size - 1)
    choose_prev = np.abs(idx - valid_idx[prev_idx]) <= np.abs(idx - valid_idx[nearest])
    nearest = np.where(choose_prev, prev_idx, nearest)
    out[~finite] = out[valid_idx[nearest[~finite]]]
    return out


def wrap_to_180_deg(deg: np.ndarray) -> np.ndarray:
    arr = np.asarray(deg, dtype=np.float64)
    return (arr + 180.0) % 360.0 - 180.0


def eul_xyz_to_rotm(eul_rad: np.ndarray) -> np.ndarray:
    """Match ref/gsdc2023/functions/eul2rotm.m (Rx * Ry * Rz)."""
    eul = np.asarray(eul_rad, dtype=np.float64).reshape(-1, 3)
    out = np.zeros((eul.shape[0], 3, 3), dtype=np.float64)
    c = np.cos(eul)
    s = np.sin(eul)
    for i in range(eul.shape[0]):
        rx = np.array(
            [[1.0, 0.0, 0.0], [0.0, c[i, 0], -s[i, 0]], [0.0, s[i, 0], c[i, 0]]],
            dtype=np.float64,
        )
        ry = np.array(
            [[c[i, 1], 0.0, s[i, 1]], [0.0, 1.0, 0.0], [-s[i, 1], 0.0, c[i, 1]]],
            dtype=np.float64,
        )
        rz = np.array(
            [[c[i, 2], -s[i, 2], 0.0], [s[i, 2], c[i, 2], 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        out[i] = rx @ ry @ rz
    return out


def gtsam_rzryrx_to_rotm(rpy_rad: np.ndarray) -> np.ndarray:
    """Match GTSAM ``Rot3.RzRyRx(roll, pitch, yaw)`` (Rz * Ry * Rx)."""
    rpy = np.asarray(rpy_rad, dtype=np.float64).reshape(-1, 3)
    out = np.zeros((rpy.shape[0], 3, 3), dtype=np.float64)
    c = np.cos(rpy)
    s = np.sin(rpy)
    for i in range(rpy.shape[0]):
        cr, cp, cy = c[i]
        sr, sp, sy = s[i]
        rx = np.array(
            [[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]],
            dtype=np.float64,
        )
        ry = np.array(
            [[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]],
            dtype=np.float64,
        )
        rz = np.array(
            [[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        out[i] = rz @ ry @ rx
    return out


def _skew3(v: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(v, dtype=np.float64).reshape(3)
    return np.array(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        dtype=np.float64,
    )


def rotvec_to_rotm(rotvec_rad: np.ndarray) -> np.ndarray:
    """Rodrigues rotation matrix for a 3D rotation vector."""

    rv = np.asarray(rotvec_rad, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(rv))
    kx = _skew3(rv)
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64) + kx
    a = np.sin(theta) / theta
    b = (1.0 - np.cos(theta)) / (theta * theta)
    return np.eye(3, dtype=np.float64) + a * kx + b * (kx @ kx)


def rotm_to_rotvec(rotm: np.ndarray) -> np.ndarray:
    """SO(3) logarithm as a 3D rotation vector."""

    rot = np.asarray(rotm, dtype=np.float64).reshape(3, 3)
    cos_theta = 0.5 * (float(np.trace(rot)) - 1.0)
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    theta = float(np.arccos(cos_theta))
    vee = np.array(
        [
            rot[2, 1] - rot[1, 2],
            rot[0, 2] - rot[2, 0],
            rot[1, 0] - rot[0, 1],
        ],
        dtype=np.float64,
    )
    if theta < 1e-12:
        return 0.5 * vee
    return theta / (2.0 * np.sin(theta)) * vee


def right_jacobian_so3(rotvec_rad: np.ndarray) -> np.ndarray:
    phi = np.asarray(rotvec_rad, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(phi))
    kx = _skew3(phi)
    kx2 = kx @ kx
    if theta < 1e-8:
        return np.eye(3, dtype=np.float64) - 0.5 * kx + (1.0 / 6.0) * kx2
    a = (1.0 - np.cos(theta)) / (theta * theta)
    b = (theta - np.sin(theta)) / (theta * theta * theta)
    return np.eye(3, dtype=np.float64) - a * kx + b * kx2


def right_jacobian_inverse_so3(rotvec_rad: np.ndarray) -> np.ndarray:
    phi = np.asarray(rotvec_rad, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(phi))
    kx = _skew3(phi)
    kx2 = kx @ kx
    if theta < 1e-8:
        return np.eye(3, dtype=np.float64) + 0.5 * kx + (1.0 / 12.0) * kx2
    half_theta = 0.5 * theta
    b = (1.0 / (theta * theta)) * (1.0 - half_theta / np.tan(half_theta))
    return np.eye(3, dtype=np.float64) + 0.5 * kx + b * kx2


def positive_gyro_bias_jacobian_so3(increment_rotvecs_rad: np.ndarray, dt_s: np.ndarray) -> np.ndarray:
    """Return ``-d Log(prod Exp((omega-bias)dt)) / d bias`` at zero bias."""

    phis = np.asarray(increment_rotvecs_rad, dtype=np.float64).reshape(-1, 3)
    dts = np.asarray(dt_s, dtype=np.float64).reshape(-1)
    if phis.shape[0] != dts.size:
        raise ValueError("increment_rotvecs_rad and dt_s must have the same length")
    delta_rot = np.eye(3, dtype=np.float64)
    increment_rots: list[np.ndarray] = []
    for phi in phis:
        incr = rotvec_to_rotm(phi)
        increment_rots.append(incr)
        delta_rot = delta_rot @ incr
    rho = rotm_to_rotvec(delta_rot)
    jac_right = np.zeros((3, 3), dtype=np.float64)
    suffix_rot = np.eye(3, dtype=np.float64)
    for phi, dt_i, incr in zip(phis[::-1], dts[::-1], increment_rots[::-1]):
        jac_right += suffix_rot.T @ right_jacobian_so3(phi) * float(dt_i)
        suffix_rot = incr @ suffix_rot
    return right_jacobian_inverse_so3(rho) @ jac_right


def ecef_to_enu_relative(xyz: np.ndarray, origin_xyz: np.ndarray) -> np.ndarray:
    lat, lon, _ = ecef_to_lla(float(origin_xyz[0]), float(origin_xyz[1]), float(origin_xyz[2]))
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)
    rot = np.array(
        [
            [-sin_lon, cos_lon, 0.0],
            [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
            [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat],
        ],
        dtype=np.float64,
    )
    return (np.asarray(xyz, dtype=np.float64) - np.asarray(origin_xyz, dtype=np.float64)) @ rot.T


def enu_to_ecef_relative(enu: np.ndarray, origin_xyz: np.ndarray) -> np.ndarray:
    lat, lon, _ = ecef_to_lla(float(origin_xyz[0]), float(origin_xyz[1]), float(origin_xyz[2]))
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)
    rot = np.array(
        [
            [-sin_lon, cos_lon, 0.0],
            [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
            [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat],
        ],
        dtype=np.float64,
    )
    return np.asarray(enu, dtype=np.float64) @ rot + np.asarray(origin_xyz, dtype=np.float64)


def ecef_delta_from_enu_delta(delta_enu: np.ndarray, origin_xyz: np.ndarray) -> np.ndarray:
    delta = np.asarray(delta_enu, dtype=np.float64).reshape(-1, 3)
    origin = np.asarray(origin_xyz, dtype=np.float64).reshape(3)
    return enu_to_ecef_relative(delta, origin) - origin


def read_device_imu_frame(path: Path, *, read_csv_fn: RawCsvReader) -> pd.DataFrame:
    header = read_csv_fn(path, nrows=0)
    available = set(header.columns)
    missing = [col for col in DEVICE_IMU_COLUMNS if col not in available]
    if missing:
        raise RuntimeError(f"device_imu.csv missing columns: {missing}")
    return read_csv_fn(path, usecols=DEVICE_IMU_COLUMNS)


def extract_imu_measurements(df: pd.DataFrame, message_type: str) -> IMUMeasurements | None:
    rows = df[df["MessageType"] == message_type].copy()
    if rows.empty:
        return None
    rows = rows.sort_values("utcTimeMillis").drop_duplicates(subset=["utcTimeMillis"], keep="first")
    return IMUMeasurements(
        times_ms=rows["utcTimeMillis"].to_numpy(dtype=np.float64),
        elapsed_ns=rows["elapsedRealtimeNanos"].to_numpy(dtype=np.float64),
        xyz=rows[["MeasurementX", "MeasurementY", "MeasurementZ"]].to_numpy(dtype=np.float64),
        bias=rows[["BiasX", "BiasY", "BiasZ"]].to_numpy(dtype=np.float64),
    )


def load_device_imu_measurements(
    trip_dir: Path,
    *,
    read_csv_fn: RawCsvReader,
) -> tuple[IMUMeasurements | None, IMUMeasurements | None, IMUMeasurements | None]:
    path = Path(trip_dir) / "device_imu.csv"
    if not path.is_file():
        return None, None, None
    df = read_device_imu_frame(path, read_csv_fn=read_csv_fn)
    acc = extract_imu_measurements(df, "UncalAccel")
    gyro = extract_imu_measurements(df, "UncalGyro")
    mag = extract_imu_measurements(df, "UncalMag")
    return acc, gyro, mag


def interp_vectors(source_t: np.ndarray, source_xyz: np.ndarray, query_t: np.ndarray) -> np.ndarray:
    out = np.zeros((query_t.size, source_xyz.shape[1]), dtype=np.float64)
    for axis in range(source_xyz.shape[1]):
        out[:, axis] = np.interp(query_t, source_t, source_xyz[:, axis])
    return out


def rolling_std(values: np.ndarray, window: int) -> np.ndarray:
    # MATLAB movstd(A, k) uses a centered sliding window with shortened
    # endpoints; Taroz IMU stop detection depends on that alignment.
    return (
        pd.Series(np.asarray(values, dtype=np.float64))
        .rolling(window, center=True, min_periods=1)
        .std(ddof=1)
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
    )


def process_device_imu(
    acc: IMUMeasurements,
    gyro: IMUMeasurements,
    gnss_times_ms: np.ndarray,
    gnss_elapsed_ns: np.ndarray | None,
    *,
    imu_sync: str = IMU_SYNC_MODE,
) -> tuple[ProcessedIMU, ProcessedIMU, np.ndarray]:
    gnss_times_ms = np.asarray(gnss_times_ms, dtype=np.float64)
    gnss_elapsed_ns_arr = (
        np.asarray(gnss_elapsed_ns, dtype=np.float64).reshape(-1) if gnss_elapsed_ns is not None else np.array([], dtype=np.float64)
    )
    has_gnss_elapsed = gnss_elapsed_ns_arr.size == gnss_times_ms.size and np.isfinite(gnss_elapsed_ns_arr).any()

    if has_gnss_elapsed:
        acc_times_ms = np.interp(acc.elapsed_ns, gnss_elapsed_ns_arr, gnss_times_ms)
        gyro_times_ms = np.interp(gyro.elapsed_ns, gnss_elapsed_ns_arr, gnss_times_ms)
        sync_coefficient = IMU_SYNC_COEFFICIENT
    else:
        acc_times_ms = acc.times_ms + ACC_TIME_OFFSET_MS
        gyro_times_ms = gyro.times_ms + GYRO_TIME_OFFSET_MS
        sync_coefficient = 1.0

    if imu_sync == "acc":
        sync_times_ms = acc_times_ms
        acc_xyz_sync = acc.xyz
        gyro_xyz_sync = interp_vectors(gyro_times_ms, gyro.xyz, sync_times_ms)
        acc_bias_sync = acc.bias
        gyro_bias_sync = interp_vectors(gyro_times_ms, gyro.bias, sync_times_ms)
    elif imu_sync == "gyro":
        sync_times_ms = gyro_times_ms
        acc_xyz_sync = interp_vectors(acc_times_ms, acc.xyz, sync_times_ms)
        gyro_xyz_sync = gyro.xyz
        acc_bias_sync = interp_vectors(acc_times_ms, acc.bias, sync_times_ms)
        gyro_bias_sync = gyro.bias
    else:
        raise ValueError(f"unsupported imu_sync: {imu_sync}")

    dt_s = np.diff(sync_times_ms) / 1000.0
    if dt_s.size == 0:
        dt_s = np.array([0.0], dtype=np.float64)
    else:
        dt_s = np.concatenate([dt_s, dt_s[-1:]])
    bad_dt = (~np.isfinite(dt_s)) | (dt_s <= 0.0)
    dt_s[bad_dt] = np.nanmedian(dt_s[~bad_dt]) if np.any(~bad_dt) else 0.01

    acc_norm = np.linalg.norm(acc_xyz_sync, axis=1)
    gyro_norm = np.linalg.norm(gyro_xyz_sync, axis=1)
    acc_std = rolling_std(acc_norm, STOP_WINDOW_SIZE)
    gyro_std = rolling_std(gyro_norm, STOP_WINDOW_SIZE)
    acc_stop_th = float(np.nanmin(acc_std) + STOP_ACC_STD_OFFSET)
    gyro_stop_th = float(np.nanmin(gyro_std) + STOP_GYRO_STD_OFFSET)
    idx_stop = (acc_std < acc_stop_th) & (gyro_std < gyro_stop_th) & (gyro_norm < STOP_GYRO_MAX)

    acc_processed = ProcessedIMU(
        times_ms=sync_times_ms,
        xyz=acc_xyz_sync,
        dt_s=dt_s,
        norm_3d=acc_norm,
        norm_std=acc_std,
        sync_coefficient=sync_coefficient,
        bias=acc_bias_sync,
    )
    gyro_processed = ProcessedIMU(
        times_ms=sync_times_ms,
        xyz=gyro_xyz_sync,
        dt_s=dt_s,
        norm_3d=gyro_norm,
        norm_std=gyro_std,
        sync_coefficient=sync_coefficient,
        bias=gyro_bias_sync,
    )
    return acc_processed, gyro_processed, idx_stop


def project_stop_to_epochs(
    imu_times_ms: np.ndarray,
    idx_stop: np.ndarray,
    epoch_times_ms: np.ndarray,
) -> np.ndarray:
    if imu_times_ms.size == 0 or idx_stop.size == 0 or epoch_times_ms.size == 0:
        return np.zeros(epoch_times_ms.size, dtype=bool)
    stop_values = np.asarray(idx_stop, dtype=np.float64)
    nearest = np.searchsorted(imu_times_ms, epoch_times_ms)
    nearest = np.clip(nearest, 0, imu_times_ms.size - 1)
    prev_idx = np.clip(nearest - 1, 0, imu_times_ms.size - 1)
    choose_prev = np.abs(epoch_times_ms - imu_times_ms[prev_idx]) <= np.abs(epoch_times_ms - imu_times_ms[nearest])
    nearest = np.where(choose_prev, prev_idx, nearest)
    return stop_values[nearest] > 0.5


def estimate_rpy_from_velocity(vel_enu: np.ndarray) -> np.ndarray:
    venu = np.asarray(vel_enu, dtype=np.float64).reshape(-1, 3)
    if venu.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    smoothed = np.zeros_like(venu)
    for axis in range(3):
        smoothed[:, axis] = (
            pd.Series(venu[:, axis])
            .rolling(VELOCITY_SMOOTH_WINDOW, center=True, min_periods=1)
            .mean()
            .to_numpy(dtype=np.float64)
        )
    head_deg = np.rad2deg(np.arctan2(smoothed[:, 1], smoothed[:, 0]))
    speed = np.linalg.norm(smoothed, axis=1)
    head_deg[speed < VELOCITY_THRESHOLD_MPS] = np.nan
    head_deg = fill_nearest(head_deg)
    yaw = np.deg2rad(wrap_to_180_deg(head_deg + 180.0))
    rpy = np.zeros((venu.shape[0], 3), dtype=np.float64)
    rpy[:, 2] = yaw
    return rpy


def preintegrate_processed_imu(
    acc: ProcessedIMU,
    gyro: ProcessedIMU,
    epoch_times_ms: np.ndarray,
    *,
    delta_frame: str = "body",
    reference_xyz_ecef: np.ndarray | None = None,
    gravity_mps2: float = IMU_GRAVITY_MPS2,
    mounting_angle_rad: np.ndarray = IMU_MOUNTING_ANGLE_RAD,
    sample_dt_mode: str = "bounded",
) -> IMUPreintegration:
    """Integrate aligned IMU samples between GNSS epochs.

    ``delta_frame="body"`` keeps the legacy raw sensor-frame deltas. The opt-in
    ``"ecef"`` mode approximates the older bridge path by using velocity-derived
    yaw, the MATLAB mounting angle, and gravity compensation before converting
    ENU deltas to ECEF. ``"taroz_body"`` applies the MATLAB mounting angle but
    keeps p/v/angle deltas in the body frame and stores navigation-frame ECEF
    gravity for the native GTSAM-style IMU residual. Synchronized raw bias
    columns are retained as interval telemetry, not subtracted from the samples.
    """

    if delta_frame not in IMU_DELTA_FRAMES:
        raise ValueError(f"unsupported IMU delta frame: {delta_frame}")
    if sample_dt_mode not in IMU_SAMPLE_DT_MODES:
        raise ValueError(f"unsupported IMU sample dt mode: {sample_dt_mode}")
    epoch_times = np.asarray(epoch_times_ms, dtype=np.float64).reshape(-1)
    n_interval = max(epoch_times.size - 1, 0)
    delta_t_s = np.zeros(n_interval, dtype=np.float64)
    delta_v = np.zeros((n_interval, 3), dtype=np.float64)
    delta_p = np.zeros((n_interval, 3), dtype=np.float64)
    delta_angle = np.zeros((n_interval, 3), dtype=np.float64)
    sample_count = np.zeros(n_interval, dtype=np.int32)
    acc_bias_mean = np.full((n_interval, 3), np.nan, dtype=np.float64)
    gyro_bias_mean = np.full((n_interval, 3), np.nan, dtype=np.float64)
    delta_p_bias_accel_jac = np.zeros((n_interval, 3, 3), dtype=np.float64)
    delta_v_bias_accel_jac = np.zeros((n_interval, 3, 3), dtype=np.float64)
    delta_p_bias_gyro_jac = np.zeros((n_interval, 3, 3), dtype=np.float64)
    delta_v_bias_gyro_jac = np.zeros((n_interval, 3, 3), dtype=np.float64)
    delta_angle_bias_gyro_jac = np.zeros((n_interval, 3, 3), dtype=np.float64)
    pva_accel_noise_cov = np.zeros((n_interval, 9, 9), dtype=np.float64)
    pva_gyro_noise_cov = np.zeros((n_interval, 9, 9), dtype=np.float64)
    pva_integration_noise_cov = np.zeros((n_interval, 9, 9), dtype=np.float64)
    if n_interval == 0 or acc.times_ms.size == 0 or gyro.times_ms.size == 0:
        return IMUPreintegration(
            epoch_times,
            delta_t_s,
            delta_v,
            delta_p,
            delta_angle,
            sample_count,
            delta_frame,
            acc_bias_mean,
            gyro_bias_mean,
            delta_p_bias_accel_jac,
            delta_v_bias_accel_jac,
            delta_p_bias_gyro_jac,
            delta_v_bias_gyro_jac,
            delta_angle_bias_gyro_jac,
            pva_accel_noise_cov,
            pva_gyro_noise_cov,
            pva_integration_noise_cov,
        )

    imu_t = np.asarray(acc.times_ms, dtype=np.float64).reshape(-1)
    acc_xyz = np.asarray(acc.xyz, dtype=np.float64).reshape(-1, 3)
    gyro_xyz = np.asarray(gyro.xyz, dtype=np.float64).reshape(-1, 3)
    acc_bias = (
        np.asarray(acc.bias, dtype=np.float64).reshape(-1, 3)
        if acc.bias is not None
        else np.full_like(acc_xyz, np.nan, dtype=np.float64)
    )
    gyro_bias = (
        np.asarray(gyro.bias, dtype=np.float64).reshape(-1, 3)
        if gyro.bias is not None
        else np.full_like(gyro_xyz, np.nan, dtype=np.float64)
    )
    n = min(imu_t.size, acc_xyz.shape[0], gyro_xyz.shape[0], acc_bias.shape[0], gyro_bias.shape[0])
    imu_t = imu_t[:n]
    acc_xyz = acc_xyz[:n]
    gyro_xyz = gyro_xyz[:n]
    acc_bias = acc_bias[:n]
    gyro_bias = gyro_bias[:n]
    if n == 0:
        return IMUPreintegration(
            epoch_times,
            delta_t_s,
            delta_v,
            delta_p,
            delta_angle,
            sample_count,
            delta_frame,
            acc_bias_mean,
            gyro_bias_mean,
            delta_p_bias_accel_jac,
            delta_v_bias_accel_jac,
            delta_p_bias_gyro_jac,
            delta_v_bias_gyro_jac,
            delta_angle_bias_gyro_jac,
            pva_accel_noise_cov,
            pva_gyro_noise_cov,
            pva_integration_noise_cov,
        )

    use_ecef = False
    origin_xyz = None
    acc_nav = acc_xyz
    gyro_nav = gyro_xyz
    gravity_ecef = None
    rot_body_sensor_for_bias = None
    if delta_frame == IMU_TAROZ_BODY_DELTA_FRAME:
        rot_body_sensor = gtsam_rzryrx_to_rotm(np.asarray(mounting_angle_rad, dtype=np.float64).reshape(1, 3))[0]
        rot_body_sensor_for_bias = rot_body_sensor
        acc_nav = np.einsum("ij,nj->ni", rot_body_sensor, acc_xyz)
        gyro_nav = np.einsum("ij,nj->ni", rot_body_sensor, gyro_xyz)
    if delta_frame in ("ecef", IMU_TAROZ_BODY_DELTA_FRAME) and reference_xyz_ecef is not None:
        ref_xyz = np.asarray(reference_xyz_ecef, dtype=np.float64).reshape(-1, 3)
        finite_ref = np.isfinite(ref_xyz).all(axis=1)
        if ref_xyz.shape[0] == epoch_times.size and finite_ref.any():
            origin_xyz = ref_xyz[np.flatnonzero(finite_ref)[0]]
            gravity_vec = ecef_delta_from_enu_delta(
                np.array([[0.0, 0.0, -float(gravity_mps2)]], dtype=np.float64),
                origin_xyz,
            )[0]
            gravity_ecef = np.tile(gravity_vec.reshape(1, 3), (n_interval, 1))
            if delta_frame == "ecef":
                ref_enu = ecef_to_enu_relative(ref_xyz, origin_xyz)
                ref_vel_enu = np.zeros_like(ref_enu)
                ref_dt = np.diff(epoch_times) / 1000.0
                valid_dt = np.isfinite(ref_dt) & (ref_dt > 0.0)
                if ref_enu.shape[0] > 1 and valid_dt.any():
                    step_vel = np.zeros((ref_enu.shape[0] - 1, 3), dtype=np.float64)
                    step_vel[valid_dt] = (ref_enu[1:][valid_dt] - ref_enu[:-1][valid_dt]) / ref_dt[valid_dt, None]
                    ref_vel_enu[:-1] = step_vel
                    ref_vel_enu[-1] = step_vel[-1]
                rpy_epoch = estimate_rpy_from_velocity(ref_vel_enu)
                rpy_epoch[~np.isfinite(rpy_epoch)] = 0.0
                rpy_samples = np.column_stack(
                    [
                        np.interp(imu_t, epoch_times, rpy_epoch[:, axis])
                        for axis in range(3)
                    ],
                )
                rot_nav_body = eul_xyz_to_rotm(rpy_samples)
                rot_body_sensor = eul_xyz_to_rotm(np.asarray(mounting_angle_rad, dtype=np.float64).reshape(1, 3))[0]
                acc_body = np.einsum("ij,nj->ni", rot_body_sensor, acc_xyz)
                gyro_body = np.einsum("ij,nj->ni", rot_body_sensor, gyro_xyz)
                acc_nav = np.einsum("nij,nj->ni", rot_nav_body, acc_body)
                gyro_nav = np.einsum("nij,nj->ni", rot_nav_body, gyro_body)
                acc_nav[:, 2] -= float(gravity_mps2)
                use_ecef = True

    for interval_idx in range(n_interval):
        t0 = float(epoch_times[interval_idx])
        t1 = float(epoch_times[interval_idx + 1])
        if not np.isfinite(t0) or not np.isfinite(t1) or t1 <= t0:
            continue
        idx = np.flatnonzero((imu_t >= t0) & (imu_t <= t1))
        if idx.size == 0:
            continue
        times = imu_t[idx]
        if sample_dt_mode == "taroz":
            dt = np.asarray(acc.dt_s[idx], dtype=np.float64)
            acc_samples = acc_nav[idx]
            gyro_samples = gyro_nav[idx]
            acc_bias_samples = acc_bias[idx]
            gyro_bias_samples = gyro_bias[idx]
        elif times.size == 1:
            dt = np.array([(t1 - t0) * 1e-3], dtype=np.float64)
            acc_samples = acc_nav[idx]
            gyro_samples = gyro_nav[idx]
            acc_bias_samples = acc_bias[idx]
            gyro_bias_samples = gyro_bias[idx]
        else:
            segment = np.concatenate(([t0], times, [t1]))
            dt = np.diff(segment) * 1e-3
            bad_dt = (~np.isfinite(dt)) | (dt < 0.0)
            dt[bad_dt] = 0.0
            sample_dt = 0.5 * (dt[:-1] + dt[1:])
            acc_samples = acc_nav[idx]
            gyro_samples = gyro_nav[idx]
            acc_bias_samples = acc_bias[idx]
            gyro_bias_samples = gyro_bias[idx]
            dt = sample_dt
        finite = (
            np.isfinite(dt)
            & np.isfinite(acc_samples).all(axis=1)
            & np.isfinite(gyro_samples).all(axis=1)
        )
        if not finite.any():
            continue
        dt = dt[finite]
        acc_samples = acc_samples[finite]
        gyro_samples = gyro_samples[finite]
        acc_bias_samples = acc_bias_samples[finite]
        gyro_bias_samples = gyro_bias_samples[finite]
        interval_dt = float(np.sum(dt))
        vel = np.zeros(3, dtype=np.float64)
        pos = np.zeros(3, dtype=np.float64)
        angle = np.zeros(3, dtype=np.float64)
        acc_jac_v = np.zeros((3, 3), dtype=np.float64)
        acc_jac_p = np.zeros((3, 3), dtype=np.float64)
        gyro_jac_v = np.zeros((3, 3), dtype=np.float64)
        gyro_jac_p = np.zeros((3, 3), dtype=np.float64)
        gyro_jac_angle = np.zeros((3, 3), dtype=np.float64)
        eye3 = np.eye(3, dtype=np.float64)
        delta_rot = np.eye(3, dtype=np.float64)
        delta_rot_bias_deriv = np.zeros((3, 3, 3), dtype=np.float64)
        vel_bias_gyro_deriv = np.zeros((3, 3), dtype=np.float64)
        pos_bias_gyro_deriv = np.zeros((3, 3), dtype=np.float64)
        accel_noise_cov = np.zeros((9, 9), dtype=np.float64)
        gyro_noise_cov = np.zeros((9, 9), dtype=np.float64)
        integration_noise_cov = np.zeros((9, 9), dtype=np.float64)
        rotate_accel_by_delta = sample_dt_mode == "taroz"
        gyro_increments: list[np.ndarray] = []
        for dt_s, a_body, w_body in zip(dt, acc_samples, gyro_samples):
            gyro_increment = w_body * dt_s
            increment_rot = rotvec_to_rotm(gyro_increment) if rotate_accel_by_delta else eye3
            increment_jr = right_jacobian_so3(gyro_increment) if rotate_accel_by_delta else eye3
            rot_accel = delta_rot if rotate_accel_by_delta else eye3
            if dt_s > 0.0:
                f_cov = np.eye(9, dtype=np.float64)
                f_cov[0:3, 3:6] = eye3 * dt_s
                accel_theta_jac = -rot_accel @ _skew3(a_body)
                f_cov[0:3, 6:9] = 0.5 * accel_theta_jac * dt_s * dt_s
                f_cov[3:6, 6:9] = accel_theta_jac * dt_s
                f_cov[6:9, 6:9] = increment_rot.T
                g_acc = np.zeros((9, 3), dtype=np.float64)
                g_acc[0:3, :] = 0.5 * rot_accel * dt_s * dt_s
                g_acc[3:6, :] = rot_accel * dt_s
                g_gyro = np.zeros((9, 3), dtype=np.float64)
                g_gyro[6:9, :] = increment_jr * dt_s
                inv_dt = 1.0 / dt_s
                accel_noise_cov = f_cov @ accel_noise_cov @ f_cov.T + (g_acc @ g_acc.T) * inv_dt
                gyro_noise_cov = f_cov @ gyro_noise_cov @ f_cov.T + (g_gyro @ g_gyro.T) * inv_dt
                integration_noise_cov = f_cov @ integration_noise_cov @ f_cov.T
                integration_noise_cov[0:3, 0:3] += eye3 * dt_s
            acc_delta = delta_rot @ a_body if rotate_accel_by_delta else a_body
            acc_bias_jac = delta_rot if rotate_accel_by_delta else eye3
            if rotate_accel_by_delta:
                acc_gyro_deriv = np.zeros((3, 3), dtype=np.float64)
                for axis in range(3):
                    acc_gyro_deriv[:, axis] = delta_rot_bias_deriv[axis] @ a_body
                pos_bias_gyro_deriv += vel_bias_gyro_deriv * dt_s + 0.5 * acc_gyro_deriv * dt_s * dt_s
                vel_bias_gyro_deriv += acc_gyro_deriv * dt_s
            pos += vel * dt_s + 0.5 * acc_delta * dt_s * dt_s
            vel += acc_delta * dt_s
            angle += gyro_increment
            acc_jac_p += acc_jac_v * dt_s + 0.5 * acc_bias_jac * dt_s * dt_s
            acc_jac_v += acc_bias_jac * dt_s
            gyro_jac_angle += eye3 * dt_s
            if rotate_accel_by_delta:
                gyro_increments.append(gyro_increment)
                next_delta_rot_bias_deriv = np.zeros_like(delta_rot_bias_deriv)
                for axis in range(3):
                    basis = eye3[:, axis]
                    d_increment_db = increment_rot @ _skew3(-(increment_jr @ basis) * dt_s)
                    next_delta_rot_bias_deriv[axis] = delta_rot_bias_deriv[axis] @ increment_rot + delta_rot @ d_increment_db
                delta_rot = delta_rot @ increment_rot
                delta_rot_bias_deriv = next_delta_rot_bias_deriv
        if rotate_accel_by_delta:
            angle = rotm_to_rotvec(delta_rot)
            gyro_jac_p = -pos_bias_gyro_deriv
            gyro_jac_v = -vel_bias_gyro_deriv
            gyro_jac_angle = positive_gyro_bias_jacobian_so3(np.asarray(gyro_increments), dt)
            angle_cov_map = np.eye(9, dtype=np.float64)
            angle_cov_map[6:9, 6:9] = right_jacobian_inverse_so3(angle)
            accel_noise_cov = angle_cov_map @ accel_noise_cov @ angle_cov_map.T
            gyro_noise_cov = angle_cov_map @ gyro_noise_cov @ angle_cov_map.T
            integration_noise_cov = angle_cov_map @ integration_noise_cov @ angle_cov_map.T
        delta_t_s[interval_idx] = interval_dt
        if use_ecef and origin_xyz is not None:
            delta_v[interval_idx] = ecef_delta_from_enu_delta(vel.reshape(1, 3), origin_xyz)[0]
            delta_p[interval_idx] = ecef_delta_from_enu_delta(pos.reshape(1, 3), origin_xyz)[0]
            delta_angle[interval_idx] = ecef_delta_from_enu_delta(angle.reshape(1, 3), origin_xyz)[0]
        else:
            delta_v[interval_idx] = vel
            delta_p[interval_idx] = pos
            delta_angle[interval_idx] = angle
        if rot_body_sensor_for_bias is not None:
            acc_jac_p = acc_jac_p @ rot_body_sensor_for_bias
            acc_jac_v = acc_jac_v @ rot_body_sensor_for_bias
            gyro_jac_p = gyro_jac_p @ rot_body_sensor_for_bias
            gyro_jac_v = gyro_jac_v @ rot_body_sensor_for_bias
            gyro_jac_angle = gyro_jac_angle @ rot_body_sensor_for_bias
        delta_p_bias_accel_jac[interval_idx] = acc_jac_p
        delta_v_bias_accel_jac[interval_idx] = acc_jac_v
        delta_p_bias_gyro_jac[interval_idx] = gyro_jac_p
        delta_v_bias_gyro_jac[interval_idx] = gyro_jac_v
        delta_angle_bias_gyro_jac[interval_idx] = gyro_jac_angle
        pva_accel_noise_cov[interval_idx] = 0.5 * (accel_noise_cov + accel_noise_cov.T)
        pva_gyro_noise_cov[interval_idx] = 0.5 * (gyro_noise_cov + gyro_noise_cov.T)
        pva_integration_noise_cov[interval_idx] = 0.5 * (integration_noise_cov + integration_noise_cov.T)
        sample_count[interval_idx] = int(dt.size)

        positive_dt = np.isfinite(dt) & (dt > 0.0)
        if positive_dt.any():
            acc_bias_finite = positive_dt & np.isfinite(acc_bias_samples).all(axis=1)
            gyro_bias_finite = positive_dt & np.isfinite(gyro_bias_samples).all(axis=1)
            if acc_bias_finite.any():
                acc_bias_mean[interval_idx] = np.average(
                    acc_bias_samples[acc_bias_finite],
                    axis=0,
                    weights=dt[acc_bias_finite],
                )
            if gyro_bias_finite.any():
                gyro_bias_mean[interval_idx] = np.average(
                    gyro_bias_samples[gyro_bias_finite],
                    axis=0,
                    weights=dt[gyro_bias_finite],
                )

    return IMUPreintegration(
        epoch_times,
        delta_t_s,
        delta_v,
        delta_p,
        delta_angle,
        sample_count,
        "ecef" if use_ecef else (IMU_TAROZ_BODY_DELTA_FRAME if delta_frame == IMU_TAROZ_BODY_DELTA_FRAME else "body"),
        acc_bias_mean,
        gyro_bias_mean,
        delta_p_bias_accel_jac,
        delta_v_bias_accel_jac,
        delta_p_bias_gyro_jac,
        delta_v_bias_gyro_jac,
        delta_angle_bias_gyro_jac,
        pva_accel_noise_cov,
        pva_gyro_noise_cov,
        pva_integration_noise_cov,
        gravity_ecef if delta_frame == IMU_TAROZ_BODY_DELTA_FRAME else None,
    )


def imu_preintegration_segment(
    preintegration: IMUPreintegration | None,
    start: int,
    end: int,
) -> tuple[np.ndarray | None, np.ndarray | None, int]:
    """Return native VD IMU delta-prior slices for a local epoch segment."""

    delta_p, delta_v, _delta_angle, count = imu_preintegration_segment_with_angle(
        preintegration,
        start,
        end,
    )
    return delta_p, delta_v, count


def imu_preintegration_segment_with_angle(
    preintegration: IMUPreintegration | None,
    start: int,
    end: int,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, int]:
    """Return native VD IMU delta-prior slices, including gyro delta angle."""

    delta_p, delta_v, delta_angle, _delta_t, count = imu_preintegration_segment_with_angle_and_dt(
        preintegration,
        start,
        end,
    )
    return delta_p, delta_v, delta_angle, count


def imu_preintegration_segment_with_angle_and_dt(
    preintegration: IMUPreintegration | None,
    start: int,
    end: int,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None, int]:
    """Return native VD IMU delta-prior slices plus preintegration delta time."""

    (
        delta_p,
        delta_v,
        delta_angle,
        delta_t,
        _delta_p_bias_accel_jac,
        _delta_v_bias_accel_jac,
        _delta_p_bias_gyro_jac,
        _delta_v_bias_gyro_jac,
        _delta_angle_bias_gyro_jac,
        count,
    ) = imu_preintegration_segment_with_bias_jacobians(preintegration, start, end)
    return delta_p, delta_v, delta_angle, delta_t, count


def imu_preintegration_segment_with_bias_jacobians(
    preintegration: IMUPreintegration | None,
    start: int,
    end: int,
) -> tuple[
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    int,
]:
    """Return native VD IMU delta-prior slices plus preintegrated bias Jacobians."""

    if preintegration is None or end - start <= 1:
        return None, None, None, None, None, None, None, None, None, 0
    i0 = max(int(start), 0)
    i1 = max(int(end) - 1, i0)
    n_interval = int(preintegration.delta_t_s.size)
    if i0 >= n_interval:
        return None, None, None, None, None, None, None, None, None, 0
    i1 = min(i1, n_interval)
    if i1 <= i0:
        return None, None, None, None, None, None, None, None, None, 0

    delta_p = np.asarray(preintegration.delta_p_body[i0:i1], dtype=np.float64).copy()
    delta_v = np.asarray(preintegration.delta_v_body[i0:i1], dtype=np.float64).copy()
    delta_angle = np.asarray(preintegration.delta_angle_rad[i0:i1], dtype=np.float64).copy()
    delta_t = np.asarray(preintegration.delta_t_s[i0:i1], dtype=np.float64).copy()
    delta_p_bias_accel_jac = (
        np.asarray(preintegration.delta_p_bias_accel_jac[i0:i1], dtype=np.float64).copy()
        if preintegration.delta_p_bias_accel_jac is not None
        else None
    )
    delta_v_bias_accel_jac = (
        np.asarray(preintegration.delta_v_bias_accel_jac[i0:i1], dtype=np.float64).copy()
        if preintegration.delta_v_bias_accel_jac is not None
        else None
    )
    delta_p_bias_gyro_jac = (
        np.asarray(preintegration.delta_p_bias_gyro_jac[i0:i1], dtype=np.float64).copy()
        if preintegration.delta_p_bias_gyro_jac is not None
        else None
    )
    delta_v_bias_gyro_jac = (
        np.asarray(preintegration.delta_v_bias_gyro_jac[i0:i1], dtype=np.float64).copy()
        if preintegration.delta_v_bias_gyro_jac is not None
        else None
    )
    delta_angle_bias_gyro_jac = (
        np.asarray(preintegration.delta_angle_bias_gyro_jac[i0:i1], dtype=np.float64).copy()
        if preintegration.delta_angle_bias_gyro_jac is not None
        else None
    )
    sample_count = np.asarray(preintegration.sample_count[i0:i1], dtype=np.int32)
    valid = (
        (sample_count > 0)
        & np.isfinite(delta_t)
        & (delta_t > 0.0)
        & np.isfinite(delta_p).all(axis=1)
        & np.isfinite(delta_v).all(axis=1)
        & np.isfinite(delta_angle).all(axis=1)
    )
    if delta_p_bias_accel_jac is not None:
        valid &= np.isfinite(delta_p_bias_accel_jac).all(axis=(1, 2))
    if delta_v_bias_accel_jac is not None:
        valid &= np.isfinite(delta_v_bias_accel_jac).all(axis=(1, 2))
    if delta_p_bias_gyro_jac is not None:
        valid &= np.isfinite(delta_p_bias_gyro_jac).all(axis=(1, 2))
    if delta_v_bias_gyro_jac is not None:
        valid &= np.isfinite(delta_v_bias_gyro_jac).all(axis=(1, 2))
    if delta_angle_bias_gyro_jac is not None:
        valid &= np.isfinite(delta_angle_bias_gyro_jac).all(axis=(1, 2))
    if not valid.any():
        return None, None, None, None, None, None, None, None, None, 0
    delta_p[~valid, :] = np.nan
    delta_v[~valid, :] = np.nan
    delta_angle[~valid, :] = np.nan
    delta_t[~valid] = np.nan
    if delta_p_bias_accel_jac is not None:
        delta_p_bias_accel_jac[~valid, :, :] = 0.0
    if delta_v_bias_accel_jac is not None:
        delta_v_bias_accel_jac[~valid, :, :] = 0.0
    if delta_p_bias_gyro_jac is not None:
        delta_p_bias_gyro_jac[~valid, :, :] = 0.0
    if delta_v_bias_gyro_jac is not None:
        delta_v_bias_gyro_jac[~valid, :, :] = 0.0
    if delta_angle_bias_gyro_jac is not None:
        delta_angle_bias_gyro_jac[~valid, :, :] = 0.0
    return (
        delta_p,
        delta_v,
        delta_angle,
        delta_t,
        delta_p_bias_accel_jac,
        delta_v_bias_accel_jac,
        delta_p_bias_gyro_jac,
        delta_v_bias_gyro_jac,
        delta_angle_bias_gyro_jac,
        int(np.count_nonzero(valid)),
    )


def imu_preintegration_gravity_segment(
    preintegration: IMUPreintegration | None,
    start: int,
    end: int,
) -> np.ndarray | None:
    """Return ECEF gravity slices aligned with native IMU delta intervals."""

    if preintegration is None or preintegration.gravity_ecef is None or end - start <= 1:
        return None
    i0 = max(int(start), 0)
    i1 = max(int(end) - 1, i0)
    n_interval = int(preintegration.delta_t_s.size)
    if i0 >= n_interval:
        return None
    i1 = min(i1, n_interval)
    if i1 <= i0:
        return None

    gravity = np.asarray(preintegration.gravity_ecef[i0:i1], dtype=np.float64).copy()
    delta_t = np.asarray(preintegration.delta_t_s[i0:i1], dtype=np.float64)
    sample_count = np.asarray(preintegration.sample_count[i0:i1], dtype=np.int32)
    valid = (sample_count > 0) & np.isfinite(delta_t) & (delta_t > 0.0) & np.isfinite(gravity).all(axis=1)
    if not valid.any():
        return None
    gravity[~valid, :] = np.nan
    return gravity
