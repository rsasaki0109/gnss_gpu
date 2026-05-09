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
IMU_DELTA_FRAMES = ("body", "ecef")
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
    return (
        pd.Series(np.asarray(values, dtype=np.float64))
        .rolling(window, center=False, min_periods=1)
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
) -> IMUPreintegration:
    """Integrate aligned IMU samples between GNSS epochs.

    ``delta_frame="body"`` keeps the legacy raw sensor-frame deltas. The opt-in
    ``"ecef"`` mode approximates MATLAB/GTSAM setup by using velocity-derived
    yaw, the MATLAB mounting angle, and gravity compensation before converting
    ENU deltas to ECEF. It still does not estimate pose attitude or IMU bias
    states. Synchronized raw bias columns are retained as interval telemetry,
    not subtracted from the IMU samples.
    """

    if delta_frame not in IMU_DELTA_FRAMES:
        raise ValueError(f"unsupported IMU delta frame: {delta_frame}")
    epoch_times = np.asarray(epoch_times_ms, dtype=np.float64).reshape(-1)
    n_interval = max(epoch_times.size - 1, 0)
    delta_t_s = np.zeros(n_interval, dtype=np.float64)
    delta_v = np.zeros((n_interval, 3), dtype=np.float64)
    delta_p = np.zeros((n_interval, 3), dtype=np.float64)
    delta_angle = np.zeros((n_interval, 3), dtype=np.float64)
    sample_count = np.zeros(n_interval, dtype=np.int32)
    acc_bias_mean = np.full((n_interval, 3), np.nan, dtype=np.float64)
    gyro_bias_mean = np.full((n_interval, 3), np.nan, dtype=np.float64)
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
        )

    use_ecef = False
    origin_xyz = None
    acc_nav = acc_xyz
    gyro_nav = gyro_xyz
    if delta_frame == "ecef" and reference_xyz_ecef is not None:
        ref_xyz = np.asarray(reference_xyz_ecef, dtype=np.float64).reshape(-1, 3)
        finite_ref = np.isfinite(ref_xyz).all(axis=1)
        if ref_xyz.shape[0] == epoch_times.size and finite_ref.any():
            origin_xyz = ref_xyz[np.flatnonzero(finite_ref)[0]]
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
        if times.size == 1:
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
        for dt_s, a_body, w_body in zip(dt, acc_samples, gyro_samples):
            pos += vel * dt_s + 0.5 * a_body * dt_s * dt_s
            vel += a_body * dt_s
            angle += w_body * dt_s
        delta_t_s[interval_idx] = interval_dt
        if use_ecef and origin_xyz is not None:
            delta_v[interval_idx] = ecef_delta_from_enu_delta(vel.reshape(1, 3), origin_xyz)[0]
            delta_p[interval_idx] = ecef_delta_from_enu_delta(pos.reshape(1, 3), origin_xyz)[0]
            delta_angle[interval_idx] = ecef_delta_from_enu_delta(angle.reshape(1, 3), origin_xyz)[0]
        else:
            delta_v[interval_idx] = vel
            delta_p[interval_idx] = pos
            delta_angle[interval_idx] = angle
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
        "ecef" if use_ecef else "body",
        acc_bias_mean,
        gyro_bias_mean,
    )


def imu_preintegration_segment(
    preintegration: IMUPreintegration | None,
    start: int,
    end: int,
) -> tuple[np.ndarray | None, np.ndarray | None, int]:
    """Return native VD IMU delta-prior slices for a local epoch segment."""

    if preintegration is None or end - start <= 1:
        return None, None, 0
    i0 = max(int(start), 0)
    i1 = max(int(end) - 1, i0)
    n_interval = int(preintegration.delta_t_s.size)
    if i0 >= n_interval:
        return None, None, 0
    i1 = min(i1, n_interval)
    if i1 <= i0:
        return None, None, 0

    delta_p = np.asarray(preintegration.delta_p_body[i0:i1], dtype=np.float64).copy()
    delta_v = np.asarray(preintegration.delta_v_body[i0:i1], dtype=np.float64).copy()
    delta_t = np.asarray(preintegration.delta_t_s[i0:i1], dtype=np.float64)
    sample_count = np.asarray(preintegration.sample_count[i0:i1], dtype=np.int32)
    valid = (
        (sample_count > 0)
        & np.isfinite(delta_t)
        & (delta_t > 0.0)
        & np.isfinite(delta_p).all(axis=1)
        & np.isfinite(delta_v).all(axis=1)
    )
    if not valid.any():
        return None, None, 0
    delta_p[~valid, :] = np.nan
    delta_v[~valid, :] = np.nan
    return delta_p, delta_v, int(np.count_nonzero(valid))
