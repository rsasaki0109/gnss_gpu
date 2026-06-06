#!/usr/bin/env python3
"""Audit native FGO residuals against Taroz fixed-linearized factors."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from experiments.gsdc2023_imu import IMU_TAROZ_BODY_DELTA_FRAME, rotm_to_rotvec, rotvec_to_rotm
from experiments.gsdc2023_observation_matrix import TripArrays
from experiments.gsdc2023_raw_bridge import (
    DEFAULT_ROOT,
    _apply_taroz_factor_mask_to_batch,
    _apply_taroz_imu_factor_mask_to_batch,
    _load_taroz_fgo_seed_state,
    _repair_baseline_wls,
    _seed_vd_state,
    build_trip_arrays,
    fit_state_with_clock_bias,
    run_wls,
)
from experiments.gsdc2023_residual_model import (
    EARTH_ROTATION_RATE_RAD_S,
    LIGHT_SPEED_MPS,
    fill_clock_design,
)
from experiments.gsdc2023_signal_model import constellation_to_matlab_sys, slot_frequency_label


@dataclass(frozen=True)
class LinearizationDeltaStats:
    label: str
    factor: str
    count: int
    native_weighted_rms: float | None
    taroz_linear_weighted_rms: float | None
    delta_weighted_rms: float | None
    delta_median_abs: float | None
    delta_p95_abs: float | None
    delta_max_abs: float | None


def _active_weights(batch: TripArrays, attr_fgo: str, attr_base: str) -> np.ndarray | None:
    fgo_value = getattr(batch, attr_fgo, None)
    if fgo_value is not None:
        return np.asarray(fgo_value, dtype=np.float64)
    base_value = getattr(batch, attr_base, None)
    if base_value is None:
        return None
    return np.asarray(base_value, dtype=np.float64)


def _state_parts(state: np.ndarray, n_clock: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(state, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 7 + n_clock:
        raise ValueError(f"expected VD state with at least {7 + n_clock} columns")
    return values[:, :3], values[:, 3:6], values[:, 6 : 6 + n_clock], values[:, 6 + n_clock]


def native_range_and_jacobian(sat_ecef: np.ndarray, receiver_ecef: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return native PR range and d(range)/d(receiver_xyz).

    This mirrors ``src/positioning/fgo.cu``: satellite coordinates are rotated
    by the signal transit angle and the rotation derivative is intentionally
    ignored in the Jacobian, as in the native solver.
    """
    sat = np.asarray(sat_ecef, dtype=np.float64)
    rx = np.asarray(receiver_ecef, dtype=np.float64)
    sat_b, rx_b = np.broadcast_arrays(sat, rx)
    dx0 = rx_b[..., 0] - sat_b[..., 0]
    dy0 = rx_b[..., 1] - sat_b[..., 1]
    dz0 = rx_b[..., 2] - sat_b[..., 2]
    r0 = np.sqrt(dx0 * dx0 + dy0 * dy0 + dz0 * dz0)
    theta = EARTH_ROTATION_RATE_RAD_S * (r0 / LIGHT_SPEED_MPS)
    sx_rot = sat_b[..., 0] * np.cos(theta) + sat_b[..., 1] * np.sin(theta)
    sy_rot = -sat_b[..., 0] * np.sin(theta) + sat_b[..., 1] * np.cos(theta)
    delta = np.stack((rx_b[..., 0] - sx_rot, rx_b[..., 1] - sy_rot, rx_b[..., 2] - sat_b[..., 2]), axis=-1)
    ranges = np.linalg.norm(delta, axis=-1)
    jac = np.full_like(delta, np.nan, dtype=np.float64)
    valid = np.isfinite(ranges) & (ranges > 1.0e-6) & np.isfinite(delta).all(axis=-1)
    jac[valid] = delta[valid] / ranges[valid, None]
    return ranges, jac


def native_doppler_geom_and_jacobian(
    sat_ecef: np.ndarray,
    receiver_ecef: np.ndarray,
    sat_vel: np.ndarray,
    receiver_vel: np.ndarray,
    sat_clock_drift_mps: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return native geometric range-rate and d(rate)/d(receiver_velocity)."""
    sat = np.asarray(sat_ecef, dtype=np.float64)
    rx = np.asarray(receiver_ecef, dtype=np.float64)
    sv = np.asarray(sat_vel, dtype=np.float64)
    rv = np.asarray(receiver_vel, dtype=np.float64)
    sat_b, rx_b, sv_b, rv_b = np.broadcast_arrays(sat, rx, sv, rv)
    delta = sat_b - rx_b
    ranges = np.linalg.norm(delta, axis=-1)
    los = np.full_like(delta, np.nan, dtype=np.float64)
    valid = np.isfinite(ranges) & (ranges > 1.0e-6) & np.isfinite(delta).all(axis=-1)
    los[valid] = delta[valid] / ranges[valid, None]

    euclidean_rate = np.sum(los * (sv_b - rv_b), axis=-1)
    sagnac_rate = EARTH_ROTATION_RATE_RAD_S * (
        sv_b[..., 0] * rx_b[..., 1]
        + sat_b[..., 0] * rv_b[..., 1]
        - sv_b[..., 1] * rx_b[..., 0]
        - sat_b[..., 1] * rv_b[..., 0]
    ) / LIGHT_SPEED_MPS
    geom = euclidean_rate - sagnac_rate
    if sat_clock_drift_mps is not None:
        drift = np.asarray(sat_clock_drift_mps, dtype=np.float64)
        drift_b = np.broadcast_to(drift, geom.shape)
        finite = np.isfinite(drift_b)
        geom = geom.copy()
        geom[finite] -= drift_b[finite]

    jac = np.stack(
        (
            -los[..., 0] + EARTH_ROTATION_RATE_RAD_S * sat_b[..., 1] / LIGHT_SPEED_MPS,
            -los[..., 1] - EARTH_ROTATION_RATE_RAD_S * sat_b[..., 0] / LIGHT_SPEED_MPS,
            -los[..., 2],
        ),
        axis=-1,
    )
    jac[~valid] = np.nan
    return geom, jac


def _clock_value_for_slots(clock: np.ndarray, sys_kind: np.ndarray, n_clock: int) -> np.ndarray:
    out = np.full(sys_kind.shape, np.nan, dtype=np.float64)
    for epoch_idx in range(sys_kind.shape[0]):
        design = fill_clock_design(np.asarray(sys_kind[epoch_idx], dtype=np.int32), n_clock)
        out[epoch_idx] = design @ clock[epoch_idx]
    return out


def _slot_metadata(batch: TripArrays, slot_idx: int) -> tuple[int, int, str]:
    constellation_type, svid, signal_type = batch.slot_keys[slot_idx]
    return constellation_to_matlab_sys(int(constellation_type)), int(svid), slot_frequency_label(str(signal_type))


def taroz_imu_body_gravity_residuals(
    state_i: np.ndarray,
    state_j: np.ndarray,
    *,
    attitude_idx: int,
    dt_s: float,
    gravity_nav: np.ndarray,
    delta_p_body: np.ndarray,
    delta_v_body: np.ndarray,
    delta_angle: np.ndarray,
    delta_p_bias_accel_jac: np.ndarray | None = None,
    delta_v_bias_accel_jac: np.ndarray | None = None,
    delta_p_bias_gyro_jac: np.ndarray | None = None,
    delta_v_bias_gyro_jac: np.ndarray | None = None,
    delta_angle_bias_gyro_jac: np.ndarray | None = None,
    accel_bias: np.ndarray | None = None,
    gyro_bias: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return Taroz/GTSAM-style body-frame IMU residuals for one interval."""

    s_i = np.asarray(state_i, dtype=np.float64).reshape(-1)
    s_j = np.asarray(state_j, dtype=np.float64).reshape(-1)
    dt = float(dt_s)
    gravity = np.asarray(gravity_nav, dtype=np.float64).reshape(3)
    rot_i = rotvec_to_rotm(s_i[attitude_idx : attitude_idx + 3])
    rot_j = rotvec_to_rotm(s_j[attitude_idx : attitude_idx + 3])
    delta_r = _correct_imu_angle_delta(
        delta_angle,
        delta_angle_bias_gyro_jac,
        gyro_bias,
        fallback_diag=dt,
    )
    delta_p = _correct_imu_pva_delta(
        delta_p_body,
        delta_p_bias_accel_jac,
        accel_bias,
        delta_p_bias_gyro_jac,
        gyro_bias,
        accel_fallback_diag=0.5 * dt * dt,
    )
    delta_v = _correct_imu_pva_delta(
        delta_v_body,
        delta_v_bias_accel_jac,
        accel_bias,
        delta_v_bias_gyro_jac,
        gyro_bias,
        accel_fallback_diag=dt,
    )
    predicted_rot_j = rot_i @ rotvec_to_rotm(delta_r)
    predicted_p_j = s_i[:3] + s_i[3:6] * dt + 0.5 * gravity * dt * dt + rot_i @ delta_p
    predicted_v_j = s_i[3:6] + gravity * dt + rot_i @ delta_v
    res_p = rot_j.T @ (predicted_p_j - s_j[:3])
    res_v = rot_j.T @ (predicted_v_j - s_j[3:6])
    res_r = rotm_to_rotvec(rot_j.T @ predicted_rot_j)
    return (
        res_p,
        res_v,
        res_r,
    )


def native_imu_body_gravity_residuals(
    state_i: np.ndarray,
    state_j: np.ndarray,
    *,
    attitude_idx: int,
    dt_s: float,
    gravity_nav: np.ndarray,
    delta_p_body: np.ndarray,
    delta_v_body: np.ndarray,
    delta_angle: np.ndarray,
    delta_p_bias_accel_jac: np.ndarray | None = None,
    delta_v_bias_accel_jac: np.ndarray | None = None,
    delta_p_bias_gyro_jac: np.ndarray | None = None,
    delta_v_bias_gyro_jac: np.ndarray | None = None,
    delta_angle_bias_gyro_jac: np.ndarray | None = None,
    accel_bias: np.ndarray | None = None,
    gyro_bias: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mirror the native CUDA ``imu_gravity`` P/V/R topology."""

    return taroz_imu_body_gravity_residuals(
        state_i,
        state_j,
        attitude_idx=attitude_idx,
        dt_s=dt_s,
        gravity_nav=gravity_nav,
        delta_p_body=delta_p_body,
        delta_v_body=delta_v_body,
        delta_angle=delta_angle,
        delta_p_bias_accel_jac=delta_p_bias_accel_jac,
        delta_v_bias_accel_jac=delta_v_bias_accel_jac,
        delta_p_bias_gyro_jac=delta_p_bias_gyro_jac,
        delta_v_bias_gyro_jac=delta_v_bias_gyro_jac,
        delta_angle_bias_gyro_jac=delta_angle_bias_gyro_jac,
        accel_bias=accel_bias,
        gyro_bias=gyro_bias,
    )


def _finite_jacobian_or_diag(value: np.ndarray | None, fallback_diag: float) -> np.ndarray:
    if value is None:
        return np.eye(3, dtype=np.float64) * float(fallback_diag)
    jac = np.asarray(value, dtype=np.float64).reshape(3, 3)
    return np.where(np.isfinite(jac), jac, 0.0)


def _jacobian_interval(value: np.ndarray | None, epoch_idx: int) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 2 and arr.shape == (3, 3):
        return arr
    if arr.ndim >= 3 and epoch_idx < arr.shape[0]:
        return np.asarray(arr[epoch_idx], dtype=np.float64).reshape(3, 3)
    return None


def _has_nonzero_jacobian(value: np.ndarray | None) -> bool:
    if value is None:
        return False
    jac = np.asarray(value, dtype=np.float64)
    finite = np.isfinite(jac)
    return bool(np.any(finite & (np.abs(jac) > 0.0)))


def _finite_vector_or_none(value: np.ndarray | None) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64).reshape(3)
    if not np.isfinite(arr).all():
        return None
    return arr


def _correct_imu_angle_delta(
    delta: np.ndarray,
    gyro_jac: np.ndarray | None,
    gyro_bias: np.ndarray | None,
    *,
    fallback_diag: float,
) -> np.ndarray:
    corrected = np.asarray(delta, dtype=np.float64).reshape(3).copy()
    bg = _finite_vector_or_none(gyro_bias)
    if bg is not None:
        corrected -= _finite_jacobian_or_diag(gyro_jac, fallback_diag) @ bg
    return corrected


def _correct_imu_pva_delta(
    delta: np.ndarray,
    accel_jac: np.ndarray | None,
    accel_bias: np.ndarray | None,
    gyro_jac: np.ndarray | None,
    gyro_bias: np.ndarray | None,
    *,
    accel_fallback_diag: float,
) -> np.ndarray:
    corrected = np.asarray(delta, dtype=np.float64).reshape(3).copy()
    ba = _finite_vector_or_none(accel_bias)
    if ba is not None:
        corrected -= _finite_jacobian_or_diag(accel_jac, accel_fallback_diag) @ ba
    bg = _finite_vector_or_none(gyro_bias)
    if bg is not None and _has_nonzero_jacobian(gyro_jac):
        corrected -= _finite_jacobian_or_diag(gyro_jac, 0.0) @ bg
    return corrected


def imu_body_gravity_residual_frame(
    batch: TripArrays,
    state: np.ndarray,
    *,
    label: str,
    weight: float = 1.0,
    imu_factor_use_next_bias: bool = True,
) -> pd.DataFrame:
    """Return native-vs-Taroz residual rows for body-frame IMU intervals."""

    preint = getattr(batch, "imu_preintegration", None)
    if preint is None or getattr(preint, "gravity_ecef", None) is None:
        return pd.DataFrame()
    values = np.asarray(state, dtype=np.float64)
    attitude_idx = 7 + int(batch.n_clock)
    if values.ndim != 2 or values.shape[1] < attitude_idx + 3:
        return pd.DataFrame()

    delta_t = np.asarray(preint.delta_t_s, dtype=np.float64).reshape(-1)
    delta_p = np.asarray(preint.delta_p_body, dtype=np.float64).reshape(-1, 3)
    delta_v = np.asarray(preint.delta_v_body, dtype=np.float64).reshape(-1, 3)
    delta_angle = np.asarray(preint.delta_angle_rad, dtype=np.float64).reshape(-1, 3)
    gravity = np.asarray(preint.gravity_ecef, dtype=np.float64).reshape(-1, 3)
    sample_count = np.asarray(preint.sample_count, dtype=np.int32).reshape(-1)
    delta_p_bias_accel_jac = getattr(preint, "delta_p_bias_accel_jac", None)
    delta_v_bias_accel_jac = getattr(preint, "delta_v_bias_accel_jac", None)
    delta_p_bias_gyro_jac = getattr(preint, "delta_p_bias_gyro_jac", None)
    delta_v_bias_gyro_jac = getattr(preint, "delta_v_bias_gyro_jac", None)
    delta_angle_bias_gyro_jac = getattr(preint, "delta_angle_bias_gyro_jac", None)
    n_interval = min(
        delta_t.size,
        delta_p.shape[0],
        delta_v.shape[0],
        delta_angle.shape[0],
        gravity.shape[0],
        sample_count.size,
        max(values.shape[0] - 1, 0),
    )
    if n_interval <= 0:
        return pd.DataFrame()

    graph_dt = np.ones(n_interval, dtype=np.float64)
    if batch.dt is not None and np.asarray(batch.dt).size > 1:
        graph_dt_src = np.asarray(batch.dt, dtype=np.float64).reshape(-1)[:n_interval]
        graph_dt[: graph_dt_src.size] = graph_dt_src

    accel_bias_idx = attitude_idx + 3
    gyro_bias_idx = attitude_idx + 6
    rows: list[dict[str, object]] = []
    for epoch_idx in range(n_interval):
        valid = (
            sample_count[epoch_idx] > 0
            and np.isfinite(delta_t[epoch_idx])
            and delta_t[epoch_idx] > 0.0
            and np.isfinite(graph_dt[epoch_idx])
            and graph_dt[epoch_idx] > 0.0
            and np.isfinite(delta_p[epoch_idx]).all()
            and np.isfinite(delta_v[epoch_idx]).all()
            and np.isfinite(delta_angle[epoch_idx]).all()
            and np.isfinite(gravity[epoch_idx]).all()
        )
        if not valid:
            continue
        bias_epoch_idx = epoch_idx + 1 if imu_factor_use_next_bias else epoch_idx
        accel_bias = None
        gyro_bias = None
        if bias_epoch_idx < values.shape[0] and values.shape[1] >= accel_bias_idx + 3:
            candidate = values[bias_epoch_idx, accel_bias_idx : accel_bias_idx + 3]
            if np.isfinite(candidate).all():
                accel_bias = candidate
        if bias_epoch_idx < values.shape[0] and values.shape[1] >= gyro_bias_idx + 3:
            candidate = values[bias_epoch_idx, gyro_bias_idx : gyro_bias_idx + 3]
            if np.isfinite(candidate).all():
                gyro_bias = candidate
        p_acc_jac = _jacobian_interval(delta_p_bias_accel_jac, epoch_idx)
        v_acc_jac = _jacobian_interval(delta_v_bias_accel_jac, epoch_idx)
        p_gyro_jac = _jacobian_interval(delta_p_bias_gyro_jac, epoch_idx)
        v_gyro_jac = _jacobian_interval(delta_v_bias_gyro_jac, epoch_idx)
        angle_gyro_jac = _jacobian_interval(delta_angle_bias_gyro_jac, epoch_idx)
        native_p, native_v, native_r = native_imu_body_gravity_residuals(
            values[epoch_idx],
            values[epoch_idx + 1],
            attitude_idx=attitude_idx,
            dt_s=float(delta_t[epoch_idx]),
            gravity_nav=gravity[epoch_idx],
            delta_p_body=delta_p[epoch_idx],
            delta_v_body=delta_v[epoch_idx],
            delta_angle=delta_angle[epoch_idx],
            delta_p_bias_accel_jac=p_acc_jac,
            delta_v_bias_accel_jac=v_acc_jac,
            delta_p_bias_gyro_jac=p_gyro_jac,
            delta_v_bias_gyro_jac=v_gyro_jac,
            delta_angle_bias_gyro_jac=angle_gyro_jac,
            accel_bias=accel_bias,
            gyro_bias=gyro_bias,
        )
        taroz_p, taroz_v, taroz_r = taroz_imu_body_gravity_residuals(
            values[epoch_idx],
            values[epoch_idx + 1],
            attitude_idx=attitude_idx,
            dt_s=float(delta_t[epoch_idx]),
            gravity_nav=gravity[epoch_idx],
            delta_p_body=delta_p[epoch_idx],
            delta_v_body=delta_v[epoch_idx],
            delta_angle=delta_angle[epoch_idx],
            delta_p_bias_accel_jac=p_acc_jac,
            delta_v_bias_accel_jac=v_acc_jac,
            delta_p_bias_gyro_jac=p_gyro_jac,
            delta_v_bias_gyro_jac=v_gyro_jac,
            delta_angle_bias_gyro_jac=angle_gyro_jac,
            accel_bias=accel_bias,
            gyro_bias=gyro_bias,
        )
        for factor, native_res, taroz_res in (
            ("IMU_P", native_p, taroz_p),
            ("IMU_V", native_v, taroz_v),
            ("IMU_R", native_r, taroz_r),
        ):
            for axis, (native_value, taroz_value) in enumerate(zip(native_res, taroz_res)):
                rows.append(
                    {
                        "label": label,
                        "factor": factor,
                        "freq": "IMU",
                        "epoch_index": int(getattr(batch, "build_start_epoch", 0)) + int(epoch_idx) + 1,
                        "utcTimeMillis": int(round(float(batch.times_ms[epoch_idx]))),
                        "nextUtcTimeMillis": int(round(float(batch.times_ms[epoch_idx + 1]))),
                        "sys": 0,
                        "svid": 0,
                        "slot_index": -1,
                        "axis": int(axis),
                        "weight": float(weight),
                        "native_residual": float(native_value),
                        "taroz_linear_residual": float(taroz_value),
                        "native_minus_taroz_linear": float(native_value - taroz_value),
                    }
                )
    return pd.DataFrame(rows)


def taroz_linearization_residual_frame(
    batch: TripArrays,
    origin_state: np.ndarray,
    eval_state: np.ndarray,
    *,
    label: str,
    tdcp_use_drift: bool = False,
    tdcp_native_ref_ecef: np.ndarray | None = None,
) -> pd.DataFrame:
    """Return native-vs-Taroz-linear residual rows for active P/D/L factors."""
    origin_x, origin_v, _origin_clk, _origin_drift = _state_parts(origin_state, batch.n_clock)
    eval_x, eval_v, eval_clk, eval_drift = _state_parts(eval_state, batch.n_clock)
    n_epoch = int(batch.times_ms.size)
    n_slot = len(batch.slot_keys)
    sys_kind = (
        np.asarray(batch.sys_kind, dtype=np.int32)
        if batch.sys_kind is not None
        else np.zeros((n_epoch, n_slot), dtype=np.int32)
    )

    rows: list[dict[str, object]] = []

    pr_weights = _active_weights(batch, "weights_fgo", "weights")
    if pr_weights is not None:
        origin_range, origin_jac = native_range_and_jacobian(batch.sat_ecef, origin_x[:, None, :])
        eval_range, _ = native_range_and_jacobian(batch.sat_ecef, eval_x[:, None, :])
        clock_eval = _clock_value_for_slots(eval_clk, sys_kind, batch.n_clock)
        pre_res = np.asarray(batch.pseudorange, dtype=np.float64) - origin_range
        native = np.asarray(batch.pseudorange, dtype=np.float64) - (eval_range + clock_eval)
        linear = pre_res - np.sum(origin_jac * (eval_x - origin_x)[:, None, :], axis=2) - clock_eval
        active = np.isfinite(pr_weights) & (pr_weights > 0.0)
        for epoch_idx, slot_idx in zip(*np.nonzero(active)):
            _append_row(rows, batch, label, "P", epoch_idx, slot_idx, native, linear, pr_weights)

    doppler_weights = _active_weights(batch, "doppler_weights_fgo", "doppler_weights")
    if (
        doppler_weights is not None
        and batch.doppler is not None
        and batch.sat_vel is not None
    ):
        origin_geom, origin_jac = native_doppler_geom_and_jacobian(
            batch.sat_ecef,
            origin_x[:, None, :],
            batch.sat_vel,
            origin_v[:, None, :],
            batch.sat_clock_drift_mps,
        )
        eval_geom, _ = native_doppler_geom_and_jacobian(
            batch.sat_ecef,
            eval_x[:, None, :],
            batch.sat_vel,
            eval_v[:, None, :],
            batch.sat_clock_drift_mps,
        )
        doppler = np.asarray(batch.doppler, dtype=np.float64)
        pre_res = doppler - origin_geom
        native = doppler - (eval_drift[:, None] + eval_geom)
        linear = pre_res - np.sum(origin_jac * (eval_v - origin_v)[:, None, :], axis=2) - eval_drift[:, None]
        active = np.isfinite(doppler_weights) & (doppler_weights > 0.0)
        for epoch_idx, slot_idx in zip(*np.nonzero(active)):
            _append_row(rows, batch, label, "D", epoch_idx, slot_idx, native, linear, doppler_weights)

    tdcp_weights = _active_weights(batch, "tdcp_weights_fgo", "tdcp_weights")
    if tdcp_weights is not None and batch.tdcp_meas is not None and n_epoch > 1:
        origin_range_next, origin_jac_next = native_range_and_jacobian(batch.sat_ecef[1:], origin_x[1:, None, :])
        del origin_range_next
        if tdcp_native_ref_ecef is not None:
            native_ref = np.asarray(tdcp_native_ref_ecef, dtype=np.float64).reshape(n_epoch, 3)
            native_range_next, eval_jac_next = native_range_and_jacobian(batch.sat_ecef[1:], native_ref[1:, None, :])
            del native_range_next
            dx_eval = (eval_x[1:] - native_ref[1:]) - (eval_x[:-1] - native_ref[:-1])
        else:
            eval_range_next, eval_jac_next = native_range_and_jacobian(batch.sat_ecef[1:], eval_x[1:, None, :])
            del eval_range_next
            dx_eval = eval_x[1:] - eval_x[:-1]
        dx_linear = (eval_x[1:] - origin_x[1:]) - (eval_x[:-1] - origin_x[:-1])
        native_pred = np.sum(eval_jac_next * dx_eval[:, None, :], axis=2)
        linear_pred = np.sum(origin_jac_next * dx_linear[:, None, :], axis=2)
        valid_time = np.ones(native_pred.shape[0], dtype=bool)
        if tdcp_use_drift:
            if batch.dt is None:
                valid_time[:] = False
            else:
                dt = np.asarray(batch.dt, dtype=np.float64).reshape(-1)[: native_pred.shape[0]]
                valid_time = np.isfinite(dt) & (dt > 0.0)
                clock_term = 0.5 * dt[:, None] * (eval_drift[:-1, None] + eval_drift[1:, None])
                native_pred += clock_term
                linear_pred += clock_term
        else:
            for epoch_idx in range(native_pred.shape[0]):
                design = fill_clock_design(np.asarray(sys_kind[epoch_idx + 1], dtype=np.int32), batch.n_clock)
                clock_delta = eval_clk[epoch_idx + 1] - eval_clk[epoch_idx]
                clock_term = design @ clock_delta
                native_pred[epoch_idx] += clock_term
                linear_pred[epoch_idx] += clock_term
        tdcp = np.asarray(batch.tdcp_meas, dtype=np.float64)
        native = tdcp - native_pred
        linear = tdcp - linear_pred
        active = np.isfinite(tdcp_weights) & (tdcp_weights > 0.0)
        active[: valid_time.size] &= valid_time[:, None]
        for epoch_idx, slot_idx in zip(*np.nonzero(active)):
            _append_row(rows, batch, label, "L", epoch_idx, slot_idx, native, linear, tdcp_weights)

    return pd.DataFrame(
        rows,
        columns=[
            "label",
            "factor",
            "freq",
            "epoch_index",
            "utcTimeMillis",
            "nextUtcTimeMillis",
            "sys",
            "svid",
            "slot_index",
            "weight",
            "native_residual",
            "taroz_linear_residual",
            "native_minus_taroz_linear",
        ],
    )


def _append_row(
    rows: list[dict[str, object]],
    batch: TripArrays,
    label: str,
    factor: str,
    epoch_idx: int,
    slot_idx: int,
    native: np.ndarray,
    linear: np.ndarray,
    weights: np.ndarray,
) -> None:
    sys, svid, freq = _slot_metadata(batch, int(slot_idx))
    next_time = 0
    if factor == "L" and epoch_idx + 1 < batch.times_ms.size:
        next_time = int(round(float(batch.times_ms[epoch_idx + 1])))
    native_value = float(native[epoch_idx, slot_idx])
    linear_value = float(linear[epoch_idx, slot_idx])
    rows.append(
        {
            "label": label,
            "factor": factor,
            "freq": freq,
            "epoch_index": int(getattr(batch, "build_start_epoch", 0)) + int(epoch_idx) + 1,
            "utcTimeMillis": int(round(float(batch.times_ms[epoch_idx]))),
            "nextUtcTimeMillis": next_time,
            "sys": sys,
            "svid": svid,
            "slot_index": int(slot_idx),
            "weight": float(weights[epoch_idx, slot_idx]),
            "native_residual": native_value,
            "taroz_linear_residual": linear_value,
            "native_minus_taroz_linear": native_value - linear_value,
        },
    )


def summarize_linearization_frame(frame: pd.DataFrame) -> list[LinearizationDeltaStats]:
    stats: list[LinearizationDeltaStats] = []
    if frame.empty:
        return stats
    for (label, factor), group in frame.groupby(["label", "factor"], sort=True):
        native = pd.to_numeric(group["native_residual"], errors="coerce").to_numpy(dtype=np.float64)
        linear = pd.to_numeric(group["taroz_linear_residual"], errors="coerce").to_numpy(dtype=np.float64)
        delta = pd.to_numeric(group["native_minus_taroz_linear"], errors="coerce").to_numpy(dtype=np.float64)
        weight = pd.to_numeric(group["weight"], errors="coerce").to_numpy(dtype=np.float64)
        valid = np.isfinite(native) & np.isfinite(linear) & np.isfinite(delta) & np.isfinite(weight) & (weight > 0.0)
        if not np.any(valid):
            stats.append(
                LinearizationDeltaStats(
                    label=str(label),
                    factor=str(factor),
                    count=0,
                    native_weighted_rms=None,
                    taroz_linear_weighted_rms=None,
                    delta_weighted_rms=None,
                    delta_median_abs=None,
                    delta_p95_abs=None,
                    delta_max_abs=None,
                )
            )
            continue
        native = native[valid]
        linear = linear[valid]
        delta = delta[valid]
        weight = weight[valid]
        weight_sum = float(np.sum(weight))
        abs_delta = np.abs(delta)
        stats.append(
            LinearizationDeltaStats(
                label=str(label),
                factor=str(factor),
                count=int(delta.size),
                native_weighted_rms=float(np.sqrt(np.sum(weight * native * native) / weight_sum)),
                taroz_linear_weighted_rms=float(np.sqrt(np.sum(weight * linear * linear) / weight_sum)),
                delta_weighted_rms=float(np.sqrt(np.sum(weight * delta * delta) / weight_sum)),
                delta_median_abs=float(np.median(abs_delta)),
                delta_p95_abs=float(np.percentile(abs_delta, 95.0)),
                delta_max_abs=float(np.max(abs_delta)),
            )
        )
    return stats


def seed_vd_state_for_batch(
    batch: TripArrays,
    *,
    imu_attitude_state: bool = False,
    imu_accel_bias_state: bool = False,
    imu_gyro_bias_state: bool = False,
) -> np.ndarray:
    raw_wls = run_wls(
        batch.sat_ecef,
        batch.pseudorange,
        batch.weights,
        sys_kind=batch.sys_kind,
        n_clock=batch.n_clock,
        fallback_xyz=batch.kaggle_wls,
    )
    raw_wls[:, :3] = _repair_baseline_wls(batch.times_ms, raw_wls[:, :3])
    baseline_state, _, _, _ = fit_state_with_clock_bias(
        batch.sat_ecef,
        batch.pseudorange,
        batch.weights,
        batch.kaggle_wls,
        sys_kind=batch.sys_kind,
        n_clock=batch.n_clock,
    )
    raw_state, _, _, _ = fit_state_with_clock_bias(
        batch.sat_ecef,
        batch.pseudorange,
        batch.weights,
        raw_wls[:, :3],
        sys_kind=batch.sys_kind,
        n_clock=batch.n_clock,
    )
    dt = batch.dt if batch.dt is not None else np.zeros(batch.times_ms.size, dtype=np.float64)
    return _seed_vd_state(
        raw_state,
        baseline_state,
        dt,
        n_clock=batch.n_clock,
        clock_drift_mps=batch.clock_drift_mps,
        imu_attitude_state=imu_attitude_state,
        imu_accel_bias_state=imu_accel_bias_state,
        imu_gyro_bias_state=imu_gyro_bias_state,
    )


def _velocity_from_positions(times_ms: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    times = np.asarray(times_ms, dtype=np.float64).reshape(-1)
    pos = np.asarray(xyz, dtype=np.float64).reshape(times.size, 3)
    vel = np.zeros_like(pos)
    if times.size <= 1:
        return vel
    for idx in range(times.size):
        if idx == 0:
            i0, i1 = 0, 1
        elif idx == times.size - 1:
            i0, i1 = times.size - 2, times.size - 1
        else:
            i0, i1 = idx - 1, idx + 1
        dt_s = (times[i1] - times[i0]) / 1000.0
        if np.isfinite(dt_s) and dt_s > 0.0 and np.isfinite(pos[[i0, i1]]).all():
            vel[idx] = (pos[i1] - pos[i0]) / dt_s
    return vel


def taroz_gnss_initial_state_for_batch(batch: TripArrays) -> np.ndarray:
    """Return the ``fgo_gnss.m`` initial state: posbl, posbl.gradient, obs.clk/dclk."""

    n_epoch = int(batch.times_ms.size)
    n_clock = int(batch.n_clock)
    state = np.zeros((n_epoch, 7 + n_clock), dtype=np.float64)
    state[:, :3] = np.asarray(batch.kaggle_wls, dtype=np.float64).reshape(n_epoch, 3)
    state[:, 3:6] = _velocity_from_positions(batch.times_ms, state[:, :3])
    if batch.clock_bias_m is not None:
        clock = np.asarray(batch.clock_bias_m, dtype=np.float64).reshape(-1)[:n_epoch]
        finite = np.isfinite(clock)
        state[finite, 6] = clock[finite]
    if batch.clock_drift_mps is not None:
        drift = np.asarray(batch.clock_drift_mps, dtype=np.float64).reshape(-1)[:n_epoch]
        finite = np.isfinite(drift)
        # Android raw bridge stores the opposite sign of Taroz obs.dclk.
        state[finite, 6 + n_clock] = -drift[finite]
    return state


def load_taroz_gnss_state_csv_for_batch(path: Path, batch: TripArrays) -> np.ndarray:
    """Load ``export_gsdc2023_taroz_gnss_state`` CSV into native VD state layout."""

    frame = pd.read_csv(path)
    required = {"utcTimeMillis", "ecef_x", "ecef_y", "ecef_z"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    key = pd.DataFrame({"utcTimeMillis": np.asarray(batch.times_ms, dtype=np.float64).round().astype(np.int64)})
    keyed = frame.copy()
    keyed["utcTimeMillis"] = pd.to_numeric(keyed["utcTimeMillis"], errors="coerce").round().astype("Int64")
    joined = key.merge(keyed, on="utcTimeMillis", how="left")
    n_epoch = int(batch.times_ms.size)
    n_clock = int(batch.n_clock)
    state = np.zeros((n_epoch, 7 + n_clock), dtype=np.float64)
    state[:, :3] = joined[["ecef_x", "ecef_y", "ecef_z"]].to_numpy(dtype=np.float64)
    vel_cols = ["velocity_ecef_x", "velocity_ecef_y", "velocity_ecef_z"]
    if set(vel_cols).issubset(joined.columns):
        state[:, 3:6] = joined[vel_cols].to_numpy(dtype=np.float64)
    else:
        state[:, 3:6] = _velocity_from_positions(batch.times_ms, state[:, :3])
    for clock_idx in range(n_clock):
        col = f"clock_bias_m_{clock_idx}"
        if col in joined.columns:
            state[:, 6 + clock_idx] = pd.to_numeric(joined[col], errors="coerce").to_numpy(dtype=np.float64)
    if "clock_drift_mps" in joined.columns:
        state[:, 6 + n_clock] = pd.to_numeric(joined["clock_drift_mps"], errors="coerce").to_numpy(dtype=np.float64)
    return state


def _resolve_eval_state_csv(path: Path, *, include_imu: bool) -> Path:
    state_path = Path(path)
    if state_path.is_file():
        return state_path
    if not state_path.is_dir():
        raise FileNotFoundError(state_path)
    candidates = (
        ("phone_data_imu_state.csv", "phone_data_gnss_graph_state.csv", "phone_data_gnss_initial_state.csv")
        if include_imu
        else ("phone_data_gnss_graph_state.csv", "phone_data_gnss_initial_state.csv")
    )
    for filename in candidates:
        candidate = state_path / filename
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(state_path)


def _overlay_finite_state(base: np.ndarray, update: np.ndarray) -> np.ndarray:
    out = np.asarray(base, dtype=np.float64).copy()
    values = np.asarray(update, dtype=np.float64)
    n_row = min(out.shape[0], values.shape[0])
    n_col = min(out.shape[1], values.shape[1])
    finite = np.isfinite(values[:n_row, :n_col])
    out[:n_row, :n_col][finite] = values[:n_row, :n_col][finite]
    return out


def load_taroz_state_csv_for_batch(
    path: Path,
    batch: TripArrays,
    *,
    trip_dir: Path,
    include_imu: bool = False,
) -> np.ndarray:
    """Load either ECEF GNSS exports or Taroz ENU FGO state exports into native layout."""

    state_path = _resolve_eval_state_csv(Path(path), include_imu=include_imu)
    columns = set(pd.read_csv(state_path, nrows=0).columns)
    if {"ecef_x", "ecef_y", "ecef_z"}.issubset(columns):
        state = load_taroz_gnss_state_csv_for_batch(state_path, batch)
    elif {"position_x", "position_y", "position_z"}.issubset(columns):
        state = _load_taroz_fgo_seed_state(state_path, batch, trip_dir=trip_dir)
    else:
        raise ValueError(f"{state_path} is not a recognized Taroz state CSV")

    if not include_imu:
        return state

    full_seed = seed_vd_state_for_batch(
        batch,
        imu_attitude_state=True,
        imu_accel_bias_state=True,
        imu_gyro_bias_state=True,
    )
    return _overlay_finite_state(full_seed, state)


def _taroz_clock_term(clock: np.ndarray, sigtype: int, n_clock: int) -> float:
    """Return ``PseudorangeFactor_XC`` clock design: c0, plus c_sigtype when nonzero."""

    values = np.asarray(clock, dtype=np.float64).reshape(-1)
    if n_clock <= 0 or values.size <= 0:
        return 0.0
    out = float(values[0])
    idx = int(sigtype)
    if 0 < idx < min(n_clock, values.size):
        out += float(values[idx])
    return out


def taroz_gtsam_gnss_factor_residual_frame(
    factor_mask_csv: Path,
    state_csv: Path,
    *,
    residual_csv: Path | None = None,
    n_clock: int = 7,
) -> pd.DataFrame:
    """Evaluate exported Taroz/GTSAM GNSS factors from CSV metadata and ENU state.

    This mirrors the custom ``gtsam_gnss`` factors exactly:
    ``PseudorangeFactor_XC``, ``DopplerFactor_VD``, ``TDCPFactor_XXCC``, and
    ``TDCPFactor_XXDD``.  It is intentionally separate from
    ``taroz_linearization_residual_frame``, which compares native ECEF geometry.
    """

    factor_frame = pd.read_csv(factor_mask_csv)
    state_frame = pd.read_csv(state_csv)
    required_factor = {
        "field",
        "freq",
        "epoch_index",
        "next_epoch_index",
        "factor_model",
        "sigtype",
        "measurement",
        "dt_s",
        "los_e",
        "los_n",
        "los_u",
        "origin1_e",
        "origin1_n",
        "origin1_u",
        "origin2_e",
        "origin2_n",
        "origin2_u",
    }
    required_state = {
        "epoch_index",
        "position_x",
        "position_y",
        "position_z",
        "velocity_x",
        "velocity_y",
        "velocity_z",
        "clock_drift_mps",
    } | {f"clock_bias_m_{idx}" for idx in range(int(n_clock))}
    missing_factor = sorted(required_factor - set(factor_frame.columns))
    missing_state = sorted(required_state - set(state_frame.columns))
    if missing_factor:
        raise ValueError(f"{factor_mask_csv} is missing columns: {missing_factor}")
    if missing_state:
        raise ValueError(f"{state_csv} is missing columns: {missing_state}")

    state_frame = state_frame.copy()
    state_frame["epoch_index"] = pd.to_numeric(state_frame["epoch_index"], errors="coerce").astype("Int64")
    state_by_epoch = state_frame.dropna(subset=["epoch_index"]).set_index("epoch_index", drop=False)
    clock_cols = [f"clock_bias_m_{idx}" for idx in range(int(n_clock))]

    if residual_csv is not None:
        residual_frame = pd.read_csv(residual_csv)
        key_cols = [
            "field",
            "freq",
            "epoch_index",
            "next_epoch_index",
            "sys",
            "svid",
            "sat_col",
            "factor_model",
        ]
        keep_cols = [col for col in key_cols + ["initial_residual", "residual", "factor_error"] if col in residual_frame.columns]
        residual_lookup = residual_frame[keep_cols]
        factor_frame = factor_frame.merge(
            residual_lookup,
            on=[col for col in key_cols if col in factor_frame.columns and col in residual_lookup.columns],
            how="left",
        )

    rows: list[dict[str, object]] = []
    for _, row in factor_frame.iterrows():
        epoch = int(row["epoch_index"])
        next_epoch = int(row["next_epoch_index"])
        if epoch not in state_by_epoch.index:
            continue
        state_i = state_by_epoch.loc[epoch]
        if isinstance(state_i, pd.DataFrame):
            state_i = state_i.iloc[0]
        pos_i = state_i[["position_x", "position_y", "position_z"]].to_numpy(dtype=np.float64)
        vel_i = state_i[["velocity_x", "velocity_y", "velocity_z"]].to_numpy(dtype=np.float64)
        clock_i = state_i[clock_cols].to_numpy(dtype=np.float64)
        drift_i = float(state_i["clock_drift_mps"])
        los = row[["los_e", "los_n", "los_u"]].to_numpy(dtype=np.float64)
        org1 = row[["origin1_e", "origin1_n", "origin1_u"]].to_numpy(dtype=np.float64)
        measurement = float(row["measurement"])
        field = str(row["field"])
        model = str(row["factor_model"])

        if field == "P":
            clock_term = _taroz_clock_term(clock_i, int(row["sigtype"]), int(n_clock))
            computed = float(np.dot(los, pos_i - org1) + clock_term - measurement)
        elif field == "D":
            computed = float(np.dot(los, vel_i - org1) + drift_i - measurement)
        elif field == "L":
            if next_epoch not in state_by_epoch.index:
                continue
            state_j = state_by_epoch.loc[next_epoch]
            if isinstance(state_j, pd.DataFrame):
                state_j = state_j.iloc[0]
            pos_j = state_j[["position_x", "position_y", "position_z"]].to_numpy(dtype=np.float64)
            clock_j = state_j[clock_cols].to_numpy(dtype=np.float64)
            drift_j = float(state_j["clock_drift_mps"])
            org2 = row[["origin2_e", "origin2_n", "origin2_u"]].to_numpy(dtype=np.float64)
            dx = (pos_j - org2) - (pos_i - org1)
            if model == "XXDD":
                clock_delta = float(row["dt_s"]) * (drift_i + drift_j) / 2.0
            else:
                clock_delta = float(clock_j[0] - clock_i[0])
            computed = float(np.dot(los, dx) + clock_delta - measurement)
        else:
            continue

        out = {col: row[col] for col in factor_frame.columns if col not in {"residual", "initial_residual", "factor_error"}}
        out["computed_residual"] = computed
        if "residual" in row.index:
            taroz_residual = float(row["residual"])
            out["taroz_residual"] = taroz_residual
            out["computed_minus_taroz"] = computed - taroz_residual
        if "initial_residual" in row.index:
            out["taroz_initial_residual"] = float(row["initial_residual"])
        if "factor_error" in row.index:
            out["taroz_factor_error"] = float(row["factor_error"])
        rows.append(out)
    return pd.DataFrame(rows)


def taroz_huber_factor_error(residual: np.ndarray | float, sigma: np.ndarray | float, huber_k: float) -> np.ndarray:
    """Return GTSAM Huber robust factor error for scalar residuals."""

    residual_arr = np.asarray(residual, dtype=np.float64)
    sigma_arr = np.asarray(sigma, dtype=np.float64)
    z = np.abs(residual_arr / sigma_arr)
    k = float(huber_k)
    if k <= 0.0:
        return 0.5 * z * z
    return np.where(z <= k, 0.5 * z * z, k * z - 0.5 * k * k)


def taroz_gtsam_gnss_graph_cost_frame(
    factor_mask_csv: Path,
    state_csv: Path,
    *,
    residual_csv: Path | None = None,
    n_clock: int = 7,
    pr_huber_k: float = 0.1,
    doppler_huber_k: float = 0.4,
    carrier_huber_k: float = 0.2,
    motion_sigma_m: float = 0.05,
    clock_sigma_m: float = 0.1,
    time_diff_threshold_s: float = 1.5,
    constrained_tolerance: float = 1.0e-9,
) -> pd.DataFrame:
    """Evaluate Taroz GNSS-only graph cost rows for P/D/L + Motion + Clock.

    Taroz ``fgo_gnss.m`` adds P/D/L robust GNSS factors plus
    ``MotionFactor_XXVV`` and ``ClockFactor_CCDD`` between adjacent epochs. The
    initial priors use infinite sigmas and contribute zero cost, so they are not
    emitted here.
    """

    residual_frame = taroz_gtsam_gnss_factor_residual_frame(
        factor_mask_csv,
        state_csv,
        residual_csv=residual_csv,
        n_clock=n_clock,
    )
    rows: list[dict[str, object]] = []
    huber_by_field = {
        "P": float(pr_huber_k),
        "D": float(doppler_huber_k),
        "L": float(carrier_huber_k),
    }
    for _, row in residual_frame.iterrows():
        field = str(row["field"])
        if field not in huber_by_field:
            continue
        residual = float(row["computed_residual"])
        sigma = float(row["sigma"])
        huber_k = huber_by_field[field]
        rows.append(
            {
                "factor": field,
                "component": "scalar",
                "epoch_index": int(row["epoch_index"]),
                "next_epoch_index": int(row["next_epoch_index"]),
                "residual": residual,
                "sigma": sigma,
                "huber_k": huber_k,
                "cost": float(taroz_huber_factor_error(residual, sigma, huber_k)),
                "source": "gnss",
            }
        )

    state_frame = pd.read_csv(state_csv).copy()
    required_state = {
        "epoch_index",
        "utcTimeMillis",
        "position_x",
        "position_y",
        "position_z",
        "velocity_x",
        "velocity_y",
        "velocity_z",
        "clock_drift_mps",
    } | {f"clock_bias_m_{idx}" for idx in range(int(n_clock))}
    missing_state = sorted(required_state - set(state_frame.columns))
    if missing_state:
        raise ValueError(f"{state_csv} is missing columns: {missing_state}")
    state_frame = state_frame.sort_values("epoch_index")
    epoch = state_frame["epoch_index"].to_numpy(dtype=np.int64)
    utc = state_frame["utcTimeMillis"].to_numpy(dtype=np.float64)
    pos = state_frame[["position_x", "position_y", "position_z"]].to_numpy(dtype=np.float64)
    vel = state_frame[["velocity_x", "velocity_y", "velocity_z"]].to_numpy(dtype=np.float64)
    clock_cols = [f"clock_bias_m_{idx}" for idx in range(int(n_clock))]
    clock = state_frame[clock_cols].to_numpy(dtype=np.float64)
    drift = state_frame["clock_drift_mps"].to_numpy(dtype=np.float64)

    for idx in range(max(0, len(state_frame) - 1)):
        dt_s = float((utc[idx + 1] - utc[idx]) / 1000.0)
        if not np.isfinite(dt_s) or dt_s >= float(time_diff_threshold_s):
            continue
        motion_res = (pos[idx + 1] - pos[idx]) - 0.5 * (vel[idx] + vel[idx + 1]) * dt_s
        for component, residual in zip(("x", "y", "z"), motion_res):
            rows.append(
                {
                    "factor": "Motion",
                    "component": component,
                    "epoch_index": int(epoch[idx]),
                    "next_epoch_index": int(epoch[idx + 1]),
                    "residual": float(residual),
                    "sigma": float(motion_sigma_m),
                    "huber_k": 0.0,
                    "cost": float(taroz_huber_factor_error(residual, motion_sigma_m, 0.0)),
                    "source": "motion",
                }
            )

        clock_res = clock[idx + 1] - clock[idx]
        clock_res[0] -= 0.5 * (drift[idx] + drift[idx + 1]) * dt_s
        for clock_idx, residual in enumerate(clock_res):
            sigma = float(clock_sigma_m) if clock_idx == 0 else 0.0
            if sigma == 0.0:
                cost = 0.0 if abs(float(residual)) <= float(constrained_tolerance) else float("inf")
            else:
                cost = float(taroz_huber_factor_error(residual, sigma, 0.0))
            rows.append(
                {
                    "factor": "Clock",
                    "component": f"c{clock_idx}",
                    "epoch_index": int(epoch[idx]),
                    "next_epoch_index": int(epoch[idx + 1]),
                    "residual": float(residual),
                    "sigma": sigma,
                    "huber_k": 0.0,
                    "cost": cost,
                    "source": "clock",
                }
            )
    return pd.DataFrame(rows)


def summarize_taroz_gtsam_graph_cost(cost_frame: pd.DataFrame) -> pd.DataFrame:
    """Return per-factor cost sums from ``taroz_gtsam_gnss_graph_cost_frame``."""

    if cost_frame.empty:
        return pd.DataFrame(columns=["factor", "count", "cost"])
    return (
        cost_frame.groupby("factor", dropna=False)["cost"]
        .agg(count="count", cost="sum")
        .reset_index()
        .sort_values("factor")
        .reset_index(drop=True)
    )


def perturb_vd_state(
    state: np.ndarray,
    *,
    position_m: float,
    velocity_mps: float,
    clock_m: float,
    drift_mps: float,
    n_clock: int,
) -> np.ndarray:
    perturbed = np.asarray(state, dtype=np.float64).copy()
    n_epoch = perturbed.shape[0]
    if n_epoch == 0:
        return perturbed
    phase = np.arange(n_epoch, dtype=np.float64)
    directions = np.column_stack(
        (
            np.sin(0.37 * phase) + 0.25,
            np.cos(0.23 * phase) - 0.1,
            np.sin(0.19 * phase + 0.5),
        )
    )
    norms = np.linalg.norm(directions, axis=1)
    directions[norms > 0.0] /= norms[norms > 0.0, None]
    perturbed[:, :3] += float(position_m) * directions
    perturbed[:, 3:6] += float(velocity_mps) * directions
    perturbed[:, 6 : 6 + n_clock] += float(clock_m)
    perturbed[:, 6 + n_clock] += float(drift_mps)
    return perturbed


def _auto_path(data_root: Path, trip: str, filename: str) -> Path:
    return data_root / trip / filename


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--trip", required=True, help="Trip path like train/<course>/<phone>")
    parser.add_argument("--max-epochs", type=int, default=0)
    parser.add_argument("--start-epoch", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--matlab-factor-mask", type=Path, default=None)
    parser.add_argument("--matlab-imu-factor-mask", type=Path, default=None)
    parser.add_argument(
        "--apply-matlab-factor-mask",
        action="store_true",
        help="apply Taroz GNSS/IMU factor masks to the Python batch before residual evaluation",
    )
    parser.add_argument("--matlab-residual-diagnostics-mask", type=Path, default=None)
    parser.add_argument(
        "--use-matlab-residual-diagnostics-mask",
        action="store_true",
        help="apply Taroz residual diagnostics p/d/l_factor_finite flags before building Python factors",
    )
    parser.add_argument("--tdcp-use-drift", action="store_true")
    parser.add_argument("--perturb-position-m", type=float, default=5.0)
    parser.add_argument("--perturb-velocity-mps", type=float, default=0.5)
    parser.add_argument("--perturb-clock-m", type=float, default=0.0)
    parser.add_argument("--perturb-drift-mps", type=float, default=0.0)
    parser.add_argument(
        "--origin-state",
        choices=("native_seed", "taroz_gnss"),
        default="native_seed",
        help="linearization origin used for Taroz fixed-linear residuals",
    )
    parser.add_argument(
        "--origin-state-csv",
        type=Path,
        default=None,
        help="optional Taroz state CSV to use as linearization origin and IMU seed label",
    )
    parser.add_argument(
        "--eval-state-csv",
        type=Path,
        default=None,
        action="append",
        help="optional state CSV to evaluate against the selected origin; may be repeated",
    )
    parser.add_argument(
        "--include-imu",
        action="store_true",
        help="also build taroz_body IMU preintegration and compare native imu_gravity residuals",
    )
    parser.add_argument(
        "--dual-frequency",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="include L5/E5 observations; leave disabled for Taroz settings with L5=0",
    )
    return parser


def run_audit(args: argparse.Namespace) -> dict[str, object]:
    data_root = Path(args.data_root)
    trip = str(args.trip)
    max_epochs = int(args.max_epochs)
    if max_epochs <= 0:
        max_epochs = 1_000_000_000
    mask_path = args.matlab_residual_diagnostics_mask
    if args.use_matlab_residual_diagnostics_mask and mask_path is None:
        candidate = _auto_path(data_root, trip, "phone_data_residual_diagnostics.csv")
        mask_path = candidate if candidate.is_file() else None
    include_imu = bool(getattr(args, "include_imu", False))
    matlab_factor_path = getattr(args, "matlab_factor_mask", None)
    matlab_imu_factor_path = getattr(args, "matlab_imu_factor_mask", None)

    batch = build_trip_arrays(
        data_root / trip,
        max_epochs=max_epochs,
        start_epoch=int(args.start_epoch),
        constellation_type=1,
        signal_type="GPS_L1_CA",
        weight_mode="sin2el",
        fgo_weight_mode="taroz_sn",
        multi_gnss=True,
        use_tdcp=True,
        apply_observation_mask=True,
        data_root=data_root,
        trip=trip,
        dual_frequency=bool(args.dual_frequency),
        matlab_residual_diagnostics_mask_path=mask_path,
        imu_frame=IMU_TAROZ_BODY_DELTA_FRAME if include_imu else "body",
        imu_sample_dt_mode="taroz" if include_imu else "bounded",
    )
    if bool(getattr(args, "apply_matlab_factor_mask", False)):
        if matlab_factor_path is None or not Path(matlab_factor_path).is_file():
            raise ValueError("--apply-matlab-factor-mask requires a readable --matlab-factor-mask")
        batch = _apply_taroz_factor_mask_to_batch(
            batch,
            Path(matlab_factor_path),
            trip_dir=data_root / trip,
            use_fixed_values=False,
        )
        if include_imu:
            if matlab_imu_factor_path is None or not Path(matlab_imu_factor_path).is_file():
                raise ValueError("--apply-matlab-factor-mask with --include-imu requires --matlab-imu-factor-mask")
            batch = _apply_taroz_imu_factor_mask_to_batch(batch, Path(matlab_imu_factor_path))
    seed = seed_vd_state_for_batch(
        batch,
        imu_attitude_state=include_imu,
        imu_accel_bias_state=include_imu,
        imu_gyro_bias_state=include_imu,
    )
    origin_state = seed
    if str(args.origin_state) == "taroz_gnss":
        origin_state = taroz_gnss_initial_state_for_batch(batch)
    if args.origin_state_csv is not None:
        origin_state = load_taroz_state_csv_for_batch(
            Path(args.origin_state_csv),
            batch,
            trip_dir=data_root / trip,
            include_imu=include_imu,
        )
    perturbed = perturb_vd_state(
        origin_state,
        position_m=float(args.perturb_position_m),
        velocity_mps=float(args.perturb_velocity_mps),
        clock_m=float(args.perturb_clock_m),
        drift_mps=float(args.perturb_drift_mps),
        n_clock=batch.n_clock,
    )
    frames = [
        taroz_linearization_residual_frame(
            batch,
            origin_state,
            origin_state,
            label="seed",
            tdcp_use_drift=bool(args.tdcp_use_drift),
            tdcp_native_ref_ecef=(
                origin_state[:, :3].copy()
                if int(getattr(batch, "tdcp_geometry_correction_count", 0)) > 0
                else None
            ),
        ),
        taroz_linearization_residual_frame(
            batch,
            origin_state,
            perturbed,
            label="perturbed",
            tdcp_use_drift=bool(args.tdcp_use_drift),
            tdcp_native_ref_ecef=(
                origin_state[:, :3].copy()
                if int(getattr(batch, "tdcp_geometry_correction_count", 0)) > 0
                else None
            ),
        ),
    ]
    eval_state_paths = list(args.eval_state_csv or [])
    for eval_state_path in eval_state_paths:
        eval_state = load_taroz_state_csv_for_batch(
            Path(eval_state_path),
            batch,
            trip_dir=data_root / trip,
            include_imu=include_imu,
        )
        frames.append(
            taroz_linearization_residual_frame(
                batch,
                origin_state,
                eval_state,
                label=Path(eval_state_path).stem,
                tdcp_use_drift=bool(args.tdcp_use_drift),
                tdcp_native_ref_ecef=(
                    origin_state[:, :3].copy()
                    if int(getattr(batch, "tdcp_geometry_correction_count", 0)) > 0
                    else None
                ),
            )
        )
    if include_imu:
        frames.extend(
            [
                imu_body_gravity_residual_frame(batch, origin_state, label="seed"),
                imu_body_gravity_residual_frame(batch, perturbed, label="perturbed"),
            ]
        )
        for eval_state_path in eval_state_paths:
            eval_state = load_taroz_state_csv_for_batch(
                Path(eval_state_path),
                batch,
                trip_dir=data_root / trip,
                include_imu=True,
            )
            frames.append(imu_body_gravity_residual_frame(batch, eval_state, label=Path(eval_state_path).stem))
    frame = pd.concat(frames, ignore_index=True)
    stats = summarize_linearization_frame(frame)

    output = Path(args.output) if args.output is not None else None
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output.with_suffix(".residuals.csv"), index=False)

    payload: dict[str, object] = {
        "trip": trip,
        "epochs": int(batch.times_ms.size),
        "n_clock": int(batch.n_clock),
        "matlab_residual_diagnostics_mask_path": str(mask_path) if mask_path is not None else None,
        "matlab_factor_mask_path": str(matlab_factor_path) if matlab_factor_path is not None else None,
        "matlab_imu_factor_mask_path": str(matlab_imu_factor_path) if matlab_imu_factor_path is not None else None,
        "applied_matlab_factor_mask": bool(getattr(args, "apply_matlab_factor_mask", False)),
        "tdcp_geometry_correction_count": int(batch.tdcp_geometry_correction_count),
        "tdcp_native_uses_linearization_ref": bool(int(batch.tdcp_geometry_correction_count) > 0),
        "tdcp_use_drift": bool(args.tdcp_use_drift),
        "origin_state": str(args.origin_state),
        "origin_state_csv": str(args.origin_state_csv) if args.origin_state_csv is not None else None,
        "eval_state_csv": [str(path) for path in eval_state_paths],
        "dual_frequency": bool(args.dual_frequency),
        "include_imu": include_imu,
        "imu_residual_rows": int(frame["factor"].astype(str).str.startswith("IMU_").sum()) if not frame.empty else 0,
        "perturb_position_m": float(args.perturb_position_m),
        "perturb_velocity_mps": float(args.perturb_velocity_mps),
        "stats": [asdict(item) for item in stats],
    }
    if output is not None:
        output.with_suffix(".json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    payload = run_audit(args)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
