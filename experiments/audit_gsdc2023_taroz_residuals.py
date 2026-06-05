"""Audit Python raw-bridge factors against Taroz MATLAB residual exports."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from experiments.audit_gsdc2023_taroz_linearization import (
    imu_body_gravity_residual_frame,
    seed_vd_state_for_batch,
)
from experiments.gsdc2023_imu import (
    IMU_TAROZ_BODY_DELTA_FRAME,
    gtsam_rzryrx_to_rotm,
    rotm_to_rotvec,
    rotvec_to_rotm,
)
from experiments.gsdc2023_observation_matrix import TripArrays
from experiments.gsdc2023_raw_bridge import DEFAULT_ROOT, _apply_taroz_factor_mask_to_batch, build_trip_arrays
from experiments.gsdc2023_residual_model import (
    geometric_range_rate_with_sagnac,
    geometric_range_with_sagnac,
    receiver_velocity_from_reference,
)
from experiments.gsdc2023_signal_model import constellation_to_matlab_sys, slot_frequency_label


FACTOR_FIELDS = ("P", "D", "L")
FACTOR_KEY_COLUMNS = ("field", "freq", "utcTimeMillis", "nextUtcTimeMillis", "sys", "svid")
IMU_FACTOR_FIELDS = ("IMU_P", "IMU_V", "IMU_R")
IMU_FACTOR_KEY_COLUMNS = ("field", "freq", "utcTimeMillis", "nextUtcTimeMillis", "sys", "svid", "axis")


@dataclass(frozen=True)
class ResidualDeltaStats:
    column: str
    count: int
    mean_abs: float | None
    median_abs: float | None
    p95_abs: float | None
    max_abs: float | None


def _finite_stats(column: str, delta: np.ndarray) -> ResidualDeltaStats:
    arr = np.asarray(delta, dtype=np.float64)
    arr = np.abs(arr[np.isfinite(arr)])
    if arr.size == 0:
        return ResidualDeltaStats(
            column=column,
            count=0,
            mean_abs=None,
            median_abs=None,
            p95_abs=None,
            max_abs=None,
        )
    return ResidualDeltaStats(
        column=column,
        count=int(arr.size),
        mean_abs=float(np.mean(arr)),
        median_abs=float(np.median(arr)),
        p95_abs=float(np.percentile(arr, 95.0)),
        max_abs=float(np.max(arr)),
    )


def _active_weights(batch: TripArrays, attr_fgo: str, attr_base: str) -> np.ndarray | None:
    fgo_value = getattr(batch, attr_fgo, None)
    if fgo_value is not None:
        return np.asarray(fgo_value, dtype=np.float64)
    base_value = getattr(batch, attr_base, None)
    if base_value is None:
        return None
    return np.asarray(base_value, dtype=np.float64)


def _epoch_index(batch: TripArrays, epoch_idx: int) -> int:
    return int(getattr(batch, "build_start_epoch", 0)) + int(epoch_idx) + 1


def _slot_sys_svid_freq(slot_key: tuple[int, int, str]) -> tuple[int, int, str]:
    constellation_type, svid, signal_type = slot_key
    return constellation_to_matlab_sys(int(constellation_type)), int(svid), slot_frequency_label(str(signal_type))


def python_factor_mask(batch: TripArrays) -> pd.DataFrame:
    """Return Python factor keys in the same key space as Taroz factor-mask CSV."""
    rows: list[dict[str, object]] = []
    n_epoch = int(batch.times_ms.size)
    slot_keys = tuple(batch.slot_keys)

    pr_weights = _active_weights(batch, "weights_fgo", "weights")
    doppler_weights = _active_weights(batch, "doppler_weights_fgo", "doppler_weights")
    tdcp_weights = _active_weights(batch, "tdcp_weights_fgo", "tdcp_weights")

    def append_row(field: str, epoch_idx: int, slot_idx: int, next_epoch_idx: int | None = None) -> None:
        sys, svid, freq = _slot_sys_svid_freq(slot_keys[slot_idx])
        rows.append(
            {
                "field": field,
                "freq": freq,
                "epoch_index": _epoch_index(batch, epoch_idx),
                "utcTimeMillis": int(round(float(batch.times_ms[epoch_idx]))),
                "next_epoch_index": _epoch_index(batch, next_epoch_idx) if next_epoch_idx is not None else 0,
                "nextUtcTimeMillis": (
                    int(round(float(batch.times_ms[next_epoch_idx]))) if next_epoch_idx is not None else 0
                ),
                "sys": sys,
                "svid": svid,
            }
        )

    if pr_weights is not None:
        active = np.isfinite(pr_weights) & (pr_weights > 0.0)
        for epoch_idx, slot_idx in zip(*np.nonzero(active)):
            append_row("P", int(epoch_idx), int(slot_idx))

    if doppler_weights is not None:
        active = np.isfinite(doppler_weights) & (doppler_weights > 0.0)
        for epoch_idx, slot_idx in zip(*np.nonzero(active)):
            append_row("D", int(epoch_idx), int(slot_idx))

    if tdcp_weights is not None and n_epoch > 1:
        active = np.isfinite(tdcp_weights) & (tdcp_weights > 0.0)
        for epoch_idx, slot_idx in zip(*np.nonzero(active)):
            if int(epoch_idx) + 1 >= n_epoch:
                continue
            append_row("L", int(epoch_idx), int(slot_idx), int(epoch_idx) + 1)

    columns = [
        "field",
        "freq",
        "epoch_index",
        "utcTimeMillis",
        "next_epoch_index",
        "nextUtcTimeMillis",
        "sys",
        "svid",
    ]
    return pd.DataFrame(rows, columns=columns)


def compare_factor_mask(python_mask: pd.DataFrame, matlab_mask: pd.DataFrame) -> pd.DataFrame:
    """Compare Python and Taroz factor keys by field/frequency."""
    py = python_mask.loc[python_mask["field"].isin(FACTOR_FIELDS), list(FACTOR_KEY_COLUMNS)].drop_duplicates()
    ml = matlab_mask.loc[matlab_mask["field"].isin(FACTOR_FIELDS), list(FACTOR_KEY_COLUMNS)].drop_duplicates()
    py["_python"] = True
    ml["_matlab"] = True
    merged = py.merge(ml, on=list(FACTOR_KEY_COLUMNS), how="outer")
    merged["_python"] = merged["_python"].eq(True)
    merged["_matlab"] = merged["_matlab"].eq(True)

    rows: list[dict[str, object]] = []
    for (field, freq), group in merged.groupby(["field", "freq"], dropna=False):
        in_py = group["_python"].to_numpy(dtype=bool)
        in_ml = group["_matlab"].to_numpy(dtype=bool)
        rows.append(
            {
                "field": str(field),
                "freq": str(freq),
                "python_count": int(np.count_nonzero(in_py)),
                "matlab_count": int(np.count_nonzero(in_ml)),
                "matched_count": int(np.count_nonzero(in_py & in_ml)),
                "only_python_count": int(np.count_nonzero(in_py & ~in_ml)),
                "only_matlab_count": int(np.count_nonzero(~in_py & in_ml)),
            }
        )
    return pd.DataFrame(rows).sort_values(["field", "freq"]).reset_index(drop=True)


def _valid_imu_interval_indices(batch: TripArrays) -> list[int]:
    preint = getattr(batch, "imu_preintegration", None)
    if preint is None or getattr(preint, "gravity_ecef", None) is None:
        return []
    delta_t = np.asarray(preint.delta_t_s, dtype=np.float64).reshape(-1)
    delta_p = np.asarray(preint.delta_p_body, dtype=np.float64).reshape(-1, 3)
    delta_v = np.asarray(preint.delta_v_body, dtype=np.float64).reshape(-1, 3)
    delta_angle = np.asarray(preint.delta_angle_rad, dtype=np.float64).reshape(-1, 3)
    gravity = np.asarray(preint.gravity_ecef, dtype=np.float64).reshape(-1, 3)
    sample_count = np.asarray(preint.sample_count, dtype=np.int32).reshape(-1)
    n_interval = min(
        delta_t.size,
        delta_p.shape[0],
        delta_v.shape[0],
        delta_angle.shape[0],
        gravity.shape[0],
        sample_count.size,
        max(int(batch.times_ms.size) - 1, 0),
    )
    if n_interval <= 0:
        return []

    graph_dt = np.ones(n_interval, dtype=np.float64)
    if batch.dt is not None and np.asarray(batch.dt).size > 1:
        graph_dt_src = np.asarray(batch.dt, dtype=np.float64).reshape(-1)[:n_interval]
        graph_dt[: graph_dt_src.size] = graph_dt_src

    valid: list[int] = []
    for epoch_idx in range(n_interval):
        if (
            sample_count[epoch_idx] > 0
            and np.isfinite(delta_t[epoch_idx])
            and delta_t[epoch_idx] > 0.0
            and np.isfinite(graph_dt[epoch_idx])
            and graph_dt[epoch_idx] > 0.0
            and np.isfinite(delta_p[epoch_idx]).all()
            and np.isfinite(delta_v[epoch_idx]).all()
            and np.isfinite(delta_angle[epoch_idx]).all()
            and np.isfinite(gravity[epoch_idx]).all()
        ):
            valid.append(int(epoch_idx))
    return valid


def _imu_interval_metadata(batch: TripArrays) -> dict[int, dict[str, float | int]]:
    preint = getattr(batch, "imu_preintegration", None)
    if preint is None:
        return {}
    valid_indices = _valid_imu_interval_indices(batch)
    if not valid_indices:
        return {}
    delta_t = np.asarray(preint.delta_t_s, dtype=np.float64).reshape(-1)
    sample_count = np.asarray(preint.sample_count, dtype=np.int32).reshape(-1)
    graph_dt = np.ones(delta_t.size, dtype=np.float64)
    if batch.dt is not None and np.asarray(batch.dt).size > 1:
        graph_dt_src = np.asarray(batch.dt, dtype=np.float64).reshape(-1)[: delta_t.size]
        graph_dt[: graph_dt_src.size] = graph_dt_src
    return {
        int(epoch_idx): {
            "sample_count": int(sample_count[epoch_idx]),
            "graph_dt_s": float(graph_dt[epoch_idx]),
            "preintegrated_dt_s": float(delta_t[epoch_idx]),
        }
        for epoch_idx in valid_indices
    }


def python_imu_factor_mask(batch: TripArrays) -> pd.DataFrame:
    """Return body-frame IMU factor keys in a Taroz-exportable key space."""
    rows: list[dict[str, object]] = []
    interval_meta = _imu_interval_metadata(batch)
    for epoch_idx, meta in interval_meta.items():
        for field in IMU_FACTOR_FIELDS:
            for axis in range(3):
                rows.append(
                    {
                        "field": field,
                        "freq": "IMU",
                        "epoch_index": _epoch_index(batch, epoch_idx),
                        "utcTimeMillis": int(round(float(batch.times_ms[epoch_idx]))),
                        "next_epoch_index": _epoch_index(batch, epoch_idx + 1),
                        "nextUtcTimeMillis": int(round(float(batch.times_ms[epoch_idx + 1]))),
                        "sys": 0,
                        "svid": 0,
                        "axis": int(axis),
                        "sample_count": int(meta["sample_count"]),
                        "graph_dt_s": float(meta["graph_dt_s"]),
                        "preintegrated_dt_s": float(meta["preintegrated_dt_s"]),
                    }
                )
    return pd.DataFrame(
        rows,
        columns=[
            "field",
            "freq",
            "epoch_index",
            "utcTimeMillis",
            "next_epoch_index",
            "nextUtcTimeMillis",
            "sys",
            "svid",
            "axis",
            "sample_count",
            "graph_dt_s",
            "preintegrated_dt_s",
        ],
    )


def python_imu_residual_diagnostics(batch: TripArrays, state: np.ndarray) -> pd.DataFrame:
    """Return Taroz/GTSAM-style body-gravity IMU residual diagnostics."""
    frame = imu_body_gravity_residual_frame(batch, state, label="seed")
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "field",
                "freq",
                "epoch_index",
                "utcTimeMillis",
                "next_epoch_index",
                "nextUtcTimeMillis",
                "sys",
                "svid",
                "axis",
                "sample_count",
                "graph_dt_s",
                "preintegrated_dt_s",
                "weight",
                "imu_native_residual",
                "imu_taroz_reference_residual",
                "imu_native_minus_taroz_reference",
            ]
        )
    out = frame.rename(
        columns={
            "factor": "field",
            "native_residual": "imu_native_residual",
            "taroz_linear_residual": "imu_taroz_reference_residual",
            "native_minus_taroz_linear": "imu_native_minus_taroz_reference",
        }
    ).copy()
    next_time_to_epoch = {
        int(round(float(time_ms))): _epoch_index(batch, epoch_idx)
        for epoch_idx, time_ms in enumerate(np.asarray(batch.times_ms, dtype=np.float64))
    }
    out["next_epoch_index"] = [
        next_time_to_epoch.get(int(round(float(value))), 0) for value in out["nextUtcTimeMillis"].to_numpy()
    ]
    interval_meta = _imu_interval_metadata(batch)
    epoch_to_interval = {
        _epoch_index(batch, epoch_idx): meta for epoch_idx, meta in interval_meta.items()
    }
    out["sample_count"] = [
        int(epoch_to_interval.get(int(epoch), {}).get("sample_count", 0))
        for epoch in out["epoch_index"].to_numpy()
    ]
    out["graph_dt_s"] = [
        float(epoch_to_interval.get(int(epoch), {}).get("graph_dt_s", np.nan))
        for epoch in out["epoch_index"].to_numpy()
    ]
    out["preintegrated_dt_s"] = [
        float(epoch_to_interval.get(int(epoch), {}).get("preintegrated_dt_s", np.nan))
        for epoch in out["epoch_index"].to_numpy()
    ]
    columns = [
        "field",
        "freq",
        "epoch_index",
        "utcTimeMillis",
        "next_epoch_index",
        "nextUtcTimeMillis",
        "sys",
        "svid",
        "axis",
        "sample_count",
        "graph_dt_s",
        "preintegrated_dt_s",
        "weight",
        "imu_native_residual",
        "imu_taroz_reference_residual",
        "imu_native_minus_taroz_reference",
    ]
    return out[columns].reset_index(drop=True)


def _gtsam_rpy_to_rotm(rpy_rad: np.ndarray) -> np.ndarray:
    """Return matrices matching GTSAM ``Rot3.RzRyRx(roll, pitch, yaw)``."""
    return gtsam_rzryrx_to_rotm(rpy_rad)


def _row_xyz(row: pd.Series, prefix: str) -> np.ndarray:
    return np.array(
        [
            float(row[f"{prefix}_x"]),
            float(row[f"{prefix}_y"]),
            float(row[f"{prefix}_z"]),
        ],
        dtype=np.float64,
    )


def _row_rpy(row: pd.Series) -> np.ndarray:
    return np.array([float(row["roll"]), float(row["pitch"]), float(row["yaw"])], dtype=np.float64)


def python_imu_residual_diagnostics_from_matlab_exports(
    matlab_state: pd.DataFrame,
    matlab_preintegration: pd.DataFrame,
) -> pd.DataFrame:
    """Recompute Taroz/GTSAM IMU residuals from exported optimized state and PIM deltas."""

    if matlab_state.empty or matlab_preintegration.empty:
        return pd.DataFrame(
            columns=[
                "field",
                "freq",
                "epoch_index",
                "utcTimeMillis",
                "next_epoch_index",
                "nextUtcTimeMillis",
                "sys",
                "svid",
                "axis",
                "sample_count",
                "graph_dt_s",
                "preintegrated_dt_s",
                "imu_same_state_residual",
            ]
        )

    state_by_epoch = {
        int(round(float(row["epoch_index"]))): row for _, row in matlab_state.iterrows()
    }
    delta_prefix = "corrected_delta"
    required_delta_cols = {
        f"{delta_prefix}_r_x",
        f"{delta_prefix}_r_y",
        f"{delta_prefix}_r_z",
        f"{delta_prefix}_p_x",
        f"{delta_prefix}_p_y",
        f"{delta_prefix}_p_z",
        f"{delta_prefix}_v_x",
        f"{delta_prefix}_v_y",
        f"{delta_prefix}_v_z",
    }
    if not required_delta_cols.issubset(set(matlab_preintegration.columns)):
        delta_prefix = "delta"

    rows: list[dict[str, object]] = []
    for _, interval in matlab_preintegration.iterrows():
        epoch = int(round(float(interval["epoch_index"])))
        next_epoch = int(round(float(interval["next_epoch_index"])))
        state_i = state_by_epoch.get(epoch)
        state_j = state_by_epoch.get(next_epoch)
        if state_i is None or state_j is None:
            continue
        dt = float(interval["preintegrated_dt_s"])
        if not np.isfinite(dt) or dt <= 0.0:
            continue
        p_i = _row_xyz(state_i, "position")
        p_j = _row_xyz(state_j, "position")
        v_i = _row_xyz(state_i, "velocity")
        v_j = _row_xyz(state_j, "velocity")
        rot_i = _gtsam_rpy_to_rotm(_row_rpy(state_i))[0]
        rot_j = _gtsam_rpy_to_rotm(_row_rpy(state_j))[0]
        gravity = _row_xyz(interval, "gravity")
        delta_r = _row_xyz(interval, f"{delta_prefix}_r")
        delta_p = _row_xyz(interval, f"{delta_prefix}_p")
        delta_v = _row_xyz(interval, f"{delta_prefix}_v")
        if not (
            np.isfinite(p_i).all()
            and np.isfinite(p_j).all()
            and np.isfinite(v_i).all()
            and np.isfinite(v_j).all()
            and np.isfinite(rot_i).all()
            and np.isfinite(rot_j).all()
            and np.isfinite(gravity).all()
            and np.isfinite(delta_r).all()
            and np.isfinite(delta_p).all()
            and np.isfinite(delta_v).all()
        ):
            continue
        predicted_rot_j = rot_i @ rotvec_to_rotm(delta_r)
        predicted_p_j = p_i + v_i * dt + 0.5 * gravity * dt * dt + rot_i @ delta_p
        predicted_v_j = v_i + gravity * dt + rot_i @ delta_v
        residual_p = rot_j.T @ (predicted_p_j - p_j)
        residual_v = rot_j.T @ (predicted_v_j - v_j)
        residual_r = rotm_to_rotvec(rot_j.T @ predicted_rot_j)
        for field, residual in (
            ("IMU_R", residual_r),
            ("IMU_P", residual_p),
            ("IMU_V", residual_v),
        ):
            for axis, value in enumerate(residual):
                rows.append(
                    {
                        "field": field,
                        "freq": "IMU",
                        "epoch_index": epoch,
                        "utcTimeMillis": int(round(float(interval["utcTimeMillis"]))),
                        "next_epoch_index": next_epoch,
                        "nextUtcTimeMillis": int(round(float(interval["nextUtcTimeMillis"]))),
                        "sys": 0,
                        "svid": 0,
                        "axis": int(axis),
                        "sample_count": int(round(float(interval["sample_count"]))),
                        "graph_dt_s": float(interval["graph_dt_s"]),
                        "preintegrated_dt_s": dt,
                        "imu_same_state_residual": float(value),
                    }
                )
    return pd.DataFrame(
        rows,
        columns=[
            "field",
            "freq",
            "epoch_index",
            "utcTimeMillis",
            "next_epoch_index",
            "nextUtcTimeMillis",
            "sys",
            "svid",
            "axis",
            "sample_count",
            "graph_dt_s",
            "preintegrated_dt_s",
            "imu_same_state_residual",
        ],
    )


IMU_PREINTEGRATION_KEY_COLUMNS = ("epoch_index", "utcTimeMillis", "next_epoch_index", "nextUtcTimeMillis")


def _state_bias_by_epoch(matlab_state: pd.DataFrame | None) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    if matlab_state is None or matlab_state.empty:
        return {}
    required = {
        "epoch_index",
        "bias_acc_x",
        "bias_acc_y",
        "bias_acc_z",
        "bias_gyro_x",
        "bias_gyro_y",
        "bias_gyro_z",
    }
    if not required.issubset(set(matlab_state.columns)):
        return {}
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for _, row in matlab_state.iterrows():
        epoch = int(round(float(row["epoch_index"])))
        acc_bias = _row_xyz(row, "bias_acc")
        gyro_bias = _row_xyz(row, "bias_gyro")
        if np.isfinite(acc_bias).all() and np.isfinite(gyro_bias).all():
            out[epoch] = (acc_bias, gyro_bias)
    return out


def python_imu_preintegration_diagnostics(
    batch: TripArrays,
    matlab_state: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return Python taroz_body PIM deltas in the MATLAB preintegration export schema."""

    preint = getattr(batch, "imu_preintegration", None)
    if preint is None:
        return pd.DataFrame()
    valid_indices = _valid_imu_interval_indices(batch)
    if not valid_indices:
        return pd.DataFrame()

    delta_t = np.asarray(preint.delta_t_s, dtype=np.float64).reshape(-1)
    delta_r = np.asarray(preint.delta_angle_rad, dtype=np.float64).reshape(-1, 3)
    delta_p = np.asarray(preint.delta_p_body, dtype=np.float64).reshape(-1, 3)
    delta_v = np.asarray(preint.delta_v_body, dtype=np.float64).reshape(-1, 3)
    sample_count = np.asarray(preint.sample_count, dtype=np.int32).reshape(-1)
    graph_dt = np.ones(delta_t.size, dtype=np.float64)
    if batch.dt is not None and np.asarray(batch.dt).size > 1:
        graph_dt_src = np.asarray(batch.dt, dtype=np.float64).reshape(-1)[: delta_t.size]
        graph_dt[: graph_dt_src.size] = graph_dt_src

    p_acc_jac = getattr(preint, "delta_p_bias_accel_jac", None)
    v_acc_jac = getattr(preint, "delta_v_bias_accel_jac", None)
    p_gyro_jac = getattr(preint, "delta_p_bias_gyro_jac", None)
    v_gyro_jac = getattr(preint, "delta_v_bias_gyro_jac", None)
    r_gyro_jac = getattr(preint, "delta_angle_bias_gyro_jac", None)
    p_acc_jac_arr = np.asarray(p_acc_jac, dtype=np.float64) if p_acc_jac is not None else None
    v_acc_jac_arr = np.asarray(v_acc_jac, dtype=np.float64) if v_acc_jac is not None else None
    p_gyro_jac_arr = np.asarray(p_gyro_jac, dtype=np.float64) if p_gyro_jac is not None else None
    v_gyro_jac_arr = np.asarray(v_gyro_jac, dtype=np.float64) if v_gyro_jac is not None else None
    r_gyro_jac_arr = np.asarray(r_gyro_jac, dtype=np.float64) if r_gyro_jac is not None else None
    bias_by_epoch = _state_bias_by_epoch(matlab_state)

    rows: list[dict[str, object]] = []
    for epoch_idx in valid_indices:
        epoch = _epoch_index(batch, epoch_idx)
        next_epoch = _epoch_index(batch, epoch_idx + 1)
        corrected_r = np.full(3, np.nan, dtype=np.float64)
        corrected_p = np.full(3, np.nan, dtype=np.float64)
        corrected_v = np.full(3, np.nan, dtype=np.float64)
        if (
            next_epoch in bias_by_epoch
            and p_acc_jac_arr is not None
            and v_acc_jac_arr is not None
            and p_gyro_jac_arr is not None
            and v_gyro_jac_arr is not None
            and r_gyro_jac_arr is not None
        ):
            acc_bias, gyro_bias = bias_by_epoch[next_epoch]
            corrected_p = (
                delta_p[epoch_idx]
                - p_acc_jac_arr[epoch_idx] @ acc_bias
                - p_gyro_jac_arr[epoch_idx] @ gyro_bias
            )
            corrected_v = (
                delta_v[epoch_idx]
                - v_acc_jac_arr[epoch_idx] @ acc_bias
                - v_gyro_jac_arr[epoch_idx] @ gyro_bias
            )
            corrected_r = delta_r[epoch_idx] - r_gyro_jac_arr[epoch_idx] @ gyro_bias
        rows.append(
            {
                "epoch_index": epoch,
                "utcTimeMillis": int(round(float(batch.times_ms[epoch_idx]))),
                "next_epoch_index": next_epoch,
                "nextUtcTimeMillis": int(round(float(batch.times_ms[epoch_idx + 1]))),
                "sample_count": int(sample_count[epoch_idx]),
                "graph_dt_s": float(graph_dt[epoch_idx]),
                "preintegrated_dt_s": float(delta_t[epoch_idx]),
                "delta_r_x": float(delta_r[epoch_idx, 0]),
                "delta_r_y": float(delta_r[epoch_idx, 1]),
                "delta_r_z": float(delta_r[epoch_idx, 2]),
                "delta_p_x": float(delta_p[epoch_idx, 0]),
                "delta_p_y": float(delta_p[epoch_idx, 1]),
                "delta_p_z": float(delta_p[epoch_idx, 2]),
                "delta_v_x": float(delta_v[epoch_idx, 0]),
                "delta_v_y": float(delta_v[epoch_idx, 1]),
                "delta_v_z": float(delta_v[epoch_idx, 2]),
                "corrected_delta_r_x": float(corrected_r[0]),
                "corrected_delta_r_y": float(corrected_r[1]),
                "corrected_delta_r_z": float(corrected_r[2]),
                "corrected_delta_p_x": float(corrected_p[0]),
                "corrected_delta_p_y": float(corrected_p[1]),
                "corrected_delta_p_z": float(corrected_p[2]),
                "corrected_delta_v_x": float(corrected_v[0]),
                "corrected_delta_v_y": float(corrected_v[1]),
                "corrected_delta_v_z": float(corrected_v[2]),
            }
        )
    return pd.DataFrame(rows)


def compare_imu_preintegration_diagnostics(
    python_df: pd.DataFrame,
    matlab_df: pd.DataFrame,
) -> list[ResidualDeltaStats]:
    """Return PIM delta export deltas after joining by interval keys."""

    if python_df.empty or matlab_df.empty:
        return []
    key_cols = [column for column in IMU_PREINTEGRATION_KEY_COLUMNS if column in python_df.columns and column in matlab_df.columns]
    if not key_cols:
        return []
    merged = python_df.merge(matlab_df, on=key_cols, how="inner", suffixes=("_python", "_matlab"))
    stats: list[ResidualDeltaStats] = []
    for column in (
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
        "corrected_delta_r_x",
        "corrected_delta_r_y",
        "corrected_delta_r_z",
        "corrected_delta_p_x",
        "corrected_delta_p_y",
        "corrected_delta_p_z",
        "corrected_delta_v_x",
        "corrected_delta_v_y",
        "corrected_delta_v_z",
    ):
        py_col = f"{column}_python"
        ml_col = f"{column}_matlab"
        if py_col in merged.columns and ml_col in merged.columns:
            stats.append(_finite_stats(column, merged[py_col].to_numpy() - merged[ml_col].to_numpy()))
    return stats


def compare_imu_factor_mask(python_mask: pd.DataFrame, matlab_mask: pd.DataFrame) -> pd.DataFrame:
    """Compare Python and Taroz IMU factor keys by field and axis."""

    def prepare(frame: pd.DataFrame, marker: str) -> pd.DataFrame:
        out = frame.copy()
        if "factor" in out.columns and "field" not in out.columns:
            out = out.rename(columns={"factor": "field"})
        if "axis" not in out.columns:
            out["axis"] = -1
        for column in IMU_FACTOR_KEY_COLUMNS:
            if column not in out.columns:
                out[column] = "" if column == "freq" else 0
        out = out.loc[out["field"].astype(str).isin(IMU_FACTOR_FIELDS), list(IMU_FACTOR_KEY_COLUMNS)].drop_duplicates()
        out[marker] = True
        return out

    py = prepare(python_mask, "_python")
    ml = prepare(matlab_mask, "_matlab")
    merged = py.merge(ml, on=list(IMU_FACTOR_KEY_COLUMNS), how="outer")
    merged["_python"] = merged["_python"].eq(True)
    merged["_matlab"] = merged["_matlab"].eq(True)

    rows: list[dict[str, object]] = []
    for (field, axis), group in merged.groupby(["field", "axis"], dropna=False):
        in_py = group["_python"].to_numpy(dtype=bool)
        in_ml = group["_matlab"].to_numpy(dtype=bool)
        rows.append(
            {
                "field": str(field),
                "axis": int(axis),
                "python_count": int(np.count_nonzero(in_py)),
                "matlab_count": int(np.count_nonzero(in_ml)),
                "matched_count": int(np.count_nonzero(in_py & in_ml)),
                "only_python_count": int(np.count_nonzero(in_py & ~in_ml)),
                "only_matlab_count": int(np.count_nonzero(~in_py & in_ml)),
            }
        )
    return pd.DataFrame(rows).sort_values(["field", "axis"]).reset_index(drop=True)


def compare_imu_residual_diagnostics(python_df: pd.DataFrame, matlab_df: pd.DataFrame) -> list[ResidualDeltaStats]:
    """Return IMU residual deltas after joining by field/time/axis."""

    def prepare(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        if "factor" in out.columns and "field" not in out.columns:
            out = out.rename(columns={"factor": "field"})
        if "axis" not in out.columns:
            out["axis"] = -1
        return out

    py = prepare(python_df)
    ml = prepare(matlab_df)
    key_cols = [column for column in IMU_FACTOR_KEY_COLUMNS if column in py.columns and column in ml.columns]
    if not key_cols:
        return []
    merged = py.merge(ml, on=key_cols, how="inner", suffixes=("_python", "_matlab"))
    stats: list[ResidualDeltaStats] = []
    for column in (
        "imu_native_residual",
        "imu_taroz_reference_residual",
        "imu_native_minus_taroz_reference",
        "imu_same_state_residual",
        "native_residual",
        "taroz_linear_residual",
        "residual",
        "sample_count",
        "graph_dt_s",
        "preintegrated_dt_s",
    ):
        py_col = f"{column}_python"
        ml_col = f"{column}_matlab"
        if py_col in merged.columns and ml_col in merged.columns:
            stats.append(_finite_stats(column, merged[py_col].to_numpy() - merged[ml_col].to_numpy()))
    for python_residual_col, stat_name in (
        ("imu_same_state_residual_python", "imu_same_state_residual_vs_residual"),
        ("imu_same_state_residual", "imu_same_state_residual_vs_residual"),
        ("imu_taroz_reference_residual_python", "imu_taroz_reference_residual_vs_residual"),
        ("imu_taroz_reference_residual", "imu_taroz_reference_residual_vs_residual"),
    ):
        if python_residual_col not in merged.columns:
            continue
        for matlab_residual_col in ("residual_matlab", "residual"):
            if matlab_residual_col not in merged.columns:
                continue
            delta = (
                merged[python_residual_col].to_numpy()
                - merged[matlab_residual_col].to_numpy()
            )
            stats.append(_finite_stats(stat_name, delta))
            return stats
    return stats


def python_residual_diagnostics(batch: TripArrays) -> pd.DataFrame:
    """Build Python residual diagnostics comparable to Taroz export CSV."""
    n_epoch = int(batch.times_ms.size)
    n_slot = len(batch.slot_keys)
    if n_epoch == 0 or n_slot == 0:
        return pd.DataFrame()

    pr_weights = _active_weights(batch, "weights_fgo", "weights")
    doppler_weights = _active_weights(batch, "doppler_weights_fgo", "doppler_weights")
    tdcp_weights = _active_weights(batch, "tdcp_weights_fgo", "tdcp_weights")

    rx = np.asarray(batch.kaggle_wls[:, :3], dtype=np.float64)
    ranges = geometric_range_with_sagnac(batch.sat_ecef, rx[:, None, :])
    p_corrected = np.asarray(batch.pseudorange, dtype=np.float64)
    p_pre_respc = p_corrected - ranges

    d_obs = np.asarray(batch.doppler, dtype=np.float64) if batch.doppler is not None else None
    d_model = None
    d_model_alt = None
    d_pre_resd = None
    if batch.sat_vel is not None and d_obs is not None:
        rx_vel = receiver_velocity_from_reference(batch.times_ms, rx)
        geom_rate = geometric_range_rate_with_sagnac(batch.sat_ecef, rx[:, None, :], batch.sat_vel, rx_vel[:, None, :])
        if batch.sat_clock_drift_mps is not None and batch.sat_clock_drift_mps.shape == geom_rate.shape:
            finite_drift = np.isfinite(batch.sat_clock_drift_mps)
            geom_rate[finite_drift] -= batch.sat_clock_drift_mps[finite_drift]
        d_model = geom_rate
        d_model_alt = -geom_rate
        d_pre_resd = d_obs - d_model

    l_phase = np.zeros((n_epoch, n_slot), dtype=bool)
    if tdcp_weights is not None:
        active_pairs = np.isfinite(tdcp_weights) & (tdcp_weights > 0.0)
        pair_epoch_count = min(active_pairs.shape[0], max(n_epoch - 1, 0))
        for epoch_idx, slot_idx in zip(*np.nonzero(active_pairs[:pair_epoch_count])):
            l_phase[int(epoch_idx), int(slot_idx)] = True
            l_phase[int(epoch_idx) + 1, int(slot_idx)] = True

    rows: list[dict[str, object]] = []
    for epoch_idx in range(n_epoch):
        for slot_idx, slot_key in enumerate(batch.slot_keys):
            sys, svid, freq = _slot_sys_svid_freq(slot_key)
            p_active = bool(pr_weights is not None and pr_weights[epoch_idx, slot_idx] > 0.0)
            d_active = bool(doppler_weights is not None and doppler_weights[epoch_idx, slot_idx] > 0.0)
            l_active = bool(l_phase[epoch_idx, slot_idx])
            has_values = (
                p_active
                or d_active
                or l_active
                or np.isfinite(p_pre_respc[epoch_idx, slot_idx])
                or (d_pre_resd is not None and np.isfinite(d_pre_resd[epoch_idx, slot_idx]))
            )
            if not has_values:
                continue
            row = {
                "freq": freq,
                "epoch_index": _epoch_index(batch, epoch_idx),
                "utcTimeMillis": int(round(float(batch.times_ms[epoch_idx]))),
                "sys": sys,
                "svid": svid,
                "slot_index": int(slot_idx),
                "p_corrected_m": float(p_corrected[epoch_idx, slot_idx]),
                "p_range_m": float(ranges[epoch_idx, slot_idx]),
                "p_pre_respc_m": float(p_pre_respc[epoch_idx, slot_idx]),
                "p_factor_finite": p_active,
                "d_factor_finite": d_active,
                "l_factor_finite": l_active,
            }
            if d_obs is not None:
                row["d_obs_mps"] = float(d_obs[epoch_idx, slot_idx])
            if d_model is not None and d_model_alt is not None and d_pre_resd is not None:
                row["d_model_mps"] = float(d_model[epoch_idx, slot_idx])
                row["d_model_alt_mps"] = float(d_model_alt[epoch_idx, slot_idx])
                row["d_pre_resd_m"] = float(d_pre_resd[epoch_idx, slot_idx])
            rows.append(row)
    return pd.DataFrame(rows)


def compare_residual_diagnostics(python_df: pd.DataFrame, matlab_df: pd.DataFrame) -> list[ResidualDeltaStats]:
    """Return numeric deltas after joining Python and Taroz diagnostics."""
    key_cols = ["freq", "utcTimeMillis", "sys", "svid"]
    merged = python_df.merge(matlab_df, on=key_cols, how="inner", suffixes=("_python", "_matlab"))
    stats: list[ResidualDeltaStats] = []
    for column in ("p_corrected_m", "p_range_m", "p_pre_respc_m", "d_obs_mps"):
        py_col = f"{column}_python"
        ml_col = f"{column}_matlab"
        if py_col in merged.columns and ml_col in merged.columns:
            stats.append(_finite_stats(column, merged[py_col].to_numpy() - merged[ml_col].to_numpy()))
    if "d_model_mps_python" in merged.columns and "d_model_mps_matlab" in merged.columns:
        direct = merged["d_model_mps_python"].to_numpy() - merged["d_model_mps_matlab"].to_numpy()
        stats.append(_finite_stats("d_model_mps", direct))
    if "d_model_alt_mps_python" in merged.columns and "d_model_mps_matlab" in merged.columns:
        alt = merged["d_model_alt_mps_python"].to_numpy() - merged["d_model_mps_matlab"].to_numpy()
        stats.append(_finite_stats("d_model_alt_mps", alt))
    return stats


def filter_matlab_frame_to_python_window(matlab_df: pd.DataFrame, python_times_ms: np.ndarray) -> pd.DataFrame:
    """Restrict a Taroz export frame to the epoch window present in Python."""
    if "utcTimeMillis" not in matlab_df.columns:
        return matlab_df
    times = {int(round(float(value))) for value in np.asarray(python_times_ms, dtype=np.float64)}
    if not times:
        return matlab_df.iloc[0:0].copy()
    utc = pd.to_numeric(matlab_df["utcTimeMillis"], errors="coerce").round().astype("Int64")
    keep = utc.isin(times).to_numpy(dtype=bool)
    if "nextUtcTimeMillis" in matlab_df.columns:
        next_utc = pd.to_numeric(matlab_df["nextUtcTimeMillis"], errors="coerce").round().astype("Int64")
        next_required = next_utc.notna().to_numpy(dtype=bool) & (next_utc.fillna(0).to_numpy(dtype=np.int64) != 0)
        next_keep = ~next_required | next_utc.isin(times).to_numpy(dtype=bool)
        keep &= next_keep
    return matlab_df.loc[keep].copy()


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
    parser.add_argument("--matlab-residual-diagnostics", type=Path, default=None)
    parser.add_argument("--matlab-imu-factor-mask", type=Path, default=None)
    parser.add_argument("--matlab-imu-residual-diagnostics", type=Path, default=None)
    parser.add_argument("--matlab-imu-state", type=Path, default=None)
    parser.add_argument("--matlab-imu-preintegration", type=Path, default=None)
    parser.add_argument("--matlab-residual-diagnostics-mask", type=Path, default=None)
    parser.add_argument(
        "--use-matlab-residual-diagnostics-mask",
        action="store_true",
        help="apply Taroz residual diagnostics p/d/l_factor_finite flags before building Python factors",
    )
    parser.add_argument(
        "--apply-matlab-factor-mask",
        action="store_true",
        help="apply --matlab-factor-mask to the Python batch before exporting Python factor keys",
    )
    parser.add_argument(
        "--include-imu",
        action="store_true",
        help="also export taroz_body IMU factor keys and body-gravity residual diagnostics",
    )
    parser.add_argument("--no-auto-matlab", action="store_true")
    return parser


def run_audit(args: argparse.Namespace) -> dict[str, object]:
    data_root = Path(args.data_root)
    trip = str(args.trip)
    max_epochs = int(args.max_epochs)
    if max_epochs <= 0:
        max_epochs = 1_000_000_000
    matlab_factor_path = args.matlab_factor_mask
    matlab_residual_path = args.matlab_residual_diagnostics
    matlab_imu_factor_path = args.matlab_imu_factor_mask
    matlab_imu_residual_path = args.matlab_imu_residual_diagnostics
    matlab_imu_state_path = args.matlab_imu_state
    matlab_imu_preintegration_path = args.matlab_imu_preintegration
    matlab_residual_mask_path = args.matlab_residual_diagnostics_mask
    if not args.no_auto_matlab:
        if matlab_factor_path is None:
            candidate = _auto_path(data_root, trip, "phone_data_factor_mask.csv")
            matlab_factor_path = candidate if candidate.is_file() else None
        if matlab_residual_path is None:
            candidate = _auto_path(data_root, trip, "phone_data_residual_diagnostics.csv")
            matlab_residual_path = candidate if candidate.is_file() else None
        if matlab_imu_factor_path is None:
            candidate = _auto_path(data_root, trip, "phone_data_imu_factor_mask.csv")
            matlab_imu_factor_path = candidate if candidate.is_file() else None
        if matlab_imu_residual_path is None:
            candidate = _auto_path(data_root, trip, "phone_data_imu_residual_diagnostics.csv")
            matlab_imu_residual_path = candidate if candidate.is_file() else None
        if matlab_imu_state_path is None:
            candidate = _auto_path(data_root, trip, "phone_data_imu_state.csv")
            matlab_imu_state_path = candidate if candidate.is_file() else None
        if matlab_imu_preintegration_path is None:
            candidate = _auto_path(data_root, trip, "phone_data_imu_preintegration.csv")
            matlab_imu_preintegration_path = candidate if candidate.is_file() else None
    if args.use_matlab_residual_diagnostics_mask and matlab_residual_mask_path is None:
        matlab_residual_mask_path = matlab_residual_path
    include_imu = bool(args.include_imu)
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
        dual_frequency=True,
        matlab_residual_diagnostics_mask_path=matlab_residual_mask_path,
        imu_frame=IMU_TAROZ_BODY_DELTA_FRAME if include_imu else "body",
        imu_sample_dt_mode="taroz" if include_imu else "bounded",
    )
    if args.apply_matlab_factor_mask:
        if matlab_factor_path is None or not Path(matlab_factor_path).is_file():
            raise ValueError("--apply-matlab-factor-mask requires a readable --matlab-factor-mask")
        batch = _apply_taroz_factor_mask_to_batch(
            batch,
            Path(matlab_factor_path),
            trip_dir=data_root / trip,
            use_fixed_values=False,
        )
    factor_mask = python_factor_mask(batch)
    residuals = python_residual_diagnostics(batch)
    imu_factor_mask = pd.DataFrame()
    imu_residuals = pd.DataFrame()
    if include_imu:
        imu_state = seed_vd_state_for_batch(
            batch,
            imu_attitude_state=True,
            imu_accel_bias_state=True,
            imu_gyro_bias_state=True,
        )
        imu_factor_mask = python_imu_factor_mask(batch)
        imu_residuals = python_imu_residual_diagnostics(batch, imu_state)

    output = Path(args.output) if args.output is not None else None
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        factor_mask.to_csv(output.with_suffix(".factor_mask.csv"), index=False)
        residuals.to_csv(output.with_suffix(".residuals.csv"), index=False)
        if include_imu:
            imu_factor_mask.to_csv(output.with_suffix(".imu_factor_mask.csv"), index=False)
            imu_residuals.to_csv(output.with_suffix(".imu_residuals.csv"), index=False)

    payload: dict[str, object] = {
        "trip": trip,
        "epochs": int(batch.times_ms.size),
        "python_factor_rows": int(factor_mask.shape[0]),
        "python_residual_rows": int(residuals.shape[0]),
        "include_imu": include_imu,
        "python_imu_factor_rows": int(imu_factor_mask.shape[0]),
        "python_imu_residual_rows": int(imu_residuals.shape[0]),
        "matlab_residual_diagnostics_mask_path": (
            str(matlab_residual_mask_path) if matlab_residual_mask_path is not None else None
        ),
        "applied_matlab_factor_mask": bool(args.apply_matlab_factor_mask),
        "matlab_imu_factor_mask_path": str(matlab_imu_factor_path) if matlab_imu_factor_path is not None else None,
        "matlab_imu_residual_diagnostics_path": (
            str(matlab_imu_residual_path) if matlab_imu_residual_path is not None else None
        ),
        "matlab_imu_state_path": str(matlab_imu_state_path) if matlab_imu_state_path is not None else None,
        "matlab_imu_preintegration_path": (
            str(matlab_imu_preintegration_path) if matlab_imu_preintegration_path is not None else None
        ),
    }
    if matlab_factor_path is not None and Path(matlab_factor_path).is_file():
        matlab_mask = filter_matlab_frame_to_python_window(pd.read_csv(matlab_factor_path), batch.times_ms)
        comparison = compare_factor_mask(factor_mask, matlab_mask)
        payload["factor_mask_comparison"] = comparison.to_dict(orient="records")
        if output is not None:
            comparison.to_csv(output.with_suffix(".factor_mask_compare.csv"), index=False)
    if matlab_residual_path is not None and Path(matlab_residual_path).is_file():
        matlab_residuals = filter_matlab_frame_to_python_window(pd.read_csv(matlab_residual_path), batch.times_ms)
        stats = compare_residual_diagnostics(residuals, matlab_residuals)
        payload["residual_delta_stats"] = [asdict(stat) for stat in stats]
    if include_imu and matlab_imu_factor_path is not None and Path(matlab_imu_factor_path).is_file():
        matlab_imu_mask = filter_matlab_frame_to_python_window(pd.read_csv(matlab_imu_factor_path), batch.times_ms)
        comparison = compare_imu_factor_mask(imu_factor_mask, matlab_imu_mask)
        payload["imu_factor_mask_comparison"] = comparison.to_dict(orient="records")
        if output is not None:
            comparison.to_csv(output.with_suffix(".imu_factor_mask_compare.csv"), index=False)
    if include_imu and matlab_imu_residual_path is not None and Path(matlab_imu_residual_path).is_file():
        matlab_imu_residuals = filter_matlab_frame_to_python_window(pd.read_csv(matlab_imu_residual_path), batch.times_ms)
        stats = compare_imu_residual_diagnostics(imu_residuals, matlab_imu_residuals)
        payload["imu_residual_delta_stats"] = [asdict(stat) for stat in stats]
        if (
            matlab_imu_state_path is not None
            and Path(matlab_imu_state_path).is_file()
            and matlab_imu_preintegration_path is not None
            and Path(matlab_imu_preintegration_path).is_file()
        ):
            matlab_imu_state = filter_matlab_frame_to_python_window(pd.read_csv(matlab_imu_state_path), batch.times_ms)
            matlab_imu_preintegration = filter_matlab_frame_to_python_window(
                pd.read_csv(matlab_imu_preintegration_path),
                batch.times_ms,
            )
            imu_same_state_residuals = python_imu_residual_diagnostics_from_matlab_exports(
                matlab_imu_state,
                matlab_imu_preintegration,
            )
            payload["python_imu_same_state_residual_rows"] = int(imu_same_state_residuals.shape[0])
            same_state_stats = compare_imu_residual_diagnostics(
                imu_same_state_residuals,
                matlab_imu_residuals,
            )
            payload["imu_same_state_residual_delta_stats"] = [
                asdict(stat) for stat in same_state_stats
            ]
            if output is not None:
                imu_same_state_residuals.to_csv(output.with_suffix(".imu_same_state_residuals.csv"), index=False)
    if (
        include_imu
        and matlab_imu_preintegration_path is not None
        and Path(matlab_imu_preintegration_path).is_file()
    ):
        matlab_imu_state = None
        if matlab_imu_state_path is not None and Path(matlab_imu_state_path).is_file():
            matlab_imu_state = filter_matlab_frame_to_python_window(pd.read_csv(matlab_imu_state_path), batch.times_ms)
        matlab_imu_preintegration = filter_matlab_frame_to_python_window(
            pd.read_csv(matlab_imu_preintegration_path),
            batch.times_ms,
        )
        imu_preintegration = python_imu_preintegration_diagnostics(batch, matlab_imu_state)
        payload["python_imu_preintegration_rows"] = int(imu_preintegration.shape[0])
        preintegration_stats = compare_imu_preintegration_diagnostics(
            imu_preintegration,
            matlab_imu_preintegration,
        )
        payload["imu_preintegration_delta_stats"] = [
            asdict(stat) for stat in preintegration_stats
        ]
        if output is not None:
            imu_preintegration.to_csv(output.with_suffix(".imu_preintegration.csv"), index=False)

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
