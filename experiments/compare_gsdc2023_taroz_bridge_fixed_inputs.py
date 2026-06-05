#!/usr/bin/env python3
"""Compare raw-bridge fixed-linearized GNSS inputs with Taroz factor exports."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.compare_gsdc2023_taroz_imu_state import taroz_preprocessing_origin_ecef
from experiments.gsdc2023_imu import ecef_to_enu_relative, enu_to_ecef_relative
from experiments.gsdc2023_raw_bridge import (
    DEFAULT_ROOT,
    _fixed_doppler_linearization_inputs,
    _fixed_pr_linearization_inputs,
    _fixed_tdcp_linearization_inputs,
    _tdcp_unit_vectors_vd,
    build_trip_arrays,
    compute_base_pseudorange_correction_matrix,
)
from experiments.gsdc2023_signal_model import constellation_to_matlab_sys, slot_frequency_label

FACTOR_KEY_COLUMNS = [
    "field",
    "freq",
    "epoch_index",
    "utcTimeMillis",
    "next_epoch_index",
    "nextUtcTimeMillis",
    "sys",
    "svid",
]
FACTOR_VALUE_COLUMNS = [
    "sat_col",
    "factor_model",
    "sigtype",
    "sigma",
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
]
FACTOR_COLUMNS = FACTOR_KEY_COLUMNS + FACTOR_VALUE_COLUMNS
BASE_CORRECTION_KEY_COLUMNS = ["freq", "epoch_index", "utcTimeMillis", "sys", "svid"]


def _ecef_vector_to_enu(vector_ecef: np.ndarray, origin_ecef: np.ndarray) -> np.ndarray:
    vec = np.asarray(vector_ecef, dtype=np.float64)
    return ecef_to_enu_relative(np.asarray(origin_ecef, dtype=np.float64) + vec, origin_ecef)


def _enu_vector_to_ecef(vector_enu: np.ndarray, origin_ecef: np.ndarray) -> np.ndarray:
    vec = np.asarray(vector_enu, dtype=np.float64)
    return enu_to_ecef_relative(vec, origin_ecef) - np.asarray(origin_ecef, dtype=np.float64)


def _as_int_time(values: np.ndarray | pd.Series) -> np.ndarray:
    return np.asarray(np.round(np.asarray(values, dtype=np.float64)), dtype=np.int64)


def load_taroz_initial_state_ecef(
    state_csv: Path,
    *,
    batch_times_ms: np.ndarray,
    origin_ecef: np.ndarray,
    n_clock: int,
) -> np.ndarray:
    """Load Taroz local ENU initial state and convert position/velocity to ECEF."""

    frame = pd.read_csv(state_csv)
    required = {
        "utcTimeMillis",
        "position_x",
        "position_y",
        "position_z",
        "velocity_x",
        "velocity_y",
        "velocity_z",
        "clock_drift_mps",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{state_csv} is missing columns: {missing}")
    key = pd.DataFrame({"utcTimeMillis": _as_int_time(batch_times_ms)})
    keyed = frame.copy()
    keyed["utcTimeMillis"] = _as_int_time(keyed["utcTimeMillis"])
    joined = key.merge(keyed, on="utcTimeMillis", how="left", validate="one_to_one")
    if joined[["position_x", "position_y", "position_z"]].isna().any().any():
        raise ValueError("Taroz state CSV does not cover every raw-bridge epoch")

    n_epoch = int(joined.shape[0])
    state = np.zeros((n_epoch, 7 + int(n_clock)), dtype=np.float64)
    pos_enu = joined[["position_x", "position_y", "position_z"]].to_numpy(dtype=np.float64)
    vel_enu = joined[["velocity_x", "velocity_y", "velocity_z"]].to_numpy(dtype=np.float64)
    state[:, :3] = enu_to_ecef_relative(pos_enu, origin_ecef)
    state[:, 3:6] = _enu_vector_to_ecef(vel_enu, origin_ecef)
    for clock_idx in range(int(n_clock)):
        col = f"clock_bias_m_{clock_idx}"
        if col in joined.columns:
            state[:, 6 + clock_idx] = pd.to_numeric(joined[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    state[:, 6 + int(n_clock)] = pd.to_numeric(joined["clock_drift_mps"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    return state


def _slot_meta(batch: object, slot_idx: int) -> tuple[int, int, str]:
    constellation_type, svid, signal_type = batch.slot_keys[int(slot_idx)]
    return constellation_to_matlab_sys(int(constellation_type)), int(svid), slot_frequency_label(str(signal_type))


def _append_factor_row(
    rows: list[dict[str, object]],
    *,
    batch: object,
    field: str,
    epoch_idx: int,
    slot_idx: int,
    sigma: float,
    measurement: float,
    los_enu: np.ndarray,
    origin1_enu: np.ndarray,
    origin2_enu: np.ndarray | None = None,
    sigtype: int = 0,
    dt_s: float = 0.0,
) -> None:
    sys, svid, freq = _slot_meta(batch, slot_idx)
    next_epoch_index = 0
    next_time = 0
    if field == "L":
        next_epoch_index = int(getattr(batch, "build_start_epoch", 0)) + int(epoch_idx) + 2
        next_time = int(round(float(batch.times_ms[epoch_idx + 1])))
    los = np.asarray(los_enu, dtype=np.float64).reshape(3)
    origin1 = np.asarray(origin1_enu, dtype=np.float64).reshape(3)
    origin2 = np.full(3, np.nan, dtype=np.float64) if origin2_enu is None else np.asarray(origin2_enu, dtype=np.float64).reshape(3)
    rows.append(
        {
            "field": field,
            "freq": freq,
            "epoch_index": int(getattr(batch, "build_start_epoch", 0)) + int(epoch_idx) + 1,
            "utcTimeMillis": int(round(float(batch.times_ms[epoch_idx]))),
            "next_epoch_index": next_epoch_index,
            "nextUtcTimeMillis": next_time,
            "sys": int(sys),
            "svid": int(svid),
            "sat_col": int(slot_idx),
            "factor_model": {"P": "XC", "D": "VD", "L": "XXCC"}[field],
            "sigtype": int(sigtype),
            "sigma": float(sigma),
            "measurement": float(measurement),
            "dt_s": float(dt_s),
            "los_e": float(los[0]),
            "los_n": float(los[1]),
            "los_u": float(los[2]),
            "origin1_e": float(origin1[0]),
            "origin1_n": float(origin1[1]),
            "origin1_u": float(origin1[2]),
            "origin2_e": float(origin2[0]),
            "origin2_n": float(origin2[1]),
            "origin2_u": float(origin2[2]),
        }
    )


def raw_bridge_fixed_factor_frame(batch: object, origin_state_ecef: np.ndarray, origin_ecef: np.ndarray) -> pd.DataFrame:
    """Return raw-bridge P/D/L fixed-linearized solver inputs in Taroz factor-mask shape."""

    state = np.asarray(origin_state_ecef, dtype=np.float64)
    n_epoch = int(batch.times_ms.size)
    n_slot = len(batch.slot_keys)
    sys_kind = (
        np.asarray(batch.sys_kind, dtype=np.int32)
        if getattr(batch, "sys_kind", None) is not None
        else np.zeros((n_epoch, n_slot), dtype=np.int32)
    )
    pos_enu = ecef_to_enu_relative(state[:, :3], origin_ecef)
    vel_enu = _ecef_vector_to_enu(state[:, 3:6], origin_ecef)
    rows: list[dict[str, object]] = []

    pr_weights = batch.weights_fgo if getattr(batch, "weights_fgo", None) is not None else batch.weights
    pr_meas, pr_solver_weights, _pr_ref, pr_los = _fixed_pr_linearization_inputs(
        batch.sat_ecef,
        batch.pseudorange,
        pr_weights,
        state,
    )
    pr_los_enu = _ecef_vector_to_enu(pr_los, origin_ecef)
    for epoch_idx, slot_idx in zip(*np.nonzero(np.asarray(pr_solver_weights) > 0.0)):
        weight = float(pr_solver_weights[epoch_idx, slot_idx])
        if not np.isfinite(weight) or weight <= 0.0:
            continue
        _append_factor_row(
            rows,
            batch=batch,
            field="P",
            epoch_idx=int(epoch_idx),
            slot_idx=int(slot_idx),
            sigma=1.0 / np.sqrt(weight),
            measurement=float(pr_meas[epoch_idx, slot_idx]),
            los_enu=pr_los_enu[epoch_idx, slot_idx],
            origin1_enu=pos_enu[epoch_idx],
            sigtype=int(sys_kind[epoch_idx, slot_idx]),
        )

    doppler_weights_source = (
        batch.doppler_weights_fgo
        if getattr(batch, "doppler_weights_fgo", None) is not None
        else getattr(batch, "doppler_weights", None)
    )
    doppler_fixed = _fixed_doppler_linearization_inputs(
        batch.sat_ecef,
        getattr(batch, "sat_vel", None),
        getattr(batch, "doppler", None),
        doppler_weights_source,
        getattr(batch, "sat_clock_drift_mps", None),
        state,
    )
    if doppler_fixed is not None:
        dop_meas, dop_solver_weights, _dop_ref, dop_los = doppler_fixed
        dop_los_enu = _ecef_vector_to_enu(dop_los, origin_ecef)
        for epoch_idx, slot_idx in zip(*np.nonzero(np.asarray(dop_solver_weights) > 0.0)):
            weight = float(dop_solver_weights[epoch_idx, slot_idx])
            if not np.isfinite(weight) or weight <= 0.0:
                continue
            _append_factor_row(
                rows,
                batch=batch,
                field="D",
                epoch_idx=int(epoch_idx),
                slot_idx=int(slot_idx),
                sigma=1.0 / np.sqrt(weight),
                measurement=float(dop_meas[epoch_idx, slot_idx]),
                los_enu=dop_los_enu[epoch_idx, slot_idx],
                origin1_enu=vel_enu[epoch_idx],
            )

    tdcp_weights_source = (
        batch.tdcp_weights_fgo
        if getattr(batch, "tdcp_weights_fgo", None) is not None
        else getattr(batch, "tdcp_weights", None)
    )
    if (
        tdcp_weights_source is not None
        and getattr(batch, "tdcp_raw_meas", None) is not None
        and n_epoch > 1
    ):
        tdcp_meas, tdcp_solver_weights, _tdcp_ref = _fixed_tdcp_linearization_inputs(
            batch.sat_ecef,
            batch.tdcp_raw_meas,
            tdcp_weights_source,
            state,
            getattr(batch, "sat_clock_bias_matrix", None),
        )
        tdcp_los = _tdcp_unit_vectors_vd(batch.sat_ecef[:-1], state[:-1, None, :3])
        tdcp_los_enu = _ecef_vector_to_enu(tdcp_los, origin_ecef)
        dt = np.zeros(n_epoch, dtype=np.float64) if getattr(batch, "dt", None) is None else np.asarray(batch.dt, dtype=np.float64)
        for epoch_idx, slot_idx in zip(*np.nonzero(np.asarray(tdcp_solver_weights) > 0.0)):
            weight = float(tdcp_solver_weights[epoch_idx, slot_idx])
            if not np.isfinite(weight) or weight <= 0.0:
                continue
            _append_factor_row(
                rows,
                batch=batch,
                field="L",
                epoch_idx=int(epoch_idx),
                slot_idx=int(slot_idx),
                sigma=1.0 / np.sqrt(weight),
                measurement=float(tdcp_meas[epoch_idx, slot_idx]),
                los_enu=tdcp_los_enu[epoch_idx, slot_idx],
                origin1_enu=pos_enu[epoch_idx],
                origin2_enu=pos_enu[epoch_idx + 1],
                dt_s=float(dt[epoch_idx]) if epoch_idx < dt.size else 0.0,
            )

    return pd.DataFrame(rows, columns=FACTOR_COLUMNS).sort_values(FACTOR_KEY_COLUMNS + ["sat_col"]).reset_index(drop=True)


def load_taroz_factor_frame(factor_mask_csv: Path) -> pd.DataFrame:
    frame = pd.read_csv(factor_mask_csv)
    missing = sorted(set(FACTOR_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"{factor_mask_csv} is missing columns: {missing}")
    out = frame[FACTOR_COLUMNS].copy()
    for col in ("epoch_index", "utcTimeMillis", "next_epoch_index", "nextUtcTimeMillis", "sys", "svid"):
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(np.int64)
    out["field"] = out["field"].astype(str)
    out["freq"] = out["freq"].astype(str)
    return out.sort_values(FACTOR_KEY_COLUMNS + ["sat_col"]).reset_index(drop=True)


def compare_factor_frames(taroz: pd.DataFrame, bridge: pd.DataFrame) -> pd.DataFrame:
    return taroz.merge(
        bridge,
        on=FACTOR_KEY_COLUMNS,
        how="outer",
        suffixes=("_taroz", "_bridge"),
        indicator=True,
    )


def restrict_bridge_to_taroz_factor_keys(bridge: pd.DataFrame, taroz: pd.DataFrame) -> pd.DataFrame:
    keys = taroz[FACTOR_KEY_COLUMNS].drop_duplicates()
    out = bridge.merge(keys, on=FACTOR_KEY_COLUMNS, how="inner")
    return out.sort_values(FACTOR_KEY_COLUMNS + ["sat_col"]).reset_index(drop=True)


def apply_taroz_residual_diagnostics_products(batch: object, residual_diagnostics_csv: Path) -> object:
    """Overwrite bridge satellite products with Taroz residual-diagnostics products.

    This is intentionally scoped to fixed-input parity comparisons.  It lets
    the comparison consume Taroz's already-exported ``satr`` products directly
    when auditing the remaining FGO factor deltas.
    """

    frame = pd.read_csv(residual_diagnostics_csv)
    required = {
        "freq",
        "epoch_index",
        "utcTimeMillis",
        "sys",
        "svid",
        "sat_x_m",
        "sat_y_m",
        "sat_z_m",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{residual_diagnostics_csv} is missing columns: {missing}")

    keyed = frame.copy()
    keyed["freq"] = keyed["freq"].astype(str)
    for col in ("epoch_index", "utcTimeMillis", "sys", "svid"):
        keyed[col] = pd.to_numeric(keyed[col], errors="coerce").fillna(0).astype(np.int64)
    for col in (
        "sat_x_m",
        "sat_y_m",
        "sat_z_m",
        "sat_vx_mps",
        "sat_vy_mps",
        "sat_vz_mps",
        "sat_clock_bias_m",
        "sat_clock_drift_mps",
    ):
        if col in keyed.columns:
            keyed[col] = pd.to_numeric(keyed[col], errors="coerce")
    keyed = keyed.drop_duplicates(["freq", "epoch_index", "utcTimeMillis", "sys", "svid"], keep="first")
    lookup = {
        (str(row.freq), int(row.epoch_index), int(row.utcTimeMillis), int(row.sys), int(row.svid)): row
        for row in keyed.itertuples(index=False)
    }

    sat_ecef = np.asarray(batch.sat_ecef, dtype=np.float64).copy()
    sat_vel = (
        np.asarray(batch.sat_vel, dtype=np.float64).copy()
        if getattr(batch, "sat_vel", None) is not None
        else None
    )
    sat_clock_bias = (
        np.asarray(batch.sat_clock_bias_matrix, dtype=np.float64).copy()
        if getattr(batch, "sat_clock_bias_matrix", None) is not None
        else None
    )
    sat_clock_drift = (
        np.asarray(batch.sat_clock_drift_mps, dtype=np.float64).copy()
        if getattr(batch, "sat_clock_drift_mps", None) is not None
        else None
    )
    if sat_vel is None and {"sat_vx_mps", "sat_vy_mps", "sat_vz_mps"}.issubset(keyed.columns):
        sat_vel = np.full_like(sat_ecef, np.nan, dtype=np.float64)
    if sat_clock_bias is None and "sat_clock_bias_m" in keyed.columns:
        sat_clock_bias = np.full(sat_ecef.shape[:2], np.nan, dtype=np.float64)
    if sat_clock_drift is None and "sat_clock_drift_mps" in keyed.columns:
        sat_clock_drift = np.full(sat_ecef.shape[:2], np.nan, dtype=np.float64)

    build_start = int(getattr(batch, "build_start_epoch", 0))
    for epoch_idx, time_ms in enumerate(np.asarray(batch.times_ms, dtype=np.float64)):
        epoch_index = build_start + epoch_idx + 1
        utc_time = int(round(float(time_ms)))
        for slot_idx in range(len(batch.slot_keys)):
            sys, svid, freq = _slot_meta(batch, slot_idx)
            row = lookup.get((freq, epoch_index, utc_time, sys, svid))
            if row is None:
                continue
            pos = np.array([row.sat_x_m, row.sat_y_m, row.sat_z_m], dtype=np.float64)
            if np.isfinite(pos).all():
                sat_ecef[epoch_idx, slot_idx] = pos
            if sat_vel is not None and all(hasattr(row, col) for col in ("sat_vx_mps", "sat_vy_mps", "sat_vz_mps")):
                vel = np.array([row.sat_vx_mps, row.sat_vy_mps, row.sat_vz_mps], dtype=np.float64)
                if np.isfinite(vel).all():
                    sat_vel[epoch_idx, slot_idx] = vel
            if sat_clock_bias is not None and hasattr(row, "sat_clock_bias_m") and np.isfinite(row.sat_clock_bias_m):
                sat_clock_bias[epoch_idx, slot_idx] = float(row.sat_clock_bias_m)
            if sat_clock_drift is not None and hasattr(row, "sat_clock_drift_mps") and np.isfinite(row.sat_clock_drift_mps):
                sat_clock_drift[epoch_idx, slot_idx] = float(row.sat_clock_drift_mps)

    return replace(
        batch,
        sat_ecef=sat_ecef,
        sat_vel=sat_vel,
        sat_clock_bias_matrix=sat_clock_bias,
        sat_clock_drift_mps=sat_clock_drift,
    )


def apply_taroz_residual_diagnostics_pseudorange(
    batch: object,
    residual_diagnostics_csv: Path,
    *,
    base_correction: np.ndarray | None = None,
) -> object:
    """Overwrite bridge corrected pseudorange with Taroz ``p_corrected_m`` values.

    ``p_corrected_m`` is Taroz's phone-side corrected pseudorange before base
    correction.  When a base-correction matrix is supplied, this helper applies
    the same post-step as ``build_trip_arrays(..., apply_base_correction=True)``.
    """

    frame = pd.read_csv(residual_diagnostics_csv)
    required = set(BASE_CORRECTION_KEY_COLUMNS + ["p_corrected_m"])
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{residual_diagnostics_csv} is missing columns: {missing}")

    keyed = frame[BASE_CORRECTION_KEY_COLUMNS + ["p_corrected_m"]].copy()
    keyed["freq"] = keyed["freq"].astype(str)
    for col in ("epoch_index", "utcTimeMillis", "sys", "svid"):
        keyed[col] = pd.to_numeric(keyed[col], errors="coerce").fillna(0).astype(np.int64)
    keyed["p_corrected_m"] = pd.to_numeric(keyed["p_corrected_m"], errors="coerce")
    keyed = keyed.drop_duplicates(BASE_CORRECTION_KEY_COLUMNS, keep="first")
    lookup = {
        (str(row.freq), int(row.epoch_index), int(row.utcTimeMillis), int(row.sys), int(row.svid)): float(row.p_corrected_m)
        for row in keyed.itertuples(index=False)
    }

    pseudorange = np.asarray(batch.pseudorange, dtype=np.float64).copy()
    weights = np.asarray(batch.weights, dtype=np.float64)
    correction = None if base_correction is None else np.asarray(base_correction, dtype=np.float64)
    if correction is not None and correction.shape != pseudorange.shape:
        raise ValueError(f"base_correction shape {correction.shape} does not match pseudorange shape {pseudorange.shape}")

    build_start = int(getattr(batch, "build_start_epoch", 0))
    for epoch_idx, time_ms in enumerate(np.asarray(batch.times_ms, dtype=np.float64)):
        epoch_index = build_start + epoch_idx + 1
        utc_time = int(round(float(time_ms)))
        for slot_idx in range(len(batch.slot_keys)):
            sys, svid, freq = _slot_meta(batch, slot_idx)
            value = lookup.get((freq, epoch_index, utc_time, sys, svid))
            if value is None or not np.isfinite(value):
                continue
            pseudorange[epoch_idx, slot_idx] = value
            if (
                correction is not None
                and np.isfinite(correction[epoch_idx, slot_idx])
                and weights[epoch_idx, slot_idx] > 0.0
            ):
                pseudorange[epoch_idx, slot_idx] -= correction[epoch_idx, slot_idx]

    return replace(batch, pseudorange=pseudorange)


def _finite_stats(values: np.ndarray) -> dict[str, float | None]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean_abs": None, "median_abs": None, "p95_abs": None, "max_abs": None}
    abs_arr = np.abs(arr)
    return {
        "mean_abs": float(np.mean(abs_arr)),
        "median_abs": float(np.median(abs_arr)),
        "p95_abs": float(np.percentile(abs_arr, 95.0)),
        "max_abs": float(np.max(abs_arr)),
    }


def summarize_factor_comparison(joined: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if joined.empty:
        return pd.DataFrame()
    for (field, freq), group in joined.groupby(["field", "freq"], sort=True):
        both = group[group["_merge"].eq("both")]
        row: dict[str, object] = {
            "field": str(field),
            "freq": str(freq),
            "taroz_count": int(np.count_nonzero(group["_merge"].ne("right_only"))),
            "bridge_count": int(np.count_nonzero(group["_merge"].ne("left_only"))),
            "matched_count": int(both.shape[0]),
            "taroz_only_count": int(np.count_nonzero(group["_merge"].eq("left_only"))),
            "bridge_only_count": int(np.count_nonzero(group["_merge"].eq("right_only"))),
        }
        if not both.empty:
            measurement_delta = pd.to_numeric(both["measurement_bridge"], errors="coerce").to_numpy(dtype=np.float64) - pd.to_numeric(
                both["measurement_taroz"], errors="coerce"
            ).to_numpy(dtype=np.float64)
            sigma_delta = pd.to_numeric(both["sigma_bridge"], errors="coerce").to_numpy(dtype=np.float64) - pd.to_numeric(
                both["sigma_taroz"], errors="coerce"
            ).to_numpy(dtype=np.float64)
            los_delta = (
                both[["los_e_bridge", "los_n_bridge", "los_u_bridge"]].to_numpy(dtype=np.float64)
                - both[["los_e_taroz", "los_n_taroz", "los_u_taroz"]].to_numpy(dtype=np.float64)
            )
            origin1_delta = (
                both[["origin1_e_bridge", "origin1_n_bridge", "origin1_u_bridge"]].to_numpy(dtype=np.float64)
                - both[["origin1_e_taroz", "origin1_n_taroz", "origin1_u_taroz"]].to_numpy(dtype=np.float64)
            )
            origin2_delta = (
                both[["origin2_e_bridge", "origin2_n_bridge", "origin2_u_bridge"]].to_numpy(dtype=np.float64)
                - both[["origin2_e_taroz", "origin2_n_taroz", "origin2_u_taroz"]].to_numpy(dtype=np.float64)
            )
            row.update({f"measurement_delta_{k}": v for k, v in _finite_stats(measurement_delta).items()})
            row.update({f"sigma_delta_{k}": v for k, v in _finite_stats(sigma_delta).items()})
            row.update({f"los_delta_norm_{k}": v for k, v in _finite_stats(np.linalg.norm(los_delta, axis=1)).items()})
            row.update({f"origin1_delta_norm_{k}": v for k, v in _finite_stats(np.linalg.norm(origin1_delta, axis=1)).items()})
            finite_origin2 = np.isfinite(origin2_delta).all(axis=1)
            row.update({f"origin2_delta_norm_{k}": v for k, v in _finite_stats(np.linalg.norm(origin2_delta[finite_origin2], axis=1)).items()})
        rows.append(row)
    return pd.DataFrame(rows)


def bridge_base_correction_frame(
    *,
    data_root: Path,
    trip: str,
    batch: object,
    signal_type: str = "GPS_L1_CA",
) -> pd.DataFrame:
    """Return raw-bridge base corrections keyed like Taroz GNSS factor exports."""

    correction = compute_base_pseudorange_correction_matrix(
        Path(data_root),
        str(trip),
        np.asarray(batch.times_ms, dtype=np.float64),
        list(batch.slot_keys),
        signal_type,
    )
    rows: list[dict[str, object]] = []
    for epoch_idx, slot_idx in zip(*np.nonzero(np.isfinite(correction))):
        sys, svid, freq = _slot_meta(batch, int(slot_idx))
        rows.append(
            {
                "freq": freq,
                "epoch_index": int(getattr(batch, "build_start_epoch", 0)) + int(epoch_idx) + 1,
                "utcTimeMillis": int(round(float(batch.times_ms[int(epoch_idx)]))),
                "sys": int(sys),
                "svid": int(svid),
                "bridge_correction_m": float(correction[int(epoch_idx), int(slot_idx)]),
            }
        )
    return pd.DataFrame(rows, columns=BASE_CORRECTION_KEY_COLUMNS + ["bridge_correction_m"])


def infer_taroz_base_correction_frame(
    taroz_factors: pd.DataFrame,
    residual_diagnostics_csv: Path,
) -> pd.DataFrame:
    """Infer Taroz ``correct_pseudorange`` values from pre-base residuals and P factors."""

    residuals = pd.read_csv(residual_diagnostics_csv)
    required = set(BASE_CORRECTION_KEY_COLUMNS + ["p_pre_respc_m"])
    missing = sorted(required - set(residuals.columns))
    if missing:
        raise ValueError(f"{residual_diagnostics_csv} is missing columns: {missing}")

    p_factors = taroz_factors[taroz_factors["field"].eq("P")].copy()
    p_factors = p_factors[BASE_CORRECTION_KEY_COLUMNS + ["measurement"]].rename(
        columns={"measurement": "taroz_post_respc_m"}
    )
    pre = residuals[BASE_CORRECTION_KEY_COLUMNS + ["p_pre_respc_m"]].copy()
    for frame in (p_factors, pre):
        frame["freq"] = frame["freq"].astype(str)
        for col in ("epoch_index", "utcTimeMillis", "sys", "svid"):
            frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0).astype(np.int64)
    pre["p_pre_respc_m"] = pd.to_numeric(pre["p_pre_respc_m"], errors="coerce")
    p_factors["taroz_post_respc_m"] = pd.to_numeric(p_factors["taroz_post_respc_m"], errors="coerce")

    joined = p_factors.merge(pre, on=BASE_CORRECTION_KEY_COLUMNS, how="left", validate="many_to_one")
    joined["taroz_correction_m"] = joined["p_pre_respc_m"] - joined["taroz_post_respc_m"]
    finite = np.isfinite(joined["taroz_correction_m"].to_numpy(dtype=np.float64))
    return joined.loc[finite, BASE_CORRECTION_KEY_COLUMNS + ["taroz_correction_m"]].reset_index(drop=True)


def compare_base_correction_frames(taroz: pd.DataFrame, bridge: pd.DataFrame) -> pd.DataFrame:
    joined = taroz.merge(bridge, on=BASE_CORRECTION_KEY_COLUMNS, how="outer", indicator=True)
    joined["correction_delta_m"] = joined["bridge_correction_m"] - joined["taroz_correction_m"]
    return joined.sort_values(BASE_CORRECTION_KEY_COLUMNS).reset_index(drop=True)


def summarize_base_correction_comparison(joined: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if joined.empty:
        return pd.DataFrame()
    for freq, group in joined.groupby("freq", sort=True, dropna=False):
        both = group[group["_merge"].eq("both")]
        row: dict[str, object] = {
            "freq": str(freq),
            "taroz_count": int(np.count_nonzero(group["_merge"].ne("right_only"))),
            "bridge_count": int(np.count_nonzero(group["_merge"].ne("left_only"))),
            "matched_count": int(both.shape[0]),
            "taroz_only_count": int(np.count_nonzero(group["_merge"].eq("left_only"))),
            "bridge_only_count": int(np.count_nonzero(group["_merge"].eq("right_only"))),
        }
        if not both.empty:
            delta = pd.to_numeric(both["correction_delta_m"], errors="coerce").to_numpy(dtype=np.float64)
            row.update({f"correction_delta_{k}": v for k, v in _finite_stats(delta).items()})
        rows.append(row)
    return pd.DataFrame(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export_dir", type=Path)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--trip", required=True)
    parser.add_argument("--max-epochs", type=int, default=0)
    parser.add_argument("--start-epoch", type=int, default=0)
    parser.add_argument("--multi-gnss", action="store_true")
    parser.add_argument("--dual-frequency", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--apply-observation-mask", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--apply-base-correction", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tdcp-consistency-threshold-m", type=float, default=1.0e9)
    parser.add_argument("--tdcp-weight-scale", type=float, default=1.0)
    parser.add_argument("--state-csv", type=Path, default=None)
    parser.add_argument("--factor-mask-csv", type=Path, default=None)
    parser.add_argument("--residual-diagnostics-csv", type=Path, default=None)
    parser.add_argument("--matlab-residual-diagnostics-mask", type=Path, default=None)
    parser.add_argument("--restrict-bridge-to-taroz-factors", action="store_true")
    parser.add_argument("--use-taroz-residual-diagnostics-products", action="store_true")
    parser.add_argument("--use-taroz-residual-diagnostics-pseudorange", action="store_true")
    parser.add_argument("--joined-output", type=Path, default=None)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--bridge-factor-output", type=Path, default=None)
    parser.add_argument("--base-correction-joined-output", type=Path, default=None)
    parser.add_argument("--base-correction-summary-output", type=Path, default=None)
    parser.add_argument("--json-output", type=Path, default=None)
    return parser


def run_comparison(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    export_dir = Path(args.export_dir)
    data_root = Path(args.data_root)
    trip = str(args.trip)
    max_epochs = int(args.max_epochs)
    if max_epochs <= 0:
        max_epochs = 1_000_000_000
    trip_dir = data_root / trip
    origin_ecef = taroz_preprocessing_origin_ecef(trip_dir)
    batch = build_trip_arrays(
        trip_dir,
        max_epochs=max_epochs,
        start_epoch=int(args.start_epoch),
        constellation_type=1,
        signal_type="GPS_L1_CA",
        weight_mode="taroz_sn",
        fgo_weight_mode="taroz_sn",
        multi_gnss=bool(args.multi_gnss),
        use_tdcp=True,
        apply_observation_mask=bool(args.apply_observation_mask),
        apply_base_correction=bool(args.apply_base_correction),
        tdcp_consistency_threshold_m=float(args.tdcp_consistency_threshold_m),
        tdcp_weight_scale=float(args.tdcp_weight_scale),
        matlab_residual_diagnostics_mask_path=(
            Path(args.matlab_residual_diagnostics_mask)
            if args.matlab_residual_diagnostics_mask is not None
            else None
        ),
        data_root=data_root,
        trip=trip,
        dual_frequency=bool(args.dual_frequency),
    )
    residual_csv = Path(args.residual_diagnostics_csv) if args.residual_diagnostics_csv is not None else trip_dir / "phone_data_residual_diagnostics.csv"
    if bool(args.use_taroz_residual_diagnostics_products):
        if not residual_csv.is_file():
            raise FileNotFoundError(f"Taroz residual diagnostics CSV not found: {residual_csv}")
        batch = apply_taroz_residual_diagnostics_products(batch, residual_csv)
    if bool(args.use_taroz_residual_diagnostics_pseudorange):
        if not residual_csv.is_file():
            raise FileNotFoundError(f"Taroz residual diagnostics CSV not found: {residual_csv}")
        base_correction = None
        if bool(args.apply_base_correction):
            base_correction = compute_base_pseudorange_correction_matrix(
                data_root,
                trip,
                np.asarray(batch.times_ms, dtype=np.float64),
                list(batch.slot_keys),
                "GPS_L1_CA",
            )
        batch = apply_taroz_residual_diagnostics_pseudorange(
            batch,
            residual_csv,
            base_correction=base_correction,
        )
    state_csv = Path(args.state_csv) if args.state_csv is not None else export_dir / "phone_data_gnss_initial_state.csv"
    origin_state = load_taroz_initial_state_ecef(
        state_csv,
        batch_times_ms=batch.times_ms,
        origin_ecef=origin_ecef,
        n_clock=batch.n_clock,
    )
    factor_csv = Path(args.factor_mask_csv) if args.factor_mask_csv is not None else export_dir / "phone_data_gnss_factor_mask.csv"
    taroz = load_taroz_factor_frame(factor_csv)
    bridge = raw_bridge_fixed_factor_frame(batch, origin_state, origin_ecef)
    if bool(args.restrict_bridge_to_taroz_factors):
        bridge = restrict_bridge_to_taroz_factor_keys(bridge, taroz)
    joined = compare_factor_frames(taroz, bridge)
    summary = summarize_factor_comparison(joined)
    return joined, summary, bridge


def run_base_correction_comparison(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    residual_csv = Path(args.residual_diagnostics_csv) if args.residual_diagnostics_csv is not None else None
    if residual_csv is None:
        candidate = Path(args.data_root) / str(args.trip) / "phone_data_residual_diagnostics.csv"
        residual_csv = candidate if candidate.is_file() else None
    if residual_csv is None or not residual_csv.is_file():
        return None

    export_dir = Path(args.export_dir)
    data_root = Path(args.data_root)
    trip = str(args.trip)
    max_epochs = int(args.max_epochs)
    if max_epochs <= 0:
        max_epochs = 1_000_000_000
    trip_dir = data_root / trip
    batch = build_trip_arrays(
        trip_dir,
        max_epochs=max_epochs,
        start_epoch=int(args.start_epoch),
        constellation_type=1,
        signal_type="GPS_L1_CA",
        weight_mode="taroz_sn",
        fgo_weight_mode="taroz_sn",
        multi_gnss=bool(args.multi_gnss),
        use_tdcp=True,
        apply_observation_mask=bool(args.apply_observation_mask),
        apply_base_correction=False,
        tdcp_consistency_threshold_m=float(args.tdcp_consistency_threshold_m),
        tdcp_weight_scale=float(args.tdcp_weight_scale),
        data_root=data_root,
        trip=trip,
        dual_frequency=bool(args.dual_frequency),
    )
    factor_csv = Path(args.factor_mask_csv) if args.factor_mask_csv is not None else export_dir / "phone_data_gnss_factor_mask.csv"
    taroz = infer_taroz_base_correction_frame(load_taroz_factor_frame(factor_csv), residual_csv)
    bridge = bridge_base_correction_frame(data_root=data_root, trip=trip, batch=batch)
    joined = compare_base_correction_frames(taroz, bridge)
    summary = summarize_base_correction_comparison(joined)
    return joined, summary


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    joined, summary, bridge = run_comparison(args)
    base_comparison = run_base_correction_comparison(args)
    if args.joined_output is not None:
        args.joined_output.parent.mkdir(parents=True, exist_ok=True)
        joined.to_csv(args.joined_output, index=False)
    if args.summary_output is not None:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.summary_output, index=False)
    if args.bridge_factor_output is not None:
        args.bridge_factor_output.parent.mkdir(parents=True, exist_ok=True)
        bridge.to_csv(args.bridge_factor_output, index=False)
    if base_comparison is not None:
        base_joined, base_summary = base_comparison
        if args.base_correction_joined_output is not None:
            args.base_correction_joined_output.parent.mkdir(parents=True, exist_ok=True)
            base_joined.to_csv(args.base_correction_joined_output, index=False)
        if args.base_correction_summary_output is not None:
            args.base_correction_summary_output.parent.mkdir(parents=True, exist_ok=True)
            base_summary.to_csv(args.base_correction_summary_output, index=False)
    payload = {
        "summary": summary.to_dict(orient="records"),
        "total": {
            "taroz_count": int(np.count_nonzero(joined["_merge"].ne("right_only"))) if not joined.empty else 0,
            "bridge_count": int(np.count_nonzero(joined["_merge"].ne("left_only"))) if not joined.empty else 0,
            "matched_count": int(np.count_nonzero(joined["_merge"].eq("both"))) if not joined.empty else 0,
            "taroz_only_count": int(np.count_nonzero(joined["_merge"].eq("left_only"))) if not joined.empty else 0,
            "bridge_only_count": int(np.count_nonzero(joined["_merge"].eq("right_only"))) if not joined.empty else 0,
        },
    }
    if base_comparison is not None:
        base_joined, base_summary = base_comparison
        payload["base_correction_summary"] = base_summary.to_dict(orient="records")
        payload["base_correction_total"] = {
            "taroz_count": int(np.count_nonzero(base_joined["_merge"].ne("right_only"))) if not base_joined.empty else 0,
            "bridge_count": int(np.count_nonzero(base_joined["_merge"].ne("left_only"))) if not base_joined.empty else 0,
            "matched_count": int(np.count_nonzero(base_joined["_merge"].eq("both"))) if not base_joined.empty else 0,
            "taroz_only_count": int(np.count_nonzero(base_joined["_merge"].eq("left_only"))) if not base_joined.empty else 0,
            "bridge_only_count": int(np.count_nonzero(base_joined["_merge"].eq("right_only"))) if not base_joined.empty else 0,
        }
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
