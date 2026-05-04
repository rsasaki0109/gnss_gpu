#!/usr/bin/env python3
"""Compare MATLAB residual diagnostics values against raw-bridge residual values."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from experiments.gsdc2023_raw_bridge import (  # noqa: E402
    DEFAULT_ROOT,
    _build_trip_arrays,
    _gps_sat_clock_bias_adjustment_m,
    _gps_tgd_m_by_svid_for_trip,
)
from experiments.gsdc2023_audit_output import (  # noqa: E402
    print_summary_and_output_dir as _print_summary_and_output_dir,
    timestamped_output_dir as _timestamped_output_dir,
    write_summary_json as _write_summary_json,
)
from experiments.gsdc2023_audit_cli import (  # noqa: E402
    add_data_root_trip_args as _add_data_root_trip_args,
    add_max_epochs_arg as _add_max_epochs_arg,
    add_multi_gnss_arg as _add_multi_gnss_arg,
    add_output_dir_arg as _add_output_dir_arg,
    nonnegative_max_epochs as _nonnegative_max_epochs,
    resolve_trip_dir as _resolve_trip_dir,
    resolved_output_root as _resolved_output_root,
)
from experiments.gsdc2023_observation_matrix import load_raw_gnss_frame as _load_raw_gnss_frame  # noqa: E402
from experiments.gsdc2023_observation_matrix import (  # noqa: E402
    RTKLIB_SAT_VELOCITY_FORWARD_DIFF_HALF_STEP_S as _SAT_VEL_HALF_STEP_S,
)
from experiments.gsdc2023_residual_audit import (  # noqa: E402
    RESIDUAL_KEY_COLUMNS as _KEY_COLUMNS,
    append_bridge_residual_row as _append_bridge_residual_rows,
    matlab_residual_frame as _matlab_residual_frame,
    merge_residual_value_frames as _merge_residual_value_frames,
    residual_value_summary_frame as _summary_frame,
)
from experiments.gsdc2023_residual_model import (  # noqa: E402
    geometric_range_rate_with_sagnac as _geometric_range_rate_with_sagnac,
    geometric_range_with_sagnac as _geometric_range_with_sagnac,
    median_clock_prediction as _median_clock_prediction,
    pseudorange_global_isb_by_group as _pseudorange_global_isb_by_group,
    receiver_velocity_from_reference as _receiver_velocity_from_reference,
)
from experiments.gsdc2023_signal_model import (  # noqa: E402
    constellation_to_matlab_sys as _constellation_to_matlab_sys,
    multi_gnss_mask as _multi_gnss_mask,
    signal_types_for_constellation as _signal_types_for_constellation,
    slot_frequency_label as _freq_label,
    slot_pseudorange_common_bias_groups as _slot_pseudorange_common_bias_groups,
)
from experiments.gsdc2023_trip_window import (  # noqa: E402
    FULL_EPOCH_COUNT,
    settings_epoch_window_for_trip as _settings_epoch_window_for_trip,
    trim_epoch_window as _trim_epoch_window,
)


def _frame_sat_velocity_forward_difference_lookup(
    frame: pd.DataFrame,
    *,
    half_step_s: float = _SAT_VEL_HALF_STEP_S,
) -> dict[tuple[int, int, int, str], np.ndarray]:
    velocity_cols = [
        "SvVelocityXEcefMetersPerSecond",
        "SvVelocityYEcefMetersPerSecond",
        "SvVelocityZEcefMetersPerSecond",
    ]
    if not set(velocity_cols).issubset(frame.columns):
        return {}

    out: dict[tuple[int, int, int, str], np.ndarray] = {}
    group_cols = ["ConstellationType", "Svid", "SignalType"]
    for (constellation_type, svid, signal_type), group in frame.groupby(group_cols, sort=False):
        group = group.sort_values("utcTimeMillis")
        times_s = pd.to_numeric(group["utcTimeMillis"], errors="coerce").to_numpy(dtype=np.float64) * 1.0e-3
        velocities = group[velocity_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
        valid = np.isfinite(times_s) & np.isfinite(velocities).all(axis=1)
        times_s = times_s[valid]
        velocities = velocities[valid]
        utc_ms = group.loc[valid, "utcTimeMillis"].to_numpy(dtype=np.int64)
        if velocities.size == 0:
            continue
        corrected = velocities.copy()
        if velocities.shape[0] >= 2 and np.all(np.diff(times_s) > 0.0):
            acceleration = np.zeros_like(velocities)
            for axis in range(3):
                acceleration[:, axis] = np.gradient(velocities[:, axis], times_s, edge_order=1)
            corrected = velocities + float(half_step_s) * acceleration
        for time_ms, velocity in zip(utc_ms, corrected):
            out[(int(time_ms), int(constellation_type), int(svid), str(signal_type))] = velocity
    return out


def _finite_matrix_value(matrix: np.ndarray | None, epoch_idx: int, slot_idx: int) -> float | None:
    if matrix is None:
        return None
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or not (0 <= epoch_idx < values.shape[0]) or not (0 <= slot_idx < values.shape[1]):
        return None
    value = float(values[epoch_idx, slot_idx])
    return value if np.isfinite(value) else None


def _finite_vector_value(matrix: np.ndarray | None, epoch_idx: int, slot_idx: int) -> np.ndarray | None:
    if matrix is None:
        return None
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 3 or values.shape[2] != 3:
        return None
    if not (0 <= epoch_idx < values.shape[0]) or not (0 <= slot_idx < values.shape[1]):
        return None
    value = values[epoch_idx, slot_idx]
    return value if np.isfinite(value).all() else None


def _bridge_component_frame(
    trip_dir: Path,
    times_ms: np.ndarray,
    *,
    multi_gnss: bool = False,
    dual_frequency: bool = True,
    receiver_xyz: np.ndarray | None = None,
    receiver_vel: np.ndarray | None = None,
    slot_keys: tuple[tuple[int, int, str], ...] | None = None,
    sat_ecef: np.ndarray | None = None,
    sat_vel: np.ndarray | None = None,
    sat_clock_bias_m: np.ndarray | None = None,
    sat_clock_drift_mps: np.ndarray | None = None,
    rtklib_iono_m: np.ndarray | None = None,
    rtklib_tropo_m: np.ndarray | None = None,
    epoch_offset: int = 0,
) -> pd.DataFrame:
    raw = _load_raw_gnss_frame(trip_dir / "device_gnss.csv")
    if multi_gnss:
        frame = raw[_multi_gnss_mask(raw, dual_frequency=dual_frequency)].copy()
    elif dual_frequency:
        signal_types = _signal_types_for_constellation(
            constellation_type=1,
            signal_type="GPS_L1_CA",
            dual_frequency=True,
        )
        frame = raw[(raw["ConstellationType"] == 1) & (raw["SignalType"].isin(signal_types))].copy()
    else:
        frame = raw[(raw["ConstellationType"] == 1) & (raw["SignalType"] == "GPS_L1_CA")].copy()

    required = [
        "RawPseudorangeMeters",
        "SvPositionXEcefMeters",
        "SvPositionYEcefMeters",
        "SvPositionZEcefMeters",
        "SvClockBiasMeters",
        "IonosphericDelayMeters",
        "TroposphericDelayMeters",
    ]
    for col in required:
        frame = frame[np.isfinite(pd.to_numeric(frame[col], errors="coerce"))]
    if frame.empty:
        return pd.DataFrame(columns=_KEY_COLUMNS)

    frame = frame.sort_values(["utcTimeMillis", "ConstellationType", "Svid", "Cn0DbHz"]).groupby(
        ["utcTimeMillis", "ConstellationType", "Svid", "SignalType"],
        as_index=False,
    ).tail(1)
    gps_l1_product_lookup = {
        (int(getattr(item, "utcTimeMillis")), int(item.Svid)): item
        for item in frame.itertuples(index=False)
        if int(item.ConstellationType) == 1 and _freq_label(str(item.SignalType)) == "L1"
    }
    sat_velocity_lookup = _frame_sat_velocity_forward_difference_lookup(frame)
    slot_lookup = (
        {
            (int(constellation), int(svid), str(signal_type)): idx
            for idx, (constellation, svid, signal_type) in enumerate(slot_keys)
        }
        if slot_keys is not None
        else {}
    )
    epoch_lookup = {
        int(round(float(time_ms))): (idx + 1 + int(epoch_offset), idx)
        for idx, time_ms in enumerate(times_ms)
    }
    gps_tgd_m_by_svid = _gps_tgd_m_by_svid_for_trip(trip_dir)
    rows: list[dict[str, object]] = []
    for row in frame.itertuples(index=False):
        utc_ms = int(getattr(row, "utcTimeMillis"))
        epoch_item = epoch_lookup.get(utc_ms)
        if epoch_item is None:
            continue
        epoch_index, array_idx = epoch_item
        product_row = (
            gps_l1_product_lookup.get((utc_ms, int(row.Svid)), row)
            if int(row.ConstellationType) == 1 and _freq_label(str(row.SignalType)) == "L5"
            else row
        )
        product_velocity = sat_velocity_lookup.get(
            (
                utc_ms,
                int(product_row.ConstellationType),
                int(product_row.Svid),
                str(product_row.SignalType),
            ),
        )
        slot_idx = slot_lookup.get((int(row.ConstellationType), int(row.Svid), str(row.SignalType)))
        raw_pr = float(row.RawPseudorangeMeters)
        sv_clock = float(row.SvClockBiasMeters) + _gps_sat_clock_bias_adjustment_m(
            int(row.ConstellationType),
            int(row.Svid),
            str(row.SignalType),
            gps_tgd_m_by_svid,
        )
        iono = float(row.IonosphericDelayMeters)
        trop = float(row.TroposphericDelayMeters)
        if slot_idx is not None:
            actual_sv_clock = _finite_matrix_value(sat_clock_bias_m, array_idx, slot_idx)
            if actual_sv_clock is not None:
                sv_clock = actual_sv_clock
            actual_iono = _finite_matrix_value(rtklib_iono_m, array_idx, slot_idx)
            if actual_iono is not None:
                iono = actual_iono
            actual_trop = _finite_matrix_value(rtklib_tropo_m, array_idx, slot_idx)
            if actual_trop is not None:
                trop = actual_trop
            actual_velocity = _finite_vector_value(sat_vel, array_idx, slot_idx)
            if actual_velocity is not None:
                product_velocity = actual_velocity
        base = {
            "freq": _freq_label(str(row.SignalType)),
            "epoch_index": int(epoch_index),
            "utcTimeMillis": utc_ms,
            "sys": _constellation_to_matlab_sys(int(row.ConstellationType)),
            "svid": int(row.Svid),
            "bridge_raw_pseudorange": raw_pr,
            "bridge_corrected_pseudorange": raw_pr + sv_clock - iono - trop,
            "bridge_sat_x": float(product_row.SvPositionXEcefMeters),
            "bridge_sat_y": float(product_row.SvPositionYEcefMeters),
            "bridge_sat_z": float(product_row.SvPositionZEcefMeters),
            "bridge_sat_clock_bias": sv_clock,
            "bridge_sat_iono": iono,
            "bridge_sat_trop": trop,
            "bridge_sat_elevation": float(getattr(product_row, "SvElevationDegrees", np.nan)),
        }
        if slot_idx is not None:
            actual_sat_xyz = _finite_vector_value(sat_ecef, array_idx, slot_idx)
            if actual_sat_xyz is not None:
                base.update(
                    {
                        "bridge_sat_x": float(actual_sat_xyz[0]),
                        "bridge_sat_y": float(actual_sat_xyz[1]),
                        "bridge_sat_z": float(actual_sat_xyz[2]),
                    },
                )
        if receiver_xyz is not None and 0 <= array_idx < receiver_xyz.shape[0]:
            base.update(
                {
                    "bridge_rcv_x": float(receiver_xyz[array_idx, 0]),
                    "bridge_rcv_y": float(receiver_xyz[array_idx, 1]),
                    "bridge_rcv_z": float(receiver_xyz[array_idx, 2]),
                },
            )
        if receiver_vel is not None and 0 <= array_idx < receiver_vel.shape[0]:
            base.update(
                {
                    "bridge_rcv_vx": float(receiver_vel[array_idx, 0]),
                    "bridge_rcv_vy": float(receiver_vel[array_idx, 1]),
                    "bridge_rcv_vz": float(receiver_vel[array_idx, 2]),
                },
            )
        if hasattr(product_row, "SvVelocityXEcefMetersPerSecond"):
            if product_velocity is None:
                product_velocity = np.array(
                    [
                        float(product_row.SvVelocityXEcefMetersPerSecond),
                        float(product_row.SvVelocityYEcefMetersPerSecond),
                        float(product_row.SvVelocityZEcefMetersPerSecond),
                    ],
                    dtype=np.float64,
                )
            base.update(
                {
                    "bridge_sat_vx": float(product_velocity[0]),
                    "bridge_sat_vy": float(product_velocity[1]),
                    "bridge_sat_vz": float(product_velocity[2]),
                },
            )
        if hasattr(product_row, "SvClockDriftMetersPerSecond"):
            base["bridge_sat_clock_drift"] = float(product_row.SvClockDriftMetersPerSecond)
        if slot_idx is not None:
            actual_drift = _finite_matrix_value(sat_clock_drift_mps, array_idx, slot_idx)
            if actual_drift is not None:
                base["bridge_sat_clock_drift"] = actual_drift
        for field in ("P", "D"):
            item = dict(base)
            item["field"] = field
            rows.append(item)
    if not rows:
        return pd.DataFrame(columns=_KEY_COLUMNS)
    return pd.DataFrame(rows).drop_duplicates(_KEY_COLUMNS)


def build_bridge_residual_frame(
    trip_dir: Path,
    *,
    max_epochs: int = 0,
    multi_gnss: bool = False,
    dual_frequency: bool = True,
) -> pd.DataFrame:
    start_epoch, bridge_max_epochs = _settings_epoch_window_for_trip(trip_dir, max_epochs)

    def build_batch(
        *,
        batch_start_epoch: int,
        batch_max_epochs: int,
        apply_observation_mask: bool = True,
    ):
        return _build_trip_arrays(
            trip_dir,
            max_epochs=batch_max_epochs,
            start_epoch=batch_start_epoch,
            constellation_type=1,
            signal_type="GPS_L1_CA",
            weight_mode="sin2el",
            multi_gnss=multi_gnss,
            use_tdcp=False,
            apply_observation_mask=apply_observation_mask,
            pseudorange_residual_mask_m=0.0,
            doppler_residual_mask_mps=0.0,
            pseudorange_doppler_mask_m=0.0,
            dual_frequency=dual_frequency,
            raw_frame_epoch_window=True,
        )

    batch = build_batch(batch_start_epoch=start_epoch, batch_max_epochs=bridge_max_epochs)
    times_ms = np.asarray(batch.times_ms, dtype=np.float64)
    receiver_velocity = _receiver_velocity_from_reference(times_ms, np.asarray(batch.kaggle_wls, dtype=np.float64))
    clock_drift_mps = batch.clock_drift_mps
    if start_epoch > 0 or bridge_max_epochs < FULL_EPOCH_COUNT:
        context_start_epoch = max(int(start_epoch) - 1, 0)
        leading_context = int(start_epoch) - context_start_epoch
        trailing_context = 1 if bridge_max_epochs < FULL_EPOCH_COUNT else 0
        context_max_epochs = int(bridge_max_epochs) + leading_context + trailing_context
        context_batch = build_batch(
            batch_start_epoch=context_start_epoch,
            batch_max_epochs=context_max_epochs,
            apply_observation_mask=True,
        )
        context_velocity = _receiver_velocity_from_reference(
            context_batch.times_ms,
            np.asarray(context_batch.kaggle_wls, dtype=np.float64),
        )
        context_lookup = {int(round(float(time_ms))): idx for idx, time_ms in enumerate(context_batch.times_ms)}
        for epoch_idx in {0, times_ms.size - 1}:
            context_idx = context_lookup.get(int(round(float(times_ms[epoch_idx]))))
            if context_idx is None:
                continue
            candidate = context_velocity[context_idx]
            if np.isfinite(candidate).all():
                receiver_velocity[epoch_idx] = candidate
        if batch.clock_drift_mps is not None and context_batch.clock_drift_mps is not None:
            clock_drift_mps = np.asarray(batch.clock_drift_mps, dtype=np.float64).copy()
            for epoch_idx in {0, times_ms.size - 1}:
                context_idx = context_lookup.get(int(round(float(times_ms[epoch_idx]))))
                if context_idx is None:
                    continue
                candidate = float(context_batch.clock_drift_mps[context_idx])
                if np.isfinite(candidate):
                    clock_drift_mps[epoch_idx] = candidate
    slot_keys = tuple(batch.slot_keys)
    slot_freq = np.array([_freq_label(key[2]) for key in slot_keys], dtype=object)
    pseudorange_bias_groups = _slot_pseudorange_common_bias_groups(slot_keys)
    rows: list[dict[str, object]] = []
    p_records: list[dict[str, object]] = []

    for epoch_idx in range(batch.weights.shape[0]):
        idx = np.flatnonzero(batch.weights[epoch_idx] > 0.0)
        if idx.size == 0:
            continue
        rx = np.asarray(batch.kaggle_wls[epoch_idx], dtype=np.float64)
        ranges = _geometric_range_with_sagnac(batch.sat_ecef[epoch_idx, idx], rx)
        valid = np.isfinite(ranges) & (ranges > 1.0e6) & np.isfinite(batch.pseudorange[epoch_idx, idx])
        if not np.any(valid):
            continue
        idx = idx[valid]
        ranges = ranges[valid]
        residual0 = batch.pseudorange[epoch_idx, idx] - ranges
        clock_bias = (
            float(batch.clock_bias_m[epoch_idx])
            if batch.clock_bias_m is not None
            and epoch_idx < len(batch.clock_bias_m)
            and np.isfinite(batch.clock_bias_m[epoch_idx])
            else np.nan
        )
        if np.isfinite(clock_bias):
            pred_bias = np.full(idx.size, np.nan, dtype=np.float64)
            residual = residual0.copy()
        elif idx.size >= 4:
            weights = batch.weights[epoch_idx, idx]
            sk = (
                batch.sys_kind[epoch_idx, idx].astype(np.int32, copy=False)
                if batch.sys_kind is not None
                else np.zeros(idx.size, dtype=np.int32)
            )
            pred_bias = _median_clock_prediction(residual0, weights, sk, batch.n_clock)
            residual = residual0 - pred_bias
        else:
            continue
        for slot_idx, value, pre_value, bias_value, range_value in zip(
            idx,
            residual,
            residual0,
            pred_bias,
            ranges,
        ):
            if np.isfinite(value):
                constellation, svid, _signal_type = slot_keys[int(slot_idx)]
                p_records.append(
                    {
                        "field": "P",
                        "freq": str(slot_freq[int(slot_idx)]),
                        "epoch_index": int(epoch_idx) + 1 + int(start_epoch),
                        "utcTimeMillis": int(round(float(times_ms[int(epoch_idx)]))),
                        "sys": _constellation_to_matlab_sys(int(constellation)),
                        "svid": int(svid),
                        "bridge_residual": float(value),
                        "bridge_pre_residual": float(pre_value),
                        "bridge_common_bias": float(bias_value),
                        "bridge_observation": float(batch.pseudorange[epoch_idx, slot_idx]),
                        "bridge_model": float(range_value),
                        "_clock_bias": clock_bias,
                        "_bias_group": int(pseudorange_bias_groups[int(slot_idx)]),
                    },
                )

    if p_records:
        p_frame = pd.DataFrame(p_records)
        if p_frame["_clock_bias"].notna().any():
            p_frame["bridge_common_bias"] = np.nan
            isb_by_group = getattr(batch, "pseudorange_isb_by_group", None)
            if not isb_by_group:
                isb_by_group = _pseudorange_global_isb_by_group(
                    batch.sat_ecef,
                    batch.pseudorange,
                    (
                        batch.pseudorange_bias_weights
                        if getattr(batch, "pseudorange_bias_weights", None) is not None
                        else batch.weights
                    ),
                    batch.kaggle_wls,
                    batch.clock_bias_m,
                    sys_kind=batch.sys_kind,
                    common_bias_group=pseudorange_bias_groups,
                )
            for group_id, isb in isb_by_group.items():
                idx = p_frame["_bias_group"] == int(group_id)
                p_frame.loc[idx, "bridge_common_bias"] = p_frame.loc[idx, "_clock_bias"] + float(isb)
            finite_common = np.isfinite(p_frame["bridge_common_bias"].to_numpy(dtype=np.float64))
            p_frame.loc[finite_common, "bridge_residual"] = (
                p_frame.loc[finite_common, "bridge_pre_residual"].to_numpy(dtype=np.float64)
                - p_frame.loc[finite_common, "bridge_common_bias"].to_numpy(dtype=np.float64)
            )
        rows.extend(p_frame.drop(columns=["_clock_bias", "_bias_group"], errors="ignore").to_dict("records"))

    if batch.sat_vel is not None and batch.doppler is not None and batch.doppler_weights is not None:
        rx_xyz = np.asarray(batch.kaggle_wls, dtype=np.float64)
        rx_vel = receiver_velocity
        delta = batch.sat_ecef - rx_xyz[:, None, :]
        ranges = np.linalg.norm(delta, axis=2)
        valid_range = (
            np.isfinite(ranges)
            & (ranges > 1.0e6)
            & np.isfinite(batch.sat_ecef).all(axis=2)
            & np.isfinite(rx_xyz).all(axis=1)[:, None]
        )
        geom_rate = _geometric_range_rate_with_sagnac(
            batch.sat_ecef,
            rx_xyz[:, None, :],
            batch.sat_vel,
            rx_vel[:, None, :],
        )
        sat_clock_drift = getattr(batch, "sat_clock_drift_mps", None)
        if sat_clock_drift is not None:
            sat_clock_drift = np.asarray(sat_clock_drift, dtype=np.float64)
            if sat_clock_drift.shape == geom_rate.shape:
                finite_clock_drift = np.isfinite(sat_clock_drift)
                geom_rate[finite_clock_drift] -= sat_clock_drift[finite_clock_drift]
        valid = (
            valid_range
            & (batch.doppler_weights > 0.0)
            & np.isfinite(batch.doppler)
            & np.isfinite(geom_rate)
        )
        for epoch_idx in range(batch.doppler_weights.shape[0]):
            idx = np.flatnonzero(valid[epoch_idx])
            if idx.size == 0:
                continue
            # _build_trip_arrays stores Doppler in the native VD solver convention
            # (-Android pseudorange-rate). MATLAB residual diagnostics use the
            # gnsslog2obs/resD convention: raw pseudorange-rate - satr.rate.
            observation = -batch.doppler[epoch_idx, idx]
            model = geom_rate[epoch_idx, idx]
            residual0 = observation - model
            if clock_drift_mps is not None and np.isfinite(clock_drift_mps[epoch_idx]):
                drift = -float(clock_drift_mps[epoch_idx])
            elif idx.size >= 4:
                drift = float(np.median(residual0))
            else:
                continue
            residual = residual0 - drift
            for slot_idx, value, pre_value, obs_value, model_value in zip(
                idx,
                residual,
                residual0,
                observation,
                model,
            ):
                if np.isfinite(value):
                    _append_bridge_residual_rows(
                        rows,
                        field="D",
                        freq=str(slot_freq[int(slot_idx)]),
                        times_ms=times_ms,
                        slot_keys=slot_keys,
                        epoch_idx=epoch_idx,
                        slot_idx=int(slot_idx),
                        residual=float(value),
                        pre_residual=float(pre_value),
                        common_bias=drift,
                        observation=float(obs_value),
                        model=float(model_value),
                        epoch_offset=start_epoch,
                    )

    component_frame = _bridge_component_frame(
        trip_dir,
        times_ms,
        multi_gnss=multi_gnss,
        dual_frequency=dual_frequency,
        receiver_xyz=np.asarray(batch.kaggle_wls, dtype=np.float64),
        receiver_vel=receiver_velocity,
        slot_keys=slot_keys,
        sat_ecef=np.asarray(batch.sat_ecef, dtype=np.float64),
        sat_vel=np.asarray(batch.sat_vel, dtype=np.float64) if batch.sat_vel is not None else None,
        sat_clock_bias_m=(
            np.asarray(getattr(batch, "sat_clock_bias_matrix", None), dtype=np.float64)
            if getattr(batch, "sat_clock_bias_matrix", None) is not None
            else None
        ),
        sat_clock_drift_mps=(
            np.asarray(getattr(batch, "sat_clock_drift_mps", None), dtype=np.float64)
            if getattr(batch, "sat_clock_drift_mps", None) is not None
            else None
        ),
        rtklib_iono_m=(
            np.asarray(getattr(batch, "rtklib_iono_m", None), dtype=np.float64)
            if getattr(batch, "rtklib_iono_m", None) is not None
            else None
        ),
        rtklib_tropo_m=(
            np.asarray(getattr(batch, "rtklib_tropo_m", None), dtype=np.float64)
            if getattr(batch, "rtklib_tropo_m", None) is not None
            else None
        ),
        epoch_offset=start_epoch,
    )
    if not rows:
        return pd.DataFrame(
            columns=_KEY_COLUMNS
            + [
                "bridge_residual",
                "bridge_pre_residual",
                "bridge_common_bias",
                "bridge_observation",
                "bridge_model",
            ],
        )
    out = pd.DataFrame(rows).drop_duplicates(_KEY_COLUMNS)
    if component_frame.empty:
        return out
    return out.merge(component_frame, on=_KEY_COLUMNS, how="left")


def compare_residual_values(
    trip_dir: Path,
    *,
    diagnostics_path: Path | None = None,
    max_epochs: int = 0,
    multi_gnss: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    if diagnostics_path is None:
        diagnostics_path = trip_dir / "phone_data_residual_diagnostics.csv"
    matlab = _matlab_residual_frame(diagnostics_path)
    start_epoch, bridge_max_epochs = _settings_epoch_window_for_trip(trip_dir, max_epochs)
    matlab = _trim_epoch_window(matlab, start_epoch, bridge_max_epochs)
    bridge = build_bridge_residual_frame(trip_dir, max_epochs=max_epochs, multi_gnss=multi_gnss)
    merged = _merge_residual_value_frames(matlab, bridge)
    summary, payload = _summary_frame(merged)
    payload.update(
        {
            "trip_dir": str(trip_dir),
            "diagnostics_path": str(diagnostics_path),
            "max_epochs": int(max_epochs),
            "multi_gnss": bool(multi_gnss),
        },
    )
    return merged, summary, payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _add_data_root_trip_args(parser, default_root=DEFAULT_ROOT)
    parser.add_argument("--diagnostics", type=Path, default=None)
    _add_max_epochs_arg(parser)
    _add_multi_gnss_arg(parser)
    _add_output_dir_arg(parser)
    args = parser.parse_args()

    trip_dir = _resolve_trip_dir(args)
    out_dir = _timestamped_output_dir(_resolved_output_root(args), "gsdc2023_residual_value_parity")

    merged, summary, payload = compare_residual_values(
        trip_dir,
        diagnostics_path=args.diagnostics,
        max_epochs=_nonnegative_max_epochs(args),
        multi_gnss=args.multi_gnss,
    )
    merged.to_csv(out_dir / "residual_value_join.csv", index=False)
    merged[merged["side"] == "matlab_only"].to_csv(out_dir / "matlab_only.csv", index=False)
    merged[merged["side"] == "bridge_only"].to_csv(out_dir / "bridge_only.csv", index=False)
    summary.to_csv(out_dir / "summary_by_field.csv", index=False)
    _write_summary_json(out_dir, payload)
    _print_summary_and_output_dir(payload, out_dir)


if __name__ == "__main__":
    main()
