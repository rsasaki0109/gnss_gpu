#!/usr/bin/env python3
"""Export per-observation VD factor residual diagnostics for GSDC2023 trips."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from experiments.gsdc2023_observation_matrix import LIGHT_SPEED_MPS, EARTH_ROTATION_RATE_RAD_S, repair_baseline_wls
from experiments.gsdc2023_raw_bridge import (
    DEFAULT_ROOT,
    DEFAULT_TDCP_GEOMETRY_CORRECTION,
    DEFAULT_TDCP_WEIGHT_SCALE,
    OBS_MASK_DOPPLER_RESIDUAL_THRESHOLD_MPS,
    OBS_MASK_MIN_CN0_DBHZ,
    OBS_MASK_MIN_ELEVATION_DEG,
    OBS_MASK_PSEUDORANGE_DOPPLER_THRESHOLD_M,
    OBS_MASK_RESIDUAL_THRESHOLD_L5_M,
    OBS_MASK_RESIDUAL_THRESHOLD_M,
    build_trip_arrays,
    fit_state_with_clock_bias,
    run_wls,
    weighted_mse,
    _seed_vd_state,
    _tdcp_use_drift_for_phone,
)
from experiments.gsdc2023_residual_model import (
    fill_clock_design,
    geometric_range_rate_with_sagnac,
)
from experiments.gsdc2023_validation_context import max_epochs_for_build


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _weighted_quantile_abs(abs_residual: np.ndarray, weights: np.ndarray, q: float) -> float:
    if abs_residual.size == 0:
        return float("nan")
    order = np.argsort(abs_residual)
    vals = abs_residual[order]
    w = weights[order]
    total = float(np.sum(w))
    if total <= 0.0:
        return float(np.quantile(vals, q))
    cutoff = q * total
    return float(vals[min(np.searchsorted(np.cumsum(w), cutoff, side="left"), vals.size - 1)])


def _summary(residual: np.ndarray, weights: np.ndarray) -> dict[str, float | int]:
    valid = np.isfinite(residual) & np.isfinite(weights) & (weights > 0.0)
    if not valid.any():
        return {
            "count": 0,
            "weight_sum": 0.0,
            "weighted_rms": float("nan"),
            "abs_p50": float("nan"),
            "abs_p95": float("nan"),
            "abs_max": float("nan"),
            "max_weighted_contribution": float("nan"),
        }
    r = residual[valid]
    w = weights[valid]
    abs_r = np.abs(r)
    return {
        "count": int(r.size),
        "weight_sum": float(np.sum(w)),
        "weighted_rms": float(np.sqrt(np.sum(w * r * r) / np.sum(w))),
        "abs_p50": _weighted_quantile_abs(abs_r, w, 0.50),
        "abs_p95": _weighted_quantile_abs(abs_r, w, 0.95),
        "abs_max": float(np.max(abs_r)),
        "max_weighted_contribution": float(np.max(w * r * r)),
    }


def _slot_payload(slot_key: Any) -> dict[str, Any]:
    if isinstance(slot_key, (tuple, list)) and len(slot_key) >= 3:
        return {
            "constellation_type": int(slot_key[0]),
            "svid": int(slot_key[1]),
            "signal_type": str(slot_key[2]),
        }
    if isinstance(slot_key, (tuple, list)) and len(slot_key) >= 2:
        return {
            "constellation_type": int(slot_key[0]),
            "svid": int(slot_key[1]),
            "signal_type": "",
        }
    return {"constellation_type": "", "svid": "", "signal_type": str(slot_key)}


def _top_rows(
    *,
    factor: str,
    times_ms: np.ndarray,
    slot_keys: tuple[Any, ...],
    residual: np.ndarray,
    predicted: np.ndarray,
    observed: np.ndarray,
    weights: np.ndarray,
    limit: int,
    component_a: np.ndarray | None = None,
    component_b: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    valid = np.isfinite(residual) & np.isfinite(weights) & (weights > 0.0)
    if not valid.any():
        return []
    contribution = np.zeros_like(residual, dtype=np.float64)
    contribution[valid] = weights[valid] * residual[valid] * residual[valid]
    flat = np.flatnonzero(valid.reshape(-1))
    order = flat[np.argsort(contribution.reshape(-1)[flat])[::-1]]
    rows: list[dict[str, Any]] = []
    n_sat = residual.shape[1]
    for flat_idx in order[:limit]:
        epoch_idx = int(flat_idx // n_sat)
        slot_idx = int(flat_idx % n_sat)
        slot = _slot_payload(slot_keys[slot_idx] if slot_idx < len(slot_keys) else slot_idx)
        rows.append(
            {
                "factor": factor,
                "epoch_index": epoch_idx,
                "unix_time_millis": int(round(float(times_ms[epoch_idx]))),
                "slot_index": slot_idx,
                **slot,
                "residual": float(residual[epoch_idx, slot_idx]),
                "abs_residual": float(abs(residual[epoch_idx, slot_idx])),
                "weight": float(weights[epoch_idx, slot_idx]),
                "weighted_contribution": float(contribution[epoch_idx, slot_idx]),
                "observed": float(observed[epoch_idx, slot_idx]),
                "predicted": float(predicted[epoch_idx, slot_idx]),
                "component_a": (
                    float(component_a[epoch_idx, slot_idx])
                    if component_a is not None and np.isfinite(component_a[epoch_idx, slot_idx])
                    else ""
                ),
                "component_b": (
                    float(component_b[epoch_idx, slot_idx])
                    if component_b is not None and np.isfinite(component_b[epoch_idx, slot_idx])
                    else ""
                ),
            },
        )
    return rows


def _doppler_residuals(
    batch: Any,
    state: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if batch.doppler is None or batch.doppler_weights is None or batch.sat_vel is None:
        empty = np.zeros_like(batch.weights, dtype=np.float64)
        return empty, empty, empty, empty, empty, empty
    geom_rate = geometric_range_rate_with_sagnac(
        batch.sat_ecef,
        state[:, None, :3],
        batch.sat_vel,
        state[:, None, 3:6],
    )
    if batch.sat_clock_drift_mps is not None and batch.sat_clock_drift_mps.shape == geom_rate.shape:
        finite = np.isfinite(batch.sat_clock_drift_mps)
        geom_rate[finite] -= batch.sat_clock_drift_mps[finite]
    drift = state[:, 6 + batch.n_clock]
    drift_component = np.broadcast_to(drift[:, None], geom_rate.shape).copy()
    rate_component = -geom_rate
    predicted = drift_component + rate_component
    residual = batch.doppler - predicted
    valid = (
        (batch.doppler_weights > 0.0)
        & np.isfinite(batch.doppler)
        & np.isfinite(predicted)
        & np.isfinite(drift)[:, None]
    )
    weights = np.where(valid, batch.doppler_weights, 0.0)
    return residual, predicted, batch.doppler, weights, rate_component, drift_component


def _sagnac_unit_vectors_like_vd_tdcp(sat_ecef: np.ndarray, receiver_ecef: np.ndarray) -> np.ndarray:
    sat = np.asarray(sat_ecef, dtype=np.float64)
    rx = np.asarray(receiver_ecef, dtype=np.float64)
    dx0 = rx[..., 0] - sat[..., 0]
    dy0 = rx[..., 1] - sat[..., 1]
    dz0 = rx[..., 2] - sat[..., 2]
    r0 = np.sqrt(dx0 * dx0 + dy0 * dy0 + dz0 * dz0)
    theta = EARTH_ROTATION_RATE_RAD_S * (r0 / LIGHT_SPEED_MPS)
    sx_rot = sat[..., 0] * np.cos(theta) + sat[..., 1] * np.sin(theta)
    sy_rot = -sat[..., 0] * np.sin(theta) + sat[..., 1] * np.cos(theta)
    delta = np.stack((rx[..., 0] - sx_rot, rx[..., 1] - sy_rot, rx[..., 2] - sat[..., 2]), axis=-1)
    ranges = np.linalg.norm(delta, axis=-1)
    unit = np.zeros_like(delta, dtype=np.float64)
    valid = np.isfinite(ranges) & (ranges > 1.0e-6) & np.isfinite(delta).all(axis=-1)
    unit[valid] = delta[valid] / ranges[valid, None]
    unit[~valid] = np.nan
    return unit


def _tdcp_residuals(
    batch: Any,
    state: np.ndarray,
    *,
    tdcp_use_drift: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if batch.tdcp_meas is None or batch.tdcp_weights is None or batch.times_ms.size <= 1:
        shape = (max(batch.times_ms.size - 1, 0), batch.weights.shape[1])
        empty = np.zeros(shape, dtype=np.float64)
        return empty, empty, empty, empty, empty, empty

    x0 = state[:-1, :3]
    x1 = state[1:, :3]
    los = _sagnac_unit_vectors_like_vd_tdcp(batch.sat_ecef[1:], x1[:, None, :])
    position_component = np.sum(los * (x1 - x0)[:, None, :], axis=2)
    clock_component = np.zeros_like(position_component, dtype=np.float64)
    predicted = position_component.copy()
    if tdcp_use_drift:
        dt = np.asarray(batch.dt, dtype=np.float64).reshape(-1)[: predicted.shape[0]]
        drift = state[:, 6 + batch.n_clock]
        clock_component = np.broadcast_to(
            0.5 * dt[:, None] * (drift[:-1, None] + drift[1:, None]),
            predicted.shape,
        ).copy()
        predicted += clock_component
        valid_time = np.isfinite(dt) & (dt > 0.0)
    else:
        sys_kind = batch.sys_kind[1:] if batch.sys_kind is not None else np.zeros_like(batch.tdcp_meas, dtype=np.int32)
        for t in range(predicted.shape[0]):
            design = fill_clock_design(np.asarray(sys_kind[t], dtype=np.int32), batch.n_clock)
            clock_delta = state[t + 1, 6 : 6 + batch.n_clock] - state[t, 6 : 6 + batch.n_clock]
            clock_component[t] = design @ clock_delta
            predicted[t] += clock_component[t]
        valid_time = np.ones(predicted.shape[0], dtype=bool)

    observed = batch.tdcp_meas
    residual = observed - predicted
    weights = np.asarray(batch.tdcp_weights, dtype=np.float64)
    valid = (
        valid_time[:, None]
        & (weights > 0.0)
        & np.isfinite(observed)
        & np.isfinite(predicted)
    )
    return residual, predicted, observed, np.where(valid, weights, 0.0), position_component, clock_component


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "factor",
        "epoch_index",
        "unix_time_millis",
        "slot_index",
        "constellation_type",
        "svid",
        "signal_type",
        "residual",
        "abs_residual",
        "weight",
        "weighted_contribution",
        "observed",
        "predicted",
        "component_a",
        "component_b",
    ]
    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_vd_factor_residual_diagnosis(args: Any) -> dict[str, Any]:
    trip_dir = args.data_root / args.trip
    batch = build_trip_arrays(
        trip_dir,
        max_epochs=max_epochs_for_build(args.max_epochs),
        start_epoch=args.start_epoch,
        constellation_type=args.constellation_type,
        signal_type=args.signal_type,
        weight_mode=args.weight_mode,
        multi_gnss=args.multi_gnss,
        use_tdcp=args.tdcp,
        tdcp_consistency_threshold_m=args.tdcp_consistency_threshold_m,
        tdcp_weight_scale=args.tdcp_weight_scale,
        tdcp_geometry_correction=args.tdcp_geometry_correction,
        data_root=args.data_root,
        trip=args.trip,
        apply_observation_mask=args.observation_mask,
        observation_min_cn0_dbhz=args.observation_min_cn0_dbhz,
        observation_min_elevation_deg=args.observation_min_elevation_deg,
        pseudorange_residual_mask_m=args.pseudorange_residual_mask_m,
        pseudorange_residual_mask_l5_m=args.pseudorange_residual_mask_l5_m,
        doppler_residual_mask_mps=args.doppler_residual_mask_mps,
        pseudorange_doppler_mask_m=args.pseudorange_doppler_mask_m,
        dual_frequency=args.dual_frequency,
    )

    baseline_state, baseline_sse, baseline_weight_sum, _ = fit_state_with_clock_bias(
        batch.sat_ecef,
        batch.pseudorange,
        batch.weights,
        batch.kaggle_wls,
        sys_kind=batch.sys_kind,
        n_clock=batch.n_clock,
    )
    raw_wls = run_wls(
        batch.sat_ecef,
        batch.pseudorange,
        batch.weights,
        sys_kind=batch.sys_kind,
        n_clock=batch.n_clock,
        fallback_xyz=batch.kaggle_wls,
    )
    raw_wls[:, :3] = repair_baseline_wls(batch.times_ms, raw_wls[:, :3])
    raw_state, raw_sse, raw_weight_sum, _ = fit_state_with_clock_bias(
        batch.sat_ecef,
        batch.pseudorange,
        batch.weights,
        raw_wls[:, :3],
        sys_kind=batch.sys_kind,
        n_clock=batch.n_clock,
    )
    dt = batch.dt if batch.dt is not None else np.diff(batch.times_ms) * 1e-3
    seed_state = _seed_vd_state(
        raw_state,
        baseline_state,
        dt,
        n_clock=batch.n_clock,
        clock_drift_mps=batch.clock_drift_mps,
    )

    if args.tdcp_use_drift == "auto":
        tdcp_use_drift = _tdcp_use_drift_for_phone(Path(args.trip).name)
    else:
        tdcp_use_drift = args.tdcp_use_drift == "yes"
    dop_res, dop_pred, dop_obs, dop_w, dop_rate_component, dop_drift_component = _doppler_residuals(batch, seed_state)
    tdcp_res, tdcp_pred, tdcp_obs, tdcp_w, tdcp_position_component, tdcp_clock_component = _tdcp_residuals(
        batch,
        seed_state,
        tdcp_use_drift=tdcp_use_drift,
    )

    top_rows = []
    top_rows.extend(
        _top_rows(
            factor="doppler",
            times_ms=batch.times_ms,
            slot_keys=tuple(batch.slot_keys),
            residual=dop_res,
            predicted=dop_pred,
            observed=dop_obs,
            weights=dop_w,
            limit=args.top,
            component_a=dop_rate_component,
            component_b=dop_drift_component,
        ),
    )
    if tdcp_res.size > 0:
        top_rows.extend(
            _top_rows(
                factor="tdcp",
                times_ms=batch.times_ms[:-1],
                slot_keys=tuple(batch.slot_keys),
                residual=tdcp_res,
                predicted=tdcp_pred,
                observed=tdcp_obs,
                weights=tdcp_w,
                limit=args.top,
                component_a=tdcp_position_component,
                component_b=tdcp_clock_component,
            ),
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "trip": args.trip,
        "start_epoch": args.start_epoch,
        "n_epochs": int(batch.times_ms.size),
        "n_sat_slots": int(batch.weights.shape[1]),
        "n_clock": int(batch.n_clock),
        "multi_gnss": bool(args.multi_gnss),
        "dual_frequency": bool(args.dual_frequency),
        "observation_mask": bool(args.observation_mask),
        "tdcp_enabled": bool(args.tdcp and batch.tdcp_meas is not None),
        "tdcp_use_drift": bool(tdcp_use_drift),
        "mask_counts": {
            "observation_mask": int(batch.observation_mask_count),
            "pseudorange_residual": int(batch.residual_mask_count),
            "doppler_residual": int(batch.doppler_residual_mask_count),
            "pseudorange_doppler": int(batch.pseudorange_doppler_mask_count),
            "tdcp_consistency": int(batch.tdcp_consistency_mask_count),
            "tdcp_geometry_correction": int(batch.tdcp_geometry_correction_count),
        },
        "pseudorange_mse": {
            "baseline": weighted_mse(baseline_sse, baseline_weight_sum),
            "raw_wls": weighted_mse(raw_sse, raw_weight_sum),
        },
        "doppler_seed_residual": _summary(dop_res, dop_w),
        "tdcp_seed_residual": _summary(tdcp_res, tdcp_w),
        "csv_component_columns": {
            "doppler": {
                "component_a": "-geometric_range_rate_with_sat_clock_drift",
                "component_b": "receiver_clock_drift",
            },
            "tdcp": {
                "component_a": "los_position_delta",
                "component_b": "clock_delta_or_average_drift_times_dt",
            },
        },
        "outputs": {
            "summary_json": str(args.output_dir / "vd_factor_residual_summary.json"),
            "top_residuals_csv": str(args.output_dir / "vd_factor_top_residuals.csv"),
        },
    }
    with (args.output_dir / "vd_factor_residual_summary.json").open("w") as fp:
        json.dump(_json_safe(summary), fp, indent=2, sort_keys=True)
        fp.write("\n")
    _write_csv(args.output_dir / "vd_factor_top_residuals.csv", top_rows)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--trip", required=True)
    parser.add_argument("--start-epoch", type=int, default=0)
    parser.add_argument("--max-epochs", type=int, default=200, help="0 means full trip")
    parser.add_argument("--signal-type", type=str, default="GPS_L1_CA")
    parser.add_argument("--constellation-type", type=int, default=1)
    parser.add_argument("--weight-mode", choices=("sin2el", "cn0"), default="sin2el")
    parser.add_argument("--multi-gnss", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dual-frequency", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tdcp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--tdcp-use-drift",
        choices=("auto", "yes", "no"),
        default="auto",
        help="TDCP prediction clock model: auto uses phone policy, yes uses average drift*dt, no uses clock-state delta",
    )
    parser.add_argument("--tdcp-consistency-threshold-m", type=float, default=1.5)
    parser.add_argument("--tdcp-weight-scale", type=float, default=DEFAULT_TDCP_WEIGHT_SCALE)
    parser.add_argument("--tdcp-geometry-correction", action=argparse.BooleanOptionalAction, default=DEFAULT_TDCP_GEOMETRY_CORRECTION)
    parser.add_argument("--observation-mask", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--observation-min-cn0-dbhz", type=float, default=OBS_MASK_MIN_CN0_DBHZ)
    parser.add_argument("--observation-min-elevation-deg", type=float, default=OBS_MASK_MIN_ELEVATION_DEG)
    parser.add_argument("--pseudorange-residual-mask-m", type=float, default=OBS_MASK_RESIDUAL_THRESHOLD_M)
    parser.add_argument("--pseudorange-residual-mask-l5-m", type=float, default=OBS_MASK_RESIDUAL_THRESHOLD_L5_M)
    parser.add_argument("--doppler-residual-mask-mps", type=float, default=OBS_MASK_DOPPLER_RESIDUAL_THRESHOLD_MPS)
    parser.add_argument("--pseudorange-doppler-mask-m", type=float, default=OBS_MASK_PSEUDORANGE_DOPPLER_THRESHOLD_M)
    parser.add_argument("--output-dir", type=Path, default=Path("tmp/vd_factor_diagnostics"))
    parser.add_argument("--top", type=int, default=50)
    args = parser.parse_args()

    trip_dir = args.data_root / args.trip
    batch = build_trip_arrays(
        trip_dir,
        max_epochs=max_epochs_for_build(args.max_epochs),
        start_epoch=args.start_epoch,
        constellation_type=args.constellation_type,
        signal_type=args.signal_type,
        weight_mode=args.weight_mode,
        multi_gnss=args.multi_gnss,
        use_tdcp=args.tdcp,
        tdcp_consistency_threshold_m=args.tdcp_consistency_threshold_m,
        tdcp_weight_scale=args.tdcp_weight_scale,
        tdcp_geometry_correction=args.tdcp_geometry_correction,
        data_root=args.data_root,
        trip=args.trip,
        apply_observation_mask=args.observation_mask,
        observation_min_cn0_dbhz=args.observation_min_cn0_dbhz,
        observation_min_elevation_deg=args.observation_min_elevation_deg,
        pseudorange_residual_mask_m=args.pseudorange_residual_mask_m,
        pseudorange_residual_mask_l5_m=args.pseudorange_residual_mask_l5_m,
        doppler_residual_mask_mps=args.doppler_residual_mask_mps,
        pseudorange_doppler_mask_m=args.pseudorange_doppler_mask_m,
        dual_frequency=args.dual_frequency,
    )

    baseline_state, baseline_sse, baseline_weight_sum, _ = fit_state_with_clock_bias(
        batch.sat_ecef,
        batch.pseudorange,
        batch.weights,
        batch.kaggle_wls,
        sys_kind=batch.sys_kind,
        n_clock=batch.n_clock,
    )
    raw_wls = run_wls(
        batch.sat_ecef,
        batch.pseudorange,
        batch.weights,
        sys_kind=batch.sys_kind,
        n_clock=batch.n_clock,
        fallback_xyz=batch.kaggle_wls,
    )
    raw_wls[:, :3] = repair_baseline_wls(batch.times_ms, raw_wls[:, :3])
    raw_state, raw_sse, raw_weight_sum, _ = fit_state_with_clock_bias(
        batch.sat_ecef,
        batch.pseudorange,
        batch.weights,
        raw_wls[:, :3],
        sys_kind=batch.sys_kind,
        n_clock=batch.n_clock,
    )
    dt = batch.dt if batch.dt is not None else np.diff(batch.times_ms) * 1e-3
    seed_state = _seed_vd_state(
        raw_state,
        baseline_state,
        dt,
        n_clock=batch.n_clock,
        clock_drift_mps=batch.clock_drift_mps,
    )

    if args.tdcp_use_drift == "auto":
        tdcp_use_drift = _tdcp_use_drift_for_phone(Path(args.trip).name)
    else:
        tdcp_use_drift = args.tdcp_use_drift == "yes"
    dop_res, dop_pred, dop_obs, dop_w, dop_rate_component, dop_drift_component = _doppler_residuals(batch, seed_state)
    tdcp_res, tdcp_pred, tdcp_obs, tdcp_w, tdcp_position_component, tdcp_clock_component = _tdcp_residuals(
        batch,
        seed_state,
        tdcp_use_drift=tdcp_use_drift,
    )

    top_rows = []
    top_rows.extend(
        _top_rows(
            factor="doppler",
            times_ms=batch.times_ms,
            slot_keys=tuple(batch.slot_keys),
            residual=dop_res,
            predicted=dop_pred,
            observed=dop_obs,
            weights=dop_w,
            limit=args.top,
            component_a=dop_rate_component,
            component_b=dop_drift_component,
        ),
    )
    if tdcp_res.size > 0:
        top_rows.extend(
            _top_rows(
                factor="tdcp",
                times_ms=batch.times_ms[:-1],
                slot_keys=tuple(batch.slot_keys),
                residual=tdcp_res,
                predicted=tdcp_pred,
                observed=tdcp_obs,
                weights=tdcp_w,
                limit=args.top,
                component_a=tdcp_position_component,
                component_b=tdcp_clock_component,
            ),
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "trip": args.trip,
        "start_epoch": args.start_epoch,
        "n_epochs": int(batch.times_ms.size),
        "n_sat_slots": int(batch.weights.shape[1]),
        "n_clock": int(batch.n_clock),
        "multi_gnss": bool(args.multi_gnss),
        "dual_frequency": bool(args.dual_frequency),
        "observation_mask": bool(args.observation_mask),
        "tdcp_enabled": bool(args.tdcp and batch.tdcp_meas is not None),
        "tdcp_use_drift": bool(tdcp_use_drift),
        "mask_counts": {
            "observation_mask": int(batch.observation_mask_count),
            "pseudorange_residual": int(batch.residual_mask_count),
            "doppler_residual": int(batch.doppler_residual_mask_count),
            "pseudorange_doppler": int(batch.pseudorange_doppler_mask_count),
            "tdcp_consistency": int(batch.tdcp_consistency_mask_count),
            "tdcp_geometry_correction": int(batch.tdcp_geometry_correction_count),
        },
        "pseudorange_mse": {
            "baseline": weighted_mse(baseline_sse, baseline_weight_sum),
            "raw_wls": weighted_mse(raw_sse, raw_weight_sum),
        },
        "doppler_seed_residual": _summary(dop_res, dop_w),
        "tdcp_seed_residual": _summary(tdcp_res, tdcp_w),
        "csv_component_columns": {
            "doppler": {
                "component_a": "-geometric_range_rate_with_sat_clock_drift",
                "component_b": "receiver_clock_drift",
            },
            "tdcp": {
                "component_a": "los_position_delta",
                "component_b": "clock_delta_or_average_drift_times_dt",
            },
        },
        "outputs": {
            "summary_json": str(args.output_dir / "vd_factor_residual_summary.json"),
            "top_residuals_csv": str(args.output_dir / "vd_factor_top_residuals.csv"),
        },
    }
    with (args.output_dir / "vd_factor_residual_summary.json").open("w") as fp:
        json.dump(_json_safe(summary), fp, indent=2, sort_keys=True)
        fp.write("\n")
    _write_csv(args.output_dir / "vd_factor_top_residuals.csv", top_rows)

    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
