"""Residual model and masking helpers for GSDC2023 raw parity.

The functions here are numeric kernels: no trip discovery, no CSV loading, and
no solver orchestration.  Keeping them independent makes mask and residual
parity testable without building full trip arrays.
"""

from __future__ import annotations

import numpy as np


LIGHT_SPEED_MPS = 299792458.0
EARTH_ROTATION_RATE_RAD_S = 7.2921151467e-5
DEFAULT_PSEUDORANGE_RESIDUAL_THRESHOLD_M = 20.0
DEFAULT_DOPPLER_RESIDUAL_THRESHOLD_MPS = 3.0
DEFAULT_PSEUDORANGE_DOPPLER_THRESHOLD_M = 40.0


def sagnac_correction_m(sat_ecef: np.ndarray, receiver_ecef: np.ndarray) -> np.ndarray:
    sat = np.asarray(sat_ecef, dtype=np.float64)
    rx = np.asarray(receiver_ecef, dtype=np.float64)
    return EARTH_ROTATION_RATE_RAD_S * (sat[..., 0] * rx[..., 1] - sat[..., 1] * rx[..., 0]) / LIGHT_SPEED_MPS


def geometric_range_with_sagnac(sat_ecef: np.ndarray, receiver_ecef: np.ndarray) -> np.ndarray:
    sat = np.asarray(sat_ecef, dtype=np.float64)
    rx = np.asarray(receiver_ecef, dtype=np.float64)
    return np.linalg.norm(sat - rx, axis=-1) + sagnac_correction_m(sat, rx)


def geometric_range_rate_with_sagnac(
    sat_ecef: np.ndarray,
    receiver_ecef: np.ndarray,
    sat_vel: np.ndarray,
    receiver_vel: np.ndarray,
) -> np.ndarray:
    sat = np.asarray(sat_ecef, dtype=np.float64)
    rx = np.asarray(receiver_ecef, dtype=np.float64)
    sv = np.asarray(sat_vel, dtype=np.float64)
    rv = np.asarray(receiver_vel, dtype=np.float64)
    delta = sat - rx
    ranges = np.linalg.norm(delta, axis=-1)
    los = np.zeros_like(delta, dtype=np.float64)
    valid = np.isfinite(ranges) & (ranges > 0.0) & np.isfinite(delta).all(axis=-1)
    los[valid] = delta[valid] / ranges[valid, None]
    euclidean_rate = np.sum(los * (sv - rv), axis=-1)
    sagnac_rate = EARTH_ROTATION_RATE_RAD_S * (
        sv[..., 0] * rx[..., 1]
        + sat[..., 0] * rv[..., 1]
        - sv[..., 1] * rx[..., 0]
        - sat[..., 1] * rv[..., 0]
    ) / LIGHT_SPEED_MPS
    return euclidean_rate - sagnac_rate


def interpolate_series(times_s: np.ndarray, values: np.ndarray) -> np.ndarray:
    times = np.asarray(times_s, dtype=np.float64)
    vals = np.asarray(values, dtype=np.float64)
    out = vals.copy()
    valid = np.isfinite(times) & np.isfinite(vals)
    if not valid.any():
        return out
    if np.count_nonzero(valid) == 1:
        out[~valid] = float(vals[valid][0])
        return out
    out[~valid] = np.interp(times[~valid], times[valid], vals[valid])
    return out


def gradient_with_dt(values: np.ndarray, times_s: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    times_s = np.asarray(times_s, dtype=np.float64)
    if values.size <= 1:
        return np.zeros(values.size, dtype=np.float64)
    return np.gradient(values, times_s, edge_order=1)


def matlab_epoch_interval_s(times_ms: np.ndarray) -> float:
    times_s = np.asarray(times_ms, dtype=np.float64).reshape(-1) * 1e-3
    if times_s.size <= 1:
        return 0.0
    diffs = np.diff(times_s)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
    if diffs.size == 0:
        return 0.0
    return float(np.round(np.median(diffs), 2))


def gradient_with_matlab_interval(values: np.ndarray, dt_s: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size <= 1 or not np.isfinite(dt_s) or dt_s <= 0.0:
        return np.zeros(values.size, dtype=np.float64)
    return np.gradient(values, edge_order=1) / float(dt_s)


def receiver_velocity_from_reference(times_ms: np.ndarray, reference_xyz: np.ndarray) -> np.ndarray:
    times_s = np.asarray(times_ms, dtype=np.float64) * 1e-3
    rx_xyz = np.asarray(reference_xyz, dtype=np.float64).reshape(-1, 3)
    rx_vel = np.zeros_like(rx_xyz)
    if rx_xyz.shape[0] <= 1:
        return rx_vel
    matlab_dt_s = matlab_epoch_interval_s(times_ms)
    for axis in range(3):
        series = interpolate_series(times_s, rx_xyz[:, axis])
        rx_vel[:, axis] = gradient_with_matlab_interval(series, matlab_dt_s)
    rx_vel[~np.isfinite(rx_vel)] = 0.0
    return rx_vel


def estimate_residual_clock_series(
    times_ms: np.ndarray,
    baseline_xyz: np.ndarray,
    sat_ecef: np.ndarray,
    pseudorange: np.ndarray,
    sat_vel: np.ndarray | None,
    doppler: np.ndarray | None,
    *,
    sat_clock_drift_mps: np.ndarray | None = None,
    sys_kind: np.ndarray | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    times_s = np.asarray(times_ms, dtype=np.float64) * 1e-3
    rx_xyz = np.asarray(baseline_xyz, dtype=np.float64).reshape(-1, 3)
    sat_xyz = np.asarray(sat_ecef, dtype=np.float64)
    pr = np.asarray(pseudorange, dtype=np.float64)
    if rx_xyz.shape[0] == 0 or sat_xyz.shape[:2] != pr.shape:
        return None, None

    gps_mask = np.ones(pr.shape, dtype=bool) if sys_kind is None else (np.asarray(sys_kind, dtype=np.int32) == 0)
    delta = sat_xyz - rx_xyz[:, None, :]
    ranges = geometric_range_with_sagnac(sat_xyz, rx_xyz[:, None, :])
    valid_pr = (
        gps_mask
        & np.isfinite(pr)
        & (pr > 1.0e7)
        & (pr < 4.0e7)
        & np.isfinite(ranges)
        & (ranges > 1.0e6)
    )

    clock_bias = np.full(rx_xyz.shape[0], np.nan, dtype=np.float64)
    for epoch_idx in range(rx_xyz.shape[0]):
        vals = pr[epoch_idx, valid_pr[epoch_idx]] - ranges[epoch_idx, valid_pr[epoch_idx]]
        if vals.size > 0:
            clock_bias[epoch_idx] = float(np.median(vals))

    if np.isfinite(clock_bias).any():
        clock_bias = interpolate_series(times_s, clock_bias)
    else:
        clock_bias = None

    if sat_vel is None or doppler is None:
        return clock_bias, None

    sat_velocity = np.asarray(sat_vel, dtype=np.float64)
    dop = np.asarray(doppler, dtype=np.float64)
    if sat_velocity.shape != sat_xyz.shape or dop.shape != pr.shape:
        return clock_bias, None

    rx_vel = np.zeros_like(rx_xyz)
    matlab_dt_s = matlab_epoch_interval_s(times_ms)
    for axis in range(3):
        rx_vel[:, axis] = gradient_with_matlab_interval(
            interpolate_series(times_s, rx_xyz[:, axis]),
            matlab_dt_s,
        )
    euclidean_ranges = np.linalg.norm(delta, axis=2)
    valid_range = valid_pr & (euclidean_ranges > 0.0)
    geom_rate = geometric_range_rate_with_sagnac(sat_xyz, rx_xyz[:, None, :], sat_velocity, rx_vel[:, None, :])
    if sat_clock_drift_mps is not None:
        sat_clock_drift = np.asarray(sat_clock_drift_mps, dtype=np.float64)
        if sat_clock_drift.shape == geom_rate.shape:
            finite_clock_drift = np.isfinite(sat_clock_drift)
            geom_rate[finite_clock_drift] -= sat_clock_drift[finite_clock_drift]
    valid_doppler = valid_range & np.isfinite(dop) & np.isfinite(geom_rate)

    clock_drift = np.full(rx_xyz.shape[0], np.nan, dtype=np.float64)
    for epoch_idx in range(rx_xyz.shape[0]):
        vals = dop[epoch_idx, valid_doppler[epoch_idx]] + geom_rate[epoch_idx, valid_doppler[epoch_idx]]
        if vals.size > 0:
            clock_drift[epoch_idx] = float(np.median(vals))

    if np.isfinite(clock_drift).any():
        clock_drift = interpolate_series(times_s, clock_drift)
    else:
        clock_drift = None
    return clock_bias, clock_drift


def fill_clock_design(system_kind: np.ndarray, n_clock: int) -> np.ndarray:
    design = np.ones((system_kind.size, n_clock), dtype=np.float64)
    if n_clock > 1:
        design[:, 1:] = 0.0
        for row_idx, sk in enumerate(system_kind.astype(np.int32, copy=False)):
            if 0 < sk < n_clock:
                design[row_idx, sk] = 1.0
    return design


def solve_clock_biases(residual: np.ndarray, weights: np.ndarray, system_kind: np.ndarray, n_clock: int) -> np.ndarray:
    if n_clock <= 1:
        weight_sum = float(np.sum(weights))
        if weight_sum <= 0.0:
            return np.zeros(1, dtype=np.float64)
        return np.array([float(np.sum(weights * residual) / weight_sum)], dtype=np.float64)

    design = fill_clock_design(system_kind, n_clock)
    active = np.zeros(n_clock, dtype=bool)
    active[0] = True
    active[1:] = np.any(design[:, 1:] != 0.0, axis=0)
    design_active = design[:, active]
    sqrt_w = np.sqrt(weights)
    lhs = design_active * sqrt_w[:, None]
    rhs = residual * sqrt_w
    sol_active, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)
    sol = np.zeros(n_clock, dtype=np.float64)
    sol[active] = sol_active
    return sol


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    vals = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(vals) & np.isfinite(w) & (w > 0.0)
    if not valid.any():
        return float("nan")
    vals = vals[valid]
    w = w[valid]
    order = np.argsort(vals)
    vals = vals[order]
    cumulative = np.cumsum(w[order])
    cutoff = 0.5 * float(cumulative[-1])
    idx = min(int(np.searchsorted(cumulative, cutoff, side="left")), vals.size - 1)
    return float(vals[idx])


def median_clock_prediction(residual: np.ndarray, weights: np.ndarray, system_kind: np.ndarray, n_clock: int) -> np.ndarray:
    pred = np.zeros_like(residual, dtype=np.float64)
    if n_clock <= 1:
        bias = weighted_median(residual, weights)
        if np.isfinite(bias):
            pred.fill(float(bias))
        return pred

    for kind in sorted({int(value) for value in system_kind if 0 <= int(value) < n_clock}):
        idx = system_kind == kind
        if not np.any(idx):
            continue
        bias = weighted_median(residual[idx], weights[idx])
        if np.isfinite(bias):
            pred[idx] = float(bias)
    return pred


def pseudorange_global_isb_by_group(
    sat_ecef: np.ndarray,
    pseudorange: np.ndarray,
    sample_weights: np.ndarray,
    reference_xyz: np.ndarray,
    receiver_clock_bias_m: np.ndarray | None,
    *,
    sys_kind: np.ndarray | None = None,
    common_bias_group: np.ndarray | None = None,
) -> dict[int, float]:
    if receiver_clock_bias_m is None:
        return {}
    receiver_clock = np.asarray(receiver_clock_bias_m, dtype=np.float64).reshape(-1)
    samples = np.asarray(sample_weights, dtype=np.float64)
    if receiver_clock.size != samples.shape[0]:
        raise ValueError("receiver_clock_bias_m must have one value per epoch")
    if common_bias_group is not None:
        common_bias_group = np.asarray(common_bias_group, dtype=np.int32).reshape(-1)
        if common_bias_group.size != samples.shape[1]:
            raise ValueError("common_bias_group must have one value per satellite slot")

    group_samples: dict[int, list[float]] = {}
    n_epoch = samples.shape[0]
    for epoch_idx in range(n_epoch):
        clock_bias = float(receiver_clock[epoch_idx])
        if not np.isfinite(clock_bias):
            continue
        idx = np.flatnonzero(samples[epoch_idx] > 0.0)
        if idx.size == 0:
            continue
        rx = np.asarray(reference_xyz[epoch_idx], dtype=np.float64)
        if not np.isfinite(rx).all() or np.linalg.norm(rx) <= 1.0e3:
            continue
        ranges = geometric_range_with_sagnac(sat_ecef[epoch_idx, idx], rx)
        valid = np.isfinite(ranges) & (ranges > 1.0e6) & np.isfinite(pseudorange[epoch_idx, idx])
        if not np.any(valid):
            continue
        idx = idx[valid]
        residual_seed = pseudorange[epoch_idx, idx] - ranges[valid] - clock_bias
        if common_bias_group is not None:
            group_values = common_bias_group[idx]
        elif sys_kind is not None:
            group_values = sys_kind[epoch_idx, idx].astype(np.int32, copy=False)
        else:
            group_values = np.zeros(idx.size, dtype=np.int32)
        for group_id, value in zip(group_values, residual_seed):
            if int(group_id) < 0 or not np.isfinite(value):
                continue
            group_samples.setdefault(int(group_id), []).append(float(value))

    global_isb_by_group: dict[int, float] = {}
    for group_id, values in group_samples.items():
        arr = np.asarray(values, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            global_isb_by_group[int(group_id)] = float(np.median(arr))
    return global_isb_by_group


def mask_pseudorange_residual_outliers(
    sat_ecef: np.ndarray,
    pseudorange: np.ndarray,
    weights: np.ndarray,
    reference_xyz: np.ndarray,
    *,
    sys_kind: np.ndarray | None = None,
    n_clock: int = 1,
    threshold_m: float | np.ndarray = DEFAULT_PSEUDORANGE_RESIDUAL_THRESHOLD_M,
    receiver_clock_bias_m: np.ndarray | None = None,
    common_bias_group: np.ndarray | None = None,
    common_bias_sample_weights: np.ndarray | None = None,
    common_bias_by_group: dict[int, float] | None = None,
) -> int:
    threshold = np.asarray(threshold_m, dtype=np.float64)
    if threshold.size == 1 and float(threshold.reshape(-1)[0]) <= 0.0:
        return 0
    if threshold.size > 1 and not np.any(threshold > 0.0):
        return 0
    n_epoch = weights.shape[0]
    if threshold.size not in (1, weights.shape[1]):
        raise ValueError("threshold_m must be scalar or have one value per satellite slot")
    if common_bias_group is not None:
        common_bias_group = np.asarray(common_bias_group, dtype=np.int32).reshape(-1)
        if common_bias_group.size != weights.shape[1]:
            raise ValueError("common_bias_group must have one value per satellite slot")
    receiver_clock = None
    if receiver_clock_bias_m is not None:
        receiver_clock = np.asarray(receiver_clock_bias_m, dtype=np.float64).reshape(-1)
        if receiver_clock.size != n_epoch:
            raise ValueError("receiver_clock_bias_m must have one value per epoch")
    if common_bias_by_group is None:
        sample_weights = weights if common_bias_sample_weights is None else common_bias_sample_weights
        global_isb_by_group = pseudorange_global_isb_by_group(
            sat_ecef,
            pseudorange,
            sample_weights,
            reference_xyz,
            receiver_clock_bias_m,
            sys_kind=sys_kind,
            common_bias_group=common_bias_group,
        )
    else:
        global_isb_by_group = {
            int(group_id): float(value)
            for group_id, value in common_bias_by_group.items()
            if np.isfinite(value)
        }

    masked_count = 0
    for epoch_idx in range(n_epoch):
        idx = np.flatnonzero(weights[epoch_idx] > 0.0)
        if idx.size == 0:
            continue
        rx = np.asarray(reference_xyz[epoch_idx], dtype=np.float64)
        if not np.isfinite(rx).all() or np.linalg.norm(rx) <= 1.0e3:
            continue
        ranges = geometric_range_with_sagnac(sat_ecef[epoch_idx, idx], rx)
        valid = np.isfinite(ranges) & (ranges > 1.0e6) & np.isfinite(pseudorange[epoch_idx, idx])
        if not np.any(valid):
            continue
        idx = idx[valid]
        ranges = ranges[valid]
        resid0 = pseudorange[epoch_idx, idx] - ranges
        w = weights[epoch_idx, idx]
        sk = (
            sys_kind[epoch_idx, idx].astype(np.int32, copy=False)
            if sys_kind is not None
            else np.zeros(idx.size, dtype=np.int32)
        )
        pred_bias = np.full(idx.size, np.nan, dtype=np.float64)
        if receiver_clock is not None and np.isfinite(receiver_clock[epoch_idx]) and global_isb_by_group:
            if common_bias_group is not None:
                group_values = common_bias_group[idx]
            else:
                group_values = sk
            clock_pred = np.full(idx.size, np.nan, dtype=np.float64)
            for local_idx, group_id in enumerate(group_values):
                isb = global_isb_by_group.get(int(group_id))
                if isb is not None and np.isfinite(isb):
                    clock_pred[local_idx] = float(receiver_clock[epoch_idx]) + float(isb)
            finite_pred = np.isfinite(clock_pred)
            if finite_pred.any():
                pred_bias[finite_pred] = clock_pred[finite_pred]
        needs_median = ~np.isfinite(pred_bias)
        if needs_median.any():
            if idx.size < 4:
                if not np.isfinite(pred_bias).any():
                    continue
            else:
                median_pred = median_clock_prediction(resid0, w, sk, n_clock)
                pred_bias[needs_median] = median_pred[needs_median]
        residual = resid0 - pred_bias
        threshold_i = float(threshold.reshape(-1)[0]) if threshold.size == 1 else threshold[idx]
        reject = np.isfinite(residual) & (threshold_i > 0.0) & (np.abs(residual) > threshold_i)
        if not reject.any():
            continue
        reject_idx = idx[reject]
        weights[epoch_idx, reject_idx] = 0.0
        masked_count += int(reject_idx.size)
    return masked_count


def mask_doppler_residual_outliers(
    times_ms: np.ndarray,
    sat_ecef: np.ndarray,
    sat_vel: np.ndarray | None,
    doppler: np.ndarray | None,
    doppler_weights: np.ndarray | None,
    reference_xyz: np.ndarray,
    *,
    threshold_mps: float = DEFAULT_DOPPLER_RESIDUAL_THRESHOLD_MPS,
    receiver_clock_drift_mps: np.ndarray | None = None,
    sat_clock_drift_mps: np.ndarray | None = None,
    velocity_times_ms: np.ndarray | None = None,
    velocity_reference_xyz: np.ndarray | None = None,
    clock_drift_times_ms: np.ndarray | None = None,
    clock_drift_reference_mps: np.ndarray | None = None,
) -> int:
    if threshold_mps <= 0.0 or sat_vel is None or doppler is None or doppler_weights is None:
        return 0
    sat_xyz = np.asarray(sat_ecef, dtype=np.float64)
    sat_velocity = np.asarray(sat_vel, dtype=np.float64)
    dop = np.asarray(doppler, dtype=np.float64)
    weights = np.asarray(doppler_weights, dtype=np.float64)
    rx_xyz = np.asarray(reference_xyz, dtype=np.float64).reshape(-1, 3)
    if sat_velocity.shape != sat_xyz.shape or dop.shape != weights.shape or sat_xyz.shape[:2] != weights.shape:
        return 0
    if rx_xyz.shape[0] != weights.shape[0]:
        return 0

    rx_vel = receiver_velocity_from_reference(times_ms, rx_xyz)
    if velocity_times_ms is not None and velocity_reference_xyz is not None:
        vel_times = np.asarray(velocity_times_ms, dtype=np.float64).reshape(-1)
        vel_xyz = np.asarray(velocity_reference_xyz, dtype=np.float64).reshape(-1, 3)
        if vel_times.size == vel_xyz.shape[0] and vel_times.size >= times_ms.size:
            full_rx_vel = receiver_velocity_from_reference(vel_times, vel_xyz)
            if full_rx_vel.shape == vel_xyz.shape:
                full_lookup = {int(round(float(time_ms))): idx for idx, time_ms in enumerate(vel_times)}
                edge_epochs = {0, weights.shape[0] - 1}
                for epoch_idx, time_ms in enumerate(times_ms):
                    if epoch_idx not in edge_epochs:
                        continue
                    full_idx = full_lookup.get(int(round(float(time_ms))))
                    if full_idx is None:
                        continue
                    candidate = full_rx_vel[full_idx]
                    if np.isfinite(candidate).all():
                        rx_vel[epoch_idx] = candidate
    delta = sat_xyz - rx_xyz[:, None, :]
    ranges = np.linalg.norm(delta, axis=2)
    valid_range = (
        np.isfinite(ranges)
        & (ranges > 1.0e6)
        & np.isfinite(rx_xyz).all(axis=1)[:, None]
        & np.isfinite(sat_xyz).all(axis=2)
    )
    geom_rate = geometric_range_rate_with_sagnac(sat_xyz, rx_xyz[:, None, :], sat_velocity, rx_vel[:, None, :])
    if sat_clock_drift_mps is not None:
        sat_clock_drift = np.asarray(sat_clock_drift_mps, dtype=np.float64)
        if sat_clock_drift.shape == geom_rate.shape:
            finite_clock_drift = np.isfinite(sat_clock_drift)
            geom_rate[finite_clock_drift] -= sat_clock_drift[finite_clock_drift]
    valid = valid_range & (weights > 0.0) & np.isfinite(dop) & np.isfinite(geom_rate)

    clock_drift = None
    if receiver_clock_drift_mps is not None:
        arr = np.asarray(receiver_clock_drift_mps, dtype=np.float64).reshape(-1)
        if arr.size == weights.shape[0]:
            clock_drift = arr.copy()
    if clock_drift is not None and clock_drift_times_ms is not None and clock_drift_reference_mps is not None:
        drift_times = np.asarray(clock_drift_times_ms, dtype=np.float64).reshape(-1)
        drift_context = np.asarray(clock_drift_reference_mps, dtype=np.float64).reshape(-1)
        if drift_times.size == drift_context.size and drift_times.size >= times_ms.size:
            drift_lookup = {int(round(float(time_ms))): idx for idx, time_ms in enumerate(drift_times)}
            edge_epochs = {0, weights.shape[0] - 1}
            for epoch_idx, time_ms in enumerate(times_ms):
                if epoch_idx not in edge_epochs:
                    continue
                full_idx = drift_lookup.get(int(round(float(time_ms))))
                if full_idx is None:
                    continue
                candidate = float(drift_context[full_idx])
                if np.isfinite(candidate):
                    clock_drift[epoch_idx] = candidate

    masked_count = 0
    for epoch_idx in range(weights.shape[0]):
        idx = np.flatnonzero(valid[epoch_idx])
        if idx.size == 0:
            continue
        residual0 = dop[epoch_idx, idx] + geom_rate[epoch_idx, idx]
        if clock_drift is not None and np.isfinite(clock_drift[epoch_idx]):
            drift = float(clock_drift[epoch_idx])
        elif idx.size >= 4:
            drift = weighted_median(residual0, weights[epoch_idx, idx])
        else:
            continue
        if not np.isfinite(drift):
            continue
        residual = residual0 - drift
        reject = np.isfinite(residual) & (np.abs(residual) > float(threshold_mps))
        if not reject.any():
            continue
        reject_idx = idx[reject]
        doppler_weights[epoch_idx, reject_idx] = 0.0
        masked_count += int(reject_idx.size)
    return masked_count


def min_pseudorange_keep_count(system_kind: np.ndarray | None, n_clock: int) -> int:
    min_keep = 4
    if system_kind is not None and n_clock > 1:
        active_clocks = len({int(val) for val in system_kind if 0 <= int(val) < n_clock})
        min_keep = max(min_keep, 3 + active_clocks)
    return min_keep


def mask_pseudorange_doppler_consistency(
    times_ms: np.ndarray,
    pseudorange: np.ndarray,
    weights: np.ndarray,
    doppler: np.ndarray | None,
    doppler_weights: np.ndarray | None,
    *,
    phone: str,
    sys_kind: np.ndarray | None = None,
    n_clock: int = 1,
    threshold_m: float | np.ndarray = DEFAULT_PSEUDORANGE_DOPPLER_THRESHOLD_M,
) -> int:
    del sys_kind, n_clock
    threshold = np.asarray(threshold_m, dtype=np.float64)
    if threshold.size == 1 and float(threshold.reshape(-1)[0]) <= 0.0:
        return 0
    if threshold.size > 1 and not np.any(threshold > 0.0):
        return 0
    if doppler is None or doppler_weights is None:
        return 0
    if phone.lower() in {"sm-a205u", "sm-a505u"}:
        return 0
    pr = np.asarray(pseudorange, dtype=np.float64)
    pr_weights = np.asarray(weights, dtype=np.float64)
    dop = np.asarray(doppler, dtype=np.float64)
    dop_w = np.asarray(doppler_weights, dtype=np.float64)
    if pr.shape != pr_weights.shape or dop.shape != pr.shape or dop_w.shape != pr.shape:
        return 0
    if threshold.size not in (1, pr.shape[1]):
        raise ValueError("threshold_m must be scalar or have one value per satellite slot")
    times_s = np.asarray(times_ms, dtype=np.float64) * 1e-3
    if times_s.size != pr.shape[0] or times_s.size <= 1:
        return 0
    matlab_dt_s = matlab_epoch_interval_s(times_ms)

    reject_all = np.zeros_like(pr_weights, dtype=bool)
    for epoch_idx in range(pr.shape[0] - 1):
        dt_s = float(times_s[epoch_idx + 1] - times_s[epoch_idx])
        if not np.isfinite(dt_s) or dt_s <= 0.0 or dt_s > 30.0:
            continue
        valid = (
            (pr_weights[epoch_idx] > 0.0)
            & (pr_weights[epoch_idx + 1] > 0.0)
            & (dop_w[epoch_idx] > 0.0)
            & (dop_w[epoch_idx + 1] > 0.0)
            & np.isfinite(pr[epoch_idx])
            & np.isfinite(pr[epoch_idx + 1])
            & np.isfinite(dop[epoch_idx])
            & np.isfinite(dop[epoch_idx + 1])
        )
        idx = np.flatnonzero(valid)
        if idx.size == 0:
            continue
        doppler_delta = -0.5 * (dop[epoch_idx, idx] + dop[epoch_idx + 1, idx]) * matlab_dt_s
        pseudorange_delta = pr[epoch_idx + 1, idx] - pr[epoch_idx, idx]
        residual = doppler_delta - pseudorange_delta
        threshold_i = float(threshold.reshape(-1)[0]) if threshold.size == 1 else threshold[idx]
        reject = np.isfinite(residual) & (threshold_i > 0.0) & (np.abs(residual) > threshold_i)
        if not reject.any():
            continue
        reject_idx = idx[reject]
        reject_all[epoch_idx, reject_idx] = True
        reject_all[epoch_idx + 1, reject_idx] = True
    masked_count = int(np.count_nonzero(reject_all & (pr_weights > 0.0)))
    pr_weights[reject_all] = 0.0
    return masked_count
