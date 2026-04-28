"""Per-trip solver execution context for GSDC2023 raw bridge runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from experiments.gsdc2023_clock_state import (
    CLOCK_DRIFT_BLOCKLIST_PHONES,
    clock_aid_enabled,
    clock_drift_seed_enabled,
    combine_clock_jump_masks,
    detect_clock_jumps_from_clock_bias,
)
from experiments.gsdc2023_imu import (
    VELOCITY_SMOOTH_WINDOW,
    VELOCITY_THRESHOLD_MPS,
    ecef_to_enu_relative,
)
from experiments.gsdc2023_observation_matrix import TripArrays
from experiments.gsdc2023_tdcp import tdcp_use_drift_for_phone


@dataclass(frozen=True)
class SolverExecutionContext:
    tdcp_use_drift: bool
    clock_use_average_drift: bool
    stop_mask: np.ndarray | None
    clock_jump: np.ndarray | None
    clock_drift_seed_mps: np.ndarray | None

    def run_kwargs(self) -> dict[str, Any]:
        return {
            "clock_jump": self.clock_jump,
            "clock_drift_seed_mps": self.clock_drift_seed_mps,
            "clock_use_average_drift": self.clock_use_average_drift,
            "tdcp_use_drift": self.tdcp_use_drift,
            "stop_mask": self.stop_mask,
        }


def estimate_speed_mps(xyz_ecef: np.ndarray, times_ms: np.ndarray | None = None) -> np.ndarray:
    xyz = np.asarray(xyz_ecef, dtype=np.float64).reshape(-1, 3)
    if xyz.size == 0:
        return np.zeros(0, dtype=np.float64)
    finite_rows = np.isfinite(xyz).all(axis=1)
    if not finite_rows.any():
        return np.zeros(xyz.shape[0], dtype=np.float64)
    origin_xyz = xyz[np.flatnonzero(finite_rows)[0]]
    enu = ecef_to_enu_relative(xyz, origin_xyz)
    if times_ms is None:
        times_s = np.arange(enu.shape[0], dtype=np.float64)
    else:
        times_s = np.asarray(times_ms, dtype=np.float64).reshape(-1) * 1e-3
        if times_s.size != enu.shape[0] or np.any(~np.isfinite(times_s)) or np.any(np.diff(times_s) <= 0.0):
            times_s = np.arange(enu.shape[0], dtype=np.float64)
    vel_enu = np.zeros_like(enu)
    if enu.shape[0] > 1:
        dt_s = np.diff(times_s)
        valid_dt = np.isfinite(dt_s) & (dt_s > 0.0)
        for axis in range(3):
            diff_axis = np.diff(enu[:, axis])
            vel_axis = np.zeros(enu.shape[0], dtype=np.float64)
            vel_axis[1:][valid_dt] = diff_axis[valid_dt] / dt_s[valid_dt]
            vel_axis = (
                pd.Series(vel_axis)
                .rolling(VELOCITY_SMOOTH_WINDOW, center=False, min_periods=1)
                .mean()
                .to_numpy(dtype=np.float64)
            )
            vel_enu[:, axis] = vel_axis
    return np.linalg.norm(vel_enu, axis=1)


def solver_stop_mask(
    stop_epochs: np.ndarray | None,
    reference_xyz_ecef: np.ndarray,
    times_ms: np.ndarray | None = None,
) -> np.ndarray | None:
    if stop_epochs is None:
        return None
    stop_mask = np.asarray(stop_epochs, dtype=bool).reshape(-1)
    if stop_mask.size == 0:
        return None
    speed_mps = estimate_speed_mps(reference_xyz_ecef, times_ms)
    if speed_mps.size != stop_mask.size:
        return stop_mask
    return stop_mask & np.isfinite(speed_mps) & (speed_mps < VELOCITY_THRESHOLD_MPS)


def build_solver_execution_context(
    phone_name: str,
    batch: TripArrays,
    baseline_state: np.ndarray,
) -> SolverExecutionContext:
    tdcp_use_drift = tdcp_use_drift_for_phone(phone_name)
    stop_mask = solver_stop_mask(batch.stop_epochs, batch.kaggle_wls, batch.times_ms)

    effective_clock_jump = None
    effective_clock_drift_mps = None
    if clock_aid_enabled(phone_name):
        phone_l = phone_name.lower()
        if phone_l in CLOCK_DRIFT_BLOCKLIST_PHONES and batch.clock_bias_m is not None:
            jump_bias = batch.clock_bias_m
        else:
            baseline = np.asarray(baseline_state, dtype=np.float64)
            jump_bias = baseline[:, 3]
        effective_clock_jump = combine_clock_jump_masks(
            batch.clock_jump,
            detect_clock_jumps_from_clock_bias(jump_bias, phone_name),
        )
        effective_clock_drift_mps = batch.clock_drift_mps if clock_drift_seed_enabled(phone_name) else None

    return SolverExecutionContext(
        tdcp_use_drift=tdcp_use_drift,
        clock_use_average_drift=tdcp_use_drift,
        stop_mask=stop_mask,
        clock_jump=effective_clock_jump,
        clock_drift_seed_mps=effective_clock_drift_mps,
    )


__all__ = [
    "SolverExecutionContext",
    "build_solver_execution_context",
    "estimate_speed_mps",
    "solver_stop_mask",
]
