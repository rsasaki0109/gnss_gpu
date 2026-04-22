"""TDCP motion-estimation helpers shared by PF smoother experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from gnss_gpu.pf_smoother_common import finite_float as _finite_float
from gnss_gpu.tdcp_velocity import estimate_velocity_from_tdcp_with_metrics


@dataclass(frozen=True)
class TdcpPositionUpdateDecision:
    apply_update: bool
    predicted_position: np.ndarray | None
    gate_reason: str
    gate_skipped: bool = False


@dataclass(frozen=True)
class TdcpMotionEstimate:
    velocity: np.ndarray | None
    rms: float
    reason: str
    spp_diff_mps: float | None

    @property
    def ok(self) -> bool:
        return self.velocity is not None


def _estimate_tdcp_motion_velocity(
    receiver_position_ecef: np.ndarray,
    prev_measurements: list | None,
    measurements: list,
    dt: float,
    spp_lookup: dict[float, np.ndarray],
    *,
    prev_tow: float | None,
    tow_key: float,
    elevation_weight: bool,
    el_sin_floor: float,
    rms_max_m: float | None,
    spp_max_diff_mps: float | None,
) -> tuple[np.ndarray | None, float, str, float | None]:
    if prev_measurements is None or dt <= 0:
        return None, float("nan"), "no_previous_epoch", None
    receiver_position = np.asarray(receiver_position_ecef, dtype=np.float64).ravel()[:3]
    if receiver_position.shape[0] != 3 or not np.isfinite(receiver_position).all():
        return None, float("nan"), "invalid_receiver_position", None

    velocity, rms = estimate_velocity_from_tdcp_with_metrics(
        receiver_position,
        prev_measurements,
        measurements,
        dt=dt,
        elevation_weight=elevation_weight,
        el_sin_floor=el_sin_floor,
    )
    rms_f = float(rms) if np.isfinite(rms) else float("nan")
    if velocity is None:
        return None, rms_f, "tdcp_failed", None
    velocity_arr = np.asarray(velocity, dtype=np.float64).ravel()[:3]
    if velocity_arr.shape[0] != 3 or not np.isfinite(velocity_arr).all():
        return None, rms_f, "tdcp_failed", None

    spp_diff_mps: float | None = None
    if prev_tow is not None:
        prev_key = round(prev_tow, 1)
        if tow_key in spp_lookup and prev_key in spp_lookup:
            spp_delta = np.asarray(spp_lookup[tow_key][:3] - spp_lookup[prev_key][:3], dtype=np.float64)
            if np.isfinite(spp_delta).all():
                spp_fd_velocity = spp_delta / float(dt)
                spp_diff_mps = float(np.linalg.norm(velocity_arr - spp_fd_velocity))
                if spp_max_diff_mps is not None and spp_diff_mps > float(spp_max_diff_mps):
                    return None, rms_f, "spp_guard", spp_diff_mps

    if rms_max_m is not None and (not np.isfinite(rms_f) or rms_f > float(rms_max_m)):
        return None, rms_f, "rms_guard", spp_diff_mps
    return velocity_arr, rms_f, "ok", spp_diff_mps


def estimate_tdcp_position_update_motion(
    receiver_position_ecef: np.ndarray,
    prev_measurements: list | None,
    measurements: list,
    dt: float,
    spp_lookup: dict[float, np.ndarray],
    *,
    prev_tow: float | None,
    tow_key: float,
    elevation_weight: bool,
    el_sin_floor: float,
    rms_max_m: float,
    spp_max_diff_mps: float | None,
) -> TdcpMotionEstimate:
    velocity, rms, reason, spp_diff_mps = _estimate_tdcp_motion_velocity(
        receiver_position_ecef,
        prev_measurements,
        measurements,
        dt,
        spp_lookup,
        prev_tow=prev_tow,
        tow_key=tow_key,
        elevation_weight=elevation_weight,
        el_sin_floor=el_sin_floor,
        rms_max_m=rms_max_m,
        spp_max_diff_mps=(
            None
            if spp_max_diff_mps is None or float(spp_max_diff_mps) <= 0.0
            else float(spp_max_diff_mps)
        ),
    )
    return TdcpMotionEstimate(velocity, rms, reason, spp_diff_mps)


def estimate_local_fgo_tdcp_motion(
    receiver_position_ecef: np.ndarray,
    prev_measurements: list | None,
    measurements: list,
    dt: float,
    spp_lookup: dict[float, np.ndarray],
    *,
    prev_tow: float | None,
    tow_key: float,
    elevation_weight: bool,
    el_sin_floor: float,
    rms_max_m: float,
    spp_max_diff_mps: float | None,
) -> TdcpMotionEstimate:
    velocity, rms, reason, spp_diff_mps = _estimate_tdcp_motion_velocity(
        receiver_position_ecef,
        prev_measurements,
        measurements,
        dt,
        spp_lookup,
        prev_tow=prev_tow,
        tow_key=tow_key,
        elevation_weight=elevation_weight,
        el_sin_floor=el_sin_floor,
        rms_max_m=rms_max_m,
        spp_max_diff_mps=spp_max_diff_mps,
    )
    return TdcpMotionEstimate(velocity, rms, reason, spp_diff_mps)


def evaluate_tdcp_position_update(
    prev_estimate: np.ndarray,
    tdcp_velocity: np.ndarray | None,
    tdcp_rms: float,
    dt: float,
    *,
    rms_max: float,
    tdcp_reason: str | None = None,
    dd_gate_stats: Any | None = None,
    dd_cp_input_pairs: int = 0,
    dd_pr_gate_stats: Any | None = None,
    dd_pr_input_pairs: int = 0,
    gate_spread_m: float | None = None,
    gate_ess_ratio: float | None = None,
    dd_pr_raw_abs_res_median_m: float | None = None,
    dd_cp_raw_abs_afv_median_cycles: float | None = None,
    gate_dd_carrier_min_pairs: int | None = None,
    gate_dd_carrier_max_pairs: int | None = None,
    gate_dd_pseudorange_max_pairs: int | None = None,
    gate_min_spread_m: float | None = None,
    gate_max_spread_m: float | None = None,
    gate_min_ess_ratio: float | None = None,
    gate_max_ess_ratio: float | None = None,
    gate_dd_pr_max_raw_median_m: float | None = None,
    gate_dd_cp_max_raw_afv_median_cycles: float | None = None,
    gate_logic: str = "any",
    gate_stop_mode: str = "any",
    imu_stop_detected: bool = False,
) -> TdcpPositionUpdateDecision:
    velocity = None if tdcp_velocity is None else np.asarray(tdcp_velocity, dtype=np.float64).ravel()[:3]
    if (
        velocity is None
        or velocity.shape[0] != 3
        or not np.isfinite(velocity).all()
        or not np.isfinite(tdcp_rms)
        or float(tdcp_rms) >= float(rms_max)
    ):
        return TdcpPositionUpdateDecision(
            apply_update=False,
            predicted_position=None,
            gate_reason=tdcp_reason or "invalid_tdcp",
            gate_skipped=False,
        )

    dd_cp_kept_for_gate = (
        int(dd_gate_stats.n_kept_pairs)
        if dd_gate_stats is not None
        else int(dd_cp_input_pairs)
    )
    dd_pr_kept_for_gate = (
        int(dd_pr_gate_stats.n_kept_pairs)
        if dd_pr_gate_stats is not None
        else int(dd_pr_input_pairs)
    )

    gate_conditions: list[bool] = []
    if gate_dd_carrier_min_pairs is not None:
        gate_conditions.append(dd_cp_kept_for_gate >= int(gate_dd_carrier_min_pairs))
    if gate_dd_carrier_max_pairs is not None:
        gate_conditions.append(dd_cp_kept_for_gate <= int(gate_dd_carrier_max_pairs))
    if gate_dd_pseudorange_max_pairs is not None:
        gate_conditions.append(dd_pr_kept_for_gate <= int(gate_dd_pseudorange_max_pairs))
    if gate_min_spread_m is not None:
        gate_conditions.append(
            gate_spread_m is not None and float(gate_spread_m) >= float(gate_min_spread_m)
        )
    if gate_max_spread_m is not None:
        gate_conditions.append(
            gate_spread_m is not None and float(gate_spread_m) <= float(gate_max_spread_m)
        )
    if gate_min_ess_ratio is not None:
        gate_conditions.append(
            gate_ess_ratio is not None and float(gate_ess_ratio) >= float(gate_min_ess_ratio)
        )
    if gate_max_ess_ratio is not None:
        gate_conditions.append(
            gate_ess_ratio is not None and float(gate_ess_ratio) <= float(gate_max_ess_ratio)
        )
    if gate_dd_pr_max_raw_median_m is not None:
        raw_dd_pr_median = _finite_float(dd_pr_raw_abs_res_median_m)
        gate_conditions.append(
            raw_dd_pr_median is not None
            and raw_dd_pr_median <= float(gate_dd_pr_max_raw_median_m)
        )
    if gate_dd_cp_max_raw_afv_median_cycles is not None:
        raw_dd_cp_median = _finite_float(dd_cp_raw_abs_afv_median_cycles)
        gate_conditions.append(
            raw_dd_cp_median is not None
            and raw_dd_cp_median <= float(gate_dd_cp_max_raw_afv_median_cycles)
        )

    logic = str(gate_logic).strip().lower()
    gate_ok = (
        True
        if not gate_conditions
        else (all(gate_conditions) if logic == "all" else any(gate_conditions))
    )
    stop_mode = str(gate_stop_mode).strip().lower()
    if stop_mode == "stopped" and not imu_stop_detected:
        return TdcpPositionUpdateDecision(
            apply_update=False,
            predicted_position=None,
            gate_reason="gate_not_stopped",
            gate_skipped=True,
        )
    if stop_mode == "moving" and imu_stop_detected:
        return TdcpPositionUpdateDecision(
            apply_update=False,
            predicted_position=None,
            gate_reason="gate_not_moving",
            gate_skipped=True,
        )
    if not gate_ok:
        return TdcpPositionUpdateDecision(
            apply_update=False,
            predicted_position=None,
            gate_reason="gate_no_match",
            gate_skipped=True,
        )

    predicted_position = np.asarray(prev_estimate, dtype=np.float64).ravel()[:3] + velocity * float(dt)
    if predicted_position.shape[0] != 3 or not np.isfinite(predicted_position).all():
        return TdcpPositionUpdateDecision(
            apply_update=False,
            predicted_position=None,
            gate_reason="invalid_predicted_position",
            gate_skipped=False,
        )

    return TdcpPositionUpdateDecision(
        apply_update=True,
        predicted_position=predicted_position,
        gate_reason="ok",
        gate_skipped=False,
    )
