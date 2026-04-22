"""Predict-motion selection helpers for PF smoother forward epochs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

import numpy as np

from gnss_gpu.imu import ComplementaryHeadingFilter
from gnss_gpu.pf_smoother_runtime import spp_heading_from_velocity
from gnss_gpu.tdcp_motion import (
    estimate_local_fgo_tdcp_motion,
    estimate_tdcp_position_update_motion,
)
from gnss_gpu.tdcp_velocity import estimate_velocity_from_tdcp_with_metrics


@dataclass(frozen=True)
class EpochPredictMotionOptions:
    predict_guide: str
    tdcp_position_update: bool
    tdcp_elevation_weight: bool
    tdcp_el_sin_floor: float
    tdcp_rms_threshold: float
    tdcp_pu_rms_max: float
    tdcp_pu_spp_max_diff_mps: float | None
    need_fgo_tdcp_motion: bool
    fgo_local_tdcp_rms_max_m: float
    fgo_local_tdcp_spp_max_diff_mps: float | None


@dataclass(frozen=True)
class ImuPredictDecision:
    velocity: np.ndarray | None = None
    imu_velocity: np.ndarray | None = None
    used_imu: bool = False
    stop_detected: bool = False
    spp_fd_velocity: np.ndarray | None = None


@dataclass(frozen=True)
class TdcpPredictDecision:
    velocity: np.ndarray | None = None
    used_tdcp: bool = False
    adaptive_fallback: bool = False
    tdcp_rms: float = float("nan")
    tdcp_pu_velocity: np.ndarray | None = None
    tdcp_pu_rms: float = float("nan")
    tdcp_pu_spp_diff_mps: float | None = None


def evaluate_imu_predict_velocity(
    imu_filter,
    predict_guide: str,
    prev_tow: float,
    tow: float,
    current_position_ecef: np.ndarray,
    spp_lookup: dict[float, np.ndarray],
    ecef_to_lla_func: Callable[[float, float, float], tuple[float, float, float]],
    *,
    dt: float,
    stop_speed_mps: float = 0.01,
    spp_speed_max_mps: float = 50.0,
) -> ImuPredictDecision:
    if predict_guide not in ("imu", "imu_spp_blend") or imu_filter is None or dt <= 0:
        return ImuPredictDecision()

    current_position = np.asarray(current_position_ecef, dtype=np.float64).ravel()[:3]
    lat_r, lon_r, _ = ecef_to_lla_func(
        float(current_position[0]),
        float(current_position[1]),
        float(current_position[2]),
    )

    tow_key = round(float(tow), 1)
    prev_key = round(float(prev_tow), 1)
    spp_fd_velocity = spp_finite_difference_velocity(
        spp_lookup,
        prev_tow=prev_tow,
        tow_key=tow_key,
        dt=dt,
        max_speed_mps=None,
    )
    if spp_fd_velocity is not None:
        spp_heading = spp_heading_from_velocity(spp_fd_velocity, lat_r, lon_r)
        if spp_heading is not None:
            imu_filter.correct_heading_spp(spp_heading)
    elif tow_key not in spp_lookup or prev_key not in spp_lookup:
        spp_fd_velocity = None

    vel_enu = imu_filter.get_velocity_enu(prev_tow, tow)
    speed_enu = float(np.linalg.norm(np.asarray(vel_enu, dtype=np.float64)[:2]))
    if speed_enu <= float(stop_speed_mps):
        stopped = np.zeros(3)
        return ImuPredictDecision(
            velocity=stopped,
            imu_velocity=stopped,
            used_imu=True,
            stop_detected=True,
            spp_fd_velocity=spp_fd_velocity,
        )

    imu_velocity = ComplementaryHeadingFilter.velocity_enu_to_ecef(vel_enu, lat_r, lon_r)
    if predict_guide == "imu":
        return ImuPredictDecision(
            velocity=imu_velocity,
            imu_velocity=imu_velocity,
            used_imu=True,
            spp_fd_velocity=spp_fd_velocity,
        )

    velocity = imu_velocity
    if spp_fd_velocity is not None and float(np.linalg.norm(spp_fd_velocity)) < float(spp_speed_max_mps):
        velocity = 0.5 * imu_velocity + 0.5 * spp_fd_velocity
    return ImuPredictDecision(
        velocity=velocity,
        imu_velocity=velocity,
        used_imu=True,
        spp_fd_velocity=spp_fd_velocity,
    )


def evaluate_tdcp_predict_guide(
    predict_guide: str,
    receiver_position_ecef: np.ndarray,
    prev_measurements: list | None,
    measurements: list,
    dt: float,
    spp_lookup: dict[float, np.ndarray],
    *,
    prev_tow: float,
    tow_key: float,
    elevation_weight: bool,
    el_sin_floor: float,
    tdcp_rms_threshold: float,
    spp_guard_mps: float = 6.0,
) -> TdcpPredictDecision:
    if predict_guide not in ("tdcp", "tdcp_adaptive") or prev_measurements is None:
        return TdcpPredictDecision()

    receiver_position = np.asarray(receiver_position_ecef, dtype=np.float64).ravel()[:3]
    if not np.isfinite(receiver_position).all():
        return TdcpPredictDecision(
            adaptive_fallback=bool(predict_guide == "tdcp_adaptive"),
        )

    spp_fd_velocity = spp_finite_difference_velocity(
        spp_lookup,
        prev_tow=prev_tow,
        tow_key=tow_key,
        dt=dt,
        max_speed_mps=None,
    )
    velocity, rms = estimate_velocity_from_tdcp_with_metrics(
        receiver_position,
        prev_measurements,
        measurements,
        dt=dt,
        elevation_weight=elevation_weight,
        el_sin_floor=el_sin_floor,
    )
    spp_diff_mps = None
    if velocity is not None and spp_fd_velocity is not None:
        spp_diff_mps = float(np.linalg.norm(np.asarray(velocity, dtype=np.float64) - spp_fd_velocity))
        if spp_diff_mps > float(spp_guard_mps):
            velocity = None

    if velocity is None:
        return TdcpPredictDecision(tdcp_pu_spp_diff_mps=spp_diff_mps)

    velocity_arr = np.asarray(velocity, dtype=np.float64).ravel()[:3]
    if predict_guide == "tdcp_adaptive" and float(rms) >= float(tdcp_rms_threshold):
        return TdcpPredictDecision(
            adaptive_fallback=True,
            tdcp_pu_spp_diff_mps=spp_diff_mps,
        )

    return TdcpPredictDecision(
        velocity=velocity_arr,
        used_tdcp=True,
        tdcp_rms=float(rms),
        tdcp_pu_velocity=velocity_arr,
        tdcp_pu_rms=float(rms),
        tdcp_pu_spp_diff_mps=spp_diff_mps,
    )


def apply_epoch_predict_motion(
    epoch_state: Any,
    stats: Any,
    history: Any,
    *,
    imu_filter: Any,
    options: EpochPredictMotionOptions,
    tow: float,
    tow_key: float,
    dt: float,
    receiver_position_ecef: np.ndarray,
    current_pf_position_ecef: np.ndarray,
    measurements: Iterable[Any],
    spp_lookup: dict[float, np.ndarray],
    ecef_to_lla_func: Callable[[float, float, float], tuple[float, float, float]],
) -> None:
    if not history.has_previous_motion(dt):
        return

    prev_tow = history.prev_tow
    if prev_tow is None:
        return

    receiver_position = np.asarray(receiver_position_ecef, dtype=np.float64).ravel()[:3]
    current_measurements = list(measurements)

    imu_decision = evaluate_imu_predict_velocity(
        imu_filter,
        options.predict_guide,
        prev_tow,
        tow,
        current_pf_position_ecef,
        spp_lookup,
        ecef_to_lla_func,
        dt=dt,
    )
    if imu_decision.used_imu:
        epoch_state.velocity = imu_decision.velocity
        epoch_state.imu_velocity = imu_decision.imu_velocity
        epoch_state.used_imu = True
        stats.n_imu_used += 1
        if imu_decision.stop_detected:
            stats.n_imu_stop_detected += 1
            epoch_state.imu_stop_detected = True

    tdcp_guide_decision = evaluate_tdcp_predict_guide(
        options.predict_guide,
        receiver_position,
        history.prev_measurements,
        current_measurements,
        dt,
        spp_lookup,
        prev_tow=prev_tow,
        tow_key=tow_key,
        elevation_weight=options.tdcp_elevation_weight,
        el_sin_floor=options.tdcp_el_sin_floor,
        tdcp_rms_threshold=options.tdcp_rms_threshold,
    )
    if tdcp_guide_decision.tdcp_pu_spp_diff_mps is not None:
        epoch_state.tdcp_pu_spp_diff_mps = tdcp_guide_decision.tdcp_pu_spp_diff_mps
    if tdcp_guide_decision.adaptive_fallback:
        stats.n_tdcp_fallback += 1
    if tdcp_guide_decision.used_tdcp:
        epoch_state.velocity = tdcp_guide_decision.velocity
        epoch_state.used_tdcp = True
        epoch_state.tdcp_rms = tdcp_guide_decision.tdcp_rms
        epoch_state.tdcp_pu_velocity = tdcp_guide_decision.tdcp_pu_velocity
        epoch_state.tdcp_pu_rms = tdcp_guide_decision.tdcp_pu_rms
        stats.n_tdcp_used += 1

    if options.tdcp_position_update and epoch_state.tdcp_pu_velocity is None:
        tdcp_pu_estimate = estimate_tdcp_position_update_motion(
            receiver_position,
            history.prev_measurements,
            current_measurements,
            float(dt),
            spp_lookup,
            prev_tow=prev_tow,
            tow_key=tow_key,
            elevation_weight=options.tdcp_elevation_weight,
            el_sin_floor=options.tdcp_el_sin_floor,
            rms_max_m=options.tdcp_pu_rms_max,
            spp_max_diff_mps=options.tdcp_pu_spp_max_diff_mps,
        )
        epoch_state.tdcp_pu_velocity = tdcp_pu_estimate.velocity
        epoch_state.tdcp_pu_rms = tdcp_pu_estimate.rms
        epoch_state.tdcp_pu_reason = tdcp_pu_estimate.reason
        epoch_state.tdcp_pu_spp_diff_mps = tdcp_pu_estimate.spp_diff_mps

    if options.need_fgo_tdcp_motion:
        fgo_tdcp_estimate = estimate_local_fgo_tdcp_motion(
            receiver_position,
            history.prev_measurements,
            current_measurements,
            float(dt),
            spp_lookup,
            prev_tow=prev_tow,
            tow_key=tow_key,
            elevation_weight=options.tdcp_elevation_weight,
            el_sin_floor=options.tdcp_el_sin_floor,
            rms_max_m=options.fgo_local_tdcp_rms_max_m,
            spp_max_diff_mps=options.fgo_local_tdcp_spp_max_diff_mps,
        )
        epoch_state.fgo_tdcp_motion_velocity = fgo_tdcp_estimate.velocity
        if epoch_state.fgo_tdcp_motion_velocity is not None:
            stats.n_fgo_tdcp_motion_used += 1
        else:
            stats.n_fgo_tdcp_motion_skip += 1

    if epoch_state.velocity is None:
        epoch_state.velocity = spp_finite_difference_velocity(
            spp_lookup,
            prev_tow=prev_tow,
            tow_key=tow_key,
            dt=dt,
        )


def spp_finite_difference_velocity(
    spp_lookup: dict[float, np.ndarray],
    *,
    prev_tow: float,
    tow_key: float,
    dt: float,
    max_speed_mps: float | None = 50.0,
) -> np.ndarray | None:
    if dt <= 0 or tow_key not in spp_lookup:
        return None
    prev_key = round(float(prev_tow), 1)
    if prev_key not in spp_lookup:
        return None
    velocity = (np.asarray(spp_lookup[tow_key][:3], dtype=np.float64) - np.asarray(
        spp_lookup[prev_key][:3],
        dtype=np.float64,
    )) / float(dt)
    if not np.isfinite(velocity).all():
        return None
    if max_speed_mps is not None and float(np.linalg.norm(velocity)) >= float(max_speed_mps):
        return None
    return velocity


def select_predict_sigma(
    sigma_pos: float,
    *,
    imu_stop_detected: bool,
    imu_stop_sigma_pos: float | None,
    used_tdcp: bool,
    sigma_pos_tdcp: float | None,
    sigma_pos_tdcp_tight: float | None,
    tdcp_rms: float,
    tdcp_tight_rms_max_m: float,
) -> float:
    sigma = float(sigma_pos)
    if imu_stop_detected and imu_stop_sigma_pos is not None:
        sigma = float(imu_stop_sigma_pos)
    elif used_tdcp and sigma_pos_tdcp is not None:
        sigma = float(sigma_pos_tdcp)
    if (
        used_tdcp
        and sigma_pos_tdcp_tight is not None
        and np.isfinite(tdcp_rms)
        and float(tdcp_rms) < float(tdcp_tight_rms_max_m)
    ):
        sigma = float(sigma_pos_tdcp_tight)
    return sigma
