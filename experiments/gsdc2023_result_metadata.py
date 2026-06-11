"""BridgeResult metadata and sensor-summary helpers for GSDC2023."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from experiments.gsdc2023_bridge_config import BridgeConfig, TAROZ_FGO_WEIGHT_MODE
from experiments.gsdc2023_observation_matrix import TripArrays


@dataclass(frozen=True)
class ImuResultSummary:
    prior_interval_count: int
    acc_bias_mean_norm_mps2: float
    gyro_bias_mean_norm_radps: float


def mean_finite_row_norm(values: np.ndarray | None, mask: np.ndarray | None = None) -> float:
    if values is None:
        return float("nan")
    arr = np.asarray(values, dtype=np.float64).reshape(-1, 3)
    valid = np.isfinite(arr).all(axis=1)
    if mask is not None:
        mask_arr = np.asarray(mask, dtype=bool).reshape(-1)
        n = min(valid.size, mask_arr.size)
        valid[:n] &= mask_arr[:n]
        if n < valid.size:
            valid[n:] = False
    if not valid.any():
        return float("nan")
    return float(np.mean(np.linalg.norm(arr[valid], axis=1)))


def imu_result_summary(config: BridgeConfig, batch: TripArrays) -> ImuResultSummary:
    if batch.imu_preintegration is None:
        return ImuResultSummary(
            prior_interval_count=0,
            acc_bias_mean_norm_mps2=float("nan"),
            gyro_bias_mean_norm_radps=float("nan"),
        )

    preintegration = batch.imu_preintegration
    graph_dt = np.ones_like(preintegration.delta_t_s, dtype=np.float64)
    if batch.dt is not None and batch.dt.size > 1:
        dt_src = np.asarray(batch.dt[:-1], dtype=np.float64)
        n_dt = min(graph_dt.size, dt_src.size)
        graph_dt[:n_dt] = dt_src[:n_dt]
        if n_dt < graph_dt.size:
            graph_dt[n_dt:] = 0.0

    valid_interval_mask = (
        (np.asarray(preintegration.sample_count, dtype=np.int32) > 0)
        & np.isfinite(preintegration.delta_t_s)
        & (preintegration.delta_t_s > 0.0)
        & np.isfinite(graph_dt)
        & (graph_dt > 0.0)
    )
    prior_interval_count = int(np.count_nonzero(valid_interval_mask)) if config.apply_imu_prior else 0
    return ImuResultSummary(
        prior_interval_count=prior_interval_count,
        acc_bias_mean_norm_mps2=mean_finite_row_norm(
            preintegration.acc_bias_mean_sensor,
            valid_interval_mask,
        ),
        gyro_bias_mean_norm_radps=mean_finite_row_norm(
            preintegration.gyro_bias_mean_sensor,
            valid_interval_mask,
        ),
    )


def bridge_result_metadata_kwargs(config: BridgeConfig, batch: TripArrays) -> dict[str, Any]:
    imu = imu_result_summary(config, batch)
    imu_gravity = getattr(batch.imu_preintegration, "gravity_ecef", None) if batch.imu_preintegration is not None else None
    imu_gravity_available = bool(
        imu_gravity is not None
        and np.asarray(imu_gravity, dtype=np.float64).reshape(-1, 3).size > 0
        and np.isfinite(np.asarray(imu_gravity, dtype=np.float64).reshape(-1, 3)).all(axis=1).any()
    )
    absolute_height_applied = bool(config.apply_absolute_height and batch.absolute_height_ref_count > 0)
    graph_relative_height_applied = bool(config.graph_relative_height and not absolute_height_applied)
    return {
        "factor_dt_max_s": config.factor_dt_max_s,
        "factor_dt_gap_count": batch.factor_dt_gap_count,
        "fgo_tol": config.fgo_tol,
        "stop_velocity_sigma_mps": config.stop_velocity_sigma_mps,
        "stop_position_sigma_m": config.stop_position_sigma_m,
        "stop_attitude_sigma_rad": config.stop_attitude_sigma_rad,
        "stop_velocity_huber_k": config.stop_velocity_huber_k,
        "stop_position_huber_k": config.stop_position_huber_k,
        "imu_prior_applied": bool(config.apply_imu_prior and imu.prior_interval_count > 0),
        "imu_prior_interval_count": imu.prior_interval_count,
        "imu_frame": config.imu_frame,
        "imu_sample_dt_mode": config.imu_sample_dt_mode,
        "imu_gravity_applied": bool(
            config.use_vd
            and config.apply_imu_prior
            and config.imu_attitude_state
            and imu.prior_interval_count > 0
            and imu_gravity_available
        ),
        "imu_position_sigma_m": config.imu_position_sigma_m,
        "imu_velocity_sigma_mps": config.imu_velocity_sigma_mps,
        "imu_attitude_state_applied": bool(
            config.use_vd
            and config.apply_imu_prior
            and config.imu_attitude_state
            and imu.prior_interval_count > 0
        ),
        "imu_attitude_sigma_rad": config.imu_attitude_sigma_rad,
        "imu_diagonal_covariance_applied": bool(
            config.use_vd
            and config.apply_imu_prior
            and config.imu_diagonal_covariance
            and not config.imu_preintegration_covariance
            and imu.prior_interval_count > 0
        ),
        "imu_preintegration_covariance_applied": bool(
            config.use_vd
            and config.apply_imu_prior
            and config.imu_preintegration_covariance
            and imu.prior_interval_count > 0
        ),
        "imu_preintegration_delta_t_applied": bool(
            config.use_vd
            and config.apply_imu_prior
            and imu.prior_interval_count > 0
        ),
        "imu_preintegration_bias_jacobian_applied": bool(
            config.use_vd
            and config.apply_imu_prior
            and imu.prior_interval_count > 0
        ),
        "imu_factor_use_next_bias_applied": bool(
            config.use_vd
            and config.apply_imu_prior
            and config.imu_factor_use_next_bias
            and (config.imu_accel_bias_state or config.imu_gyro_bias_state)
            and imu.prior_interval_count > 0
        ),
        "imu_delta_pv_gyro_bias_correction_applied": bool(
            config.use_vd
            and config.apply_imu_prior
            and config.imu_attitude_state
            and config.imu_gyro_bias_state
            and imu.prior_interval_count > 0
        ),
        "imu_bias_between_sample_count_scaling_applied": bool(
            config.use_vd
            and config.apply_imu_prior
            and config.imu_bias_between_sample_count_scaling
            and (config.imu_accel_bias_state or config.imu_gyro_bias_state)
            and imu.prior_interval_count > 0
        ),
        "imu_accel_bias_state_applied": bool(
            config.use_vd
            and config.apply_imu_prior
            and config.imu_accel_bias_state
            and imu.prior_interval_count > 0
        ),
        "imu_accel_bias_prior_sigma_mps2": config.imu_accel_bias_prior_sigma_mps2,
        "imu_accel_bias_between_sigma_mps2": config.imu_accel_bias_between_sigma_mps2,
        "imu_gyro_bias_state_applied": bool(
            config.use_vd
            and config.apply_imu_prior
            and config.imu_gyro_bias_state
            and imu.prior_interval_count > 0
        ),
        "imu_gyro_bias_prior_sigma_radps": config.imu_gyro_bias_prior_sigma_radps,
        "imu_gyro_bias_between_sigma_radps": config.imu_gyro_bias_between_sigma_radps,
        "taroz_imu_noise_enabled": config.taroz_imu_noise_enabled,
        "imu_acc_sigma_mps2_sqrt_hz": config.imu_acc_sigma_mps2_sqrt_hz,
        "imu_gyro_sigma_radps_sqrt_hz": config.imu_gyro_sigma_radps_sqrt_hz,
        "imu_acc_sync_coefficient": config.imu_acc_sync_coefficient,
        "imu_gyro_sync_coefficient": config.imu_gyro_sync_coefficient,
        "imu_effective_acc_sigma_mps2_sqrt_hz": (
            config.imu_acc_sync_coefficient * config.imu_acc_sigma_mps2_sqrt_hz
        ),
        "imu_effective_gyro_sigma_radps_sqrt_hz": (
            config.imu_gyro_sync_coefficient * config.imu_gyro_sigma_radps_sqrt_hz
        ),
        "imu_acc_bias_mean_norm_mps2": imu.acc_bias_mean_norm_mps2,
        "imu_gyro_bias_mean_norm_radps": imu.gyro_bias_mean_norm_radps,
        "absolute_height_applied": absolute_height_applied,
        "absolute_height_ref_count": batch.absolute_height_ref_count,
        "absolute_height_sigma_m": config.absolute_height_sigma_m,
        "absolute_height_dist_m": config.absolute_height_dist_m,
        "absolute_height_huber_k": config.absolute_height_huber_k,
        "relative_height_applied": config.apply_relative_height,
        "relative_height_huber_k": config.relative_height_huber_k,
        "position_offset_applied": config.apply_position_offset,
        "base_correction_applied": config.apply_base_correction,
        "base_correction_count": batch.base_correction_count,
        "observation_mask_applied": config.apply_observation_mask,
        "observation_mask_count": batch.observation_mask_count,
        "residual_mask_count": batch.residual_mask_count,
        "doppler_residual_mask_count": batch.doppler_residual_mask_count,
        "pseudorange_doppler_mask_count": batch.pseudorange_doppler_mask_count,
        "tdcp_consistency_mask_count": batch.tdcp_consistency_mask_count,
        "tdcp_weight_scale": config.tdcp_weight_scale,
        "tdcp_l5_weight_scale": config.tdcp_l5_weight_scale,
        "tdcp_geometry_correction_applied": config.tdcp_geometry_correction,
        "tdcp_geometry_correction_count": batch.tdcp_geometry_correction_count,
        "tdcp_scale_candidate_enabled": config.tdcp_scale_candidate_enabled,
        "tdcp_scale_candidate_weight_scale": config.tdcp_scale_candidate_weight_scale,
        "taroz_qzss_other_clock_enabled": bool(
            config.multi_gnss
            and config.weight_mode == TAROZ_FGO_WEIGHT_MODE
            and (config.fgo_weight_mode is None or config.fgo_weight_mode == TAROZ_FGO_WEIGHT_MODE)
        ),
        "fgo_weight_mode": config.fgo_weight_mode or config.weight_mode,
        "fgo_robust_kernel": config.fgo_robust_kernel,
        "fgo_huber_k_pr": config.fgo_huber_k_pr,
        "fgo_huber_k_doppler": config.fgo_huber_k_doppler,
        "fgo_huber_k_tdcp": config.fgo_huber_k_tdcp,
        "fgo_fixed_linearization": config.fgo_fixed_linearization,
        "per_type_kernel_enabled": config.per_type_kernel_enabled,
        "per_type_kernel_huber_enabled": config.per_type_kernel_huber_enabled,
        "per_type_kernel_motion_enabled": config.per_type_kernel_motion_enabled,
        "clock_drift_sigma_m": config.clock_drift_sigma_m,
        "clock_use_average_drift": config.clock_use_average_drift,
        "fgo_raw_wls_proxy_rescue_enabled": config.fgo_raw_wls_proxy_rescue_enabled,
        "fgo_raw_wls_proxy_rescue_mse_ratio_max": config.fgo_raw_wls_proxy_rescue_mse_ratio_max,
        "fgo_raw_wls_proxy_rescue_gap_step_p95_ratio_max": config.fgo_raw_wls_proxy_rescue_gap_step_p95_ratio_max,
        "fgo_raw_wls_proxy_rescue_quality_delta_max": config.fgo_raw_wls_proxy_rescue_quality_delta_max,
        "fgo_raw_wls_proxy_rescue_mse_delta_vs_baseline_max": (
            config.fgo_raw_wls_proxy_rescue_mse_delta_vs_baseline_max
        ),
        "taroz_fgo_candidate_enabled": config.taroz_fgo_candidate_enabled,
        "taroz_fgo_candidate_sources": tuple(config.taroz_fgo_candidate_sources),
        "dual_frequency": config.dual_frequency,
        "graph_relative_height": graph_relative_height_applied,
        "ct_rbpf_fgo_enabled": config.ct_rbpf_fgo_enabled,
        "ct_rbpf_motion_sigma_m": config.ct_rbpf_motion_sigma_m,
        "dd_carrier_fgo_enabled": config.dd_carrier_fgo_enabled,
        "dd_carrier_base_obs_template": config.dd_carrier_base_obs_template,
        "dd_carrier_require_base_obs_template": config.dd_carrier_require_base_obs_template,
    }


__all__ = [
    "ImuResultSummary",
    "bridge_result_metadata_kwargs",
    "imu_result_summary",
    "mean_finite_row_norm",
]
