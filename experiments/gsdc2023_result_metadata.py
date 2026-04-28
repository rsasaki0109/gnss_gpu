"""BridgeResult metadata and sensor-summary helpers for GSDC2023."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from experiments.gsdc2023_bridge_config import BridgeConfig
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
    return {
        "factor_dt_max_s": config.factor_dt_max_s,
        "factor_dt_gap_count": batch.factor_dt_gap_count,
        "stop_velocity_sigma_mps": config.stop_velocity_sigma_mps,
        "stop_position_sigma_m": config.stop_position_sigma_m,
        "imu_prior_applied": bool(config.apply_imu_prior and imu.prior_interval_count > 0),
        "imu_prior_interval_count": imu.prior_interval_count,
        "imu_frame": config.imu_frame,
        "imu_position_sigma_m": config.imu_position_sigma_m,
        "imu_velocity_sigma_mps": config.imu_velocity_sigma_mps,
        "imu_accel_bias_state_applied": bool(
            config.use_vd
            and config.apply_imu_prior
            and config.imu_accel_bias_state
            and imu.prior_interval_count > 0
        ),
        "imu_accel_bias_prior_sigma_mps2": config.imu_accel_bias_prior_sigma_mps2,
        "imu_accel_bias_between_sigma_mps2": config.imu_accel_bias_between_sigma_mps2,
        "imu_acc_bias_mean_norm_mps2": imu.acc_bias_mean_norm_mps2,
        "imu_gyro_bias_mean_norm_radps": imu.gyro_bias_mean_norm_radps,
        "absolute_height_applied": bool(config.apply_absolute_height and batch.absolute_height_ref_count > 0),
        "absolute_height_ref_count": batch.absolute_height_ref_count,
        "absolute_height_sigma_m": config.absolute_height_sigma_m,
        "absolute_height_dist_m": config.absolute_height_dist_m,
        "relative_height_applied": config.apply_relative_height,
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
        "tdcp_geometry_correction_applied": config.tdcp_geometry_correction,
        "tdcp_geometry_correction_count": batch.tdcp_geometry_correction_count,
        "dual_frequency": config.dual_frequency,
        "graph_relative_height": config.graph_relative_height,
    }


__all__ = [
    "ImuResultSummary",
    "bridge_result_metadata_kwargs",
    "imu_result_summary",
    "mean_finite_row_norm",
]
