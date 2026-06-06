from __future__ import annotations

import numpy as np

from experiments.gsdc2023_bridge_config import BridgeConfig
from experiments.gsdc2023_imu import IMUPreintegration
from experiments.gsdc2023_observation_matrix import TripArrays
from experiments.gsdc2023_result_metadata import (
    bridge_result_metadata_kwargs,
    imu_result_summary,
    mean_finite_row_norm,
)


def _batch(*, imu_preintegration: IMUPreintegration | None = None, dt: np.ndarray | None = None) -> TripArrays:
    n_epoch = 4
    n_sat = 2
    return TripArrays(
        times_ms=np.arange(n_epoch, dtype=np.float64) * 1000.0,
        sat_ecef=np.zeros((n_epoch, n_sat, 3), dtype=np.float64),
        pseudorange=np.zeros((n_epoch, n_sat), dtype=np.float64),
        weights=np.ones((n_epoch, n_sat), dtype=np.float64),
        kaggle_wls=np.zeros((n_epoch, 3), dtype=np.float64),
        truth=np.zeros((n_epoch, 3), dtype=np.float64),
        max_sats=n_sat,
        has_truth=False,
        dt=dt,
        imu_preintegration=imu_preintegration,
        factor_dt_gap_count=2,
        absolute_height_ref_count=3,
        base_correction_count=4,
        observation_mask_count=5,
        residual_mask_count=6,
        doppler_residual_mask_count=7,
        pseudorange_doppler_mask_count=8,
        tdcp_consistency_mask_count=9,
        tdcp_geometry_correction_count=10,
        dual_frequency=True,
    )


def _preintegration() -> IMUPreintegration:
    return IMUPreintegration(
        epoch_times_ms=np.arange(4, dtype=np.float64) * 1000.0,
        delta_t_s=np.array([1.0, 1.5, 1.0], dtype=np.float64),
        delta_v_body=np.zeros((3, 3), dtype=np.float64),
        delta_p_body=np.zeros((3, 3), dtype=np.float64),
        delta_angle_rad=np.zeros((3, 3), dtype=np.float64),
        sample_count=np.array([5, 7, 0], dtype=np.int32),
        acc_bias_mean_sensor=np.array(
            [
                [3.0, 4.0, 0.0],
                [0.0, 6.0, 8.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        ),
        gyro_bias_mean_sensor=np.array(
            [
                [0.0, 0.0, 2.0],
                [0.0, 3.0, 4.0],
                [9.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        ),
        gravity_ecef=np.tile(np.array([[0.0, 0.0, -9.81]], dtype=np.float64), (3, 1)),
    )


def test_mean_finite_row_norm_applies_mask_and_ignores_nonfinite_rows() -> None:
    values = np.array(
        [
            [3.0, 4.0, 0.0],
            [0.0, np.nan, 0.0],
            [0.0, 0.0, 12.0],
        ],
        dtype=np.float64,
    )

    assert mean_finite_row_norm(values, np.array([True, True, False])) == 5.0
    assert np.isnan(mean_finite_row_norm(values, np.array([False, True, False])))
    assert np.isnan(mean_finite_row_norm(None))


def test_imu_result_summary_counts_valid_intervals_after_graph_dt_mask() -> None:
    cfg = BridgeConfig(apply_imu_prior=True)
    batch = _batch(
        imu_preintegration=_preintegration(),
        dt=np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float64),
    )

    summary = imu_result_summary(cfg, batch)

    assert summary.prior_interval_count == 1
    assert summary.acc_bias_mean_norm_mps2 == 5.0
    assert summary.gyro_bias_mean_norm_radps == 2.0


def test_imu_result_summary_keeps_bias_summary_when_prior_disabled() -> None:
    batch = _batch(
        imu_preintegration=_preintegration(),
        dt=np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float64),
    )

    summary = imu_result_summary(BridgeConfig(apply_imu_prior=False), batch)

    assert summary.prior_interval_count == 0
    assert summary.acc_bias_mean_norm_mps2 == 5.0


def test_bridge_result_metadata_kwargs_maps_config_and_batch_counts() -> None:
    cfg = BridgeConfig(
        factor_dt_max_s=1.25,
        fgo_tol=2e-8,
        stop_velocity_sigma_mps=0.2,
        stop_position_sigma_m=0.3,
        stop_attitude_sigma_rad=0.03,
        stop_velocity_huber_k=0.4,
        stop_position_huber_k=0.5,
        apply_imu_prior=True,
        imu_accel_bias_state=True,
        imu_gyro_bias_state=True,
        imu_gyro_bias_prior_sigma_radps=0.003,
        imu_frame="taroz_body",
        imu_sample_dt_mode="taroz",
        imu_position_sigma_m=11.0,
        imu_velocity_sigma_mps=1.2,
        imu_attitude_state=True,
        imu_attitude_sigma_rad=0.002,
        imu_diagonal_covariance=True,
        imu_factor_use_next_bias=True,
        imu_bias_between_sample_count_scaling=True,
        taroz_imu_noise_enabled=True,
        imu_acc_sigma_mps2_sqrt_hz=0.05,
        imu_gyro_sigma_radps_sqrt_hz=0.001,
        imu_acc_sync_coefficient=0.5,
        imu_gyro_sync_coefficient=0.5,
        imu_gyro_bias_between_sigma_radps=0.0000005,
        apply_absolute_height=True,
        absolute_height_huber_k=0.6,
        apply_relative_height=True,
        relative_height_huber_k=0.7,
        apply_position_offset=True,
        apply_base_correction=True,
        apply_observation_mask=True,
        tdcp_weight_scale=0.5,
        tdcp_geometry_correction=True,
        weight_mode="taroz_sn",
        fgo_weight_mode="taroz_sn",
        fgo_huber_k_pr=0.1,
        fgo_huber_k_doppler=0.4,
        fgo_huber_k_tdcp=0.2,
        fgo_fixed_linearization=True,
        per_type_kernel_enabled=True,
        per_type_kernel_motion_enabled=True,
        clock_drift_sigma_m=0.1,
        clock_use_average_drift=True,
        fgo_raw_wls_proxy_rescue_enabled=True,
        fgo_raw_wls_proxy_rescue_mse_ratio_max=1.12,
        fgo_raw_wls_proxy_rescue_gap_step_p95_ratio_max=1.25,
        fgo_raw_wls_proxy_rescue_quality_delta_max=-0.4,
        fgo_raw_wls_proxy_rescue_mse_delta_vs_baseline_max=-1.0,
        dual_frequency=True,
        graph_relative_height=True,
    )
    batch = _batch(
        imu_preintegration=_preintegration(),
        dt=np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float64),
    )

    kwargs = bridge_result_metadata_kwargs(cfg, batch)

    assert kwargs["factor_dt_max_s"] == 1.25
    assert kwargs["factor_dt_gap_count"] == 2
    assert kwargs["fgo_tol"] == 2e-8
    assert kwargs["stop_velocity_sigma_mps"] == 0.2
    assert kwargs["stop_position_sigma_m"] == 0.3
    assert kwargs["stop_attitude_sigma_rad"] == 0.03
    assert kwargs["stop_velocity_huber_k"] == 0.4
    assert kwargs["stop_position_huber_k"] == 0.5
    assert kwargs["imu_prior_applied"] is True
    assert kwargs["imu_prior_interval_count"] == 1
    assert kwargs["imu_frame"] == "taroz_body"
    assert kwargs["imu_sample_dt_mode"] == "taroz"
    assert kwargs["imu_gravity_applied"] is True
    assert kwargs["imu_position_sigma_m"] == 11.0
    assert kwargs["imu_velocity_sigma_mps"] == 1.2
    assert kwargs["imu_attitude_state_applied"] is True
    assert kwargs["imu_attitude_sigma_rad"] == 0.002
    assert kwargs["imu_diagonal_covariance_applied"] is True
    assert kwargs["imu_preintegration_covariance_applied"] is False
    assert kwargs["imu_preintegration_delta_t_applied"] is True
    assert kwargs["imu_preintegration_bias_jacobian_applied"] is True
    assert kwargs["imu_factor_use_next_bias_applied"] is True
    assert kwargs["imu_delta_pv_gyro_bias_correction_applied"] is True
    assert kwargs["imu_bias_between_sample_count_scaling_applied"] is True
    assert kwargs["imu_accel_bias_state_applied"] is True
    assert kwargs["imu_gyro_bias_state_applied"] is True
    assert kwargs["imu_gyro_bias_prior_sigma_radps"] == 0.003
    assert kwargs["taroz_imu_noise_enabled"] is True
    assert kwargs["imu_acc_sigma_mps2_sqrt_hz"] == 0.05
    assert kwargs["imu_gyro_sigma_radps_sqrt_hz"] == 0.001
    assert kwargs["imu_acc_sync_coefficient"] == 0.5
    assert kwargs["imu_gyro_sync_coefficient"] == 0.5
    assert kwargs["imu_effective_acc_sigma_mps2_sqrt_hz"] == 0.025
    assert kwargs["imu_effective_gyro_sigma_radps_sqrt_hz"] == 0.0005
    assert kwargs["imu_gyro_bias_between_sigma_radps"] == 0.0000005
    assert kwargs["imu_acc_bias_mean_norm_mps2"] == 5.0
    assert kwargs["imu_gyro_bias_mean_norm_radps"] == 2.0
    assert kwargs["absolute_height_applied"] is True
    assert kwargs["absolute_height_ref_count"] == 3
    assert kwargs["absolute_height_huber_k"] == 0.6
    assert kwargs["relative_height_applied"] is True
    assert kwargs["relative_height_huber_k"] == 0.7
    assert kwargs["position_offset_applied"] is True
    assert kwargs["base_correction_applied"] is True
    assert kwargs["base_correction_count"] == 4
    assert kwargs["observation_mask_applied"] is True
    assert kwargs["observation_mask_count"] == 5
    assert kwargs["residual_mask_count"] == 6
    assert kwargs["doppler_residual_mask_count"] == 7
    assert kwargs["pseudorange_doppler_mask_count"] == 8
    assert kwargs["tdcp_consistency_mask_count"] == 9
    assert kwargs["tdcp_weight_scale"] == 0.5
    assert kwargs["tdcp_geometry_correction_applied"] is True
    assert kwargs["tdcp_geometry_correction_count"] == 10
    assert kwargs["taroz_qzss_other_clock_enabled"] is True
    assert kwargs["fgo_weight_mode"] == "taroz_sn"
    assert kwargs["fgo_robust_kernel"] == "huber"
    assert kwargs["fgo_huber_k_pr"] == 0.1
    assert kwargs["fgo_huber_k_doppler"] == 0.4
    assert kwargs["fgo_huber_k_tdcp"] == 0.2
    assert kwargs["fgo_fixed_linearization"] is True
    assert kwargs["per_type_kernel_enabled"] is True
    assert kwargs["per_type_kernel_huber_enabled"] is True
    assert kwargs["per_type_kernel_motion_enabled"] is True
    assert kwargs["clock_drift_sigma_m"] == 0.1
    assert kwargs["clock_use_average_drift"] is True
    assert kwargs["fgo_raw_wls_proxy_rescue_enabled"] is True
    assert kwargs["fgo_raw_wls_proxy_rescue_mse_ratio_max"] == 1.12
    assert kwargs["fgo_raw_wls_proxy_rescue_gap_step_p95_ratio_max"] == 1.25
    assert kwargs["fgo_raw_wls_proxy_rescue_quality_delta_max"] == -0.4
    assert kwargs["fgo_raw_wls_proxy_rescue_mse_delta_vs_baseline_max"] == -1.0
    assert kwargs["dual_frequency"] is True
    assert kwargs["graph_relative_height"] is False


def test_bridge_result_metadata_kwargs_has_no_imu_applied_without_imu() -> None:
    kwargs = bridge_result_metadata_kwargs(
        BridgeConfig(apply_imu_prior=True, imu_accel_bias_state=True),
        _batch(imu_preintegration=None),
    )

    assert kwargs["imu_prior_applied"] is False
    assert kwargs["imu_prior_interval_count"] == 0
    assert kwargs["imu_gravity_applied"] is False
    assert kwargs["imu_accel_bias_state_applied"] is False
    assert np.isnan(kwargs["imu_acc_bias_mean_norm_mps2"])
