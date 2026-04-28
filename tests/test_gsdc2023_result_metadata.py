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
        stop_velocity_sigma_mps=0.2,
        stop_position_sigma_m=0.3,
        apply_imu_prior=True,
        imu_accel_bias_state=True,
        imu_frame="ecef",
        imu_position_sigma_m=11.0,
        imu_velocity_sigma_mps=1.2,
        apply_absolute_height=True,
        apply_relative_height=True,
        apply_position_offset=True,
        apply_base_correction=True,
        apply_observation_mask=True,
        tdcp_weight_scale=0.5,
        tdcp_geometry_correction=True,
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
    assert kwargs["stop_velocity_sigma_mps"] == 0.2
    assert kwargs["stop_position_sigma_m"] == 0.3
    assert kwargs["imu_prior_applied"] is True
    assert kwargs["imu_prior_interval_count"] == 1
    assert kwargs["imu_frame"] == "ecef"
    assert kwargs["imu_position_sigma_m"] == 11.0
    assert kwargs["imu_velocity_sigma_mps"] == 1.2
    assert kwargs["imu_accel_bias_state_applied"] is True
    assert kwargs["imu_acc_bias_mean_norm_mps2"] == 5.0
    assert kwargs["imu_gyro_bias_mean_norm_radps"] == 2.0
    assert kwargs["absolute_height_applied"] is True
    assert kwargs["absolute_height_ref_count"] == 3
    assert kwargs["relative_height_applied"] is True
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
    assert kwargs["dual_frequency"] is True
    assert kwargs["graph_relative_height"] is True


def test_bridge_result_metadata_kwargs_has_no_imu_applied_without_imu() -> None:
    kwargs = bridge_result_metadata_kwargs(
        BridgeConfig(apply_imu_prior=True, imu_accel_bias_state=True),
        _batch(imu_preintegration=None),
    )

    assert kwargs["imu_prior_applied"] is False
    assert kwargs["imu_prior_interval_count"] == 0
    assert kwargs["imu_accel_bias_state_applied"] is False
    assert np.isnan(kwargs["imu_acc_bias_mean_norm_mps2"])
