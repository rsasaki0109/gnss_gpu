from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.audit_gsdc2023_taroz_residuals import (
    _gtsam_rpy_to_rotm,
    compare_imu_factor_mask,
    compare_imu_preintegration_diagnostics,
    compare_imu_residual_diagnostics,
    compare_factor_mask,
    compare_residual_diagnostics,
    filter_matlab_frame_to_python_window,
    python_imu_factor_mask,
    python_imu_preintegration_diagnostics,
    python_imu_residual_diagnostics,
    python_imu_residual_diagnostics_from_matlab_exports,
    python_factor_mask,
    python_residual_diagnostics,
)
from experiments.gsdc2023_imu import IMUPreintegration, rotvec_to_rotm
from experiments.gsdc2023_observation_matrix import TripArrays


def _batch() -> TripArrays:
    return TripArrays(
        times_ms=np.array([1000.0, 2000.0], dtype=np.float64),
        sat_ecef=np.array(
            [
                [[10.0, 0.0, 0.0]],
                [[11.0, 0.0, 0.0]],
            ],
            dtype=np.float64,
        ),
        pseudorange=np.array([[12.0], [13.0]], dtype=np.float64),
        weights=np.zeros((2, 1), dtype=np.float64),
        weights_fgo=np.array([[4.0], [0.0]], dtype=np.float64),
        kaggle_wls=np.zeros((2, 3), dtype=np.float64),
        truth=np.zeros((2, 3), dtype=np.float64),
        max_sats=1,
        has_truth=True,
        slot_keys=((1, 7, "GPS_L1_CA"),),
        n_sat_slots=1,
        sat_vel=np.array(
            [
                [[1.0, 0.0, 0.0]],
                [[1.0, 0.0, 0.0]],
            ],
            dtype=np.float64,
        ),
        doppler=np.array([[-2.0], [-3.0]], dtype=np.float64),
        doppler_weights=np.zeros((2, 1), dtype=np.float64),
        doppler_weights_fgo=np.array([[9.0], [0.0]], dtype=np.float64),
        tdcp_meas=np.array([[0.5]], dtype=np.float64),
        tdcp_weights=np.zeros((1, 1), dtype=np.float64),
        tdcp_weights_fgo=np.array([[16.0]], dtype=np.float64),
        build_start_epoch=10,
        build_max_epochs=2,
    )


def _imu_batch_and_state() -> tuple[TripArrays, np.ndarray]:
    n_clock = 1
    attitude_idx = 7 + n_clock
    state = np.zeros((3, 16 + n_clock), dtype=np.float64)
    state[0, :3] = np.array([10.0, -4.0, 2.0], dtype=np.float64)
    state[0, 3:6] = np.array([1.5, -0.25, 0.5], dtype=np.float64)
    state[0, attitude_idx : attitude_idx + 3] = np.array([0.15, -0.05, 0.20], dtype=np.float64)
    dt = 0.8
    gravity = np.array([0.2, -0.1, -9.7], dtype=np.float64)
    delta_p = np.array([0.35, -0.12, 3.05], dtype=np.float64)
    delta_v = np.array([0.8, -0.2, 8.0], dtype=np.float64)
    delta_angle = np.array([0.02, -0.01, 0.03], dtype=np.float64)
    injected_p_residual = np.array([0.10, -0.20, 0.30], dtype=np.float64)
    injected_v_residual = np.array([-0.30, 0.20, 0.10], dtype=np.float64)
    rot_i = rotvec_to_rotm(state[0, attitude_idx : attitude_idx + 3])
    state[1, attitude_idx : attitude_idx + 3] = np.array([0.16, -0.06, 0.19], dtype=np.float64)
    rot_j = rotvec_to_rotm(state[1, attitude_idx : attitude_idx + 3])
    predicted_p_j = state[0, :3] + state[0, 3:6] * dt + 0.5 * gravity * dt * dt + rot_i @ delta_p
    predicted_v_j = state[0, 3:6] + gravity * dt + rot_i @ delta_v
    state[1, :3] = predicted_p_j - rot_j @ injected_p_residual
    state[1, 3:6] = predicted_v_j - rot_j @ injected_v_residual
    batch = TripArrays(
        times_ms=np.array([1000.0, 1800.0, 2600.0], dtype=np.float64),
        sat_ecef=np.zeros((3, 0, 3), dtype=np.float64),
        pseudorange=np.zeros((3, 0), dtype=np.float64),
        weights=np.zeros((3, 0), dtype=np.float64),
        kaggle_wls=state[:, :3],
        truth=state[:, :3],
        max_sats=0,
        has_truth=True,
        n_clock=n_clock,
        build_start_epoch=20,
        dt=np.array([0.8, 0.0, 0.0], dtype=np.float64),
        imu_preintegration=IMUPreintegration(
            epoch_times_ms=np.array([1000.0, 1800.0, 2600.0], dtype=np.float64),
            delta_t_s=np.array([dt, dt], dtype=np.float64),
            delta_v_body=np.vstack([delta_v, np.full(3, 9.0, dtype=np.float64)]),
            delta_p_body=np.vstack([delta_p, np.full(3, 9.0, dtype=np.float64)]),
            delta_angle_rad=np.vstack([delta_angle, np.full(3, 9.0, dtype=np.float64)]),
            sample_count=np.array([12, 12], dtype=np.int32),
            gravity_ecef=np.vstack([gravity, gravity]),
        ),
    )
    return batch, state


def test_python_factor_mask_uses_fgo_weights_and_matlab_key_space() -> None:
    mask = python_factor_mask(_batch())

    assert mask[["field", "freq", "epoch_index", "utcTimeMillis", "next_epoch_index", "nextUtcTimeMillis", "sys", "svid"]].to_dict(
        orient="records"
    ) == [
        {
            "field": "P",
            "freq": "L1",
            "epoch_index": 11,
            "utcTimeMillis": 1000,
            "next_epoch_index": 0,
            "nextUtcTimeMillis": 0,
            "sys": 1,
            "svid": 7,
        },
        {
            "field": "D",
            "freq": "L1",
            "epoch_index": 11,
            "utcTimeMillis": 1000,
            "next_epoch_index": 0,
            "nextUtcTimeMillis": 0,
            "sys": 1,
            "svid": 7,
        },
        {
            "field": "L",
            "freq": "L1",
            "epoch_index": 11,
            "utcTimeMillis": 1000,
            "next_epoch_index": 12,
            "nextUtcTimeMillis": 2000,
            "sys": 1,
            "svid": 7,
        },
    ]


def test_compare_factor_mask_counts_matches_and_misses() -> None:
    python_mask = python_factor_mask(_batch())
    matlab_mask = pd.concat(
        [
            python_mask.iloc[[0, 2]],
            pd.DataFrame(
                [
                    {
                        "field": "D",
                        "freq": "L1",
                        "epoch_index": 12,
                        "utcTimeMillis": 2000,
                        "next_epoch_index": 0,
                        "nextUtcTimeMillis": 0,
                        "sys": 1,
                        "svid": 7,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    comparison = compare_factor_mask(python_mask, matlab_mask)

    d_row = comparison[comparison["field"].eq("D")].iloc[0]
    assert int(d_row["python_count"]) == 1
    assert int(d_row["matlab_count"]) == 1
    assert int(d_row["matched_count"]) == 0
    assert int(d_row["only_python_count"]) == 1
    assert int(d_row["only_matlab_count"]) == 1


def test_python_residual_diagnostics_emits_pseudorange_and_factor_flags() -> None:
    residuals = python_residual_diagnostics(_batch())

    row = residuals[residuals["utcTimeMillis"].eq(1000)].iloc[0]
    assert row["freq"] == "L1"
    assert int(row["epoch_index"]) == 11
    assert row["p_corrected_m"] == pytest.approx(12.0)
    assert row["p_range_m"] == pytest.approx(10.0)
    assert row["p_pre_respc_m"] == pytest.approx(2.0)
    assert bool(row["p_factor_finite"])
    assert bool(row["d_factor_finite"])
    assert bool(row["l_factor_finite"])


def test_compare_residual_diagnostics_summarizes_numeric_deltas() -> None:
    python_df = python_residual_diagnostics(_batch())
    matlab_df = python_df.copy()
    matlab_df["p_corrected_m"] += 0.5
    matlab_df["p_range_m"] -= 0.25

    stats = {stat.column: stat for stat in compare_residual_diagnostics(python_df, matlab_df)}

    assert stats["p_corrected_m"].count == len(python_df)
    assert stats["p_corrected_m"].mean_abs == pytest.approx(0.5)
    assert stats["p_range_m"].mean_abs == pytest.approx(0.25)


def test_filter_matlab_frame_to_python_window_drops_outside_and_open_l_pairs() -> None:
    frame = pd.DataFrame(
        [
            {"field": "P", "utcTimeMillis": 1000, "nextUtcTimeMillis": 0},
            {"field": "P", "utcTimeMillis": 3000, "nextUtcTimeMillis": 0},
            {"field": "L", "utcTimeMillis": 1000, "nextUtcTimeMillis": 2000},
            {"field": "L", "utcTimeMillis": 2000, "nextUtcTimeMillis": 3000},
        ]
    )

    filtered = filter_matlab_frame_to_python_window(frame, np.array([1000.0, 2000.0], dtype=np.float64))

    assert filtered[["field", "utcTimeMillis", "nextUtcTimeMillis"]].to_dict(orient="records") == [
        {"field": "P", "utcTimeMillis": 1000, "nextUtcTimeMillis": 0},
        {"field": "L", "utcTimeMillis": 1000, "nextUtcTimeMillis": 2000},
    ]


def test_python_imu_factor_mask_uses_axis_and_next_epoch_key_space() -> None:
    batch, _state = _imu_batch_and_state()

    mask = python_imu_factor_mask(batch)

    assert mask.shape[0] == 9
    assert set(mask["field"]) == {"IMU_P", "IMU_V", "IMU_R"}
    assert set(mask["axis"]) == {0, 1, 2}
    assert mask[["freq", "epoch_index", "utcTimeMillis", "next_epoch_index", "nextUtcTimeMillis", "sys", "svid"]].drop_duplicates().to_dict(
        orient="records"
    ) == [
        {
            "freq": "IMU",
            "epoch_index": 21,
            "utcTimeMillis": 1000,
            "next_epoch_index": 22,
            "nextUtcTimeMillis": 1800,
            "sys": 0,
            "svid": 0,
        }
    ]
    assert mask["sample_count"].eq(12).all()
    assert mask["graph_dt_s"].eq(0.8).all()
    assert mask["preintegrated_dt_s"].eq(0.8).all()


def test_python_imu_residual_diagnostics_emits_taroz_reference_residuals() -> None:
    batch, state = _imu_batch_and_state()

    residuals = python_imu_residual_diagnostics(batch, state)

    assert residuals.shape[0] == 9
    np.testing.assert_allclose(residuals["imu_native_minus_taroz_reference"].to_numpy(dtype=np.float64), 0.0, atol=1e-12)
    by_factor = {
        factor: group.sort_values("axis")["imu_native_residual"].to_numpy(dtype=np.float64)
        for factor, group in residuals.groupby("field")
    }
    np.testing.assert_allclose(by_factor["IMU_P"], [0.10, -0.20, 0.30], atol=1e-12)
    np.testing.assert_allclose(by_factor["IMU_V"], [-0.30, 0.20, 0.10], atol=1e-12)
    assert residuals["next_epoch_index"].eq(22).all()
    assert residuals["sample_count"].eq(12).all()
    assert residuals["graph_dt_s"].eq(0.8).all()
    assert residuals["preintegrated_dt_s"].eq(0.8).all()


def test_compare_imu_factor_mask_and_residuals_summarize_deltas() -> None:
    batch, state = _imu_batch_and_state()
    mask = python_imu_factor_mask(batch)
    residuals = python_imu_residual_diagnostics(batch, state)

    matlab_mask = mask.iloc[:-1].copy()
    mask_comparison = compare_imu_factor_mask(mask, matlab_mask)
    imu_r_axis_2 = mask_comparison[(mask_comparison["field"].eq("IMU_R")) & (mask_comparison["axis"].eq(2))].iloc[0]
    assert int(imu_r_axis_2["only_python_count"]) == 1

    matlab_residuals = residuals.copy()
    matlab_residuals["imu_taroz_reference_residual"] += 0.5
    matlab_residuals["preintegrated_dt_s"] += 0.25
    stats = {stat.column: stat for stat in compare_imu_residual_diagnostics(residuals, matlab_residuals)}
    assert stats["imu_taroz_reference_residual"].count == 9
    assert stats["imu_taroz_reference_residual"].mean_abs == pytest.approx(0.5)
    assert stats["preintegrated_dt_s"].count == 9
    assert stats["preintegrated_dt_s"].mean_abs == pytest.approx(0.25)

    matlab_export = residuals[
        ["field", "freq", "utcTimeMillis", "nextUtcTimeMillis", "sys", "svid", "axis"]
    ].copy()
    matlab_export["residual"] = residuals["imu_taroz_reference_residual"] - 0.25
    stats = {stat.column: stat for stat in compare_imu_residual_diagnostics(residuals, matlab_export)}
    assert stats["imu_taroz_reference_residual_vs_residual"].count == 9
    assert stats["imu_taroz_reference_residual_vs_residual"].mean_abs == pytest.approx(0.25)


def test_python_imu_preintegration_diagnostics_compares_raw_and_corrected_deltas() -> None:
    dt = 0.8
    delta_r = np.array([0.01, -0.02, 0.03], dtype=np.float64)
    delta_p = np.array([0.35, -0.12, 3.05], dtype=np.float64)
    delta_v = np.array([0.8, -0.2, 8.0], dtype=np.float64)
    acc_bias = np.array([0.1, -0.2, 0.3], dtype=np.float64)
    gyro_bias = np.array([-0.01, 0.02, -0.03], dtype=np.float64)
    p_acc_jac = np.eye(3, dtype=np.float64) * 2.0
    p_gyro_jac = np.eye(3, dtype=np.float64) * 0.5
    v_acc_jac = np.eye(3, dtype=np.float64) * 3.0
    v_gyro_jac = np.eye(3, dtype=np.float64) * 0.25
    r_gyro_jac = np.eye(3, dtype=np.float64) * 0.75
    batch = TripArrays(
        times_ms=np.array([1000.0, 1800.0], dtype=np.float64),
        sat_ecef=np.zeros((2, 0, 3), dtype=np.float64),
        pseudorange=np.zeros((2, 0), dtype=np.float64),
        weights=np.zeros((2, 0), dtype=np.float64),
        kaggle_wls=np.zeros((2, 3), dtype=np.float64),
        truth=np.zeros((2, 3), dtype=np.float64),
        max_sats=0,
        has_truth=True,
        build_start_epoch=20,
        dt=np.array([dt, 0.0], dtype=np.float64),
        imu_preintegration=IMUPreintegration(
            epoch_times_ms=np.array([1000.0, 1800.0], dtype=np.float64),
            delta_t_s=np.array([dt], dtype=np.float64),
            delta_v_body=delta_v.reshape(1, 3),
            delta_p_body=delta_p.reshape(1, 3),
            delta_angle_rad=delta_r.reshape(1, 3),
            sample_count=np.array([12], dtype=np.int32),
            delta_p_bias_accel_jac=p_acc_jac.reshape(1, 3, 3),
            delta_v_bias_accel_jac=v_acc_jac.reshape(1, 3, 3),
            delta_p_bias_gyro_jac=p_gyro_jac.reshape(1, 3, 3),
            delta_v_bias_gyro_jac=v_gyro_jac.reshape(1, 3, 3),
            delta_angle_bias_gyro_jac=r_gyro_jac.reshape(1, 3, 3),
            gravity_ecef=np.array([[0.0, 0.0, -9.80665]], dtype=np.float64),
        ),
    )
    matlab_state = pd.DataFrame(
        [
            {
                "epoch_index": 22,
                "utcTimeMillis": 1800,
                "bias_acc_x": acc_bias[0],
                "bias_acc_y": acc_bias[1],
                "bias_acc_z": acc_bias[2],
                "bias_gyro_x": gyro_bias[0],
                "bias_gyro_y": gyro_bias[1],
                "bias_gyro_z": gyro_bias[2],
            }
        ]
    )

    diagnostics = python_imu_preintegration_diagnostics(batch, matlab_state)

    assert diagnostics.shape[0] == 1
    row = diagnostics.iloc[0]
    np.testing.assert_allclose(row[["delta_r_x", "delta_r_y", "delta_r_z"]].to_numpy(dtype=np.float64), delta_r)
    np.testing.assert_allclose(row[["delta_p_x", "delta_p_y", "delta_p_z"]].to_numpy(dtype=np.float64), delta_p)
    np.testing.assert_allclose(row[["delta_v_x", "delta_v_y", "delta_v_z"]].to_numpy(dtype=np.float64), delta_v)
    np.testing.assert_allclose(
        row[["corrected_delta_p_x", "corrected_delta_p_y", "corrected_delta_p_z"]].to_numpy(dtype=np.float64),
        delta_p - p_acc_jac @ acc_bias - p_gyro_jac @ gyro_bias,
    )
    np.testing.assert_allclose(
        row[["corrected_delta_v_x", "corrected_delta_v_y", "corrected_delta_v_z"]].to_numpy(dtype=np.float64),
        delta_v - v_acc_jac @ acc_bias - v_gyro_jac @ gyro_bias,
    )
    np.testing.assert_allclose(
        row[["corrected_delta_r_x", "corrected_delta_r_y", "corrected_delta_r_z"]].to_numpy(dtype=np.float64),
        delta_r - r_gyro_jac @ gyro_bias,
    )

    matlab_export = diagnostics.copy()
    matlab_export["delta_p_x"] += 0.25
    stats = {stat.column: stat for stat in compare_imu_preintegration_diagnostics(diagnostics, matlab_export)}
    assert stats["delta_p_x"].count == 1
    assert stats["delta_p_x"].mean_abs == pytest.approx(0.25)


def test_python_imu_residual_diagnostics_from_matlab_exports_uses_same_state() -> None:
    epoch = 21
    next_epoch = 22
    dt = 0.8
    gravity = np.array([0.0, 0.0, -9.80665], dtype=np.float64)
    position_i = np.array([10.0, -4.0, 2.0], dtype=np.float64)
    velocity_i = np.array([1.5, -0.25, 0.5], dtype=np.float64)
    rpy_i = np.array([0.10, -0.20, 0.30], dtype=np.float64)
    delta_p = np.array([0.35, -0.12, 3.05], dtype=np.float64)
    delta_v = np.array([0.8, -0.2, 8.0], dtype=np.float64)
    delta_r = np.zeros(3, dtype=np.float64)
    injected_p_residual = np.array([0.10, -0.20, 0.30], dtype=np.float64)
    injected_v_residual = np.array([-0.30, 0.20, 0.10], dtype=np.float64)
    rot_i = _gtsam_rpy_to_rotm(rpy_i)[0]
    position_j = position_i + velocity_i * dt + 0.5 * gravity * dt * dt + rot_i @ (
        delta_p - injected_p_residual
    )
    velocity_j = velocity_i + gravity * dt + rot_i @ (delta_v - injected_v_residual)

    state = pd.DataFrame(
        [
            {
                "epoch_index": epoch,
                "utcTimeMillis": 1000,
                "position_x": position_i[0],
                "position_y": position_i[1],
                "position_z": position_i[2],
                "roll": rpy_i[0],
                "pitch": rpy_i[1],
                "yaw": rpy_i[2],
                "velocity_x": velocity_i[0],
                "velocity_y": velocity_i[1],
                "velocity_z": velocity_i[2],
            },
            {
                "epoch_index": next_epoch,
                "utcTimeMillis": 1800,
                "position_x": position_j[0],
                "position_y": position_j[1],
                "position_z": position_j[2],
                "roll": rpy_i[0],
                "pitch": rpy_i[1],
                "yaw": rpy_i[2],
                "velocity_x": velocity_j[0],
                "velocity_y": velocity_j[1],
                "velocity_z": velocity_j[2],
            },
        ]
    )
    preintegration = pd.DataFrame(
        [
            {
                "epoch_index": epoch,
                "utcTimeMillis": 1000,
                "next_epoch_index": next_epoch,
                "nextUtcTimeMillis": 1800,
                "sample_count": 12,
                "graph_dt_s": dt,
                "preintegrated_dt_s": dt,
                "corrected_delta_r_x": delta_r[0],
                "corrected_delta_r_y": delta_r[1],
                "corrected_delta_r_z": delta_r[2],
                "corrected_delta_p_x": delta_p[0],
                "corrected_delta_p_y": delta_p[1],
                "corrected_delta_p_z": delta_p[2],
                "corrected_delta_v_x": delta_v[0],
                "corrected_delta_v_y": delta_v[1],
                "corrected_delta_v_z": delta_v[2],
                "gravity_x": gravity[0],
                "gravity_y": gravity[1],
                "gravity_z": gravity[2],
            }
        ]
    )

    residuals = python_imu_residual_diagnostics_from_matlab_exports(state, preintegration)

    assert residuals.shape[0] == 9
    by_factor = {
        factor: group.sort_values("axis")["imu_same_state_residual"].to_numpy(dtype=np.float64)
        for factor, group in residuals.groupby("field")
    }
    np.testing.assert_allclose(by_factor["IMU_R"], 0.0, atol=1e-12)
    np.testing.assert_allclose(by_factor["IMU_P"], injected_p_residual, atol=1e-12)
    np.testing.assert_allclose(by_factor["IMU_V"], injected_v_residual, atol=1e-12)

    matlab_export = residuals[
        ["field", "freq", "utcTimeMillis", "nextUtcTimeMillis", "sys", "svid", "axis"]
    ].copy()
    matlab_export["residual"] = residuals["imu_same_state_residual"] - 0.125
    stats = {stat.column: stat for stat in compare_imu_residual_diagnostics(residuals, matlab_export)}
    assert stats["imu_same_state_residual_vs_residual"].count == 9
    assert stats["imu_same_state_residual_vs_residual"].mean_abs == pytest.approx(0.125)
