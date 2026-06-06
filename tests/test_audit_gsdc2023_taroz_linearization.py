from __future__ import annotations

import numpy as np
import pytest

from experiments.audit_gsdc2023_taroz_linearization import (
    imu_body_gravity_residual_frame,
    load_taroz_gnss_state_csv_for_batch,
    native_doppler_geom_and_jacobian,
    native_range_and_jacobian,
    summarize_linearization_frame,
    summarize_taroz_gtsam_graph_cost,
    taroz_gnss_initial_state_for_batch,
    taroz_gtsam_gnss_graph_cost_frame,
    taroz_gtsam_gnss_factor_residual_frame,
    taroz_imu_body_gravity_residuals,
    taroz_linearization_residual_frame,
)
from experiments.gsdc2023_imu import IMUPreintegration, rotvec_to_rotm
from experiments.gsdc2023_observation_matrix import TripArrays


def _state() -> np.ndarray:
    state = np.zeros((2, 8), dtype=np.float64)
    state[0, :3] = np.array([0.0, 0.0, 0.0])
    state[1, :3] = np.array([10.0, 0.0, 0.0])
    state[:, 3:6] = np.array([1.0, 0.2, -0.1])
    state[:, 6] = 2.0
    state[:, 7] = 0.3
    return state


def _batch() -> TripArrays:
    origin = _state()
    sat_ecef = np.array(
        [
            [[20_200_000.0, 1_000.0, 0.0], [21_200_000.0, 2_000.0, 500.0]],
            [[20_200_100.0, 1_000.0, 0.0], [21_200_050.0, 2_100.0, 500.0]],
        ],
        dtype=np.float64,
    )
    sat_vel = np.array(
        [
            [[50.0, 1.0, 0.0], [45.0, 1.5, 0.0]],
            [[50.0, 1.0, 0.0], [45.0, 1.5, 0.0]],
        ],
        dtype=np.float64,
    )
    ranges, _ = native_range_and_jacobian(sat_ecef, origin[:, None, :3])
    geom, _ = native_doppler_geom_and_jacobian(sat_ecef, origin[:, None, :3], sat_vel, origin[:, None, 3:6])
    pseudorange = ranges + origin[:, None, 6] + 3.0
    doppler = geom + origin[:, None, 7] + 0.4
    return TripArrays(
        times_ms=np.array([1000.0, 2000.0], dtype=np.float64),
        sat_ecef=sat_ecef,
        pseudorange=pseudorange,
        weights=np.zeros((2, 2), dtype=np.float64),
        weights_fgo=np.ones((2, 2), dtype=np.float64),
        kaggle_wls=origin[:, :3],
        truth=origin[:, :3],
        max_sats=2,
        has_truth=True,
        slot_keys=((1, 1, "GPS_L1_CA"), (1, 2, "GPS_L1_CA")),
        n_sat_slots=2,
        n_clock=1,
        sat_vel=sat_vel,
        doppler=doppler,
        doppler_weights=np.zeros((2, 2), dtype=np.float64),
        doppler_weights_fgo=np.ones((2, 2), dtype=np.float64),
        tdcp_meas=np.zeros((1, 2), dtype=np.float64),
        tdcp_weights=np.zeros((1, 2), dtype=np.float64),
        tdcp_weights_fgo=np.array([[1.0, 0.0]], dtype=np.float64),
        dt=np.array([1.0, 0.0], dtype=np.float64),
    )


def test_seed_pr_and_doppler_match_taroz_linearization() -> None:
    batch = _batch()
    origin = _state()

    frame = taroz_linearization_residual_frame(batch, origin, origin, label="seed")

    by_factor = frame.groupby("factor")["native_minus_taroz_linear"].apply(lambda col: col.to_numpy())
    np.testing.assert_allclose(by_factor["P"], np.zeros(4), atol=1e-9)
    np.testing.assert_allclose(by_factor["D"], np.zeros(4), atol=1e-9)


def test_perturbed_pseudorange_exposes_native_nonlinear_factor() -> None:
    batch = _batch()
    origin = _state()
    perturbed = origin.copy()
    perturbed[:, :3] += np.array([1000.0, 200.0, -50.0])

    frame = taroz_linearization_residual_frame(batch, origin, perturbed, label="perturbed")
    p_delta = frame.loc[frame["factor"].eq("P"), "native_minus_taroz_linear"].to_numpy(dtype=np.float64)

    assert np.max(np.abs(p_delta)) > 1e-3


def test_corrected_tdcp_exposes_origin_relative_taroz_factor() -> None:
    batch = _batch()
    origin = _state()

    frame = taroz_linearization_residual_frame(batch, origin, origin, label="seed")
    tdcp_row = frame.loc[frame["factor"].eq("L")].iloc[0]

    assert tdcp_row["taroz_linear_residual"] == pytest.approx(0.0)
    assert abs(float(tdcp_row["native_residual"])) > 9.0
    assert abs(float(tdcp_row["native_minus_taroz_linear"])) > 9.0


def test_tdcp_linearization_reference_matches_taroz_factor() -> None:
    batch = _batch()
    origin = _state()

    frame = taroz_linearization_residual_frame(
        batch,
        origin,
        origin,
        label="seed",
        tdcp_native_ref_ecef=origin[:, :3],
    )
    tdcp_row = frame.loc[frame["factor"].eq("L")].iloc[0]

    assert tdcp_row["taroz_linear_residual"] == pytest.approx(0.0)
    assert tdcp_row["native_residual"] == pytest.approx(0.0)
    assert tdcp_row["native_minus_taroz_linear"] == pytest.approx(0.0)


def test_summarize_linearization_frame_reports_weighted_delta() -> None:
    batch = _batch()
    origin = _state()
    frame = taroz_linearization_residual_frame(batch, origin, origin, label="seed")

    stats = {(item.label, item.factor): item for item in summarize_linearization_frame(frame)}

    assert stats[("seed", "P")].count == 4
    assert stats[("seed", "P")].delta_weighted_rms == pytest.approx(0.0, abs=1e-9)
    assert stats[("seed", "L")].count == 1
    assert stats[("seed", "L")].delta_weighted_rms is not None
    assert stats[("seed", "L")].delta_weighted_rms > 9.0


def test_taroz_gnss_initial_state_uses_posbl_clock_and_taroz_drift_sign() -> None:
    batch = TripArrays(
        times_ms=np.array([1000.0, 2000.0, 3000.0], dtype=np.float64),
        sat_ecef=np.zeros((3, 0, 3), dtype=np.float64),
        pseudorange=np.zeros((3, 0), dtype=np.float64),
        weights=np.zeros((3, 0), dtype=np.float64),
        kaggle_wls=np.array(
            [
                [10.0, 0.0, 0.0],
                [13.0, 4.0, 0.0],
                [19.0, 8.0, 2.0],
            ],
            dtype=np.float64,
        ),
        truth=np.zeros((3, 3), dtype=np.float64),
        max_sats=0,
        has_truth=False,
        n_clock=7,
        clock_bias_m=np.array([1.0, 2.0, 3.0], dtype=np.float64),
        clock_drift_mps=np.array([0.5, 0.25, -0.75], dtype=np.float64),
    )

    state = taroz_gnss_initial_state_for_batch(batch)

    np.testing.assert_allclose(state[:, :3], batch.kaggle_wls)
    np.testing.assert_allclose(state[:, 6], np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(state[:, 7:13], np.zeros((3, 6)))
    np.testing.assert_allclose(state[:, 13], np.array([-0.5, -0.25, 0.75]))
    np.testing.assert_allclose(state[1, 3:6], np.array([4.5, 4.0, 1.0]))


def test_load_taroz_gnss_state_csv_for_batch_aligns_on_utc(tmp_path) -> None:
    batch = TripArrays(
        times_ms=np.array([1000.0, 2000.0], dtype=np.float64),
        sat_ecef=np.zeros((2, 0, 3), dtype=np.float64),
        pseudorange=np.zeros((2, 0), dtype=np.float64),
        weights=np.zeros((2, 0), dtype=np.float64),
        kaggle_wls=np.zeros((2, 3), dtype=np.float64),
        truth=np.zeros((2, 3), dtype=np.float64),
        max_sats=0,
        has_truth=False,
        n_clock=2,
    )
    path = tmp_path / "state.csv"
    path.write_text(
        "\n".join(
            [
                "utcTimeMillis,ecef_x,ecef_y,ecef_z,velocity_ecef_x,velocity_ecef_y,velocity_ecef_z,clock_bias_m_0,clock_bias_m_1,clock_drift_mps",
                "2000,20,21,22,2,3,4,6,7,8",
                "1000,10,11,12,1,2,3,4,5,6",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    state = load_taroz_gnss_state_csv_for_batch(path, batch)

    np.testing.assert_allclose(state[:, :3], np.array([[10.0, 11.0, 12.0], [20.0, 21.0, 22.0]]))
    np.testing.assert_allclose(state[:, 3:6], np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]))
    np.testing.assert_allclose(state[:, 6:8], np.array([[4.0, 5.0], [6.0, 7.0]]))
    np.testing.assert_allclose(state[:, 8], np.array([6.0, 8.0]))


def test_taroz_gtsam_gnss_factor_residual_frame_matches_exported_factor_formulas(tmp_path) -> None:
    state_path = tmp_path / "state.csv"
    mask_path = tmp_path / "mask.csv"
    residual_path = tmp_path / "residual.csv"
    state_path.write_text(
        "\n".join(
            [
                "epoch_index,position_x,position_y,position_z,velocity_x,velocity_y,velocity_z,clock_bias_m_0,clock_bias_m_1,clock_bias_m_2,clock_bias_m_3,clock_bias_m_4,clock_bias_m_5,clock_bias_m_6,clock_drift_mps",
                "1,10,20,30,1,2,3,4,5,6,7,8,9,10,0.5",
                "2,13,18,35,2,4,6,6,8,10,12,14,16,18,1.5",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    header = (
        "field,freq,epoch_index,utcTimeMillis,next_epoch_index,nextUtcTimeMillis,sys,svid,sat_col,"
        "factor_model,sigtype,sigma,measurement,dt_s,los_e,los_n,los_u,"
        "origin1_e,origin1_n,origin1_u,origin2_e,origin2_n,origin2_u"
    )
    rows = [
        # P: dot([1,0,0], [10,20,30] - [2,0,0]) + c0+c4 - 3 = 17
        "P,L5,1,1000,0,0,1,3,0,XC,4,1,3,0,1,0,0,2,0,0,NaN,NaN,NaN",
        # D: dot([0,1,0], [1,2,3] - [0,1,0]) + drift - 2 = -0.5
        "D,L1,1,1000,0,0,1,3,0,VD,0,1,2,0,0,1,0,0,1,0,NaN,NaN,NaN",
        # XXCC: dot([0,0,1], ([13,18,35]-[11,18,30])-([10,20,30]-[9,19,29])) + (6-4) - 4 = 2
        "L,L1,1,1000,2,2000,1,3,0,XXCC,0,1,4,1,0,0,1,9,19,29,11,18,30",
        # XXDD: dot([1,0,0], ([13,18,35]-[10,18,35])-([10,20,30]-[9,20,30])) + 2*(0.5+1.5)/2 - 3 = 1
        "L,L1,1,1000,2,2000,1,3,0,XXDD,0,1,3,2,1,0,0,9,20,30,10,18,35",
    ]
    mask_path.write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
    residual_path.write_text(
        "\n".join(
            [
                "field,freq,epoch_index,next_epoch_index,sys,svid,sat_col,factor_model,initial_residual,residual,factor_error",
                "P,L5,1,0,1,3,0,XC,0,17,1",
                "D,L1,1,0,1,3,0,VD,0,-0.5,1",
                "L,L1,1,2,1,3,0,XXCC,0,2,1",
                "L,L1,1,2,1,3,0,XXDD,0,1,1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    frame = taroz_gtsam_gnss_factor_residual_frame(mask_path, state_path, residual_csv=residual_path)

    np.testing.assert_allclose(frame["computed_residual"], np.array([17.0, -0.5, 2.0, 1.0]))
    np.testing.assert_allclose(frame["computed_minus_taroz"], np.zeros(4), atol=1e-12)


def test_taroz_gtsam_gnss_graph_cost_frame_includes_motion_and_clock(tmp_path) -> None:
    state_path = tmp_path / "state.csv"
    mask_path = tmp_path / "mask.csv"
    state_path.write_text(
        "\n".join(
            [
                "epoch_index,utcTimeMillis,position_x,position_y,position_z,velocity_x,velocity_y,velocity_z,clock_bias_m_0,clock_drift_mps",
                "1,1000,0,0,0,1,0,0,1,0.2",
                "2,2000,2,0,0,1,0,0,2,0.4",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    header = (
        "field,freq,epoch_index,utcTimeMillis,next_epoch_index,nextUtcTimeMillis,sys,svid,sat_col,"
        "factor_model,sigtype,sigma,measurement,dt_s,los_e,los_n,los_u,"
        "origin1_e,origin1_n,origin1_u,origin2_e,origin2_n,origin2_u"
    )
    rows = [
        # P: dot([1,0,0], [0,0,0]) + c0 - 0.5 = 0.5
        "P,L1,1,1000,0,0,1,3,0,XC,0,1,0.5,0,1,0,0,0,0,0,NaN,NaN,NaN",
        # D: dot([1,0,0], [1,0,0]) + drift - 1.0 = 0.2
        "D,L1,1,1000,0,0,1,3,0,VD,0,1,1.0,0,1,0,0,0,0,0,NaN,NaN,NaN",
        # XXCC: dot([1,0,0], [2,0,0]-[0,0,0]) + (2-1) - 2.5 = 0.5
        "L,L1,1,1000,2,2000,1,3,0,XXCC,0,1,2.5,1,1,0,0,0,0,0,0,0,0",
    ]
    mask_path.write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")

    frame = taroz_gtsam_gnss_graph_cost_frame(
        mask_path,
        state_path,
        n_clock=1,
        pr_huber_k=0.0,
        doppler_huber_k=0.0,
        carrier_huber_k=0.0,
        motion_sigma_m=2.0,
        clock_sigma_m=0.1,
    )
    summary = summarize_taroz_gtsam_graph_cost(frame).set_index("factor")

    assert summary.loc["P", "cost"] == pytest.approx(0.125)
    assert summary.loc["D", "cost"] == pytest.approx(0.02)
    assert summary.loc["L", "cost"] == pytest.approx(0.125)
    assert summary.loc["Motion", "cost"] == pytest.approx(0.125)
    assert summary.loc["Clock", "cost"] == pytest.approx(24.5)
    assert float(summary["cost"].sum()) == pytest.approx(24.895)


def test_imu_body_gravity_frame_matches_taroz_reference_residuals() -> None:
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

    frame = imu_body_gravity_residual_frame(batch, state, label="imu", weight=4.0)

    assert set(frame["factor"]) == {"IMU_P", "IMU_V", "IMU_R"}
    assert frame.shape[0] == 9
    np.testing.assert_allclose(frame["native_minus_taroz_linear"].to_numpy(dtype=np.float64), 0.0, atol=1e-12)
    by_factor = {
        factor: group.sort_values("axis")["native_residual"].to_numpy(dtype=np.float64)
        for factor, group in frame.groupby("factor")
    }
    expected_p, expected_v, expected_r = taroz_imu_body_gravity_residuals(
        state[0],
        state[1],
        attitude_idx=attitude_idx,
        dt_s=dt,
        gravity_nav=gravity,
        delta_p_body=delta_p,
        delta_v_body=delta_v,
        delta_angle=delta_angle,
    )
    np.testing.assert_allclose(by_factor["IMU_P"], expected_p, atol=1e-12)
    np.testing.assert_allclose(by_factor["IMU_V"], expected_v, atol=1e-12)
    np.testing.assert_allclose(by_factor["IMU_R"], expected_r, atol=1e-12)
    np.testing.assert_allclose(by_factor["IMU_P"], injected_p_residual, atol=1e-12)
    np.testing.assert_allclose(by_factor["IMU_V"], injected_v_residual, atol=1e-12)
    assert frame["weight"].eq(4.0).all()


def test_imu_body_gravity_frame_applies_next_bias_correction() -> None:
    n_clock = 1
    attitude_idx = 7 + n_clock
    accel_bias_idx = attitude_idx + 3
    gyro_bias_idx = attitude_idx + 6
    state = np.zeros((2, gyro_bias_idx + 3), dtype=np.float64)
    state[1, :3] = np.array([0.8, 0.0, 0.0], dtype=np.float64)
    state[1, 3:6] = np.array([0.3, 0.0, 0.0], dtype=np.float64)
    state[1, attitude_idx : attitude_idx + 3] = np.array([0.4, 0.0, 0.0], dtype=np.float64)
    state[1, accel_bias_idx : accel_bias_idx + 3] = np.array([0.2, 0.0, 0.0], dtype=np.float64)
    state[1, gyro_bias_idx : gyro_bias_idx + 3] = np.array([0.1, 0.0, 0.0], dtype=np.float64)

    identity_jac = np.eye(3, dtype=np.float64).reshape(1, 3, 3)
    batch = TripArrays(
        times_ms=np.array([1000.0, 2000.0], dtype=np.float64),
        sat_ecef=np.zeros((2, 0, 3), dtype=np.float64),
        pseudorange=np.zeros((2, 0), dtype=np.float64),
        weights=np.zeros((2, 0), dtype=np.float64),
        kaggle_wls=state[:, :3],
        truth=state[:, :3],
        max_sats=0,
        has_truth=True,
        n_clock=n_clock,
        dt=np.array([1.0, 0.0], dtype=np.float64),
        imu_preintegration=IMUPreintegration(
            epoch_times_ms=np.array([1000.0, 2000.0], dtype=np.float64),
            delta_t_s=np.array([1.0], dtype=np.float64),
            delta_v_body=np.array([[0.5, 0.0, 0.0]], dtype=np.float64),
            delta_p_body=np.array([[1.0, 0.0, 0.0]], dtype=np.float64),
            delta_angle_rad=np.array([[0.5, 0.0, 0.0]], dtype=np.float64),
            sample_count=np.array([8], dtype=np.int32),
            delta_p_bias_accel_jac=identity_jac,
            delta_v_bias_accel_jac=identity_jac,
            delta_p_bias_gyro_jac=np.zeros((1, 3, 3), dtype=np.float64),
            delta_v_bias_gyro_jac=np.zeros((1, 3, 3), dtype=np.float64),
            delta_angle_bias_gyro_jac=identity_jac,
            gravity_ecef=np.zeros((1, 3), dtype=np.float64),
        ),
    )

    frame = imu_body_gravity_residual_frame(batch, state, label="biased")
    residuals = frame.sort_values(["factor", "axis"])["native_residual"].to_numpy(dtype=np.float64)

    np.testing.assert_allclose(residuals, np.zeros(9), atol=1e-12)
