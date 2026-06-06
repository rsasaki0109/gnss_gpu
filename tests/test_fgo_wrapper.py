from __future__ import annotations

import numpy as np
import pytest

import gnss_gpu.fgo as fgo_mod


def test_fgo_gnss_lm_vd_forwards_tdcp_arguments(monkeypatch):
    captured: dict[str, np.ndarray | float | int | None] = {}

    def _fake_fgo_gnss_lm_vd(
        sat_ecef,
        pseudorange,
        weights,
        state,
        motion_sigma_m,
        clock_drift_sigma_m,
        clock_use_average_drift,
        stop_velocity_sigma_mps,
        stop_position_sigma_m,
        max_iter,
        tol,
        huber_k,
        enable_line_search,
        sys_kind,
        n_clock,
        sat_vel,
        doppler,
        doppler_weights,
        dt,
        stop_mask,
        tdcp_meas,
        tdcp_weights,
        tdcp_sigma_m,
        tdcp_use_drift,
        relative_height_sigma_m=0.0,
        enu_up_ecef=None,
        rel_height_edge_i=None,
        rel_height_edge_j=None,
        imu_delta_p=None,
        imu_delta_v=None,
        imu_delta_angle=None,
        imu_delta_t=None,
        imu_delta_p_bias_accel_jac=None,
        imu_delta_v_bias_accel_jac=None,
        imu_delta_p_bias_gyro_jac=None,
        imu_delta_v_bias_gyro_jac=None,
        imu_delta_angle_bias_gyro_jac=None,
        imu_position_sigma_m=0.0,
        imu_velocity_sigma_mps=0.0,
        imu_attitude_sigma_rad=0.0,
        imu_position_weights=None,
        imu_velocity_weights=None,
        imu_attitude_weights=None,
        imu_preintegration_information=None,
        imu_factor_use_next_bias=False,
        sat_clock_drift=None,
        absolute_height_ref_ecef=None,
        absolute_height_sigma_m=0.0,
        imu_accel_bias_prior_sigma_mps2=0.0,
        imu_accel_bias_between_sigma_mps2=0.0,
        imu_accel_bias_between_weights=None,
        imu_gyro_bias_prior_sigma_radps=0.0,
        imu_gyro_bias_between_sigma_radps=0.0,
        imu_gyro_bias_between_weights=None,
        doppler_huber_k=0.0,
        tdcp_huber_k=0.0,
        tdcp_linearization_ref_ecef=None,
        stop_velocity_huber_k=0.0,
        stop_position_huber_k=0.0,
        relative_height_huber_k=0.0,
        absolute_height_huber_k=0.0,
        imu_gravity=None,
        pr_linearization_ref_ecef=None,
        pr_linearization_los_ecef=None,
        doppler_linearization_ref_vel=None,
        doppler_linearization_los_ecef=None,
    ):
        captured["tdcp_meas"] = tdcp_meas
        captured["tdcp_weights"] = tdcp_weights
        captured["tdcp_sigma_m"] = tdcp_sigma_m
        captured["tdcp_use_drift"] = tdcp_use_drift
        captured["clock_use_average_drift"] = clock_use_average_drift
        captured["stop_velocity_sigma_mps"] = stop_velocity_sigma_mps
        captured["stop_position_sigma_m"] = stop_position_sigma_m
        captured["stop_mask"] = stop_mask
        captured["imu_delta_p"] = imu_delta_p
        captured["imu_delta_v"] = imu_delta_v
        captured["imu_delta_angle"] = imu_delta_angle
        captured["imu_delta_t"] = imu_delta_t
        captured["imu_delta_p_bias_accel_jac"] = imu_delta_p_bias_accel_jac
        captured["imu_delta_v_bias_accel_jac"] = imu_delta_v_bias_accel_jac
        captured["imu_delta_p_bias_gyro_jac"] = imu_delta_p_bias_gyro_jac
        captured["imu_delta_v_bias_gyro_jac"] = imu_delta_v_bias_gyro_jac
        captured["imu_delta_angle_bias_gyro_jac"] = imu_delta_angle_bias_gyro_jac
        captured["imu_position_sigma_m"] = imu_position_sigma_m
        captured["imu_velocity_sigma_mps"] = imu_velocity_sigma_mps
        captured["imu_attitude_sigma_rad"] = imu_attitude_sigma_rad
        captured["imu_position_weights"] = imu_position_weights
        captured["imu_velocity_weights"] = imu_velocity_weights
        captured["imu_attitude_weights"] = imu_attitude_weights
        captured["imu_preintegration_information"] = imu_preintegration_information
        captured["imu_factor_use_next_bias"] = imu_factor_use_next_bias
        captured["sat_clock_drift"] = sat_clock_drift
        captured["absolute_height_ref_ecef"] = absolute_height_ref_ecef
        captured["absolute_height_sigma_m"] = absolute_height_sigma_m
        captured["imu_accel_bias_prior_sigma_mps2"] = imu_accel_bias_prior_sigma_mps2
        captured["imu_accel_bias_between_sigma_mps2"] = imu_accel_bias_between_sigma_mps2
        captured["imu_accel_bias_between_weights"] = imu_accel_bias_between_weights
        captured["imu_gyro_bias_prior_sigma_radps"] = imu_gyro_bias_prior_sigma_radps
        captured["imu_gyro_bias_between_sigma_radps"] = imu_gyro_bias_between_sigma_radps
        captured["imu_gyro_bias_between_weights"] = imu_gyro_bias_between_weights
        captured["doppler_huber_k"] = doppler_huber_k
        captured["tdcp_huber_k"] = tdcp_huber_k
        captured["tdcp_linearization_ref_ecef"] = tdcp_linearization_ref_ecef
        captured["stop_velocity_huber_k"] = stop_velocity_huber_k
        captured["stop_position_huber_k"] = stop_position_huber_k
        captured["relative_height_huber_k"] = relative_height_huber_k
        captured["absolute_height_huber_k"] = absolute_height_huber_k
        captured["imu_gravity"] = imu_gravity
        captured["pr_linearization_ref_ecef"] = pr_linearization_ref_ecef
        captured["pr_linearization_los_ecef"] = pr_linearization_los_ecef
        captured["doppler_linearization_ref_vel"] = doppler_linearization_ref_vel
        captured["doppler_linearization_los_ecef"] = doppler_linearization_los_ecef
        captured["n_clock"] = n_clock
        return 2, 0.5

    monkeypatch.setattr(fgo_mod, "_fgo_gnss_lm_vd", _fake_fgo_gnss_lm_vd)

    sat_ecef = np.zeros((2, 4, 3), dtype=np.float64)
    pseudorange = np.ones((2, 4), dtype=np.float64)
    weights = np.ones((2, 4), dtype=np.float64)
    state = np.zeros((2, 17), dtype=np.float64)
    tdcp_meas = np.ones((1, 4), dtype=np.float64) * 0.25
    tdcp_weights = np.ones((1, 4), dtype=np.float64) * 10.0
    stop_mask = np.array([True, False], dtype=bool)
    imu_delta_p = np.array([[1.0, 0.0, -0.5]], dtype=np.float64)
    imu_delta_v = np.array([[0.1, 0.2, 0.3]], dtype=np.float64)
    imu_delta_angle = np.array([[0.01, 0.02, 0.03]], dtype=np.float64)
    imu_delta_t = np.array([0.95], dtype=np.float64)
    imu_delta_p_bias_accel_jac = np.eye(3, dtype=np.float64).reshape(1, 3, 3) * 0.45
    imu_delta_v_bias_accel_jac = np.eye(3, dtype=np.float64).reshape(1, 3, 3) * 0.95
    imu_delta_p_bias_gyro_jac = np.eye(3, dtype=np.float64).reshape(1, 3, 3) * 0.12
    imu_delta_v_bias_gyro_jac = np.eye(3, dtype=np.float64).reshape(1, 3, 3) * 0.34
    imu_delta_angle_bias_gyro_jac = np.eye(3, dtype=np.float64).reshape(1, 3, 3) * 0.95
    imu_position_weights = np.ones((1, 3), dtype=np.float64) * 4.0
    imu_velocity_weights = np.ones((1, 3), dtype=np.float64) * 5.0
    imu_attitude_weights = np.ones((1, 3), dtype=np.float64) * 6.0
    imu_preintegration_information = np.eye(9, dtype=np.float64).reshape(1, 9, 9)
    imu_gravity = np.array([[0.0, 0.0, -9.81]], dtype=np.float64)
    imu_accel_bias_between_weights = np.ones((1, 3), dtype=np.float64) * 7.0
    imu_gyro_bias_between_weights = np.ones((1, 3), dtype=np.float64) * 8.0
    sat_clock_drift = np.ones((2, 4), dtype=np.float64) * 0.02
    absolute_height_ref_ecef = np.array(
        [
            [10.0, 20.0, 30.0],
            [11.0, 20.0, 31.0],
        ],
        dtype=np.float64,
    )
    tdcp_linearization_ref_ecef = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        dtype=np.float64,
    )
    pr_linearization_ref_ecef = np.array(
        [
            [2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0],
        ],
        dtype=np.float64,
    )
    pr_linearization_los_ecef = np.ones((2, 4, 3), dtype=np.float64) * 0.25
    doppler_linearization_ref_vel = np.array(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ],
        dtype=np.float64,
    )
    doppler_linearization_los_ecef = np.ones((2, 4, 3), dtype=np.float64) * 0.5

    iters, mse = fgo_mod.fgo_gnss_lm_vd(
        sat_ecef,
        pseudorange,
        weights,
        state,
        n_clock=1,
        clock_use_average_drift=True,
        stop_velocity_sigma_mps=0.01,
        stop_position_sigma_m=0.02,
        dt=np.array([1.0, 0.0], dtype=np.float64),
        stop_mask=stop_mask,
        tdcp_meas=tdcp_meas,
        tdcp_weights=tdcp_weights,
        tdcp_sigma_m=0.2,
        tdcp_use_drift=True,
        imu_delta_p=imu_delta_p,
        imu_delta_v=imu_delta_v,
        imu_delta_angle=imu_delta_angle,
        imu_delta_t=imu_delta_t,
        imu_delta_p_bias_accel_jac=imu_delta_p_bias_accel_jac,
        imu_delta_v_bias_accel_jac=imu_delta_v_bias_accel_jac,
        imu_delta_p_bias_gyro_jac=imu_delta_p_bias_gyro_jac,
        imu_delta_v_bias_gyro_jac=imu_delta_v_bias_gyro_jac,
        imu_delta_angle_bias_gyro_jac=imu_delta_angle_bias_gyro_jac,
        imu_position_sigma_m=0.5,
        imu_velocity_sigma_mps=0.25,
        imu_attitude_sigma_rad=0.125,
        imu_position_weights=imu_position_weights,
        imu_velocity_weights=imu_velocity_weights,
        imu_attitude_weights=imu_attitude_weights,
        imu_preintegration_information=imu_preintegration_information,
        imu_gravity=imu_gravity,
        imu_factor_use_next_bias=True,
        sat_clock_drift=sat_clock_drift,
        enu_up_ecef=np.array([0.0, 0.0, 1.0], dtype=np.float64),
        absolute_height_ref_ecef=absolute_height_ref_ecef,
        absolute_height_sigma_m=0.1,
        imu_accel_bias_prior_sigma_mps2=1.5,
        imu_accel_bias_between_sigma_mps2=2.5,
        imu_accel_bias_between_weights=imu_accel_bias_between_weights,
        imu_gyro_bias_prior_sigma_radps=0.03,
        imu_gyro_bias_between_sigma_radps=0.04,
        imu_gyro_bias_between_weights=imu_gyro_bias_between_weights,
        doppler_huber_k=0.4,
        tdcp_huber_k=0.2,
        tdcp_linearization_ref_ecef=tdcp_linearization_ref_ecef,
        stop_velocity_huber_k=0.5,
        stop_position_huber_k=0.6,
        relative_height_huber_k=0.7,
        absolute_height_huber_k=0.8,
        pr_linearization_ref_ecef=pr_linearization_ref_ecef,
        pr_linearization_los_ecef=pr_linearization_los_ecef,
        doppler_linearization_ref_vel=doppler_linearization_ref_vel,
        doppler_linearization_los_ecef=doppler_linearization_los_ecef,
    )

    assert iters == 2
    assert mse == 0.5
    np.testing.assert_array_equal(captured["tdcp_meas"], tdcp_meas)
    np.testing.assert_array_equal(captured["tdcp_weights"], tdcp_weights)
    assert captured["tdcp_sigma_m"] == 0.2
    assert captured["tdcp_use_drift"] is True
    assert captured["clock_use_average_drift"] is True
    assert captured["stop_velocity_sigma_mps"] == 0.01
    assert captured["stop_position_sigma_m"] == 0.02
    np.testing.assert_array_equal(captured["stop_mask"], stop_mask.astype(np.uint8))
    np.testing.assert_array_equal(captured["imu_delta_p"], imu_delta_p)
    np.testing.assert_array_equal(captured["imu_delta_v"], imu_delta_v)
    np.testing.assert_array_equal(captured["imu_delta_angle"], imu_delta_angle)
    np.testing.assert_array_equal(captured["imu_delta_t"], imu_delta_t)
    np.testing.assert_array_equal(captured["imu_delta_p_bias_accel_jac"], imu_delta_p_bias_accel_jac)
    np.testing.assert_array_equal(captured["imu_delta_v_bias_accel_jac"], imu_delta_v_bias_accel_jac)
    np.testing.assert_array_equal(captured["imu_delta_p_bias_gyro_jac"], imu_delta_p_bias_gyro_jac)
    np.testing.assert_array_equal(captured["imu_delta_v_bias_gyro_jac"], imu_delta_v_bias_gyro_jac)
    np.testing.assert_array_equal(captured["imu_delta_angle_bias_gyro_jac"], imu_delta_angle_bias_gyro_jac)
    assert captured["imu_position_sigma_m"] == 0.5
    assert captured["imu_velocity_sigma_mps"] == 0.25
    assert captured["imu_attitude_sigma_rad"] == 0.125
    np.testing.assert_array_equal(captured["imu_position_weights"], imu_position_weights)
    np.testing.assert_array_equal(captured["imu_velocity_weights"], imu_velocity_weights)
    np.testing.assert_array_equal(captured["imu_attitude_weights"], imu_attitude_weights)
    np.testing.assert_array_equal(captured["imu_preintegration_information"], imu_preintegration_information)
    np.testing.assert_array_equal(captured["imu_gravity"], imu_gravity)
    assert captured["imu_factor_use_next_bias"] is True
    np.testing.assert_array_equal(captured["sat_clock_drift"], sat_clock_drift)
    np.testing.assert_array_equal(captured["absolute_height_ref_ecef"], absolute_height_ref_ecef)
    assert captured["absolute_height_sigma_m"] == 0.1
    assert captured["imu_accel_bias_prior_sigma_mps2"] == 1.5
    assert captured["imu_accel_bias_between_sigma_mps2"] == 2.5
    np.testing.assert_array_equal(captured["imu_accel_bias_between_weights"], imu_accel_bias_between_weights)
    assert captured["imu_gyro_bias_prior_sigma_radps"] == 0.03
    assert captured["imu_gyro_bias_between_sigma_radps"] == 0.04
    np.testing.assert_array_equal(captured["imu_gyro_bias_between_weights"], imu_gyro_bias_between_weights)
    assert captured["doppler_huber_k"] == 0.4
    assert captured["tdcp_huber_k"] == 0.2
    np.testing.assert_array_equal(captured["tdcp_linearization_ref_ecef"], tdcp_linearization_ref_ecef)
    assert captured["stop_velocity_huber_k"] == 0.5
    assert captured["stop_position_huber_k"] == 0.6
    assert captured["relative_height_huber_k"] == 0.7
    assert captured["absolute_height_huber_k"] == 0.8
    np.testing.assert_array_equal(captured["pr_linearization_ref_ecef"], pr_linearization_ref_ecef)
    np.testing.assert_array_equal(captured["pr_linearization_los_ecef"], pr_linearization_los_ecef)
    np.testing.assert_array_equal(captured["doppler_linearization_ref_vel"], doppler_linearization_ref_vel)
    np.testing.assert_array_equal(captured["doppler_linearization_los_ecef"], doppler_linearization_los_ecef)
    assert captured["n_clock"] == 1


def test_fgo_gnss_lm_vd_falls_back_to_legacy_native_signature(monkeypatch):
    captured: dict[str, int] = {"calls": 0}

    def _legacy_fgo_gnss_lm_vd(*args):
        captured["calls"] += 1
        if len(args) > 28:
            raise TypeError("incompatible function arguments")
        return 3, 1.25

    monkeypatch.setattr(fgo_mod, "_fgo_gnss_lm_vd", _legacy_fgo_gnss_lm_vd)

    sat_ecef = np.zeros((2, 4, 3), dtype=np.float64)
    pseudorange = np.ones((2, 4), dtype=np.float64)
    weights = np.ones((2, 4), dtype=np.float64)
    state = np.zeros((2, 8), dtype=np.float64)

    iters, mse = fgo_mod.fgo_gnss_lm_vd(
        sat_ecef,
        pseudorange,
        weights,
        state,
        n_clock=1,
    )

    assert (iters, mse) == (3, 1.25)
    assert captured["calls"] == 10


def test_fgo_gnss_lm_vd_forwards_stop_attitude_sigma(monkeypatch):
    captured: dict[str, object] = {}

    def _fake_fgo_gnss_lm_vd(*args):
        captured["args"] = args
        return 2, 0.5

    monkeypatch.setattr(fgo_mod, "_fgo_gnss_lm_vd", _fake_fgo_gnss_lm_vd)

    sat_ecef = np.zeros((2, 4, 3), dtype=np.float64)
    pseudorange = np.ones((2, 4), dtype=np.float64)
    weights = np.ones((2, 4), dtype=np.float64)
    state = np.zeros((2, 17), dtype=np.float64)

    iters, mse = fgo_mod.fgo_gnss_lm_vd(
        sat_ecef,
        pseudorange,
        weights,
        state,
        n_clock=1,
        stop_attitude_sigma_rad=0.03,
    )

    assert (iters, mse) == (2, 0.5)
    args = captured["args"]
    assert args[-1] == 0.03
    assert args[-6:-1] == (None, None, None, None, None)


def test_fgo_gnss_lm_vd_requires_rebuilt_native_for_tdcp_linearization_ref(monkeypatch):
    def _legacy_fgo_gnss_lm_vd(*args):
        raise TypeError("incompatible function arguments")

    monkeypatch.setattr(fgo_mod, "_fgo_gnss_lm_vd", _legacy_fgo_gnss_lm_vd)

    sat_ecef = np.zeros((2, 4, 3), dtype=np.float64)
    pseudorange = np.ones((2, 4), dtype=np.float64)
    weights = np.ones((2, 4), dtype=np.float64)
    state = np.zeros((2, 8), dtype=np.float64)

    with pytest.raises(RuntimeError, match="TDCP linearization reference"):
        fgo_mod.fgo_gnss_lm_vd(
            sat_ecef,
            pseudorange,
            weights,
            state,
            n_clock=1,
            tdcp_linearization_ref_ecef=np.zeros((2, 3), dtype=np.float64),
        )


def test_fgo_gnss_lm_vd_requires_rebuilt_native_for_fixed_linearization(monkeypatch):
    def _legacy_fgo_gnss_lm_vd(*args):
        raise TypeError("incompatible function arguments")

    monkeypatch.setattr(fgo_mod, "_fgo_gnss_lm_vd", _legacy_fgo_gnss_lm_vd)

    sat_ecef = np.zeros((2, 4, 3), dtype=np.float64)
    pseudorange = np.ones((2, 4), dtype=np.float64)
    weights = np.ones((2, 4), dtype=np.float64)
    state = np.zeros((2, 8), dtype=np.float64)

    with pytest.raises(RuntimeError, match="fixed-linearized P/D"):
        fgo_mod.fgo_gnss_lm_vd(
            sat_ecef,
            pseudorange,
            weights,
            state,
            n_clock=1,
            pr_linearization_ref_ecef=np.zeros((2, 3), dtype=np.float64),
            pr_linearization_los_ecef=np.zeros((2, 4, 3), dtype=np.float64),
        )


def test_fgo_gnss_lm_vd_requires_rebuilt_native_for_stop_attitude(monkeypatch):
    def _legacy_fgo_gnss_lm_vd(*args):
        raise TypeError("incompatible function arguments")

    monkeypatch.setattr(fgo_mod, "_fgo_gnss_lm_vd", _legacy_fgo_gnss_lm_vd)

    sat_ecef = np.zeros((2, 4, 3), dtype=np.float64)
    pseudorange = np.ones((2, 4), dtype=np.float64)
    weights = np.ones((2, 4), dtype=np.float64)
    state = np.zeros((2, 17), dtype=np.float64)

    with pytest.raises(RuntimeError, match="stop attitude"):
        fgo_mod.fgo_gnss_lm_vd(
            sat_ecef,
            pseudorange,
            weights,
            state,
            n_clock=1,
            stop_attitude_sigma_rad=0.03,
        )


def test_fgo_gnss_lm_vd_requires_rebuilt_native_for_factor_huber(monkeypatch):
    def _legacy_fgo_gnss_lm_vd(*args):
        raise TypeError("incompatible function arguments")

    monkeypatch.setattr(fgo_mod, "_fgo_gnss_lm_vd", _legacy_fgo_gnss_lm_vd)

    sat_ecef = np.zeros((2, 4, 3), dtype=np.float64)
    pseudorange = np.ones((2, 4), dtype=np.float64)
    weights = np.ones((2, 4), dtype=np.float64)
    state = np.zeros((2, 8), dtype=np.float64)

    with pytest.raises(RuntimeError, match="Doppler/TDCP Huber"):
        fgo_mod.fgo_gnss_lm_vd(
            sat_ecef,
            pseudorange,
            weights,
            state,
            n_clock=1,
            doppler_huber_k=0.4,
        )


def test_fgo_gnss_lm_vd_requires_rebuilt_native_for_stop_height_huber(monkeypatch):
    def _legacy_fgo_gnss_lm_vd(*args):
        raise TypeError("incompatible function arguments")

    monkeypatch.setattr(fgo_mod, "_fgo_gnss_lm_vd", _legacy_fgo_gnss_lm_vd)

    sat_ecef = np.zeros((2, 4, 3), dtype=np.float64)
    pseudorange = np.ones((2, 4), dtype=np.float64)
    weights = np.ones((2, 4), dtype=np.float64)
    state = np.zeros((2, 8), dtype=np.float64)

    with pytest.raises(RuntimeError, match="stop/height Huber"):
        fgo_mod.fgo_gnss_lm_vd(
            sat_ecef,
            pseudorange,
            weights,
            state,
            n_clock=1,
            stop_velocity_huber_k=0.5,
        )


def test_fgo_gnss_lm_vd_requires_rebuilt_native_for_accel_bias_state(monkeypatch):
    def _legacy_fgo_gnss_lm_vd(*args):
        raise TypeError("incompatible function arguments")

    monkeypatch.setattr(fgo_mod, "_fgo_gnss_lm_vd", _legacy_fgo_gnss_lm_vd)

    sat_ecef = np.zeros((2, 4, 3), dtype=np.float64)
    pseudorange = np.ones((2, 4), dtype=np.float64)
    weights = np.ones((2, 4), dtype=np.float64)
    state = np.zeros((2, 11), dtype=np.float64)

    with pytest.raises(RuntimeError, match="accel-bias"):
        fgo_mod.fgo_gnss_lm_vd(
            sat_ecef,
            pseudorange,
            weights,
            state,
            n_clock=1,
        )


def test_fgo_gnss_lm_vd_requires_rebuilt_native_for_gyro_bias_state(monkeypatch):
    def _legacy_fgo_gnss_lm_vd(*args):
        raise TypeError("incompatible function arguments")

    monkeypatch.setattr(fgo_mod, "_fgo_gnss_lm_vd", _legacy_fgo_gnss_lm_vd)

    sat_ecef = np.zeros((2, 4, 3), dtype=np.float64)
    pseudorange = np.ones((2, 4), dtype=np.float64)
    weights = np.ones((2, 4), dtype=np.float64)
    state = np.zeros((2, 14), dtype=np.float64)

    with pytest.raises(RuntimeError, match="gyro-bias"):
        fgo_mod.fgo_gnss_lm_vd(
            sat_ecef,
            pseudorange,
            weights,
            state,
            n_clock=1,
        )


def test_fgo_gnss_lm_vd_requires_rebuilt_native_for_attitude_state(monkeypatch):
    def _legacy_fgo_gnss_lm_vd(*args):
        raise TypeError("incompatible function arguments")

    monkeypatch.setattr(fgo_mod, "_fgo_gnss_lm_vd", _legacy_fgo_gnss_lm_vd)

    sat_ecef = np.zeros((2, 4, 3), dtype=np.float64)
    pseudorange = np.ones((2, 4), dtype=np.float64)
    weights = np.ones((2, 4), dtype=np.float64)
    state = np.zeros((2, 17), dtype=np.float64)

    with pytest.raises(RuntimeError, match="attitude"):
        fgo_mod.fgo_gnss_lm_vd(
            sat_ecef,
            pseudorange,
            weights,
            state,
            n_clock=1,
            imu_delta_angle=np.zeros((1, 3), dtype=np.float64),
            imu_attitude_sigma_rad=0.01,
        )


def test_fgo_gnss_lm_vd_requires_rebuilt_native_for_imu_diagonal_weights(monkeypatch):
    def _legacy_fgo_gnss_lm_vd(*args):
        raise TypeError("incompatible function arguments")

    monkeypatch.setattr(fgo_mod, "_fgo_gnss_lm_vd", _legacy_fgo_gnss_lm_vd)

    sat_ecef = np.zeros((2, 4, 3), dtype=np.float64)
    pseudorange = np.ones((2, 4), dtype=np.float64)
    weights = np.ones((2, 4), dtype=np.float64)
    state = np.zeros((2, 17), dtype=np.float64)

    with pytest.raises(RuntimeError, match="diagonal covariance"):
        fgo_mod.fgo_gnss_lm_vd(
            sat_ecef,
            pseudorange,
            weights,
            state,
            n_clock=1,
            imu_position_weights=np.ones((1, 3), dtype=np.float64),
        )


def test_fgo_gnss_lm_vd_requires_rebuilt_native_for_imu_bias_between_weights(monkeypatch):
    def _legacy_fgo_gnss_lm_vd(*args):
        raise TypeError("incompatible function arguments")

    monkeypatch.setattr(fgo_mod, "_fgo_gnss_lm_vd", _legacy_fgo_gnss_lm_vd)

    sat_ecef = np.zeros((2, 4, 3), dtype=np.float64)
    pseudorange = np.ones((2, 4), dtype=np.float64)
    weights = np.ones((2, 4), dtype=np.float64)
    state = np.zeros((2, 14), dtype=np.float64)

    with pytest.raises(RuntimeError, match="bias-between interval weights"):
        fgo_mod.fgo_gnss_lm_vd(
            sat_ecef,
            pseudorange,
            weights,
            state,
            n_clock=1,
            imu_accel_bias_between_weights=np.ones((1, 3), dtype=np.float64),
        )


def test_fgo_gnss_lm_vd_requires_rebuilt_native_for_imu_next_bias_mode(monkeypatch):
    def _legacy_fgo_gnss_lm_vd(*args):
        raise TypeError("incompatible function arguments")

    monkeypatch.setattr(fgo_mod, "_fgo_gnss_lm_vd", _legacy_fgo_gnss_lm_vd)

    sat_ecef = np.zeros((2, 4, 3), dtype=np.float64)
    pseudorange = np.ones((2, 4), dtype=np.float64)
    weights = np.ones((2, 4), dtype=np.float64)
    state = np.zeros((2, 14), dtype=np.float64)

    with pytest.raises(RuntimeError, match="next-bias"):
        fgo_mod.fgo_gnss_lm_vd(
            sat_ecef,
            pseudorange,
            weights,
            state,
            n_clock=1,
            imu_factor_use_next_bias=True,
        )


def test_fgo_gnss_lm_vd_requires_rebuilt_native_for_imu_preintegration_information(monkeypatch):
    def _legacy_fgo_gnss_lm_vd(*args):
        raise TypeError("incompatible function arguments")

    monkeypatch.setattr(fgo_mod, "_fgo_gnss_lm_vd", _legacy_fgo_gnss_lm_vd)

    sat_ecef = np.zeros((2, 4, 3), dtype=np.float64)
    pseudorange = np.ones((2, 4), dtype=np.float64)
    weights = np.ones((2, 4), dtype=np.float64)
    state = np.zeros((2, 17), dtype=np.float64)

    with pytest.raises(RuntimeError, match="preintegration information"):
        fgo_mod.fgo_gnss_lm_vd(
            sat_ecef,
            pseudorange,
            weights,
            state,
            n_clock=1,
            imu_preintegration_information=np.eye(9, dtype=np.float64).reshape(1, 9, 9),
        )


def test_fgo_gnss_lm_vd_requires_rebuilt_native_for_imu_gravity(monkeypatch):
    def _legacy_fgo_gnss_lm_vd(*args):
        raise TypeError("incompatible function arguments")

    monkeypatch.setattr(fgo_mod, "_fgo_gnss_lm_vd", _legacy_fgo_gnss_lm_vd)

    sat_ecef = np.zeros((2, 4, 3), dtype=np.float64)
    pseudorange = np.ones((2, 4), dtype=np.float64)
    weights = np.ones((2, 4), dtype=np.float64)
    state = np.zeros((2, 17), dtype=np.float64)

    with pytest.raises(RuntimeError, match="IMU gravity"):
        fgo_mod.fgo_gnss_lm_vd(
            sat_ecef,
            pseudorange,
            weights,
            state,
            n_clock=1,
            imu_gravity=np.array([[0.0, 0.0, -9.81]], dtype=np.float64),
        )


def test_fgo_gnss_lm_vd_requires_rebuilt_native_for_imu_delta_t(monkeypatch):
    def _legacy_fgo_gnss_lm_vd(*args):
        raise TypeError("incompatible function arguments")

    monkeypatch.setattr(fgo_mod, "_fgo_gnss_lm_vd", _legacy_fgo_gnss_lm_vd)

    sat_ecef = np.zeros((2, 4, 3), dtype=np.float64)
    pseudorange = np.ones((2, 4), dtype=np.float64)
    weights = np.ones((2, 4), dtype=np.float64)
    state = np.zeros((2, 8), dtype=np.float64)

    with pytest.raises(RuntimeError, match="preintegration delta times"):
        fgo_mod.fgo_gnss_lm_vd(
            sat_ecef,
            pseudorange,
            weights,
            state,
            n_clock=1,
            imu_delta_t=np.array([1.0], dtype=np.float64),
        )


def test_fgo_gnss_lm_vd_requires_rebuilt_native_for_imu_bias_jacobians(monkeypatch):
    def _legacy_fgo_gnss_lm_vd(*args):
        raise TypeError("incompatible function arguments")

    monkeypatch.setattr(fgo_mod, "_fgo_gnss_lm_vd", _legacy_fgo_gnss_lm_vd)

    sat_ecef = np.zeros((2, 4, 3), dtype=np.float64)
    pseudorange = np.ones((2, 4), dtype=np.float64)
    weights = np.ones((2, 4), dtype=np.float64)
    state = np.zeros((2, 8), dtype=np.float64)

    with pytest.raises(RuntimeError, match="preintegration bias Jacobians"):
        fgo_mod.fgo_gnss_lm_vd(
            sat_ecef,
            pseudorange,
            weights,
            state,
            n_clock=1,
            imu_delta_p_bias_accel_jac=np.eye(3, dtype=np.float64).reshape(1, 3, 3),
        )
