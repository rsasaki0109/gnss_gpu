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
        imu_position_sigma_m=0.0,
        imu_velocity_sigma_mps=0.0,
        sat_clock_drift=None,
        absolute_height_ref_ecef=None,
        absolute_height_sigma_m=0.0,
        imu_accel_bias_prior_sigma_mps2=0.0,
        imu_accel_bias_between_sigma_mps2=0.0,
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
        captured["imu_position_sigma_m"] = imu_position_sigma_m
        captured["imu_velocity_sigma_mps"] = imu_velocity_sigma_mps
        captured["sat_clock_drift"] = sat_clock_drift
        captured["absolute_height_ref_ecef"] = absolute_height_ref_ecef
        captured["absolute_height_sigma_m"] = absolute_height_sigma_m
        captured["imu_accel_bias_prior_sigma_mps2"] = imu_accel_bias_prior_sigma_mps2
        captured["imu_accel_bias_between_sigma_mps2"] = imu_accel_bias_between_sigma_mps2
        captured["n_clock"] = n_clock
        return 2, 0.5

    monkeypatch.setattr(fgo_mod, "_fgo_gnss_lm_vd", _fake_fgo_gnss_lm_vd)

    sat_ecef = np.zeros((2, 4, 3), dtype=np.float64)
    pseudorange = np.ones((2, 4), dtype=np.float64)
    weights = np.ones((2, 4), dtype=np.float64)
    state = np.zeros((2, 11), dtype=np.float64)
    tdcp_meas = np.ones((1, 4), dtype=np.float64) * 0.25
    tdcp_weights = np.ones((1, 4), dtype=np.float64) * 10.0
    stop_mask = np.array([True, False], dtype=bool)
    imu_delta_p = np.array([[1.0, 0.0, -0.5]], dtype=np.float64)
    imu_delta_v = np.array([[0.1, 0.2, 0.3]], dtype=np.float64)
    sat_clock_drift = np.ones((2, 4), dtype=np.float64) * 0.02
    absolute_height_ref_ecef = np.array(
        [
            [10.0, 20.0, 30.0],
            [11.0, 20.0, 31.0],
        ],
        dtype=np.float64,
    )

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
        imu_position_sigma_m=0.5,
        imu_velocity_sigma_mps=0.25,
        sat_clock_drift=sat_clock_drift,
        enu_up_ecef=np.array([0.0, 0.0, 1.0], dtype=np.float64),
        absolute_height_ref_ecef=absolute_height_ref_ecef,
        absolute_height_sigma_m=0.1,
        imu_accel_bias_prior_sigma_mps2=1.5,
        imu_accel_bias_between_sigma_mps2=2.5,
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
    assert captured["imu_position_sigma_m"] == 0.5
    assert captured["imu_velocity_sigma_mps"] == 0.25
    np.testing.assert_array_equal(captured["sat_clock_drift"], sat_clock_drift)
    np.testing.assert_array_equal(captured["absolute_height_ref_ecef"], absolute_height_ref_ecef)
    assert captured["absolute_height_sigma_m"] == 0.1
    assert captured["imu_accel_bias_prior_sigma_mps2"] == 1.5
    assert captured["imu_accel_bias_between_sigma_mps2"] == 2.5
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
    assert captured["calls"] == 5


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
