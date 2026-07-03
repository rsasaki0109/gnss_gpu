"""CPU-side validation tests for the Doppler wrapper."""

import numpy as np
import pytest

from gnss_gpu.doppler import doppler_velocity, L1_WAVELENGTH


def _make_doppler_scenario():
    rx_pos = np.array([-3957199.0, 3310205.0, 3737911.0])
    true_vel = np.array([0.3, -0.5, 0.8])
    true_clock_drift = 5.0

    sat_ecef = np.array([
        [-14985000.0,  -3988000.0,  21474000.0],
        [ -9575000.0,  15498000.0,  19457000.0],
        [  7624000.0, -16218000.0,  19843000.0],
        [ 16305000.0,  12037000.0,  17183000.0],
        [-20889000.0,  13759000.0,   8291000.0],
        [  5463000.0,  24413000.0,   8934000.0],
        [ 22169000.0,   3975000.0,  13781000.0],
        [-11527000.0, -19421000.0,  13682000.0],
    ])
    sat_vel = np.array([
        [ 1200.0,  -2800.0,   500.0],
        [ -800.0,   1500.0, -2700.0],
        [ 2500.0,   1800.0,  -900.0],
        [-1100.0,  -2200.0,  2100.0],
        [  600.0,   2900.0,  1300.0],
        [-2600.0,    400.0, -1800.0],
        [ 1800.0,  -1200.0, -2400.0],
        [-2000.0,   2100.0,   700.0],
    ])

    wavelength = L1_WAVELENGTH
    doppler = np.zeros(len(sat_ecef))
    for s in range(len(sat_ecef)):
        diff = sat_ecef[s] - rx_pos
        los = diff / np.linalg.norm(diff)
        range_rate = np.dot(sat_vel[s] - true_vel, los) + true_clock_drift
        doppler[s] = range_rate / wavelength

    weights = np.ones(len(sat_ecef))
    return sat_ecef, sat_vel, doppler, rx_pos, weights


def test_doppler_velocity_wrapper_rejects_invalid_inputs():
    sat_ecef, sat_vel, doppler, rx_pos, weights = _make_doppler_scenario()

    with pytest.raises(RuntimeError, match="doppler must have shape"):
        doppler_velocity(sat_ecef, sat_vel, doppler.reshape(-1, 1), rx_pos, weights)

    with pytest.raises(RuntimeError, match="sat_ecef must have shape"):
        doppler_velocity(sat_ecef.ravel()[:-1], sat_vel, doppler, rx_pos, weights)

    with pytest.raises(RuntimeError, match="sat_vel must have shape"):
        doppler_velocity(sat_ecef, sat_vel[:-1], doppler, rx_pos, weights)

    with pytest.raises(RuntimeError, match="rx_pos must have shape"):
        doppler_velocity(sat_ecef, sat_vel, doppler, rx_pos.reshape(1, 3), weights)

    with pytest.raises(RuntimeError, match="weights must have shape"):
        doppler_velocity(sat_ecef, sat_vel, doppler, rx_pos, weights[:-1])

    with pytest.raises(RuntimeError, match="wavelength must be positive"):
        doppler_velocity(sat_ecef, sat_vel, doppler, rx_pos, weights, wavelength=0.0)


def test_doppler_velocity_wrapper_rejects_nonfinite_inputs():
    sat_ecef, sat_vel, doppler, rx_pos, weights = _make_doppler_scenario()

    bad_sat = sat_ecef.copy()
    bad_sat[0, 0] = np.nan
    with pytest.raises(RuntimeError, match="satellite positions and velocities must be finite"):
        doppler_velocity(bad_sat, sat_vel, doppler, rx_pos, weights)

    bad_doppler = doppler.copy()
    bad_doppler[0] = np.inf
    with pytest.raises(RuntimeError, match="doppler values must be finite"):
        doppler_velocity(sat_ecef, sat_vel, bad_doppler, rx_pos, weights)

    bad_weights = weights.copy()
    bad_weights[0] = -1.0
    with pytest.raises(RuntimeError, match="weights must be finite and nonnegative"):
        doppler_velocity(sat_ecef, sat_vel, doppler, rx_pos, bad_weights)
