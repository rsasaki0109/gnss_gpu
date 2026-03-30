"""Tests for Doppler-based velocity estimation."""

import numpy as np
import pytest

from gnss_gpu.doppler import doppler_velocity, doppler_velocity_batch, L1_WAVELENGTH

try:
    from gnss_gpu._gnss_gpu_doppler import doppler_velocity as _native_doppler
    HAS_GPU = True
except ImportError:
    HAS_GPU = False


def _make_doppler_scenario():
    """Create a test scenario with known receiver velocity.

    True receiver: Tokyo Station area (ECEF), moving at ~1 m/s.
    Satellites at ~20200 km altitude with realistic orbital velocities.
    """
    # Receiver position (Tokyo Station, ECEF)
    rx_pos = np.array([-3957199.0, 3310205.0, 3737911.0])

    # True receiver velocity [m/s] (walking speed, roughly north-east)
    true_vel = np.array([0.3, -0.5, 0.8])
    true_clock_drift = 5.0  # [m/s]

    # Satellite ECEF positions (same as positioning test)
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

    # Satellite velocities [m/s] (typical ~3 km/s orbital velocity)
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

    n_sat = len(sat_ecef)
    wavelength = L1_WAVELENGTH

    # Compute synthetic Doppler measurements
    # doppler * lambda = (sat_vel - rx_vel) . LOS + clock_drift
    doppler = np.zeros(n_sat)
    for s in range(n_sat):
        diff = sat_ecef[s] - rx_pos
        r = np.linalg.norm(diff)
        los = diff / r
        range_rate = np.dot(sat_vel[s] - true_vel, los) + true_clock_drift
        doppler[s] = range_rate / wavelength

    weights = np.ones(n_sat)

    return sat_ecef, sat_vel, doppler, rx_pos, weights, true_vel, true_clock_drift


def test_doppler_velocity_single():
    """Test single-epoch Doppler velocity estimation."""
    sat_ecef, sat_vel, doppler, rx_pos, weights, true_vel, true_cd = _make_doppler_scenario()

    result, iters = doppler_velocity(sat_ecef, sat_vel, doppler, rx_pos, weights)

    vel = result[:3]
    cd = result[3]

    vel_err = np.linalg.norm(vel - true_vel)
    assert vel_err < 0.001, f"Velocity error {vel_err:.6f} m/s"
    assert abs(cd - true_cd) < 0.001, f"Clock drift error {abs(cd - true_cd):.6f} m/s"
    assert iters <= 10 and iters > 0


def test_doppler_velocity_with_noise():
    """Test with Doppler noise added."""
    sat_ecef, sat_vel, doppler, rx_pos, weights, true_vel, true_cd = _make_doppler_scenario()

    rng = np.random.default_rng(42)
    doppler_noisy = doppler + rng.normal(0, 0.5, doppler.shape)  # 0.5 Hz noise

    result, iters = doppler_velocity(sat_ecef, sat_vel, doppler_noisy, rx_pos, weights)

    vel = result[:3]
    vel_err = np.linalg.norm(vel - true_vel)
    # With 0.5 Hz noise (~0.095 m/s range-rate noise), expect sub-meter accuracy
    assert vel_err < 0.5, f"Velocity error {vel_err:.4f} m/s with noise"


def test_doppler_velocity_batch():
    """Test batch Doppler velocity estimation."""
    sat_ecef, sat_vel, doppler, rx_pos, weights, true_vel, true_cd = _make_doppler_scenario()
    n_epoch = 50

    rng = np.random.default_rng(123)
    sat_batch = np.tile(sat_ecef, (n_epoch, 1, 1))
    svel_batch = np.tile(sat_vel, (n_epoch, 1, 1))
    dop_batch = np.tile(doppler, (n_epoch, 1))
    dop_batch += rng.normal(0, 0.3, dop_batch.shape)  # 0.3 Hz noise
    rx_batch = np.tile(rx_pos, (n_epoch, 1))
    w_batch = np.tile(weights, (n_epoch, 1))

    results, iters = doppler_velocity_batch(
        sat_batch, svel_batch, dop_batch, rx_batch, w_batch)

    for i in range(n_epoch):
        vel_err = np.linalg.norm(results[i, :3] - true_vel)
        assert vel_err < 0.5, f"Epoch {i}: velocity error {vel_err:.4f} m/s"


def test_doppler_insufficient_sats():
    """Test with fewer than 4 satellites."""
    sat_ecef = np.array([[1e7, 0, 0], [0, 1e7, 0], [0, 0, 1e7]])
    sat_vel = np.zeros((3, 3))
    doppler = np.zeros(3)
    rx_pos = np.array([1e6, 0, 0])
    weights = np.ones(3)

    result, iters = doppler_velocity(sat_ecef, sat_vel, doppler, rx_pos, weights)
    assert iters == -1


@pytest.mark.skipif(not HAS_GPU, reason="CUDA module not available")
def test_doppler_gpu_single():
    """Test that GPU (native) single-epoch matches expected results."""
    sat_ecef, sat_vel, doppler, rx_pos, weights, true_vel, true_cd = _make_doppler_scenario()

    result, iters = _native_doppler(
        sat_ecef.ravel(), sat_vel.ravel(), doppler, rx_pos, weights)

    vel_err = np.linalg.norm(result[:3] - true_vel)
    assert vel_err < 0.001, f"GPU velocity error {vel_err:.6f} m/s"
    assert abs(result[3] - true_cd) < 0.001
