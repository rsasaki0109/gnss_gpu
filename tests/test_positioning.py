"""Tests for WLS positioning (requires CUDA GPU)."""

import numpy as np
import pytest

from gnss_gpu.range_model import geometric_ranges_sagnac

try:
    from gnss_gpu._gnss_gpu import (
        wls_position,
        wls_batch,
        ecef_to_lla,
        lla_to_ecef,
        satellite_azel,
    )
    HAS_GPU = True
except ImportError:
    HAS_GPU = False

pytestmark = pytest.mark.skipif(not HAS_GPU, reason="CUDA module not available")


def _make_test_scenario():
    """Create a test scenario with realistic GPS satellite positions.

    True receiver: Tokyo Station area (~35.68N, 139.77E)
    Satellites at ~20200 km altitude in realistic orbital positions.
    """
    # True receiver position (Tokyo Station, ECEF)
    true_pos = np.array([-3957199.0, 3310205.0, 3737911.0])
    true_cb = 3000.0  # clock bias in meters (~10 us)

    # Realistic GPS satellite ECEF positions (altitude ~20200 km)
    # Spread across the sky as seen from Tokyo
    sat_ecef = np.array([
        [-14985000.0,  -3988000.0,  21474000.0],  # G01
        [ -9575000.0,  15498000.0,  19457000.0],  # G03
        [  7624000.0, -16218000.0,  19843000.0],  # G06
        [ 16305000.0,  12037000.0,  17183000.0],  # G09
        [-20889000.0,  13759000.0,   8291000.0],  # G11
        [  5463000.0,  24413000.0,   8934000.0],  # G14
        [ 22169000.0,   3975000.0,  13781000.0],  # G17
        [-11527000.0, -19421000.0,  13682000.0],  # G22
    ])

    # Compute pseudoranges with the same Sagnac range model used by native WLS.
    ranges = geometric_ranges_sagnac(true_pos, sat_ecef)
    pseudoranges = ranges + true_cb

    weights = np.ones(len(sat_ecef))

    return sat_ecef, pseudoranges, weights, true_pos, true_cb


def test_wls_single():
    sat_ecef, pseudoranges, weights, true_pos, true_cb = _make_test_scenario()
    result, iters = wls_position(sat_ecef.flatten(), pseudoranges, weights)
    pos = result[:3]
    cb = result[3]

    err = np.linalg.norm(pos - true_pos)
    assert err < 0.01, f"Position error {err:.4f} m"
    assert abs(cb - true_cb) < 0.01, f"Clock bias error {abs(cb - true_cb):.4f} m"
    assert iters <= 10


def test_wls_single_accepts_matrix_satellite_input():
    sat_ecef, pseudoranges, weights, true_pos, true_cb = _make_test_scenario()
    result, iters = wls_position(sat_ecef, pseudoranges, weights)
    pos = result[:3]
    cb = result[3]

    assert np.linalg.norm(pos - true_pos) < 0.01
    assert abs(cb - true_cb) < 0.01
    assert iters <= 10


def test_wls_single_rejects_shape_mismatch():
    sat_ecef, pseudoranges, weights, *_ = _make_test_scenario()

    with pytest.raises(RuntimeError, match="sat_ecef must have shape"):
        wls_position(sat_ecef.flatten()[:-1], pseudoranges, weights)

    with pytest.raises(RuntimeError, match="weights length must match"):
        wls_position(sat_ecef, pseudoranges, weights[:-1])

    with pytest.raises(RuntimeError, match="pseudoranges must have shape"):
        wls_position(sat_ecef, pseudoranges.reshape(-1, 1), weights)


def test_wls_single_rejects_invalid_options():
    sat_ecef, pseudoranges, weights, *_ = _make_test_scenario()

    with pytest.raises(RuntimeError, match="max_iter must be >= 1"):
        wls_position(sat_ecef, pseudoranges, weights, max_iter=0)

    with pytest.raises(RuntimeError, match="tol must be positive"):
        wls_position(sat_ecef, pseudoranges, weights, tol=0.0)


def test_wls_batch():
    sat_ecef, pseudoranges, weights, true_pos, true_cb = _make_test_scenario()
    n_epoch = 100

    rng = np.random.default_rng(42)
    sat_batch = np.tile(sat_ecef, (n_epoch, 1, 1))
    pr_batch = np.tile(pseudoranges, (n_epoch, 1))
    pr_batch += rng.normal(0, 3.0, pr_batch.shape)  # 3m noise
    w_batch = np.tile(weights, (n_epoch, 1))

    results, iters = wls_batch(sat_batch, pr_batch, w_batch)

    for i in range(n_epoch):
        err = np.linalg.norm(results[i, :3] - true_pos)
        assert err < 20.0, f"Epoch {i}: position error {err:.2f} m"


def test_wls_batch_rejects_shape_mismatch():
    sat_ecef, pseudoranges, weights, *_ = _make_test_scenario()
    sat_batch = np.tile(sat_ecef, (3, 1, 1))
    pr_batch = np.tile(pseudoranges, (3, 1))
    w_batch = np.tile(weights, (3, 1))

    with pytest.raises(RuntimeError, match="sat_ecef must have shape"):
        wls_batch(sat_batch.reshape(3, -1), pr_batch, w_batch)

    with pytest.raises(RuntimeError, match="pseudoranges shape must match"):
        wls_batch(sat_batch, pr_batch[:, :-1], w_batch)

    with pytest.raises(RuntimeError, match="weights shape must match"):
        wls_batch(sat_batch, pr_batch, w_batch[:, :-1])


def test_wls_batch_rejects_invalid_options():
    sat_ecef, pseudoranges, weights, *_ = _make_test_scenario()
    sat_batch = np.tile(sat_ecef, (3, 1, 1))
    pr_batch = np.tile(pseudoranges, (3, 1))
    w_batch = np.tile(weights, (3, 1))

    with pytest.raises(RuntimeError, match="max_iter must be >= 1"):
        wls_batch(sat_batch, pr_batch, w_batch, max_iter=0)

    with pytest.raises(RuntimeError, match="tol must be positive"):
        wls_batch(sat_batch, pr_batch, w_batch, tol=0.0)


def test_ecef_lla_roundtrip():
    x = np.array([-3957199.0])
    y = np.array([3310205.0])
    z = np.array([3737911.0])

    lat, lon, alt = ecef_to_lla(x, y, z)
    x2, y2, z2 = lla_to_ecef(lat, lon, alt)

    assert abs(x[0] - x2[0]) < 0.01
    assert abs(y[0] - y2[0]) < 0.01
    assert abs(z[0] - z2[0]) < 0.01


def test_ecef_to_lla_rejects_invalid_inputs():
    x = np.array([-3957199.0])
    y = np.array([3310205.0])
    z = np.array([3737911.0])

    with pytest.raises(RuntimeError, match="must each have shape"):
        ecef_to_lla(x.reshape(1, 1), y, z)

    with pytest.raises(RuntimeError, match="same length"):
        ecef_to_lla(np.array([x[0], x[0]]), y, z)

    with pytest.raises(RuntimeError, match="requires at least one coordinate"):
        ecef_to_lla(np.array([]), np.array([]), np.array([]))

    with pytest.raises(RuntimeError, match="coordinates must be finite"):
        ecef_to_lla(np.array([np.nan]), y, z)


def test_lla_to_ecef_rejects_invalid_inputs():
    lat = np.array([np.radians(35.0)])
    lon = np.array([np.radians(139.0)])
    alt = np.array([10.0])

    with pytest.raises(RuntimeError, match="must each have shape"):
        lla_to_ecef(lat.reshape(1, 1), lon, alt)

    with pytest.raises(RuntimeError, match="same length"):
        lla_to_ecef(np.array([lat[0], lat[0]]), lon, alt)

    with pytest.raises(RuntimeError, match="requires at least one coordinate"):
        lla_to_ecef(np.array([]), np.array([]), np.array([]))

    with pytest.raises(RuntimeError, match="coordinates must be finite"):
        lla_to_ecef(np.array([np.nan]), lon, alt)


def test_satellite_azel_accepts_flat_and_matrix_inputs():
    rx = np.array([6378137.0, 0.0, 0.0])
    sat_ecef = np.array([
        [6378137.0 + 20_000_000.0, 0.0, 0.0],
        [6378137.0, 20_000_000.0, 0.0],
    ])

    az_matrix, el_matrix = satellite_azel(rx[0], rx[1], rx[2], sat_ecef)
    az_flat, el_flat = satellite_azel(rx[0], rx[1], rx[2], sat_ecef.ravel())

    np.testing.assert_allclose(az_flat, az_matrix)
    np.testing.assert_allclose(el_flat, el_matrix)
    assert az_matrix.shape == (2,)
    assert el_matrix.shape == (2,)
    assert np.all(np.isfinite(az_matrix))
    assert np.all(np.isfinite(el_matrix))


def test_satellite_azel_rejects_invalid_satellite_input():
    rx = np.array([6378137.0, 0.0, 0.0])

    with pytest.raises(RuntimeError, match="flat sat_ecef length must be divisible by 3"):
        satellite_azel(rx[0], rx[1], rx[2], np.zeros(5))

    with pytest.raises(RuntimeError, match="sat_ecef must have shape"):
        satellite_azel(rx[0], rx[1], rx[2], np.zeros((2, 2)))

    with pytest.raises(RuntimeError, match="requires at least one satellite"):
        satellite_azel(rx[0], rx[1], rx[2], np.empty((0, 3)))


def test_satellite_azel_rejects_nonfinite_receiver():
    with pytest.raises(RuntimeError, match="receiver ECEF coordinates must be finite"):
        satellite_azel(np.nan, 0.0, 0.0, np.zeros((1, 3)))
