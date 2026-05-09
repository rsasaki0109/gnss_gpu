"""Tests for WLS positioning (requires CUDA GPU)."""

import numpy as np
import pytest

from gnss_gpu.range_model import geometric_ranges_sagnac

try:
    from gnss_gpu._gnss_gpu import wls_position, wls_batch, ecef_to_lla, lla_to_ecef
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


def test_ecef_lla_roundtrip():
    x = np.array([-3957199.0])
    y = np.array([3310205.0])
    z = np.array([3737911.0])

    lat, lon, alt = ecef_to_lla(x, y, z)
    x2, y2, z2 = lla_to_ecef(lat, lon, alt)

    assert abs(x[0] - x2[0]) < 0.01
    assert abs(y[0] - y2[0]) < 0.01
    assert abs(z[0] - z2[0]) < 0.01
