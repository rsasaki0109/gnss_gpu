"""Deterministic checks for the shared positioning metrics."""

import math

import numpy as np
import pytest

from gnss_gpu.metrics import (
    compute_metrics,
    ecef_errors_2d_3d,
    ecef_to_lla,
    lla_to_ecef,
)


def test_equator_prime_meridian_round_trip():
    position = lla_to_ecef(0.0, 0.0, 25.0)
    np.testing.assert_array_equal(position, [6378162.0, 0.0, 0.0])
    assert ecef_to_lla(*position) == (0.0, 0.0, 25.0)


def test_north_pole_altitude():
    polar_radius = 6378137.0 * (1.0 - 1.0 / 298.257223563)
    assert ecef_to_lla(0.0, 0.0, polar_radius) == (math.pi / 2, 0.0, 0.0)


def test_nontrivial_coordinate_round_trip():
    lla = (math.radians(35.0), math.radians(139.0), 40.0)
    result = ecef_to_lla(*lla_to_ecef(*lla))
    np.testing.assert_allclose(result[:2], lla[:2], rtol=0, atol=1e-12)
    assert result[2] == pytest.approx(lla[2], abs=1e-6)


@pytest.mark.parametrize("per_epoch_truth", [False, True])
def test_ecef_errors_in_equatorial_enu_frame(per_epoch_truth):
    # At lat=lon=0: east=ECEF y, north=ECEF z, up=ECEF x.
    truth = np.array([6378137.0, 0.0, 0.0])
    estimates = truth + np.array([[12.0, 3.0, 4.0], [0.0, 0.0, 0.0]])
    if per_epoch_truth:
        truth = np.tile(truth, (2, 1))
    horizontal, spatial = ecef_errors_2d_3d(estimates, truth)
    np.testing.assert_array_equal(horizontal, [5.0, 0.0])
    np.testing.assert_array_equal(spatial, [13.0, 0.0])


@pytest.mark.parametrize("with_clock", [False, True])
def test_compute_metrics_keys_and_values(with_clock):
    truth = np.array([6378137.0, 0.0, 0.0])
    estimates = truth + np.array([[0.0, 0.0, 0.0], [12.0, 3.0, 4.0]])
    if with_clock:
        estimates = np.column_stack([estimates, [1000.0, -2000.0]])
    metrics = compute_metrics(estimates, truth)
    expected = {
        "rms_2d": math.sqrt(12.5),
        "rms_3d": math.sqrt(84.5),
        "mean_2d": 2.5,
        "mean_3d": 6.5,
        "std_2d": 2.5,
        "p50": 2.5,
        "p67": 3.35,
        "p95": 4.75,
        "max_2d": 5.0,
        "n_epochs": 2,
    }
    assert metrics.keys() == expected.keys() | {"errors_2d", "errors_3d"}
    for key, value in expected.items():
        assert metrics[key] == pytest.approx(value)
    np.testing.assert_array_equal(metrics["errors_2d"], [0.0, 5.0])
    np.testing.assert_array_equal(metrics["errors_3d"], [0.0, 13.0])


def test_evaluate_reexports_library_functions():
    from experiments import evaluate
    from gnss_gpu import metrics

    assert evaluate.compute_metrics is metrics.compute_metrics
    assert evaluate.ecef_errors_2d_3d is metrics.ecef_errors_2d_3d
    assert evaluate.ecef_to_lla is metrics.ecef_to_lla
    assert evaluate.lla_to_ecef is metrics.lla_to_ecef
