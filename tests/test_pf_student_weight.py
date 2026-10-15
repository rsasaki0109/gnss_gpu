"""Student likelihood laws, general-kernel equivalence and shared boundaries."""

import numpy as np
import pytest
from test_pf_gaussian_weight import _inputs, _weights

native = pytest.importorskip("gnss_gpu._gnss_gpu_pf_device")
pytestmark = [pytest.mark.cuda, pytest.mark.gpu]


@pytest.mark.parametrize("count", [1, 33, 257, 8193])
@pytest.mark.parametrize("n_sat", [1, 3, 12, 13, 32, 65])
@pytest.mark.parametrize("nu", [1., 3., 10.])
def test_student_matches_numpy_and_general_kernel(count, n_sat, nu):
    before, sat, pr, weights = _inputs(count, n_sat)
    ranges = np.linalg.norm(before[:, None, :3] - sat[None], axis=2)
    residual = pr[None] - (ranges + before[:, None, 3])
    expected = np.sum(-.5 * (nu + 1.) * np.log(1. + weights[None] * residual**2 / (nu * 25.)), axis=1)
    assert np.all(np.sum(residual == 0., axis=1) < min(n_sat, 4))
    state = native.pf_device_create(count)
    try:
        native.pf_device_set_particle_states(state, before)
        native.pf_device_set_log_weights(state, np.full(count, 123.))
        native.pf_device_weight(state, sat, pr, weights, n_sat, 5., nu)
        actual = _weights(state)
        saved = actual.copy()
        np.testing.assert_allclose(actual, expected, rtol=5e-9, atol=1e-8)
        # Positive gate uses the original general kernel. Too few exact-zero
        # residuals makes its gate fall back to the full Student likelihood.
        native.pf_device_weight(state, sat, pr, weights, n_sat, 5., nu, np.nextafter(0., 1.))
        np.testing.assert_array_equal(_weights(state).view(np.uint64), saved.view(np.uint64))
        np.testing.assert_array_equal(actual.view(np.uint64), saved.view(np.uint64))
        # Student takes precedence over Huber in both dispatches.
        native.pf_device_weight(state, sat, pr, weights, n_sat, 5., nu, 0., True, .1)
        np.testing.assert_array_equal(_weights(state).view(np.uint64), saved.view(np.uint64))
        np.testing.assert_array_equal(native.pf_device_get_particle_states(state).view(np.uint64),
                                      before.view(np.uint64))
    finally:
        native.pf_device_destroy(state)


@pytest.mark.parametrize("nu,sigma", [(1e-200, 5.), (1e200, 5.), (3., 1e-200), (3., 1e200)])
def test_extreme_scale_and_zero_weight_behavior_matches_general(nu, sigma):
    before, sat, pr, weights = _inputs(33, 13)
    weights[::3] = 0.
    state = native.pf_device_create(33)
    try:
        native.pf_device_set_particle_states(state, before)
        native.pf_device_weight(state, sat, pr, weights, 13, sigma, nu)
        actual = _weights(state)
        native.pf_device_weight(state, sat, pr, weights, 13, sigma, nu, np.nextafter(0., 1.))
        np.testing.assert_array_equal(_weights(state).view(np.uint64), actual.view(np.uint64))
        if sigma == 1e-200:
            assert np.isnan(actual).all()
    finally:
        native.pf_device_destroy(state)


def test_maximum_shared_size_and_observation_growth():
    state = native.pf_device_create(33)
    try:
        for count in (12, 1228, 13, 65, 1):
            before, sat, pr, weights = _inputs(33, count)
            native.pf_device_set_particle_states(state, before)
            native.pf_device_weight(state, sat, pr, weights, count, 5., 3.)
            actual = _weights(state)
            native.pf_device_weight(state, sat, pr, weights, count, 5., 3., np.nextafter(0., 1.))
            np.testing.assert_array_equal(_weights(state).view(np.uint64), actual.view(np.uint64))
    finally:
        native.pf_device_destroy(state)
