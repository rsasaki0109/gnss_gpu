"""FP64 default PR likelihood, robust dispatch and shared-memory boundaries."""

import numpy as np
import pytest

native = pytest.importorskip("gnss_gpu._gnss_gpu_pf_device")
pytestmark = [pytest.mark.cuda, pytest.mark.gpu]


def _inputs(count, n_sat):
    rng = np.random.default_rng(count + n_sat)
    position = np.array([-3957199., 3310205., 3737911.])
    initial = rng.normal(size=(count, 16))
    initial[:, :3] += position
    initial[:, 3] += 3000.
    directions = rng.normal(size=(n_sat, 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    sat = directions * 26_560_000.
    pr = np.linalg.norm(sat - position, axis=1) + 3000.
    weights = rng.uniform(.1, 1., n_sat)
    return initial, sat, pr, weights


def _weights(state):
    result = np.empty(state.n_particles)
    native.pf_device_get_log_weights(state, result)
    return result


@pytest.mark.parametrize("count", [1, 31, 33, 255, 256, 257, 8193])
@pytest.mark.parametrize("n_sat", [1, 3, 4, 11, 12, 13, 16, 17, 64, 65])
def test_gaussian_matches_independent_reference_and_full_kernel(count, n_sat):
    initial, sat, pr, weights = _inputs(count, n_sat)
    ranges = np.linalg.norm(initial[:, None, :3] - sat[None], axis=2)
    residual = pr[None] - (ranges + initial[:, None, 3])
    expected = np.sum(-.5 * weights[None] * residual * residual / 25., axis=1)
    state = native.pf_device_create(count)
    try:
        native.pf_device_set_particle_states(state, initial)
        native.pf_device_set_log_weights(state, np.full(count, 123.))
        native.pf_device_weight(state, sat, pr, weights, n_sat, 5.)
        actual = _weights(state)
        np.testing.assert_allclose(actual, expected, rtol=5e-9, atol=1e-8)
        saved = actual.copy()
        # An extremely small positive gate routes to the original full kernel.
        # Fewer than min(n_sat, 4) exact-zero residuals disables that gate, so
        # the complete Gaussian likelihood must match, including overwrite.
        assert np.all(np.sum(residual == 0., axis=1) < min(n_sat, 4))
        native.pf_device_weight(state, sat, pr, weights, n_sat, 5., 0., np.nextafter(0., 1.))
        np.testing.assert_array_equal(_weights(state).view(np.uint64), saved.view(np.uint64))
        np.testing.assert_array_equal(actual.view(np.uint64), saved.view(np.uint64))
        np.testing.assert_array_equal(native.pf_device_get_particle_states(state).view(np.uint64),
                                      initial.view(np.uint64))
    finally:
        native.pf_device_destroy(state)


@pytest.mark.parametrize("sigma", [np.nextafter(0., 1.), 1e-200, 1e200, np.finfo(float).max])
@pytest.mark.parametrize("count", [1, 33, 257])
def test_common_reciprocal_preserves_underflow_and_overflow_contract(sigma, count):
    initial, sat, pr, weights = _inputs(count, 13)
    state = native.pf_device_create(count)
    try:
        native.pf_device_set_particle_states(state, initial)
        native.pf_device_weight(state, sat, pr, weights, 13, sigma)
        actual = _weights(state)
        native.pf_device_weight(state, sat, pr, weights, 13, sigma, 0., np.nextafter(0., 1.))
        np.testing.assert_array_equal(actual.view(np.uint64), _weights(state).view(np.uint64))
    finally:
        native.pf_device_destroy(state)


@pytest.mark.parametrize("count", [33, 257, 8193])
@pytest.mark.parametrize("arguments", [(0., -1., True, 0.), (-3., 0., False, 1.5)])
def test_disabled_robust_options_keep_gaussian_result(count, arguments):
    initial, sat, pr, weights = _inputs(count, 12)
    state = native.pf_device_create(count)
    try:
        native.pf_device_set_particle_states(state, initial)
        native.pf_device_weight(state, sat, pr, weights, 12, 5.)
        expected = _weights(state)
        native.pf_device_weight(state, sat, pr, weights, 12, 5., *arguments)
        np.testing.assert_array_equal(_weights(state).view(np.uint64), expected.view(np.uint64))
    finally:
        native.pf_device_destroy(state)


def test_gaussian_at_48kib_shared_boundary_and_after_growth():
    state = native.pf_device_create(33)
    try:
        for n_sat in (12, 1228, 13, 65, 1):
            initial, sat, pr, weights = _inputs(33, n_sat)
            native.pf_device_set_particle_states(state, initial)
            native.pf_device_weight(state, sat, pr, weights, n_sat, 5.)
            expected = _weights(state)
            native.pf_device_weight(state, sat, pr, weights, n_sat, 5., 0., np.nextafter(0., 1.))
            np.testing.assert_array_equal(_weights(state).view(np.uint64), expected.view(np.uint64))
    finally:
        native.pf_device_destroy(state)
