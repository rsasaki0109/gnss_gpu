"""Undifferenced PR gates match the independently dispatched ungated GPU paths."""

import numpy as np
import pytest

from test_pf_dd_wavelength_cache import _weights, native

pytestmark = [pytest.mark.gpu, pytest.mark.cuda]


def _states(count):
    states = np.zeros((count, 16))
    states[:, 0] = 1.
    states[:, 3] = np.linspace(-.125, .125, count)
    return states


def _obs(count, fallback=False):
    sat = np.zeros((count, 3))
    sat[:, 0] = 11. + np.arange(count)
    residual = np.resize([.125, -.25, .5, -.75, 8.], count)
    if fallback:
        residual[:] = 8.
    pr = sat[:, 0] - 1. + residual
    ws = np.random.default_rng(count).uniform(.1, 1., count)
    return sat, pr, ws, residual


def _args(mode, sigma=.75):
    # The separately compiled ungated Student kernel supplies the GPU log and
    # ordered FMA oracle; the original general kernel contracts the same sum.
    return sigma, 2.5 if mode == 'student' else 0., mode == 'huber', 1.5


def _update(state, obs, mode, threshold, sigma=.75, enabled=None, k=None):
    sigma, nu, huber, huber_k = _args(mode, sigma)
    native.pf_device_weight(state, *obs[:3], len(obs[1]), sigma, nu, threshold,
                           huber if enabled is None else enabled, huber_k if k is None else k)


def _check(state, states, obs, mode, threshold=1., sigma=.75):
    count, satellites = len(states), len(obs[1])
    prior = np.random.default_rng(763).normal(size=count)
    mask = np.abs(obs[3]) <= threshold
    # All particle clock biases stay at least .125 away from a gate boundary.
    assert np.all(np.abs(np.abs(obs[3]) - threshold) > .125)
    if mask.sum() < min(satellites, 4):
        mask[:] = True
    subset = tuple(np.ascontiguousarray(v[mask]) for v in obs[:3])
    native.pf_device_set_particle_states(state, states)
    native.pf_device_set_log_weights(state, prior)
    _update(state, subset, mode, 0., sigma)
    expected = _weights(state, count)
    native.pf_device_set_log_weights(state, prior)
    _update(state, obs, mode, threshold, sigma)
    actual = _weights(state, count)
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
    np.testing.assert_array_equal(native.pf_device_get_particle_states(state).view(np.uint64), states.view(np.uint64))
    native.pf_device_set_log_weights(state, prior + 13.)
    _update(state, obs, mode, threshold, sigma)
    np.testing.assert_array_equal(_weights(state, count).view(np.uint64), expected.view(np.uint64))


@pytest.mark.parametrize('count', [1, 31, 257])
@pytest.mark.parametrize('satellites', [3, 4, 5, 33, 1228])
@pytest.mark.parametrize('mode', ['gaussian', 'huber', 'student'])
@pytest.mark.parametrize('fallback', [False, True])
def test_gate_subset_fallback_order_overwrite_and_partial_blocks(count, satellites, mode, fallback):
    state = native.pf_device_create(count)
    try:
        _check(state, _states(count), _obs(satellites, fallback), mode)
    finally:
        native.pf_device_destroy(state)


@pytest.mark.parametrize('mode', ['gaussian', 'huber', 'student'])
def test_observation_growth_modes_and_coefficients_refresh(mode):
    owners = [native.pf_device_create(31), native.pf_device_create(257)]
    try:
        for epoch, satellites in enumerate((3, 1228, 5, 128, 4, 12)):
            for state, count in zip(owners, (31, 257), strict=True):
                obs = list(_obs(satellites))
                obs[2] *= 1. + epoch * .125
                _check(state, _states(count), obs, mode, sigma=.75*(1.+epoch*.125))
    finally:
        for state in owners:
            native.pf_device_destroy(state)


@pytest.mark.parametrize('mode', ['gaussian', 'huber', 'student'])
def test_zero_observation_weights_preserve_bits(mode):
    state = native.pf_device_create(31)
    try:
        obs = list(_obs(12))
        obs[2] *= 0.
        _check(state, _states(31), obs, mode)
    finally:
        native.pf_device_destroy(state)


@pytest.mark.parametrize('mode', ['gaussian', 'huber', 'student'])
def test_successful_gate_rejects_geometry_overflow(mode):
    state = native.pf_device_create(31)
    try:
        states = _states(31)
        obs = list(_obs(7))
        obs[0][4, 0] = 1e308
        # Geometry overflows to infinity with finite valid input. The other
        # six residuals include five inliers, so the outlier must be skipped.
        _check(state, states, obs, mode)
    finally:
        native.pf_device_destroy(state)


@pytest.mark.parametrize('variant', ['negative_weight', 'nan_observation', 'infinite_observation'])
def test_invalid_observations_rejected_before_mutation(variant):
    state = native.pf_device_create(31)
    try:
        states = _states(31)
        prior = np.linspace(-3., 3., 31)
        obs = list(_obs(7))
        if variant == 'negative_weight':
            obs[2][0] = -1.
        else:
            obs[1][0] = np.nan if variant == 'nan_observation' else np.inf
        native.pf_device_set_particle_states(state, states)
        native.pf_device_set_log_weights(state, prior)
        with pytest.raises(RuntimeError, match='must be (non-negative|finite)'):
            _update(state, obs, 'gaussian', 1.)
        np.testing.assert_array_equal(_weights(state, 31).view(np.uint64), prior.view(np.uint64))
        np.testing.assert_array_equal(native.pf_device_get_particle_states(state).view(np.uint64), states.view(np.uint64))
    finally:
        native.pf_device_destroy(state)


def test_student_priority_and_disabled_huber_match_original_dispatch():
    state = native.pf_device_create(257)
    try:
        native.pf_device_set_particle_states(state, _states(257))
        obs = _obs(12)
        _update(state, obs, 'student', 1.)
        expected = _weights(state, 257)
        _update(state, obs, 'student', 1., enabled=True, k=.25)
        np.testing.assert_array_equal(_weights(state, 257).view(np.uint64), expected.view(np.uint64))
        _update(state, obs, 'gaussian', 1.)
        expected = _weights(state, 257)
        for k in (0., -1.):
            _update(state, obs, 'gaussian', 1., enabled=True, k=k)
            np.testing.assert_array_equal(_weights(state, 257).view(np.uint64), expected.view(np.uint64))
    finally:
        native.pf_device_destroy(state)


@pytest.mark.parametrize('mode', ['gaussian', 'huber', 'student'])
def test_exact_threshold_and_each_particle_fallback_are_independent(mode):
    state, oracle = native.pf_device_create(5), native.pf_device_create(1)
    try:
        states = _states(5)
        states[:, 3] = [0., .125, 4., -4., 1.]
        obs = list(_obs(5))
        residual = np.array([1., -1., .25, -.25, 8.])
        obs[1] = obs[0][:, 0] - 1. + residual
        expected = np.empty(5)
        for index, row in enumerate(states):
            mask = np.abs(residual - row[3]) <= 1.
            if mask.sum() < 4:
                mask[:] = True
            subset = tuple(np.ascontiguousarray(v[mask]) for v in obs[:3])
            native.pf_device_set_particle_states(oracle, row[None])
            _update(oracle, subset, mode, 0.)
            expected[index] = _weights(oracle, 1)[0]
        native.pf_device_set_particle_states(state, states)
        _update(state, obs, mode, 1.)
        np.testing.assert_array_equal(_weights(state, 5).view(np.uint64), expected.view(np.uint64))
    finally:
        native.pf_device_destroy(state)
        native.pf_device_destroy(oracle)


@pytest.mark.parametrize('mode', ['gaussian', 'huber', 'student'])
def test_nonfinite_particle_clock_preserves_nan_fallback_bits(mode):
    state = native.pf_device_create(31)
    try:
        states = _states(31)
        states[:, 3] = -np.inf
        obs = list(_obs(7))
        obs[0][4, 0] = 1e308
        native.pf_device_set_particle_states(state, states)
        _update(state, obs, mode, 0.)
        expected = _weights(state, 31)
        _update(state, obs, mode, 1.)
        actual = _weights(state, 31)
        assert np.isnan(actual).all()
        np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
    finally:
        native.pf_device_destroy(state)
