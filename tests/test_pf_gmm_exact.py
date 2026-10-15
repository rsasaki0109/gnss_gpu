"""Exact GMM sums agree with the retained original kernel and neutral padding."""

import numpy as np
import pytest

from test_pf_dd_reference_range import _particles
from test_pf_dd_wavelength_cache import _weights, native

pytestmark = [pytest.mark.gpu, pytest.mark.cuda]


def _observations(count):
    rng = np.random.default_rng(count)
    sats = rng.normal(size=(count, 3)) * 1e7
    rover = np.array([-3957199., 3310205., 3737911.])
    pr = np.linalg.norm(sats - rover, axis=1) + 3000.
    weights = rng.uniform(.1, 1., count)
    return sats, pr, weights


def _gmm(state, observations, sigma=1., mu=0., mixture=.5):
    sat, pr, weights = observations
    native.pf_device_weight_gmm(state, sat, pr, weights, len(pr), sigma, mixture, mu, sigma)


def _compare_original(state, states, observations, sigma, mu):
    native.pf_device_set_particle_states(state, states)
    count = len(observations[1])
    # 1,228 satellites fit the original 48KiB kernel but leave insufficient
    # room for the six shared coefficients, selecting the unchanged kernel.
    padded = (np.zeros((1228, 3)), np.zeros(1228), np.zeros(1228))
    for target, values in zip(padded, observations, strict=True):
        target[:count] = values
    _gmm(state, padded, sigma, mu)
    expected = _weights(state, len(states))
    padding_cost = np.zeros(len(states))
    if sigma == 2.:
        # Endpoint mixture=1 also selects the original kernel. Zero weight
        # gives exactly -log(2), the padding cost for mixture=.5 and sigma=2.
        _gmm(state, (np.zeros((1, 3)), np.zeros(1), np.zeros(1)), sigma, mu, 1.)
        padding_cost = _weights(state, len(states))
    else:
        assert sigma == 1.
        # Equal sigmas=1 and mixture=.5 make each zero-weight padding cost
        # log(.5)+log(2)=0, with both logs produced by the original GPU math.
    native.pf_device_set_log_weights(state, np.linspace(-3., 3., len(states)))
    _gmm(state, observations, sigma, mu)
    actual = _weights(state, len(states))
    # Append exactly the original padding costs, in the original order and
    # using FP64 additions. No inverse summation or tolerance is involved.
    folded = actual.copy()
    for _ in range(1228 - count):
        np.add(folded, padding_cost, out=folded)
    np.testing.assert_array_equal(folded.view(np.uint64), expected.view(np.uint64))
    np.testing.assert_array_equal(native.pf_device_get_particle_states(state).view(np.uint64), states.view(np.uint64))
    return actual


@pytest.mark.parametrize('particles', [1, 31, 257])
@pytest.mark.parametrize('satellites', [1, 12, 33])
@pytest.mark.parametrize('sigma', [1., 2.])
@pytest.mark.parametrize('mu', [0., .25, 15., -15.])
def test_equal_and_unequal_finite_logits_match_original_ordered_sum(particles, satellites, sigma, mu):
    states = _particles(particles)
    states[:, 3] += 3000.
    state = native.pf_device_create(particles)
    try:
        _compare_original(state, states, _observations(satellites), sigma, mu)
    finally:
        native.pf_device_destroy(state)


@pytest.mark.parametrize('particles', [1, 31, 257])
def test_shared_memory_boundary_keeps_original_result_and_partial_block_barrier(particles):
    states = _particles(particles)
    states[:, 3] += 3000.
    state = native.pf_device_create(particles)
    try:
        _compare_original(state, states, _observations(1227), 1., 15.)
    finally:
        native.pf_device_destroy(state)


@pytest.mark.parametrize('particles', [1, 31, 257])
@pytest.mark.parametrize('zero_weight', [False, True])
def test_nonfinite_logits_retain_original_two_exp_nan_bits(particles, zero_weight):
    states = _particles(particles)
    states[:, 3] += 3000.
    observations = _observations(12)
    observations[0][0, 0] = 1e308
    if zero_weight:
        observations[2][0] = 0.
    state = native.pf_device_create(particles)
    try:
        actual = _compare_original(state, states, observations, 1., 15.)
        assert np.isnan(actual).all()
    finally:
        native.pf_device_destroy(state)


def test_coefficients_observations_growth_and_returned_arrays_refresh():
    states = _particles(31)
    states[:, 3] += 3000.
    state = native.pf_device_create(len(states))
    retained = []
    try:
        for epoch, count in enumerate((1, 128, 1227, 12, 33, 1)):
            sigma = 1. if epoch % 2 else 2.
            observations = _observations(count)
            observations[1][:] += epoch * .25
            observations[2][:] *= 1. + epoch * .125
            actual = _compare_original(state, states, observations, sigma, epoch * .25 - 1.)
            retained.append((actual, actual.copy()))
            for output, snapshot in retained:
                np.testing.assert_array_equal(output.view(np.uint64), snapshot.view(np.uint64))
    finally:
        native.pf_device_destroy(state)
