"""Individual Huber DD gates agree bitwise with mixed-reference geometry."""

import numpy as np
import pytest

from test_pf_dd_reference_range import _check_against_mixed, _observations
from test_pf_dd_wavelength_cache import _cp, _pr, _weights, native

pytestmark = [pytest.mark.gpu, pytest.mark.cuda]


def _update(operation, k=1.5, sigma=None):
    carrier = operation is _cp
    if sigma is None:
        sigma = .05 if carrier else .75

    def update(owner, obs, gate, huber):
        args = [owner, obs.sat_ecef_k, obs.sat_ecef_ref,
                obs.dd_carrier_cycles if carrier else obs.dd_pseudorange_m,
                obs.base_range_k, obs.base_range_ref, obs.dd_weights]
        if carrier:
            args.append(obs.wavelengths_m)
        weight = native.pf_device_weight_dd_carrier_afv if carrier else native.pf_device_weight_dd_pseudorange
        weight(*args, obs.n_dd, sigma, gate, huber, k)

    return update


@pytest.mark.parametrize('particles', [1, 31, 257])
@pytest.mark.parametrize('pairs', [2, 12, 33, 512])
@pytest.mark.parametrize('operation', [_pr, _cp], ids=['PR', 'CP'])
@pytest.mark.parametrize('threshold', [.25, 1e-100])
def test_huber_gate_bits_match_original_geometry_and_prior_update(particles, pairs, operation, threshold):
    state = native.pf_device_create(particles)
    try:
        _check_against_mixed(state, particles, _observations(pairs), _update(operation), threshold, True)
    finally:
        native.pf_device_destroy(state)


@pytest.mark.parametrize('operation', [_pr, _cp], ids=['PR', 'CP'])
@pytest.mark.parametrize('k', [0., np.nextafter(1.5, np.inf), 3., np.inf])
def test_huber_parameter_changes_and_fallback_match_original_bits(operation, k):
    state = native.pf_device_create(31)
    try:
        _check_against_mixed(state, 31, _observations(33), _update(operation, k), .25, True)
    finally:
        native.pf_device_destroy(state)


@pytest.mark.parametrize('operation', [_pr, _cp], ids=['PR', 'CP'])
def test_insufficient_kept_observations_matches_ungated_huber_bits(operation):
    particles, data = 257, _observations(12)
    states = np.zeros((particles, 16))
    states[:, :3] = [-3957199., 3310205., 3737911.]
    prior = np.random.default_rng(13).normal(size=particles)
    state = native.pf_device_create(particles)
    try:
        native.pf_device_set_particle_states(state, states)
        native.pf_device_set_log_weights(state, prior)
        operation(state, data, 0., True)
        expected = _weights(state, particles)
        native.pf_device_set_log_weights(state, prior)
        operation(state, data, 1e-100, True)
        np.testing.assert_array_equal(_weights(state, particles).view(np.uint64), expected.view(np.uint64))
    finally:
        native.pf_device_destroy(state)


@pytest.mark.parametrize('operation', [_pr, _cp], ids=['PR', 'CP'])
def test_geometry_sigma_and_huber_parameter_refresh_after_growth(operation):
    state = native.pf_device_create(257)
    try:
        for epoch, pairs in enumerate((2, 512, 33, 12, 2)):
            data = _observations(pairs, epoch)
            data.wavelengths_m *= 1. + epoch * .1
            sigma = (.05 if operation is _cp else .75) * (1. + epoch * .125)
            update = _update(operation, 1.5 + epoch * .5, sigma)
            _check_against_mixed(state, 257, data, update, .25, True)
    finally:
        native.pf_device_destroy(state)


@pytest.mark.parametrize('operation', [_pr, _cp], ids=['PR', 'CP'])
def test_partial_nan_retains_individual_huber_skip_if_greater_semantics(operation):
    particles = 31
    states = np.zeros((particles, 16))
    states[:, 0] = 1.
    data = _observations(4)
    data.sat_ecef_k[:] = 0.
    data.sat_ecef_k[0, 0] = 1e308
    data.sat_ecef_ref[:] = 0.
    data.base_range_k[:] = data.base_range_ref[:] = 0.
    data.dd_pseudorange_m[:] = data.dd_carrier_cycles[:] = 0.
    data.dd_weights[:] = 1.
    prior = np.linspace(-3., 3., particles)
    state = native.pf_device_create(particles)
    try:
        native.pf_device_set_particle_states(state, states)
        native.pf_device_set_log_weights(state, prior)
        operation(state, data, .25, True)
        actual = _weights(state, particles)
        if operation is _cp:
            assert np.isnan(actual).all()
        else:
            np.testing.assert_array_equal(actual.view(np.uint64), np.zeros(particles, dtype=np.uint64))
        # Signed zero selects the mixed path without changing ranges.
        data.sat_ecef_ref[-1, 0] = -0.
        native.pf_device_set_log_weights(state, prior)
        operation(state, data, .25, True)
        np.testing.assert_array_equal(_weights(state, particles).view(np.uint64), actual.view(np.uint64))
        np.testing.assert_array_equal(native.pf_device_get_particle_states(state).view(np.uint64), states.view(np.uint64))
    finally:
        native.pf_device_destroy(state)
