"""Common DD reference reuse agrees bitwise with the mixed-reference path."""

import numpy as np
import pytest

from test_pf_dd_wavelength_cache import _cp, _data, _pr, _weights, native

pytestmark = [pytest.mark.gpu, pytest.mark.cuda]


def _observations(pairs, epoch=0):
    data = _data(pairs)
    data.sat_ecef_ref[:] = [0., 2.5e7 + epoch * 100., -1.1e7]
    data.wavelengths_m[:] = np.resize([.19, .24, .25], pairs)
    rover = np.array([-3957199., 3310205., 3737911.])
    base = rover + [5., -3., 2.]
    data.base_range_k = np.linalg.norm(data.sat_ecef_k - base, axis=1)
    data.base_range_ref = np.linalg.norm(data.sat_ecef_ref - base, axis=1)
    dd_range = (np.linalg.norm(data.sat_ecef_k - rover, axis=1)
                - np.linalg.norm(data.sat_ecef_ref - rover, axis=1)
                - data.base_range_k + data.base_range_ref)
    data.dd_pseudorange_m = dd_range + np.resize([.01, .1, .4, 1.2], pairs)
    data.dd_carrier_cycles = dd_range / data.wavelengths_m + np.resize([.01, .1, .3, .45], pairs)
    return data


def _particles(count):
    states = np.random.default_rng(count).normal(size=(count, 16))
    states[:, :3] += [-3957199., 3310205., 3737911.]
    assert np.all(states[:, 0] != 0.)
    return states


def _check_against_mixed(state, count, data, operation, gate, huber):
    states = _particles(count)
    prior = np.random.default_rng(count + 12).normal(size=count)
    native.pf_device_set_particle_states(state, states)
    native.pf_device_set_log_weights(state, prior)
    operation(state, data, gate, huber)
    actual = _weights(state, count)
    np.testing.assert_array_equal(native.pf_device_get_particle_states(state).view(np.uint64),
                                  states.view(np.uint64))
    # Different zero bits select the mixed-reference path.
    # Nonzero particle x means x-(+0) == x-(-0), exactly, so the original
    # per-pair GPU expression checks the common-reference geometry bitwise.
    # Gated mixed paths can now fuse their two loops; independent original
    # ungated-cost/subset comparisons live in test_pf_dd_mixed_gate.py
    # and test_pf_dd_mixed_joint_gate.py.
    data.sat_ecef_ref[-1, 0] = -0.
    native.pf_device_set_particle_states(state, states)
    native.pf_device_set_log_weights(state, prior)
    operation(state, data, gate, huber)
    expected = _weights(state, count)
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
    data.sat_ecef_ref[-1, 0] = 0.
    return actual


@pytest.mark.parametrize('particles', [31, 257])
@pytest.mark.parametrize('pairs', [2, 12, 512])
@pytest.mark.parametrize('mode', ['plain', 'gate', 'huber', 'huber_gate'])
@pytest.mark.parametrize('operation', [_pr, _cp], ids=['PR', 'CP'])
def test_common_reference_matches_original_per_pair_geometry(particles, pairs, mode, operation):
    state = native.pf_device_create(particles)
    try:
        _check_against_mixed(state, particles, _observations(pairs), operation,
                             .25 if 'gate' in mode else 0., 'huber' in mode)
    finally:
        native.pf_device_destroy(state)


@pytest.mark.parametrize('operation', [_pr, _cp], ids=['PR', 'CP'])
def test_reference_changes_and_observation_growth_refresh_range(operation):
    particles = 257
    state = native.pf_device_create(particles)
    retained = []
    try:
        for epoch, pairs in enumerate((2, 512, 12, 257, 33, 2)):
            weights = _check_against_mixed(state, particles, _observations(pairs, epoch),
                                          operation, .25, True)
            retained.append((weights, weights.copy()))
            for actual, original in retained:
                np.testing.assert_array_equal(actual.view(np.uint64), original.view(np.uint64))
    finally:
        native.pf_device_destroy(state)
