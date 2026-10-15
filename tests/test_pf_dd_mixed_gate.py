"""Mixed gates match unchanged ungated GPU costs, including ordered subsets."""

from types import SimpleNamespace

import numpy as np
import pytest

from test_pf_dd_reference_range import _observations, _particles
from test_pf_dd_wavelength_cache import _cp, _pr, _weights, native

pytestmark = [pytest.mark.gpu, pytest.mark.cuda]


def _mixed(pairs, epoch=0):
    data = _observations(pairs, epoch)
    data.sat_ecef_ref[1::2] += [12., -4., 8.]
    rover = np.array([-3957199., 3310205., 3737911.])
    data.base_range_ref = np.linalg.norm(data.sat_ecef_ref - (rover + [5., -3., 2.]), axis=1)
    dd_range = (np.linalg.norm(data.sat_ecef_k - rover, axis=1)
                - np.linalg.norm(data.sat_ecef_ref - rover, axis=1)
                - data.base_range_k + data.base_range_ref)
    data.dd_pseudorange_m = dd_range + np.resize([.01, .1, .4, 1.2], pairs)
    data.dd_carrier_cycles = dd_range / data.wavelengths_m + np.resize([.01, .1, .3, .45], pairs)
    return data


def _subset(data, selected):
    return SimpleNamespace(n_dd=int(np.count_nonzero(selected)),
        **{name: value[selected].copy() for name, value in vars(data).items() if name != 'n_dd'})


def _compare(state, states, prior, data, reference, operation, gate, huber):
    native.pf_device_set_particle_states(state, states)
    native.pf_device_set_log_weights(state, prior)
    # Zero gate retains the preceding GPU kernel and its ordered arithmetic.
    operation(state, reference, 0., huber)
    expected = _weights(state, len(states))
    native.pf_device_set_log_weights(state, prior)
    operation(state, data, gate, huber)
    actual = _weights(state, len(states))
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
    np.testing.assert_array_equal(native.pf_device_get_particle_states(state).view(np.uint64), states.view(np.uint64))


@pytest.mark.parametrize('particles', [1, 31, 257])
@pytest.mark.parametrize('pairs', [2, 33, 512])
@pytest.mark.parametrize('operation', [_pr, _cp], ids=['PR', 'CP'])
@pytest.mark.parametrize('huber', [False, True])
def test_all_kept_and_insufficient_kept_match_unchanged_ungated_cost(particles, pairs, operation, huber):
    states = _particles(particles)
    prior = np.random.default_rng(particles + 14).normal(size=particles)
    data = _mixed(pairs)
    state = native.pf_device_create(particles)
    try:
        for gate in (1e100, 1e-100):
            _compare(state, states, prior, data, data, operation, gate, huber)
    finally:
        native.pf_device_destroy(state)


@pytest.mark.parametrize('particles', [31, 257])
@pytest.mark.parametrize('pairs', [12, 33])
@pytest.mark.parametrize('operation', [_pr, _cp], ids=['PR', 'CP'])
@pytest.mark.parametrize('huber', [False, True])
def test_partial_kept_matches_ordered_original_gpu_subset(particles, pairs, operation, huber):
    states = _particles(particles)
    states[:, :3] = [-3957199., 3310205., 3737911.]
    states[:, :3] += np.random.default_rng(particles).normal(0., 1e-4, (particles, 3))
    prior = np.linspace(-3., 3., particles)
    data = _mixed(pairs)
    # Biases .01/.1 are well inside .25; .3/.4/.45/1.2 are outside.
    # Position perturbations and FP64 range rounding are far from these margins.
    selected = np.arange(pairs) % 4 < 2
    reference = _subset(data, selected)
    assert 3 <= reference.n_dd < pairs
    state = native.pf_device_create(particles)
    try:
        _compare(state, states, prior, data, reference, operation, .25, huber)
    finally:
        native.pf_device_destroy(state)


@pytest.mark.parametrize('operation', [_pr, _cp], ids=['PR', 'CP'])
@pytest.mark.parametrize('huber', [False, True])
def test_partial_nan_keeps_individual_filter_semantics(operation, huber):
    particles = 31
    states = np.zeros((particles, 16))
    states[:, 0] = 1.
    data = _observations(4)
    data.sat_ecef_k[:] = 0.
    data.sat_ecef_k[0, 0] = 1e308
    data.sat_ecef_ref[:] = 0.
    data.sat_ecef_ref[-1, 0] = -0.
    data.base_range_k[:] = data.base_range_ref[:] = 0.
    data.dd_pseudorange_m[:] = data.dd_carrier_cycles[:] = 0.
    data.dd_weights[:] = 1.
    prior = np.linspace(-3., 3., particles)
    state = native.pf_device_create(particles)
    try:
        native.pf_device_set_particle_states(state, states)
        native.pf_device_set_log_weights(state, prior)
        operation(state, data, .25, huber)
        actual = _weights(state, particles)
        if operation is _cp:
            assert np.isnan(actual).all()
        else:
            np.testing.assert_array_equal(actual.view(np.uint64), np.zeros(particles, dtype=np.uint64))
    finally:
        native.pf_device_destroy(state)


@pytest.mark.parametrize('operation', [_pr, _cp], ids=['PR', 'CP'])
@pytest.mark.parametrize('huber', [False, True])
def test_reuse_refreshes_mixed_geometry_wavelength_sigma_and_huber_parameter(operation, huber):
    states = _particles(31)
    prior = np.linspace(-3., 3., len(states))
    state = native.pf_device_create(len(states))
    try:
        for epoch, pairs in enumerate((2, 512, 33, 2)):
            data = _mixed(pairs, epoch)
            data.wavelengths_m *= 1. + epoch * .1
            carrier = operation is _cp
            def update(owner, obs, gate, use_huber):
                args = [owner, obs.sat_ecef_k, obs.sat_ecef_ref,
                        obs.dd_carrier_cycles if carrier else obs.dd_pseudorange_m,
                        obs.base_range_k, obs.base_range_ref, obs.dd_weights]
                if carrier:
                    args.append(obs.wavelengths_m)
                weight = native.pf_device_weight_dd_carrier_afv if carrier else native.pf_device_weight_dd_pseudorange
                weight(*args, obs.n_dd, (.05 if carrier else .75) * (1. + epoch * .125),
                       gate, use_huber, 1.5 + epoch * .5)
            _compare(state, states, prior, data, data, update, 1e-100, huber)
    finally:
        native.pf_device_destroy(state)
