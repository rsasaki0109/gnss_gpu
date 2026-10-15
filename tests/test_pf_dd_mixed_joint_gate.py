"""Mixed joint gates match independently filtered, original ungated GPU sums."""

import numpy as np
import pytest

from test_pf_dd_mixed_gate import _mixed, _subset
from test_pf_dd_reference_range import _particles
from test_pf_dd_wavelength_cache import _weights, native

pytestmark = [pytest.mark.gpu, pytest.mark.cuda]


def _joint(state, data, sigmas=(.75, .05), gates=(0., 0.), huber=False, ks=(1.5, 1.5)):
    native.pf_device_weight_dd_joint(state, data.sat_ecef_k, data.sat_ecef_ref,
        data.dd_pseudorange_m, data.dd_carrier_cycles, data.base_range_k, data.base_range_ref,
        data.dd_weights, data.wavelengths_m, data.n_dd, *sigmas, *gates, huber, *ks)


def _cloud(count):
    states = _particles(count)
    states[:, :3] = [-3957199., 3310205., 3737911.]
    states[:, :3] += np.random.default_rng(count).normal(0., 1e-5, (count, 3))
    return states


def _data(pairs, epoch=0):
    data = _mixed(pairs, epoch)
    rover = np.array([-3957199., 3310205., 3737911.])
    dd_range = (np.linalg.norm(data.sat_ecef_k - rover, axis=1)
                - np.linalg.norm(data.sat_ecef_ref - rover, axis=1)
                - data.base_range_k + data.base_range_ref)
    # PR retains positions 0/1 in each group; CP independently retains 2/3.
    data.dd_carrier_cycles = dd_range / data.wavelengths_m + np.resize([.4, .45, .01, .1], pairs)
    return data


def _selected(data, gate, carrier):
    if gate == 0. or gate > 1.:
        return data
    if gate < 1e-50:
        return data  # No observations remain, so the original all-DD fallback applies.
    selected = np.arange(data.n_dd) % 4 >= 2 if carrier else np.arange(data.n_dd) % 4 < 2
    if np.count_nonzero(selected) < min(data.n_dd, 3):
        return data
    return _subset(data, selected)


def _compare(state, states, data, gates, huber, sigmas=(.75, .05), ks=(1.5, 1.5)):
    native.pf_device_set_particle_states(state, states)
    # The original ungated joint kernel computes each ordered component. The
    # other component's sigma=1e200 gives zero cost on these finite ranges,
    # including Huber's squared normalized residual underflow. This preserves
    # joint CP's multiply/add rounding, which differs from the individual API.
    _joint(state, _selected(data, gates[0], False), (sigmas[0], 1e200), huber=huber, ks=ks)
    pr = _weights(state, len(states))
    _joint(state, _selected(data, gates[1], True), (1e200, sigmas[1]), huber=huber, ks=ks)
    cp = _weights(state, len(states))
    expected = np.add(pr, cp)
    native.pf_device_set_log_weights(state, np.linspace(-3., 3., len(states)))
    _joint(state, data, sigmas, gates, huber, ks)
    np.testing.assert_array_equal(_weights(state, len(states)).view(np.uint64), expected.view(np.uint64))
    np.testing.assert_array_equal(native.pf_device_get_particle_states(state).view(np.uint64), states.view(np.uint64))


@pytest.mark.parametrize('particles', [1, 31, 257])
@pytest.mark.parametrize('pairs', [2, 33, 512])
@pytest.mark.parametrize('huber', [False, True])
@pytest.mark.parametrize('gates', [(1e100, 1e100), (.25, .25), (1e-100, 1e-100)])
def test_mixed_joint_all_partial_and_insufficient_kept_match_original_components(particles, pairs, huber, gates):
    state = native.pf_device_create(particles)
    try:
        _compare(state, _cloud(particles), _data(pairs), gates, huber)
    finally:
        native.pf_device_destroy(state)


@pytest.mark.parametrize('huber', [False, True])
@pytest.mark.parametrize('gates', [(.25, 0.), (0., .25)])
def test_one_sided_gate_does_not_filter_the_other_component(huber, gates):
    state = native.pf_device_create(31)
    try:
        _compare(state, _cloud(31), _data(33), gates, huber)
    finally:
        native.pf_device_destroy(state)


@pytest.mark.parametrize('particles', [1, 31, 257])
@pytest.mark.parametrize('huber', [False, True])
@pytest.mark.parametrize('pairs', [3, 4])
def test_partial_nan_uses_joint_le_filter_and_independent_fallback(particles, huber, pairs):
    states = np.zeros((particles, 16))
    states[:, 0] = 1.
    data = _mixed(pairs)
    data.sat_ecef_k[:] = 0.
    data.sat_ecef_k[0, 0] = 1e308
    data.sat_ecef_ref[:] = 0.
    data.sat_ecef_ref[-1, 0] = -0.
    data.base_range_k[:] = data.base_range_ref[:] = 0.
    data.dd_pseudorange_m[:] = data.dd_carrier_cycles[:] = 0.
    data.dd_weights[:] = 1.
    state = native.pf_device_create(particles)
    try:
        native.pf_device_set_particle_states(state, states)
        _joint(state, data, gates=(10., .25), huber=huber)
        actual = _weights(state, particles)
        if pairs == 4:
            np.testing.assert_array_equal(actual.view(np.uint64), np.zeros(particles, np.uint64))
        else:
            assert np.isnan(actual).all()
        np.testing.assert_array_equal(native.pf_device_get_particle_states(state).view(np.uint64), states.view(np.uint64))
    finally:
        native.pf_device_destroy(state)


@pytest.mark.parametrize('huber', [False, True])
def test_reference_wavelength_sigma_huber_and_buffer_growth_refresh(huber):
    state = native.pf_device_create(31)
    try:
        for epoch, pairs in enumerate((2, 512, 33, 2)):
            data = _data(pairs, epoch)
            data.wavelengths_m *= 1. + epoch * .1
            # All-kept preserves independent component references after changes.
            _compare(state, _cloud(31), data, (1e100, 1e100), huber,
                     (.75 * (1. + epoch * .125), .05 * (1. + epoch * .25)),
                     (1.5 + epoch * .5, 3. + epoch * .25))
    finally:
        native.pf_device_destroy(state)
