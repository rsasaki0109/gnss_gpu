"""Both-Huber gated DD retains original math and independent gate fallback."""

import json
from pathlib import Path

import numpy as np
import pytest

from test_pf_dd_reference_range import _check_against_mixed, _observations
from test_pf_dd_wavelength_cache import _weights, native

pytestmark = [pytest.mark.gpu, pytest.mark.cuda]


@pytest.mark.parametrize('particles', [31, 257])
@pytest.mark.parametrize('pairs', [2, 33, 511, 512])
@pytest.mark.parametrize('gates', [(10., 0.), (0., .25), (10., .25)])
@pytest.mark.parametrize('ks', [(1.5, 1.5), (.75, 3.)])
def test_huber_gate_bits_match_original_geometry(particles, pairs, gates, ks):
    def update(state, data, gate, huber):
        native.pf_device_weight_dd_joint(state, data.sat_ecef_k, data.sat_ecef_ref,
            data.dd_pseudorange_m, data.dd_carrier_cycles, data.base_range_k, data.base_range_ref,
            data.dd_weights, data.wavelengths_m, data.n_dd, .75, .05, *gates, True, *ks)
    state = native.pf_device_create(particles)
    try:
        _check_against_mixed(state, particles, _observations(pairs), update, 0., True)
    finally:
        native.pf_device_destroy(state)


@pytest.mark.parametrize('particles', [1, 31, 257])
@pytest.mark.parametrize('huber,ks', [(False, (1.5, 1.5)), (True, (1.5, 1.5)),
                                    (True, (1.5, 0.)), (True, (0., 1.5))])
def test_partial_geometry_overflow_nan_is_excluded_after_successful_gate(particles, huber, ks):
    fixture = json.loads((Path(__file__).parent/'fixtures/pf_dd_gate_partial_nan.json').read_text(encoding='utf8'))
    states = np.zeros((particles, 16))
    states[:, 0] = fixture['particle_x']
    sats = np.zeros((4, 3))
    sats[0, 0] = fixture['overflow_satellite_x']
    refs = np.zeros((4, 3))
    zero, weights, waves = np.zeros(4), np.ones(4), np.full(4, .19)
    state = native.pf_device_create(particles)
    try:
        native.pf_device_set_particle_states(state, states)
        native.pf_device_weight_dd_joint(state, sats, refs, zero, zero, zero, zero,
                                        weights, waves, 4, .75, .05, 10., .25, huber, *ks)
        actual = _weights(state, particles)
        expected = np.full(particles, fixture['expected_weight_bits'], np.uint64)
        assert fixture['previous_fused_weight_bits'] != fixture['expected_weight_bits']
        np.testing.assert_array_equal(actual.view(np.uint64), expected)
        np.testing.assert_array_equal(native.pf_device_get_particle_states(state).view(np.uint64), states.view(np.uint64))
        refs[-1, 0] = -0.
        native.pf_device_weight_dd_joint(state, sats, refs, zero, zero, zero, zero,
                                        weights, waves, 4, .75, .05, 10., .25, huber, *ks)
        np.testing.assert_array_equal(_weights(state, particles).view(np.uint64), expected)
    finally:
        native.pf_device_destroy(state)


@pytest.mark.parametrize('huber', [False, True])
def test_insufficient_kept_observations_still_fall_back_to_all_including_nan(huber):
    states = np.zeros((31, 16))
    states[:, 0] = 1.
    sats = np.zeros((3, 3))
    sats[0, 0] = 1e308
    refs, zeros = np.zeros((3, 3)), np.zeros(3)
    state = native.pf_device_create(31)
    try:
        native.pf_device_set_particle_states(state, states)
        native.pf_device_weight_dd_joint(state, sats, refs, zeros, zeros, zeros, zeros,
                                        np.ones(3), np.full(3, .19), 3, .75, .05, 10., .25, huber, 1.5, 1.5)
        assert np.isnan(_weights(state, 31)).all()
    finally:
        native.pf_device_destroy(state)
