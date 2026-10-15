"""Joint DD reference reuse preserves the preceding GPU rounding, including odd tails."""

import json
from pathlib import Path

import numpy as np
import pytest

from test_pf_dd_reference_range import _check_against_mixed, _observations
from test_pf_dd_wavelength_cache import _joint, _weights, native

pytestmark = [pytest.mark.gpu, pytest.mark.cuda]


def _update(state, data, gate=0., huber=False):
    _joint(state, data, 10. if gate else 0., gate, huber)


@pytest.mark.parametrize('particles', [31, 257])
@pytest.mark.parametrize('pairs', [3, 7, 12, 13, 33, 257, 511])
@pytest.mark.parametrize('mode', ['plain', 'gate', 'huber', 'huber_gate'])
def test_joint_common_range_matches_original_mixed_geometry_bits(particles, pairs, mode):
    state = native.pf_device_create(particles)
    try:
        _check_against_mixed(state, particles, _observations(pairs), _update,
                             .25 if 'gate' in mode else 0., 'huber' in mode)
    finally:
        native.pf_device_destroy(state)


def test_joint_refreshes_reference_range_after_growth_and_changes():
    particles = 257
    state = native.pf_device_create(particles)
    try:
        for epoch, pairs in enumerate((3, 511, 12, 257, 33, 7)):
            _check_against_mixed(state, particles, _observations(pairs, epoch), _update, .25, True)
    finally:
        native.pf_device_destroy(state)


@pytest.mark.parametrize('particles', [31, 257])
@pytest.mark.parametrize('pairs', [3, 33, 511])
@pytest.mark.parametrize('gated', [False, True])
def test_carrier_only_huber_keeps_original_geometry_and_rounding(particles, pairs, gated):
    def update(state, data, gate, huber):
        native.pf_device_weight_dd_joint(
            state, data.sat_ecef_k, data.sat_ecef_ref, data.dd_pseudorange_m, data.dd_carrier_cycles,
            data.base_range_k, data.base_range_ref, data.dd_weights, data.wavelengths_m,
            data.n_dd, .75, .05, 10. if gate else 0., gate, huber, 0., 1.5)
    state = native.pf_device_create(particles)
    try:
        _check_against_mixed(state, particles, _observations(pairs), update, .25 if gated else 0., True)
    finally:
        native.pf_device_destroy(state)


@pytest.mark.parametrize('particles', [1, 31, 257])
def test_odd_tail_keeps_exact_preceding_gaussian_weight(particles):
    fixture = json.loads((Path(__file__).parent / 'fixtures/pf_dd_joint_rounding.json').read_text(encoding='utf-8'))
    observations = {name: np.asarray(values, dtype=np.float64) for name, values in fixture['observations'].items()}
    states = np.repeat(np.array(fixture['particle_state'])[None, :], particles, axis=0)
    state = native.pf_device_create(particles)
    try:
        native.pf_device_set_particle_states(state, states)
        native.pf_device_set_log_weights(state, np.full(particles, 123.))
        native.pf_device_weight_dd_joint(
            state, observations['sat_ecef_k'], observations['sat_ecef_ref'], observations['dd_pseudorange_m'],
            observations['dd_carrier_cycles'], observations['base_range_k'], observations['base_range_ref'],
            observations['dd_weights'], observations['wavelengths_m'], fixture['n_dd'],
            fixture['sigma_pr'], fixture['sigma_cycles'])
        expected = np.full(particles, fixture['expected_log_weight_uint64'], dtype=np.uint64)
        assert fixture['expected_log_weight_uint64'] != fixture['rejected_log_weight_uint64']
        np.testing.assert_array_equal(_weights(state, particles).view(np.uint64), expected)
        np.testing.assert_array_equal(native.pf_device_get_particle_states(state).view(np.uint64), states.view(np.uint64))
    finally:
        native.pf_device_destroy(state)
