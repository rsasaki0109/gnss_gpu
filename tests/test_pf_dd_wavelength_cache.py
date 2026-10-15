"""Shared DD inverse wavelengths preserve independent AFV and joint results."""

from types import SimpleNamespace

import numpy as np
import pytest

from gnss_gpu.particle_filter_device import ParticleFilterDevice

native = pytest.importorskip('gnss_gpu._gnss_gpu_pf_device')
pytestmark = [pytest.mark.gpu, pytest.mark.cuda]


def _data(count, wavelength=.19):
    rng = np.random.default_rng(count)
    sat = rng.uniform(1., 100., (count, 3))
    ranges = np.linalg.norm(sat, axis=1)
    return SimpleNamespace(
        sat_ecef_k=sat, sat_ecef_ref=np.zeros_like(sat), n_dd=count,
        base_range_k=np.zeros(count), base_range_ref=np.zeros(count),
        dd_pseudorange_m=ranges + rng.normal(0., .1, count),
        dd_carrier_cycles=ranges / wavelength + np.arange(count) * 7 + rng.normal(0., .01, count),
        dd_weights=rng.uniform(.1, 1., count), wavelengths_m=np.full(count, wavelength))


def _cloud(count):
    states = np.random.default_rng(count).normal(size=(count, 16))
    states[:, :4] = 0.
    return states


def _weights(state, count):
    weights = np.empty(count)
    native.pf_device_get_log_weights(state, weights)
    return weights


def _cp(state, data, gate=0., huber=False):
    native.pf_device_weight_dd_carrier_afv(
        state, data.sat_ecef_k, data.sat_ecef_ref, data.dd_carrier_cycles,
        data.base_range_k, data.base_range_ref, data.dd_weights, data.wavelengths_m,
        data.n_dd, .05, gate, huber, 1.5)


def _pr(state, data, gate=0., huber=False):
    native.pf_device_weight_dd_pseudorange(
        state, data.sat_ecef_k, data.sat_ecef_ref, data.dd_pseudorange_m,
        data.base_range_k, data.base_range_ref, data.dd_weights, data.n_dd, .75, gate, huber, 1.5)


def _joint(state, data, gate_pr=0., gate_cp=0., huber=False):
    native.pf_device_weight_dd_joint(
        state, data.sat_ecef_k, data.sat_ecef_ref, data.dd_pseudorange_m, data.dd_carrier_cycles,
        data.base_range_k, data.base_range_ref, data.dd_weights, data.wavelengths_m,
        data.n_dd, .75, .05, gate_pr, gate_cp, huber, 1.5, 1.5)


def _single_afv(state, data):
    # Zero reference ranges/base ranges and zero clock reduce DD geometry to
    # the unchanged undifferenced AFV kernel. Its FP64 reciprocal stays private
    # to each particle, so this checks an independent production calculation.
    native.pf_device_weight_carrier_afv(
        state, data.sat_ecef_k, data.dd_carrier_cycles, data.dd_weights,
        data.n_dd, data.wavelengths_m[0], .05)


@pytest.mark.parametrize('particles', [1, 31, 255, 256, 257])
@pytest.mark.parametrize('pairs', [1, 4, 12, 33, 257, 512])
def test_dd_carrier_matches_independent_single_afv(particles, pairs):
    states, data = _cloud(particles), _data(pairs)
    prior = np.random.default_rng(particles).normal(size=particles)
    state = native.pf_device_create(particles)
    try:
        native.pf_device_set_particle_states(state, states)
        native.pf_device_set_log_weights(state, prior)
        _single_afv(state, data)
        expected = _weights(state, particles)
        native.pf_device_set_log_weights(state, prior)
        _cp(state, data)
        np.testing.assert_array_equal(_weights(state, particles).view(np.uint64), expected.view(np.uint64))
        np.testing.assert_array_equal(native.pf_device_get_particle_states(state).view(np.uint64), states.view(np.uint64))
    finally:
        native.pf_device_destroy(state)


@pytest.mark.parametrize('wavelength', [.24, .25, np.nextafter(.19, np.inf), 1e-100, 1e100])
def test_reciprocal_values_match_independent_afv(wavelength):
    data, particles = _data(17, wavelength), 33
    state = native.pf_device_create(particles)
    try:
        native.pf_device_set_particle_states(state, _cloud(particles))
        _single_afv(state, data)
        expected = _weights(state, particles)
        native.pf_device_set_log_weights(state, np.zeros(particles))
        _cp(state, data)
        np.testing.assert_array_equal(_weights(state, particles).view(np.uint64), expected.view(np.uint64))
    finally:
        native.pf_device_destroy(state)


@pytest.mark.parametrize('particles', [31, 257])
@pytest.mark.parametrize('pairs', [12, 257, 512])
def test_joint_matches_independent_pr_and_single_afv(particles, pairs):
    state = native.pf_device_create(particles)
    states, data = _cloud(particles), _data(pairs)
    try:
        native.pf_device_set_particle_states(state, states)
        _pr(state, data)
        _single_afv(state, data)
        expected = _weights(state, particles)
        native.pf_device_set_particle_states(state, states)
        _joint(state, data)
        np.testing.assert_array_equal(_weights(state, particles).view(np.uint64), expected.view(np.uint64))
    finally:
        native.pf_device_destroy(state)


@pytest.mark.parametrize('pairs', [1, 2, 12, 257])
@pytest.mark.parametrize('mode', ['plain', 'gate', 'huber', 'huber_gate'])
def test_joint_keeps_separate_gate_huber_and_mixed_wavelength_semantics(pairs, mode):
    particles = 33
    states = _cloud(particles)
    states[:, :3] = np.random.default_rng(31).normal(0., .1, (particles, 3))
    data = _data(pairs)
    data.wavelengths_m[:] = np.resize([.19, .24, .25], pairs)
    state = native.pf_device_create(particles)
    gate_pr, gate_cp = (10., .25) if 'gate' in mode else (0., 0.)
    huber = 'huber' in mode
    try:
        native.pf_device_set_particle_states(state, states)
        _pr(state, data, gate_pr, huber)
        _cp(state, data, gate_cp, huber)
        expected = _weights(state, particles)
        native.pf_device_set_particle_states(state, states)
        _joint(state, data, gate_pr, gate_cp, huber)
        # Joint and split kernels already differ by a few ulps from FMA
        # contraction on the preceding build. Strict before/after bit checks
        # are in the actual-build comparator; here check split gate semantics.
        np.testing.assert_allclose(_weights(state, particles), expected,
                                   rtol=4 * np.finfo(float).eps, atol=1e-12)
    finally:
        native.pf_device_destroy(state)


def test_changed_observations_and_buffer_growth_do_not_reuse_old_inverse_values():
    particles = 257
    state = native.pf_device_create(particles)
    retained = []
    try:
        for pairs, wavelength in ((1, .19), (512, .24), (12, .25), (257, .19), (2, .24)):
            data = _data(pairs, wavelength)
            native.pf_device_set_particle_states(state, _cloud(particles))
            _single_afv(state, data)
            expected = _weights(state, particles)
            native.pf_device_set_log_weights(state, np.zeros(particles))
            _cp(state, data)
            actual = _weights(state, particles)
            np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
            retained.append((actual, actual.copy()))
            for old, copied in retained:
                np.testing.assert_array_equal(old.view(np.uint64), copied.view(np.uint64))
    finally:
        native.pf_device_destroy(state)


def test_public_default_and_override_wavelengths_match_independent_afv():
    particles = 33
    pf = ParticleFilterDevice(n_particles=particles)
    try:
        pf.initialize(np.zeros(3), spread_pos=1e-6, spread_cb=1e-6)
        states = _cloud(particles)
        for override in (None, .24):
            data = _data(12, .190293673 if override is None else override)
            pf.set_particle_states(states)
            _single_afv(pf._state, data)
            expected = pf.get_log_weights()
            pf.set_particle_states(states)
            if override is None:
                del data.wavelengths_m
            pf.update_dd_carrier_afv(data, wavelength=override, resample=False)
            np.testing.assert_array_equal(pf.get_log_weights().view(np.uint64), expected.view(np.uint64))
    finally:
        native.pf_device_destroy(pf._state)
        pf._state = None
