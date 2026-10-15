"""Packed observation ownership survives growth, switching and asynchronous inputs."""

import numpy as np
import pytest

from test_pf_dd_wavelength_cache import _weights, native

pytestmark = [pytest.mark.gpu, pytest.mark.cuda]


def _obs(count):
    rng = np.random.default_rng(count+953)
    sat = rng.uniform(100., 10000., (count, 3))
    pr = np.linalg.norm(sat, axis=1) + rng.normal(0., .1, count)
    ws = rng.uniform(.1, 1., count)
    return sat, pr, ws


def _call(state, obs, mode):
    sat, pr, ws = obs
    if mode == 'gmm':
        native.pf_device_weight_gmm(state, sat, pr, ws, len(pr), 5., .7, 15., 30.)
    elif mode == 'afv':
        cp = pr/.190293673 + np.arange(len(pr))*7.
        native.pf_device_weight_carrier_afv(state, sat, cp, ws, len(pr), .190293673, .05)
    else:
        native.pf_device_weight(state, sat, pr, ws, len(pr), 5.,
            2.5 if mode == 'student' else 0., 30., mode == 'huber', 1.5)


@pytest.mark.parametrize('count', [1, 31, 257])
@pytest.mark.parametrize('mode', ['gaussian', 'student', 'huber', 'gmm', 'afv'])
def test_grow_shrink_aliases_match_a_fresh_owner(count, mode):
    states = np.random.default_rng(count).normal(size=(count, 16))
    prior = np.linspace(-3., 3., count)
    state = native.pf_device_create(count)
    try:
        for satellites in (64, 65, 1, 128, 33, 1228, 4, 12):
            fresh = native.pf_device_create(count)
            try:
                for owner in (fresh, state):
                    native.pf_device_set_particle_states(owner, states)
                    native.pf_device_set_log_weights(owner, prior)
                    _call(owner, _obs(satellites), mode)
                np.testing.assert_array_equal(_weights(state, count).view(np.uint64), _weights(fresh, count).view(np.uint64))
                np.testing.assert_array_equal(native.pf_device_get_particle_states(state).view(np.uint64), states.view(np.uint64))
            finally:
                native.pf_device_destroy(fresh)
    finally:
        native.pf_device_destroy(state)


@pytest.mark.parametrize('count', [31, 257])
def test_three_update_modes_refresh_columns_independently(count):
    state = native.pf_device_create(count)
    states = np.random.default_rng(count+38).normal(size=(count, 16))
    prior = np.linspace(-1., 2., count)
    try:
        for satellites, mode in ((128, 'gmm'), (4, 'afv'), (65, 'student'), (1, 'gmm'), (1228, 'afv'), (12, 'gaussian')):
            fresh = native.pf_device_create(count)
            try:
                for owner in (fresh, state):
                    native.pf_device_set_particle_states(owner, states)
                    native.pf_device_set_log_weights(owner, prior)
                    _call(owner, _obs(satellites), mode)
                np.testing.assert_array_equal(_weights(state, count).view(np.uint64), _weights(fresh, count).view(np.uint64))
            finally:
                native.pf_device_destroy(fresh)
    finally:
        native.pf_device_destroy(state)


@pytest.mark.parametrize('mode', ['gaussian', 'student', 'huber', 'gmm', 'afv'])
def test_caller_input_mutation_does_not_change_queued_update(mode):
    state, oracle = native.pf_device_create(257), native.pf_device_create(257)
    try:
        states = np.random.default_rng(913).normal(size=(257, 16))
        obs = _obs(65)
        native.pf_device_set_particle_states(oracle, states)
        native.pf_device_set_log_weights(oracle, np.zeros(257))
        _call(oracle, obs, mode)
        expected = _weights(oracle, 257)
        native.pf_device_set_particle_states(state, states)
        native.pf_device_set_log_weights(state, np.zeros(257))
        _call(state, obs, mode)
        for array in obs:
            array.fill(-900.)
        np.testing.assert_array_equal(_weights(state, 257).view(np.uint64), expected.view(np.uint64))
    finally:
        native.pf_device_destroy(state)
        native.pf_device_destroy(oracle)


def test_pinned_and_device_observations_stay_separate_from_dd_scratch():
    from test_pf_dd_wavelength_cache import _cp, _data
    state = native.pf_device_create(31)
    states = np.random.default_rng(77).normal(size=(31, 16))
    try:
        for satellites, mode in ((65, 'gmm'), (4, 'afv'), (128, 'student'), (12, 'gaussian')):
            fresh = native.pf_device_create(31)
            try:
                for owner in (fresh, state):
                    native.pf_device_set_particle_states(owner, states)
                    native.pf_device_set_log_weights(owner, np.zeros(31))
                    _call(owner, _obs(satellites), mode)
                    _cp(owner, _data(33))
                np.testing.assert_array_equal(_weights(state, 31).view(np.uint64), _weights(fresh, 31).view(np.uint64))
            finally:
                native.pf_device_destroy(fresh)
    finally:
        native.pf_device_destroy(state)


@pytest.mark.parametrize('mode', ['gmm', 'afv'])
def test_empty_unvalidated_native_modes_keep_original_results(mode):
    state = native.pf_device_create(31)
    try:
        native.pf_device_set_log_weights(state, np.linspace(-2., 2., 31))
        _call(state, _obs(0), mode)
        expected = _weights(state, 31)
        _call(state, _obs(128), 'gaussian')
        native.pf_device_set_log_weights(state, np.linspace(-2., 2., 31))
        _call(state, _obs(0), mode)
        np.testing.assert_array_equal(_weights(state, 31).view(np.uint64), expected.view(np.uint64))
    finally:
        native.pf_device_destroy(state)
