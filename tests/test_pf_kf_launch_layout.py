"""Doppler KF block boundaries agree with independent and partitioned references."""

import numpy as np
import pytest

from test_pf_doppler_geometry import _matrix_reference, _scenario

native = pytest.importorskip("gnss_gpu._gnss_gpu_pf_device")
pytestmark = [pytest.mark.cuda, pytest.mark.gpu]


def _read(state):
    particles = native.pf_device_get_particle_states(state)
    weights = np.empty(len(particles))
    native.pf_device_get_log_weights(state, weights)
    return particles, weights


def _update(state, observations):
    native.pf_device_doppler_kf_update(state, *observations, len(observations[3]), .19, .5)


@pytest.mark.parametrize("count", [63, 64, 65, 127, 128, 129, 255, 257])
@pytest.mark.parametrize("n_sat", [4, 12, 16, 17, 32, 33])
def test_partial_blocks_and_following_queries_against_matrix_reference(count, n_sat):
    before, observations = _scenario(count, n_sat)
    prior = np.linspace(-3., 0., count)
    expected, expected_weights = _matrix_reference(before, prior, observations)
    state = native.pf_device_create(count)
    try:
        native.pf_device_set_particle_states(state, before)
        native.pf_device_set_log_weights(state, prior)
        _update(state, observations)
        actual, weights = _read(state)
        np.testing.assert_array_equal(actual[:, :4].view(np.uint64), before[:, :4].view(np.uint64))
        np.testing.assert_allclose(actual[:, 4:], expected[:, 4:], rtol=2e-10, atol=2e-10)
        np.testing.assert_allclose(weights, expected_weights, rtol=2e-10, atol=2e-10)
        normalized = np.exp(expected_weights - expected_weights.max())
        ess = normalized.sum()**2 / np.sum(normalized**2)
        mean = np.sum(normalized[:, None] * before[:, :4], axis=0) / normalized.sum()
        np.testing.assert_allclose(native.pf_device_ess(state), ess, rtol=2e-9, atol=2e-9)
        np.testing.assert_allclose(native.pf_device_estimate(state), mean, rtol=2e-9, atol=2e-9)
    finally:
        native.pf_device_destroy(state)


@pytest.mark.parametrize("n_sat", [12, 17, 32, 33, 65])
@pytest.mark.parametrize("nonsymmetric", [False, True])
def test_particle_independence_across_cloud_partitions_and_output_ownership(n_sat, nonsymmetric):
    before, observations = _scenario(257, n_sat)
    if nonsymmetric:
        before[:, 8] += .3
        before[:, 12] -= .2
    prior = np.linspace(-3., 0., 257)
    state = native.pf_device_create(257)
    try:
        native.pf_device_set_particle_states(state, before)
        native.pf_device_set_log_weights(state, prior)
        _update(state, observations)
        saved = _read(state)
        copies = [array.copy() for array in saved]
        _update(state, observations)
        complete = _read(state)
        particles, weights = [], []
        offset = 0
        for count in (1, 63, 64, 129):
            chunk = native.pf_device_create(count)
            try:
                native.pf_device_set_particle_states(chunk, before[offset:offset + count].copy())
                native.pf_device_set_log_weights(chunk, prior[offset:offset + count].copy())
                for _ in range(2):
                    _update(chunk, observations)
                result, likelihood = _read(chunk)
                particles.append(result)
                weights.append(likelihood)
            finally:
                native.pf_device_destroy(chunk)
            offset += count
        for actual, expected in zip((np.concatenate(particles), np.concatenate(weights)), complete):
            np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
        for array, expected in zip(saved, copies):
            np.testing.assert_array_equal(array.view(np.uint64), expected.view(np.uint64))
    finally:
        native.pf_device_destroy(state)


def test_mixed_cache_counts_preserve_reduction_and_resampling_state():
    before, _ = _scenario(129, 12)
    prior = np.linspace(-3., 0., 129)
    state = native.pf_device_create(129)
    try:
        native.pf_device_set_particle_states(state, before)
        native.pf_device_set_log_weights(state, prior)
        expected, expected_weights = before, prior
        for n_sat in (16, 17, 32, 33, 65, 12):
            _, observations = _scenario(129, n_sat)
            expected, expected_weights = _matrix_reference(expected, expected_weights, observations)
            _update(state, observations)
        actual, weights = _read(state)
        np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=2e-10)
        np.testing.assert_allclose(weights, expected_weights, rtol=2e-10, atol=2e-10)
        native.pf_device_resample_megopolis_coalesced(state, 7, 42)
        gathered, reset = _read(state)
        lookup = {row[0].tobytes(): index for index, row in enumerate(actual)}
        indices = [lookup[row[0].tobytes()] for row in gathered]
        np.testing.assert_array_equal(gathered.view(np.uint64), actual[indices].view(np.uint64))
        np.testing.assert_array_equal(reset, np.zeros(129))
    finally:
        native.pf_device_destroy(state)


def test_empty_observations_leave_all_bits_unchanged():
    before, observations = _scenario(127, 0)
    prior = np.resize([0., -0., -3.], 127)
    state = native.pf_device_create(127)
    try:
        native.pf_device_set_particle_states(state, before)
        native.pf_device_set_log_weights(state, prior)
        _update(state, observations)
        actual, weights = _read(state)
        np.testing.assert_array_equal(actual.view(np.uint64), before.view(np.uint64))
        np.testing.assert_array_equal(weights.view(np.uint64), prior.view(np.uint64))
    finally:
        native.pf_device_destroy(state)
