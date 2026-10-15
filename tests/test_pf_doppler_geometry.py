"""Doppler KF geometry reuse agrees with an independent FP64 matrix reference."""

import numpy as np
import pytest

native = pytest.importorskip("gnss_gpu._gnss_gpu_pf_device")


def _scenario(count, n_sat):
    rng = np.random.default_rng(91 + count + n_sat)
    states = rng.normal(size=(count, 16))
    states[:, :3] *= 10.
    factors = rng.normal(size=(count, 3, 3))
    states[:, 7:] = (factors @ factors.swapaxes(1, 2) + np.eye(3) * .5).reshape(count, 9)
    directions = rng.normal(size=(n_sat, 3))
    if n_sat:
        directions /= np.linalg.norm(directions, axis=1)[:, None]
    satellites = directions * 26_560_000.
    sat_vel = rng.normal(0., 1500., (n_sat, 3))
    radial = np.sum((sat_vel - np.array([3., -2., 1.5])) * directions, axis=1) + 8.
    doppler = -(radial + .1 * np.cos(np.arange(n_sat))) / .19
    weights = rng.uniform(.1, 1., n_sat)
    return states, (satellites, sat_vel, doppler, weights)


def _matrix_reference(states, log_weights, observations, sigma=.5):
    """Weighted centering plus scalar Gaussian updates using matrix products."""
    result = states.copy()
    likelihood = log_weights.copy()
    sat, sat_vel, doppler, weights = observations
    for i in range(len(states)):
        ranges = np.linalg.norm(sat - states[i, :3], axis=1)
        valid = (weights > 0.) & (ranges >= 1.)
        if not np.any(valid):
            continue
        w = weights[valid]
        los = (sat[valid] - states[i, :3]) / ranges[valid, None]
        h = -los
        obs = -doppler[valid] * .19 - np.sum(sat_vel[valid] * los, axis=1)
        h -= np.sum(w[:, None] * h, axis=0) / np.sum(w)
        obs -= np.sum(w * obs) / np.sum(w)
        mu = result[i, 4:7].copy()
        covariance = result[i, 7:].reshape(3, 3).copy()
        for row, measurement, weight in zip(h, obs, w):
            ph = covariance @ row
            variance = row @ ph + sigma**2 / weight
            if not np.isfinite(variance) or variance <= 1e-18:
                continue
            innovation = measurement - row @ mu
            gain = ph / variance
            mu += gain * innovation
            covariance -= np.outer(gain, row @ covariance)
            covariance = .5 * (covariance + covariance.T)
            diagonal = np.diag(covariance).copy()
            np.fill_diagonal(covariance, np.maximum(diagonal, 1e-12))
            likelihood[i] -= .5 * (innovation**2 / variance + np.log(variance))
        result[i, 4:7] = mu
        result[i, 7:] = covariance.ravel()
    return result, likelihood


def _update(state, observations):
    sat, velocity, doppler, weights = observations
    native.pf_device_doppler_kf_update(state, sat, velocity, doppler, weights, len(weights), .19, .5)


def _read(state, count):
    states = native.pf_device_get_particle_states(state)
    log_weights = np.empty(count)
    native.pf_device_get_log_weights(state, log_weights)
    return states, log_weights


@pytest.mark.parametrize("count", [1, 33, 257])
@pytest.mark.parametrize("n_sat", [1, 4, 12, 16, 17, 32, 33, 64, 65, 128])
def test_kf_geometry_cache_boundaries_against_matrix_reference(count, n_sat):
    before, observations = _scenario(count, n_sat)
    log_weights = np.linspace(-2., 0., count)
    expected, expected_weights = _matrix_reference(before, log_weights, observations)
    state = native.pf_device_create(count)
    try:
        native.pf_device_set_particle_states(state, before)
        native.pf_device_set_log_weights(state, log_weights)
        _update(state, observations)
        actual, weights = _read(state, count)
        np.testing.assert_array_equal(actual[:, :4].view(np.uint64), before[:, :4].view(np.uint64))
        np.testing.assert_allclose(actual[:, 4:], expected[:, 4:], rtol=2e-10, atol=2e-10)
        np.testing.assert_allclose(weights, expected_weights, rtol=2e-10, atol=2e-10)
    finally:
        native.pf_device_destroy(state)


def test_zero_weights_leave_every_bit_unchanged():
    before, observations = _scenario(33, 32)
    observations[3][:] = 0.
    log_weights = np.linspace(-2., 0., 33)
    log_weights[::3] = -0.0
    state = native.pf_device_create(33)
    try:
        native.pf_device_set_particle_states(state, before)
        native.pf_device_set_log_weights(state, log_weights)
        _update(state, observations)
        actual, weights = _read(state, 33)
        np.testing.assert_array_equal(actual.view(np.uint64), before.view(np.uint64))
        np.testing.assert_array_equal(weights.view(np.uint64), log_weights.view(np.uint64))
    finally:
        native.pf_device_destroy(state)


def test_large_observation_shared_memory_limit_is_preserved():
    # The former 8 * n_sat FP64 shared workspace fits exactly in 48 KiB.
    # Extra shared variance storage must be disabled for this fallback path.
    before, observations = _scenario(1, 768)
    expected, expected_weights = _matrix_reference(before, np.zeros(1), observations)
    state = native.pf_device_create(1)
    try:
        native.pf_device_set_particle_states(state, before)
        _update(state, observations)
        actual, weights = _read(state, 1)
        np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=2e-10)
        np.testing.assert_allclose(weights, expected_weights, rtol=2e-10, atol=2e-10)
    finally:
        native.pf_device_destroy(state)


def test_near_satellites_and_masked_observations():
    before, observations = _scenario(33, 17)
    sat, _, _, weights = observations
    before[:3, :3] = 0.
    sat[:3] = [[.5, 0., 0.], [1., 0., 0.], [2., 0., 0.]]
    weights[3:6] = [0., -1., 0.]
    expected, expected_weights = _matrix_reference(before, np.zeros(33), observations)
    state = native.pf_device_create(33)
    try:
        native.pf_device_set_particle_states(state, before)
        _update(state, observations)
        actual, weights = _read(state, 33)
        np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=2e-10)
        np.testing.assert_allclose(weights, expected_weights, rtol=2e-10, atol=2e-10)
    finally:
        native.pf_device_destroy(state)


def test_chained_updates_scratch_growth_and_input_output_ownership():
    count = 257
    before, observations = _scenario(count, 16)
    expected, expected_weights = _matrix_reference(before, np.zeros(count), observations)
    state = native.pf_device_create(count)
    try:
        native.pf_device_set_particle_states(state, before)
        _update(state, observations)
        # Native staging must finish borrowing inputs before this call returns.
        for array in observations:
            array[:] = 0.
        saved, saved_weights = _read(state, count)
        np.testing.assert_allclose(saved, expected, rtol=2e-10, atol=2e-10)
        saved_copy = saved.copy()
        saved_weights_copy = saved_weights.copy()
        for n_sat in (32, 33, 128, 12):
            _, observations = _scenario(count, n_sat)
            expected, expected_weights = _matrix_reference(expected, expected_weights, observations)
            _update(state, observations)
        actual, weights = _read(state, count)
        np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=2e-10)
        np.testing.assert_allclose(weights, expected_weights, rtol=2e-10, atol=2e-10)
        np.testing.assert_array_equal(saved.view(np.uint64), saved_copy.view(np.uint64))
        np.testing.assert_array_equal(saved_weights.view(np.uint64), saved_weights_copy.view(np.uint64))
        native.pf_device_resample_megopolis_coalesced(state, 7, 42)
        after = native.pf_device_get_particle_states(state)
        lookup = {row[0].tobytes(): i for i, row in enumerate(actual)}
        indices = np.array([lookup[row[0].tobytes()] for row in after])
        np.testing.assert_array_equal(after.view(np.uint64), actual[indices].view(np.uint64))
    finally:
        native.pf_device_destroy(state)
