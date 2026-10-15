"""Numerical and lifetime contracts for device-side PF diagnostic reductions."""

import numpy as np
import pytest

from gnss_gpu.particle_filter_device import ParticleFilterDevice

native = pytest.importorskip("gnss_gpu._gnss_gpu_pf_device")
pytestmark = [pytest.mark.cuda, pytest.mark.gpu]


def _cloud(count):
    rng = np.random.default_rng(10 + count)
    states = rng.normal(size=(count, 16))
    states[:, :3] += [-3957199., 3310205., 3737911.]
    states[:, 3] += 3000.
    return states


def _query(state, center):
    return (native.pf_device_ess(state), native.pf_device_estimate(state),
            native.pf_device_position_spread(state, *center))


@pytest.mark.parametrize("count", [1, 31, 32, 33, 63, 64, 65, 127, 128, 129,
                                   255, 256, 257, 511, 512, 513, 8191, 8192, 8193,
                                   65536, 65537, 100000])
@pytest.mark.parametrize("distribution", ["uniform", "normal", "shifted", "dominant"])
def test_fp64_diagnostics_match_independent_weighted_reference(count, distribution):
    states = _cloud(count)
    weights = np.random.default_rng(count).normal(0., 3., count)
    if distribution == "uniform":
        weights.fill(0.)
    elif distribution == "shifted":
        weights -= 1e6
    elif distribution == "dominant":
        weights.fill(-1000.)
        weights[count // 2] = 0.
    normalized = np.exp(weights - np.max(weights))
    normalized /= normalized.sum()
    center = np.array([-3957201., 3310204., 3737913.])
    expected_mean = normalized @ states[:, :4]
    expected_ess = 1. / (normalized @ normalized)
    expected_spread = np.sqrt(normalized @ np.sum((states[:, :3] - center) ** 2, axis=1))
    state = native.pf_device_create(count)
    try:
        native.pf_device_set_particle_states(state, states)
        native.pf_device_set_log_weights(state, weights)
        ess, mean, spread = _query(state, center)
        np.testing.assert_allclose(mean, expected_mean, rtol=5e-13, atol=1e-9)
        np.testing.assert_allclose(ess, expected_ess, rtol=5e-13, atol=1e-10)
        np.testing.assert_allclose(spread, expected_spread, rtol=5e-13, atol=1e-12)
        # All three queries share scratch. Repeating them in a different order
        # must neither overwrite a returned array nor mutate particle data.
        snapshot = mean.copy()
        repeated_spread = native.pf_device_position_spread(state, *center)
        repeated_mean = native.pf_device_estimate(state)
        repeated_ess = native.pf_device_ess(state)
        np.testing.assert_array_equal(mean.view(np.uint64), snapshot.view(np.uint64))
        np.testing.assert_array_equal(repeated_mean.view(np.uint64), snapshot.view(np.uint64))
        np.testing.assert_array_equal(np.array([repeated_ess, repeated_spread]).view(np.uint64),
                                      np.array([ess, spread]).view(np.uint64))
        np.testing.assert_array_equal(native.pf_device_get_particle_states(state).view(np.uint64),
                                      states.view(np.uint64))
        actual_weights = np.empty(count)
        native.pf_device_get_log_weights(state, actual_weights)
        np.testing.assert_array_equal(actual_weights.view(np.uint64), weights.view(np.uint64))
    finally:
        native.pf_device_destroy(state)


@pytest.mark.parametrize("count", [1, 33, 257, 8193])
@pytest.mark.parametrize("nonfinite", [np.nan, np.inf, -np.inf])
def test_nonfinite_weight_contract(count, nonfinite):
    state = native.pf_device_create(count)
    try:
        native.pf_device_set_particle_states(state, _cloud(count))
        native.pf_device_set_log_weights(state, np.full(count, nonfinite))
        ess, mean, spread = _query(state, np.zeros(3))
        assert np.isnan(ess) and np.all(np.isnan(mean))
        assert spread == 0.
    finally:
        native.pf_device_destroy(state)


@pytest.mark.parametrize("count", [31, 32, 33, 255, 256, 257, 8191, 8192, 8193])
@pytest.mark.parametrize("distribution", ["uniform", "dominant", "extreme"])
def test_nonfinite_positions_keep_nan_estimate_and_spread_guard(count, distribution):
    states = _cloud(count)
    states[:, :4] = (np.resize([np.nan, np.inf, -np.inf, -0.], count)[:, None]
                     * np.array([1., -1., 1., -1.]))
    weights = np.zeros(count)
    expected_ess = count
    if distribution == "dominant":
        weights.fill(-1000.)
        weights[count // 2] = 0.
        expected_ess = 1
    elif distribution == "extreme":
        weights = np.resize([0., 1e300, -1e300, 1e-310], count)
        expected_ess = (count + 2) // 4
    state = native.pf_device_create(count)
    try:
        native.pf_device_set_particle_states(state, states)
        native.pf_device_set_log_weights(state, weights)
        ess, mean, spread = _query(state, np.zeros(3))
        assert ess == expected_ess
        assert np.all(np.isnan(mean))
        assert spread == 0.
        np.testing.assert_array_equal(native.pf_device_get_particle_states(state).view(np.uint64),
                                      states.view(np.uint64))
        actual_weights = np.empty(count)
        native.pf_device_get_log_weights(state, actual_weights)
        np.testing.assert_array_equal(actual_weights.view(np.uint64), weights.view(np.uint64))
    finally:
        native.pf_device_destroy(state)


def test_large_cancellation_and_signed_zero():
    count = 513
    state = native.pf_device_create(count)
    try:
        states = np.zeros((count, 16))
        states[:, :4] = (np.resize([1e100, -1e100, 1., -1.], count)[:, None]
                         * np.array([1., -1., 1., -1.]))
        native.pf_device_set_particle_states(state, states)
        first = native.pf_device_estimate(state)
        for _ in range(3):
            native.pf_device_ess(state)
            native.pf_device_position_spread(state, 0., 0., 0.)
            np.testing.assert_array_equal(native.pf_device_estimate(state).view(np.uint64),
                                          first.view(np.uint64))
        states[:, :4] = -0.
        native.pf_device_set_particle_states(state, states)
        result = native.pf_device_estimate(state)
        assert np.all(result == 0.)
        assert native.pf_device_ess(state) == count
        assert native.pf_device_position_spread(state, -0., 0., -0.) == 0.
    finally:
        native.pf_device_destroy(state)


def test_public_threshold_and_default_center_queries():
    count = 8193
    pf = ParticleFilterDevice(n_particles=count, seed=42)
    pf.initialize([1., 2., 3.], spread_pos=3., spread_cb=1.)
    try:
        before = pf.get_particle_states()
        assert pf.get_ess() == count
        assert not pf.resample_if_needed()
        center = pf.estimate()[:3]
        assert pf.get_position_spread() == pf.get_position_spread(center)
        weights = np.full(count, -1000.)
        weights[count // 2] = 0.
        pf.set_log_weights(weights)
        assert pf.get_ess() == 1.
        assert pf.resample_if_needed()
        assert pf.get_ess() == count
        np.testing.assert_array_equal(pf.get_log_weights(), np.zeros(count))
        assert np.all(np.isfinite(pf.estimate()))
        assert pf.get_position_spread() >= 0.
        assert np.array_equal(before, pf.get_particle_states()) is False
    finally:
        native.pf_device_destroy(pf._state)
