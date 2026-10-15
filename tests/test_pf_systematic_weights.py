"""Systematic weight preparation: exact sparse CDF choices and state ownership."""

import numpy as np
import pytest

native = pytest.importorskip("gnss_gpu._gnss_gpu_pf_device")
pytestmark = [pytest.mark.gpu, pytest.mark.cuda]


def _states(count):
    result = np.random.default_rng(count).normal(size=(count, 16))
    result[:, 0] = np.arange(count)
    result[::17, 4] = -0.0
    return result


def _ancestors(state):
    result = np.empty(state.n_particles, dtype=np.int32)
    native.pf_device_get_resample_ancestors(state, result)
    return result


def _assert_copied(state, expected):
    result = native.pf_device_get_particle_states(state)
    np.testing.assert_array_equal(result.view(np.uint64), expected.view(np.uint64))
    weights = np.empty(state.n_particles)
    native.pf_device_get_log_weights(state, weights)
    np.testing.assert_array_equal(weights.view(np.uint64), np.zeros(state.n_particles).view(np.uint64))


def _reference_indices(cdf, seed):
    count = len(cdf)
    # Preserve the existing native target definition. Use counts <= 2,048 so
    # its int denominator is in range; this test does not change the algorithm.
    u0 = (seed % 1000000) / (1000000 * count)
    return np.searchsorted(cdf, (u0 + np.arange(count)) / count, side="left")


@pytest.mark.parametrize("count", [1, 31, 32, 33, 255, 256, 257, 511, 512, 513, 2048])
@pytest.mark.parametrize("seed", [0, 1, 999999])
def test_sparse_dyadic_weights_match_reference_ancestors(count, seed):
    state = native.pf_device_create(count)
    original = _states(count)
    support = np.unique([count // 3, 2 * count // 3])
    log_weights = np.full(count, -np.inf)
    log_weights[support] = 0.
    probabilities = np.zeros(count)
    probabilities[support] = 1. / len(support)
    expected = _reference_indices(np.cumsum(probabilities), seed)
    try:
        native.pf_device_set_particle_states(state, original)
        native.pf_device_set_log_weights(state, log_weights)
        native.pf_device_resample_systematic(state, seed)
        np.testing.assert_array_equal(_ancestors(state), expected)
        _assert_copied(state, original[expected])
    finally:
        native.pf_device_destroy(state)


@pytest.mark.parametrize("count", [257, 8193, 65537, 100_000])
def test_underflow_and_partial_blocks_copy_all_state_fields(count):
    state = native.pf_device_create(count)
    original = _states(count)
    weights = np.resize(np.array([0., -1., -708., -744., -745., -746., -1000., -np.inf]), count)
    try:
        native.pf_device_set_particle_states(state, original)
        native.pf_device_set_log_weights(state, weights)
        native.pf_device_resample_systematic(state, 42)
        ancestors = _ancestors(state)
        assert np.all((ancestors >= 0) & (ancestors < count))
        _assert_copied(state, original[ancestors])
    finally:
        native.pf_device_destroy(state)


def test_chained_systematic_queries_reuse_workspace_in_stream_order():
    count = 512
    state = native.pf_device_create(count)
    original = _states(count)
    log_weights = np.full(count, -np.inf)
    log_weights[[100, 400]] = 0.
    first_cdf = np.zeros(count)
    first_cdf[100:400] = .5
    first_cdf[400:] = 1.
    first = _reference_indices(first_cdf, 42)
    second = _reference_indices(np.arange(1, count + 1) / count, 0)
    try:
        native.pf_device_set_particle_states(state, original)
        native.pf_device_set_log_weights(state, log_weights)
        native.pf_device_resample_systematic(state, 42)
        native.pf_device_resample_systematic(state, 0)
        np.testing.assert_array_equal(_ancestors(state), second)
        _assert_copied(state, original[first[second]])
        retained = native.pf_device_get_particle_states(state)
        copy = retained.copy()
        native.pf_device_resample_megopolis_coalesced(state, 15, 1)
        native.pf_device_resample_systematic(state, 42)
        np.testing.assert_array_equal(retained.view(np.uint64), copy.view(np.uint64))
    finally:
        native.pf_device_destroy(state)
    assert not state.allocated
