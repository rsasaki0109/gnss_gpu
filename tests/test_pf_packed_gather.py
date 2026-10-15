"""Packed common-state rows preserve independent ancestors and all gathered bits."""

import numpy as np
import pytest

native = pytest.importorskip('gnss_gpu._gnss_gpu_pf_device')
pytestmark = [pytest.mark.gpu, pytest.mark.cuda]


def _philox(seed, subsequence, quartet):
    # Independent integer reference: absolute counter plus64-bit subsequence.
    mask = (1 << 32)-1
    words = [quartet & mask, quartet >> 32, subsequence & mask, subsequence >> 32]
    keys = [seed & mask, seed >> 32]
    for _ in range(10):
        first = 0xD2511F53*words[0]
        second = 0xCD9E8D57*words[2]
        words = [(second >> 32)^words[1]^keys[0], second & mask,
                 (first >> 32)^words[3]^keys[1], first & mask]
        keys = [(keys[0]+0x9E3779B9) & mask, (keys[1]+0xBB67AE85) & mask]
    return words


def _offsets(count, block, seed, iterations):
    return [value % count for quartet in range((iterations+3)//4)
            for value in _philox(seed, count+block, quartet)][:iterations]


def _cloud(count):
    states = np.random.default_rng(count).normal(size=(count, 16))
    states[:, 0] = np.arange(count)
    bits = states.view(np.uint64)
    bits[::7, 8] = 0x8000000000000000
    bits[::11, 12] = 0x7FF8000000000123
    return states


def _start(count, mode):
    state = native.pf_device_create(count)
    if mode == 'common':
        native.pf_device_initialize(state, -3957199., 3310205., 3737911.,
            3000., 10., 30., 42, 1., 2., 3., .5, 2.)
        original = native.pf_device_get_particle_states(state)
    else:
        original = _cloud(count)
        native.pf_device_set_particle_states(state, original)
    return state, original


def _assert_output(state, original, indices):
    actual = native.pf_device_get_particle_states(state)
    np.testing.assert_array_equal(actual.view(np.uint64), original[indices].view(np.uint64))
    weights = np.empty(len(original))
    native.pf_device_get_log_weights(state, weights)
    np.testing.assert_array_equal(weights.view(np.uint64), np.zeros(len(original)).view(np.uint64))


@pytest.mark.parametrize('mode', ['common', 'imported'])
@pytest.mark.parametrize('count', [1, 33, 255, 257, 513])
@pytest.mark.parametrize('iterations', [1, 3, 4, 5, 59, 60, 61, 255, 256, 257])
@pytest.mark.parametrize('seed', [0, 42, 2**64-1])
def test_uniform_weights_match_independent_philox_ancestors(count, iterations, seed, mode):
    # Every proposal is accepted, so the final ancestor identifies the exact
    # last scalar/vector draw, including tails and retained B257 fallback.
    indices = np.arange(count)
    for block in range((count+255)//256):
        offset = _offsets(count, block, seed, iterations)[-1]
        first, last = block*256, min(count, (block+1)*256)
        indices[first:last] = (indices[first:last]+offset) % count
    state, original = _start(count, mode)
    try:
        native.pf_device_set_log_weights(state, np.zeros(count))
        native.pf_device_resample_megopolis_coalesced(state, iterations, seed)
        _assert_output(state, original, indices)
    finally:
        native.pf_device_destroy(state)


@pytest.mark.parametrize('mode', ['common', 'imported'])
@pytest.mark.parametrize('count', [33, 257, 8193])
@pytest.mark.parametrize('iterations', [3, 4, 60, 61, 256])
@pytest.mark.parametrize('seed', [42, 2**64-1])
def test_single_finite_weight_retains_first_accepted_ancestor(count, iterations, seed, mode):
    selected = count//2
    indices = np.arange(count)
    for block in range((count+255)//256):
        offsets = _offsets(count, block, seed, iterations)
        for particle in range(block*256, min(count, (block+1)*256)):
            if any((particle+offset) % count == selected for offset in offsets):
                indices[particle] = selected
    weights = np.full(count, -np.inf)
    weights[selected] = 0.
    state, original = _start(count, mode)
    try:
        native.pf_device_set_log_weights(state, weights)
        native.pf_device_resample_megopolis_coalesced(state, iterations, seed)
        _assert_output(state, original, indices)
    finally:
        native.pf_device_destroy(state)


def test_queued_common_packing_preserves_positive_zero_and_covariance():
    count = 513
    created = [_start(count, 'common') for _ in range(2)]
    states = [entry[0] for entry in created]
    original = created[0][1]
    np.testing.assert_array_equal(original.view(np.uint64), created[1][1].view(np.uint64))
    try:
        indices = np.arange(count)
        for iterations, seed in ((60, 42), (257, 1), (1, 2**64-1), (256, 0), (0, 3), (61, 999999)):
            if iterations:
                previous = indices.copy()
                for block in range((count+255)//256):
                    offset = _offsets(count, block, seed, iterations)[-1]
                    first, last = block*256, min(count, (block+1)*256)
                    indices[first:last] = previous[(np.arange(first, last)+offset) % count]
            for state in states:
                native.pf_device_set_log_weights(state, np.zeros(count))
                native.pf_device_resample_megopolis_coalesced(state, iterations, seed)
                if state == states[0]:
                    native.pf_device_sync(state)
        for state in states:
            _assert_output(state, original, indices)
    finally:
        for state in states:
            native.pf_device_destroy(state)
