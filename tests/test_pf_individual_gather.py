"""Former covariance sources can be reused only after all nine values are copied."""

import numpy as np
import pytest

from test_pf_packed_gather import _assert_output, _cloud, _offsets, native
from test_pf_covariance_resampling import _assert_gather

pytestmark = [pytest.mark.gpu, pytest.mark.cuda]


@pytest.mark.parametrize('count', [1, 33, 129, 257, 513, 8193, 99999, 100000, 100001])
@pytest.mark.parametrize('seed', [0, 42, 2**64-1])
def test_queued_individual_covariance_matches_independent_ancestry(count, seed):
    original = _cloud(count)
    indices = np.arange(count)
    state = native.pf_device_create(count)
    try:
        native.pf_device_set_particle_states(state, original)
        for iterations in (1, 60, 0, 257, 4, 61):
            if iterations:
                previous = indices.copy()
                for block in range((count+255)//256):
                    offset = _offsets(count, block, seed, iterations)[-1]
                    first, last = block*256, min(count, (block+1)*256)
                    indices[first:last] = previous[(np.arange(first, last)+offset) % count]
            native.pf_device_set_log_weights(state, np.zeros(count))
            native.pf_device_resample_megopolis_coalesced(state, iterations, seed)
        _assert_output(state, original, indices)
    finally:
        native.pf_device_destroy(state)


@pytest.mark.parametrize('count', [33, 257, 513, 100001])
def test_source_reuse_then_legacy_systematic_and_reinitialize(count):
    state = native.pf_device_create(count)
    try:
        native.pf_device_set_particle_states(state, _cloud(count))
        methods = ('individual', 'legacy', 'individual', 'systematic', 'individual', 'common', 'individual')
        for step, method in enumerate(methods):
            if method == 'common':
                native.pf_device_initialize(state, -3957199., 3310205., 3737911.,
                    3000., 10., 30., 42, 1., 2., 3., .5, 2.)
            elif step == len(methods)-1:
                # Only the last cycle reimports; earlier cycles keep the
                # previously reused covariance buffers and pointer swaps.
                native.pf_device_set_particle_states(state, _cloud(count))
            before = native.pf_device_get_particle_states(state)
            weights = np.zeros(count)
            if method == 'systematic':
                weights[:] = -np.inf
                weights[[count//3, 2*count//3]] = 0.
            native.pf_device_set_log_weights(state, weights)
            if method == 'legacy':
                native.pf_device_resample_megopolis(state, 3, 42)
            elif method == 'systematic':
                native.pf_device_resample_systematic(state, 42)
            else:
                native.pf_device_resample_megopolis_coalesced(state, 60, 42)
            after = native.pf_device_get_particle_states(state)
            _assert_gather(before, after)
    finally:
        native.pf_device_destroy(state)
