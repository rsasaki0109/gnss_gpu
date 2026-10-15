"""Common covariance reuses the exact retained imported-state producer."""

import numpy as np
import pytest

native = pytest.importorskip('gnss_gpu._gnss_gpu_pf_device')
pytestmark = [pytest.mark.gpu, pytest.mark.cuda]

COUNTS = (1, 31, 32, 33, 255, 256, 257, 8193)
CASES = ((.3, .17, 1.3, 2.7, .13, 0.), (2., 0., 1.3, 2.7, 0., .2),
         (-1., 1e-300, 1e-300, 1e-300, -1., .2), (.3, .17, 1e-300, 2.7, .13, 0.),
         (.3, .17, 1e150, 1e150, 1e150, 0.), (.3, 1e154, 1.3, 2.7, .13, 0.),
         (.3, 1.17, 1.3, 2.7, -1., 0.))


def _initialize(state, sigma=2.):
    native.pf_device_initialize(state, -3957199., 3310205., 3737911., 3000.,
                                10., 30., 42, 1., 2., 3., .5, sigma)


def _snapshot(state):
    particles = native.pf_device_get_particle_states(state)
    weights = np.empty(state.n_particles)
    native.pf_device_get_log_weights(state, weights)
    return particles, weights


def _bits(actual, original):
    for a, b in zip(_snapshot(actual), _snapshot(original), strict=True):
        np.testing.assert_array_equal(a.view(np.uint64), b.view(np.uint64))


@pytest.mark.parametrize('count', COUNTS)
@pytest.mark.parametrize('sigma', [0., -0., 1e-300, 1e-150, 2., 1e150, 1e154, 1e308])
@pytest.mark.parametrize('case', CASES)
def test_common_factor_matches_retained_original(count, sigma, case):
    actual, original = native.pf_device_create(count), native.pf_device_create(count)
    try:
        _initialize(actual, sigma)
        initial = native.pf_device_get_particle_states(actual)
        # Import deliberately clears the common-covariance certificate.
        native.pf_device_set_particle_states(original, initial)
        alpha, dt, sp, sc, q, sv = case
        for state in (actual, original):
            native.pf_device_predict(state, 1., 2., 3., dt, sp, sc, 2**64-1, 13,
                                     sv, alpha, True, q)
        _bits(actual, original)
    finally:
        native.pf_device_destroy(actual)
        native.pf_device_destroy(original)


@pytest.mark.parametrize('count', COUNTS)
@pytest.mark.parametrize('seed', [0, 2**64-1])
def test_queued_changing_coefficients_preserve_rng_and_block_snapshot(count, seed):
    actual, original = native.pf_device_create(count), native.pf_device_create(count)
    try:
        _initialize(actual)
        native.pf_device_set_particle_states(original, native.pf_device_get_particle_states(actual))
        for step in range(1, 9):
            for state in (actual, original):
                native.pf_device_predict(state, 1.+step, 2.-step, 3./step, .01*step,
                                         1., 3., seed, step, .2, .3, True, .13)
        _bits(actual, original)
    finally:
        native.pf_device_destroy(actual)
        native.pf_device_destroy(original)


@pytest.mark.parametrize('count', COUNTS)
@pytest.mark.parametrize('seed', [0, 2**64-1])
@pytest.mark.parametrize('resampler', ['coalesced', 'systematic', 'legacy'])
def test_resampling_import_and_reinitialization_keep_all_state_fields(count, seed, resampler):
    actual, original = native.pf_device_create(count), native.pf_device_create(count)
    try:
        _initialize(actual)
        native.pf_device_set_particle_states(original, native.pf_device_get_particle_states(actual))
        if resampler == 'systematic':
            # Parallel CDF boundary ties can select different ancestors even
            # between two original builds. Compare prediction on one exact
            # post-resample state, with the imported oracle's guarantee cleared.
            native.pf_device_resample_systematic(actual, seed)
            native.pf_device_set_particle_states(original, native.pf_device_get_particle_states(actual))
            _bits(actual, original)
        else:
            for state in (actual, original):
                if resampler == 'coalesced':
                    native.pf_device_resample_megopolis_coalesced(state, 4, seed)
                else:
                    native.pf_device_resample_megopolis(state, 4, seed)
        for state in (actual, original):
            native.pf_device_predict(state, 1., 2., 3., .17, 1., 3., seed, 1,
                                     0., .3, True, .13)
        _bits(actual, original)
        imported = native.pf_device_get_particle_states(actual)
        imported[:, 7:] += np.arange(count)[:, None] * .01
        for state in (actual, original):
            native.pf_device_set_particle_states(state, imported)
            native.pf_device_predict(state, 1., 2., 3., .17, 1., 3., seed, 2,
                                     0., .3, True, .13)
        _bits(actual, original)
        _initialize(actual, 1.5)
        native.pf_device_set_particle_states(original, native.pf_device_get_particle_states(actual))
        for state in (actual, original):
            native.pf_device_predict(state, 1., 2., 3., .17, 1., 3., seed, 3,
                                     0., .3, True, .13)
        _bits(actual, original)
    finally:
        native.pf_device_destroy(actual)
        native.pf_device_destroy(original)
