"""Resampling retains common covariance and invalidates that fact after import/KF."""

import numpy as np
import pytest

native = pytest.importorskip("gnss_gpu._gnss_gpu_pf_device")


def _initialize(state):
    native.pf_device_initialize(state, -3957199., 3310205., 3737911.,
                                3000., 1000., 30., 42, 1., 2., 3., .5, 2.)


def _assert_gather(before, after):
    # Positions are unique and unaffected by resampling; check every bit,
    # including all covariance entries and negative zero imported by callers.
    lookup = {row[0].tobytes(): i for i, row in enumerate(before)}
    indices = np.array([lookup[row[0].tobytes()] for row in after])
    np.testing.assert_array_equal(after.view(np.uint64), before[indices].view(np.uint64))


@pytest.mark.parametrize("count", [1, 31, 33, 255, 256, 257, 8193])
@pytest.mark.parametrize("iterations", [0, 1, 2, 3, 4, 5, 6, 7, 8, 60])
def test_common_covariance_after_prediction_and_repeated_resampling(count, iterations):
    state = native.pf_device_create(count)
    try:
        _initialize(state)
        weights = np.random.default_rng(count).normal(size=count)
        for step in range(1, 4):
            native.pf_device_predict(state, 1., 2., 3., .1, 1., 3., 42, step,
                                    .0, .3, True, .7)
            before = native.pf_device_get_particle_states(state)
            np.testing.assert_array_equal(before[:, 7:], np.broadcast_to(before[0, 7:], (count, 9)))
            native.pf_device_set_log_weights(state, weights)
            native.pf_device_resample_megopolis_coalesced(state, iterations, 42 + step)
            after = native.pf_device_get_particle_states(state)
            _assert_gather(before, after)
    finally:
        native.pf_device_destroy(state)


@pytest.mark.parametrize("count", [33, 257, 8193])
def test_import_after_common_covariance_then_reinitialize(count):
    state = native.pf_device_create(count)
    try:
        _initialize(state)
        native.pf_device_resample_megopolis_coalesced(state, 5, 1)
        before = np.random.default_rng(9).normal(size=(count, 16))
        before[:, 0] = np.arange(count)
        before[::17, 8] = -0.0
        native.pf_device_set_particle_states(state, before)
        native.pf_device_set_log_weights(state, np.linspace(-4., 0., count))
        native.pf_device_resample_megopolis_coalesced(state, 7, 42)
        _assert_gather(before, native.pf_device_get_particle_states(state))
        _initialize(state)
        for resample in (
            lambda: native.pf_device_resample_megopolis_coalesced(state, 6, 42),
            lambda: native.pf_device_resample_megopolis(state, 3, 7),
            lambda: native.pf_device_resample_systematic(state, 42),
            lambda: native.pf_device_resample_megopolis_coalesced(state, 5, 1),
        ):
            before = native.pf_device_get_particle_states(state)
            native.pf_device_set_log_weights(state, np.linspace(-4., 0., count))
            resample()
            _assert_gather(before, native.pf_device_get_particle_states(state))
    finally:
        native.pf_device_destroy(state)


@pytest.mark.parametrize("count", [33, 257, 8193])
def test_doppler_kf_invalidates_common_covariance(count):
    state = native.pf_device_create(count)
    try:
        _initialize(state)
        native.pf_device_resample_megopolis_coalesced(state, 5, 42)
        satellites = np.array([[2.6e7, 0., 0.], [0., 2.6e7, 0.], [0., 0., 2.6e7]])
        native.pf_device_doppler_kf_update(state, satellites, np.zeros((3, 3)),
                                          np.zeros(3), np.ones(3), 3, .19, .2)
        before = native.pf_device_get_particle_states(state)
        assert np.any(before[:, 7:] != before[0, 7:])
        native.pf_device_set_log_weights(state, np.linspace(-4., 0., count))
        native.pf_device_resample_megopolis_coalesced(state, 8, 7)
        _assert_gather(before, native.pf_device_get_particle_states(state))
    finally:
        native.pf_device_destroy(state)
