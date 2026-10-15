"""Larger Doppler caches preserve the independent retained GPU path exactly."""

import numpy as np
import pytest

from test_pf_doppler_weight_cache import (
    _assert_bits, _compare, _read, _scenario, _uncached, _update, native,
)

pytestmark = [pytest.mark.gpu, pytest.mark.cuda]


@pytest.mark.parametrize('count', [3, 33, 257])
@pytest.mark.parametrize('satellites', [17, 23, 24, 25, 31, 32])
@pytest.mark.parametrize('gain', [0., .25])
def test_larger_cache_boundaries_and_tails_match_original_gpu(count, satellites, gain):
    _compare(*_scenario(count, satellites), gain=gain)


@pytest.mark.parametrize('satellites', [17, 32])
@pytest.mark.parametrize('gain', [0., .25])
@pytest.mark.parametrize('variant', ['masked', 'empty', 'range_cutoff', 'signed_zero', 'tiny_sigma', 'ecef'])
def test_large_cache_geometry_and_weights_keep_all_bits(satellites, gain, variant):
    states, prior, observations = _scenario(33, satellites, seed=517)
    sat, vel, doppler, weights = observations
    sigma = .5
    center = np.zeros(3)
    if variant == 'masked':
        weights[::3] = 0.
        weights[1::3] = -1.
    elif variant == 'empty':
        weights[:] = 0.
    elif variant == 'range_cutoff':
        states[:3, :3] = 0.
        sat[:3] = [[np.nextafter(1., 0.), 0., 0.], [1., 0., 0.], [np.nextafter(1., np.inf), 0., 0.]]
    elif variant == 'signed_zero':
        states[:, 4:7] = -0.
        prior[::2] = -0.
        vel[:] = -0.
        doppler[::2] = -0.
    elif variant == 'tiny_sigma':
        sigma = 1e-200
    elif variant == 'ecef':
        center = np.array([6378137., -4500000., 3500000.])
        states[:, :3] += center
        sat += center
    _compare(states, prior, observations, gain=gain, sigma=sigma, center=center)


@pytest.mark.parametrize('name,value', [
    ('gain', -1.), ('gain', np.inf), ('gain', np.nan),
    ('maximum', 1e-300), ('maximum', np.nan),
    ('sigma', 1e-200), ('sigma', np.inf), ('sigma', np.nan),
    ('wavelength', np.inf), ('wavelength', np.nan),
])
def test_large_cache_scalar_extremes_match_retained_kernel(name, value):
    _compare(*_scenario(33, 32), **{name: value})


@pytest.mark.parametrize('count', [1, 33])
@pytest.mark.parametrize('gain', [0., .25])
def test_highest_validity_mask_bit_remains_visible(count, gain):
    states, prior, observations = _scenario(count, 32)
    sat, _, _, _ = observations
    sat[:-1] = 0.
    # Only bit31 is set: all other satellites are within1m of every particle.
    assert np.all(np.linalg.norm(states[:, :3], axis=1) < 1.)
    _compare(states, prior, observations, gain=gain)


def test_predict_growth_input_mutation_and_resampling_refresh_every_cache():
    states, prior, _ = _scenario(33, 32)
    owners = [native.pf_device_create(33) for _ in range(2)]
    try:
        for owner in owners:
            native.pf_device_set_particle_states(owner, states)
            native.pf_device_set_log_weights(owner, prior)
        retained = []
        for epoch, satellites in enumerate((16, 17, 24, 25, 32, 33, 48, 8), 1):
            for owner in owners:
                native.pf_device_predict(owner, 1., 2., 3., .1, 1e-12, 1e-12, 42, epoch)
            _, _, observations = _scenario(33, satellites, seed=237 + epoch)
            arrays = [a.copy() for a in observations]
            center = epoch * np.array([.1, .2, .3])
            oracle = _uncached(arrays, center) if satellites <= 64 else [a.copy() for a in arrays]
            gain = .25 if epoch % 2 else 0.
            _update(owners[0], arrays, gain)
            _update(owners[1], oracle, gain)
            for array in arrays:
                array.fill(-999.)
            if epoch % 3 == 0:
                for owner in owners:
                    native.pf_device_resample_megopolis_coalesced(owner, 7, 42)
            actual, expected = (_read(owner, 33) for owner in owners)
            _assert_bits(actual, expected)
            retained.append((actual, tuple(a.copy() for a in actual)))
            for previous, snapshot in retained:
                _assert_bits(previous, snapshot)
    finally:
        for owner in owners:
            native.pf_device_destroy(owner)
