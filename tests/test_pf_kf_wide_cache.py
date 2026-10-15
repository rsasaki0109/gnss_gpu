"""Wide velocity KF geometry reuse keeps retained uncached GPU result bits."""

import numpy as np
import pytest

from test_pf_doppler_weight_cache import _assert_bits, _read, _scenario, _uncached, native

pytestmark = [pytest.mark.gpu, pytest.mark.cuda]


def _update(owner, observations, sigma=.5, wavelength=.19):
    native.pf_device_doppler_kf_update(owner, *observations, len(observations[3]), wavelength, sigma)


def _compare(states, prior, observations, sigma=.5, wavelength=.19):
    owners = [native.pf_device_create(len(states)) for _ in range(2)]
    try:
        for owner in owners:
            native.pf_device_set_particle_states(owner, states)
            native.pf_device_set_log_weights(owner, prior)
        # Zero-weight padding selects65+ uncached/block256. Each extra row is
        # skipped before geometry and ordered accumulation, including after KF.
        oracle = _uncached(observations, (0., 0., 0.))
        for _ in range(2):
            _update(owners[0], observations, sigma, wavelength)
            _update(owners[1], oracle, sigma, wavelength)
            _assert_bits(_read(owners[0], len(states)), _read(owners[1], len(states)))
    finally:
        for owner in owners:
            native.pf_device_destroy(owner)


@pytest.mark.parametrize('count', [1, 33, 129, 257])
@pytest.mark.parametrize('satellites', [33, 47, 48, 49, 63, 64])
def test_kf_wide_boundaries_and_tails_match_original_gpu(count, satellites):
    _compare(*_scenario(count, satellites))


@pytest.mark.parametrize('satellites', [33, 64])
@pytest.mark.parametrize('variant', ['masked', 'empty', 'range_cutoff', 'signed_zero', 'tiny_sigma', 'ecef'])
def test_kf_wide_geometry_and_weights_preserve_bits(satellites, variant):
    states, prior, observations = _scenario(33, satellites, seed=632)
    sat, vel, doppler, weights = observations
    sigma = .5
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
        offset = np.array([6378137., -4500000., 3500000.])
        states[:, :3] += offset
        sat += offset
    _compare(states, prior, observations, sigma=sigma)


@pytest.mark.parametrize('satellites', [33, 64])
@pytest.mark.parametrize('variant', ['spd', 'nonsymmetric', 'invalid_S', 'tiny', 'huge', 'nonfinite'])
def test_kf_wide_covariance_and_clamping_match_original_gpu(satellites, variant):
    states, prior, observations = _scenario(33, satellites, seed=729)
    rng = np.random.default_rng(719)
    factors = rng.normal(size=(33, 3, 3))
    states[:, 7:] = (factors @ factors.swapaxes(1, 2) + np.eye(3)*.1).reshape(33, 9)
    if variant == 'nonsymmetric':
        states[:, 7:] = [4., 1., .5, 9., 3., .25, -7., 8., 2.]
    elif variant == 'invalid_S':
        states[:, 7:] = -np.eye(3).ravel()*1e10
    elif variant == 'tiny':
        states[:, 7:] *= 1e-300
    elif variant == 'huge':
        states[:, 7:] *= 1e150
    elif variant == 'nonfinite':
        states[:, 7:] = np.resize([np.nan, np.inf, -np.inf, -0.], 33*9).reshape(33, 9)
    _compare(states, prior, observations)


@pytest.mark.parametrize('name,value', [
    ('sigma', 1e-200), ('sigma', np.inf), ('sigma', np.nan),
    ('wavelength', 1e-200), ('wavelength', np.inf), ('wavelength', np.nan),
    ('sigma', 0.),
])
def test_kf_wide_scalar_extremes_keep_retained_bits(name, value):
    _compare(*_scenario(33, 64), **{name: value})


@pytest.mark.parametrize('count', [1, 33])
def test_kf_wide_highest_geometry_mask_bit_is_visible(count):
    states, prior, observations = _scenario(count, 64)
    observations[0][:-1] = 0.
    assert np.all(np.linalg.norm(states[:, :3], axis=1) < 1.)
    _compare(states, prior, observations)


def test_kf_wide_queued_predict_growth_input_mutation_resample_and_ownership():
    states, prior, _ = _scenario(33, 64)
    owners = [native.pf_device_create(33) for _ in range(2)]
    try:
        for owner in owners:
            native.pf_device_set_particle_states(owner, states)
            native.pf_device_set_log_weights(owner, prior)
        retained = []
        for epoch, satellites in enumerate((32, 33, 48, 49, 64, 65, 96, 8), 1):
            for owner in owners:
                native.pf_device_predict(owner, 1., 2., 3., .1, 1., 3., 42, epoch, 0., .3, True, .1)
            _, _, observations = _scenario(33, satellites, seed=632+epoch)
            arrays = [a.copy() for a in observations]
            oracle = _uncached(arrays, (0., 0., 0.)) if satellites <= 64 else [a.copy() for a in arrays]
            _update(owners[0], arrays)
            _update(owners[1], oracle)
            for array in arrays:
                array.fill(-999.)
            if epoch % 3 == 0:
                for owner in owners:
                    native.pf_device_resample_megopolis_coalesced(owner, 7, 42)
            actual, expected = [_read(owner, 33) for owner in owners]
            _assert_bits(actual, expected)
            retained.append((actual, tuple(a.copy() for a in actual)))
            for previous, snapshot in retained:
                _assert_bits(previous, snapshot)
    finally:
        for owner in owners:
            native.pf_device_destroy(owner)
