"""Cached Doppler geometry matches the retained uncached GPU path bit for bit."""

import numpy as np
import pytest

native = pytest.importorskip('gnss_gpu._gnss_gpu_pf_device')
pytestmark = [pytest.mark.gpu, pytest.mark.cuda]


def _scenario(count, satellites, seed=237):
    rng = np.random.default_rng(seed + count + satellites)
    states = rng.normal(size=(count, 16))
    states[:, :3] = rng.uniform(-.1, .1, (count, 3))
    states[:, 7:] = np.eye(3).ravel()
    direction = rng.normal(size=(satellites, 3))
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    sat = direction * 26_560_000.
    vel = rng.normal(0., 1500., (satellites, 3))
    doppler = rng.normal(0., 7000., satellites)
    weights = rng.uniform(.1, 1., satellites)
    prior = rng.normal(size=count)
    return states, prior, (sat, vel, doppler, weights)


def _read(owner, count):
    weights = np.empty(count)
    states = native.pf_device_get_particle_states(owner)
    native.pf_device_get_log_weights(owner, weights)
    return states, weights


def _assert_bits(actual, expected):
    for before, after in zip(expected, actual, strict=True):
        np.testing.assert_array_equal(after.view(np.uint64), before.view(np.uint64))


def _update(owner, observations, gain=.25, sigma=.5, wavelength=.19, maximum=10.):
    native.pf_device_weight_doppler(owner, *observations, len(observations[3]),
                                  wavelength, sigma, gain, maximum)


def _uncached(observations, center):
    """65 observations force the retained kernel; padding is always range<1.

    The synthetic particles stay within .18m of center. The added satellites
    coincide with center and are skipped before any observation or ordered sum.
    Thus the oracle keeps the original observation count and sum order in effect.
    """
    sat, vel, doppler, weights = observations
    missing = 65 - len(weights)
    assert missing > 0
    return (np.vstack((sat, np.repeat(np.asarray(center)[None], missing, axis=0))),
            np.vstack((vel, np.zeros((missing, 3)))),
            np.r_[doppler, np.zeros(missing)], np.r_[weights, np.zeros(missing)])


def _compare(states, prior, observations, gain=.25, sigma=.5, wavelength=.19, maximum=10., center=(0., 0., 0.)):
    assert np.all(np.linalg.norm(states[:, :3] - center, axis=1) < 1.)
    actual_owner = native.pf_device_create(len(states))
    oracle_owner = native.pf_device_create(len(states))
    try:
        for owner in (actual_owner, oracle_owner):
            native.pf_device_set_particle_states(owner, states)
            native.pf_device_set_log_weights(owner, prior)
        padded = _uncached(observations, center)
        # The original implementation omits WLS for fewer than four satellites.
        oracle_gain = 0. if len(observations[3]) < 4 else gain
        for _ in range(2):
            _update(actual_owner, observations, gain, sigma, wavelength, maximum)
            _update(oracle_owner, padded, oracle_gain, sigma, wavelength, maximum)
            actual = _read(actual_owner, len(states))
            _assert_bits(actual, _read(oracle_owner, len(states)))
            np.testing.assert_array_equal(actual[0][:, :4].view(np.uint64), states[:, :4].view(np.uint64))
            np.testing.assert_array_equal(actual[0][:, 7:].view(np.uint64), states[:, 7:].view(np.uint64))
    finally:
        native.pf_device_destroy(actual_owner)
        native.pf_device_destroy(oracle_owner)


@pytest.mark.parametrize('count', [3, 33, 257])
@pytest.mark.parametrize('satellites', [1, 3, 4, 8, 9, 12, 16])
@pytest.mark.parametrize('gain', [0., .25])
def test_bounded_cache_and_odd_tails_match_uncached_gpu(count, satellites, gain):
    _compare(*_scenario(count, satellites), gain=gain)


@pytest.mark.parametrize('satellites', [8, 16])
@pytest.mark.parametrize('gain', [0., .25])
@pytest.mark.parametrize('variant', ['masked', 'empty', 'range_cutoff', 'signed_zero', 'tiny_sigma', 'ecef'])
def test_edge_geometry_and_weights_keep_every_bit(satellites, gain, variant):
    states, prior, observations = _scenario(33, satellites)
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
    ('gain', -1.), ('gain', -0.), ('gain', np.inf), ('gain', -np.inf), ('gain', np.nan),
    ('maximum', 0.), ('maximum', -1.), ('maximum', 1e-300), ('maximum', np.inf), ('maximum', np.nan),
    ('wavelength', 1e-300), ('wavelength', np.inf), ('wavelength', np.nan),
    ('sigma', 0.), ('sigma', -1.), ('sigma', np.inf), ('sigma', np.nan),
])
def test_scalar_extremes_retain_original_gpu_semantics(name, value):
    _compare(*_scenario(33, 12), **{name: value})


def test_refresh_after_prediction_growth_resampling_and_input_mutation():
    states, prior, _ = _scenario(33, 12)
    owners = [native.pf_device_create(33) for _ in range(2)]
    try:
        for owner in owners:
            native.pf_device_set_particle_states(owner, states)
            native.pf_device_set_log_weights(owner, prior)
        retained = []
        for epoch, satellites in enumerate((8, 9, 16, 17, 32, 4, 1, 12), 1):
            _, _, observations = _scenario(33, satellites, seed=237 + epoch)
            # Every particle receives the same shift, keeping padding within1m.
            center = np.array([epoch * .125, epoch * -.0625, epoch * .03125])
            for owner in owners:
                native.pf_device_shift_position(owner, .125, -.0625, .03125)
            arrays = [a.copy() for a in observations]
            padded = _uncached(arrays, center) if satellites <= 64 else [a.copy() for a in arrays]
            gain = 0. if satellites < 4 else .25
            _update(owners[0], arrays, gain)
            _update(owners[1], padded, gain)
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


def test_zero_observations_and_shared_memory_capacity_preserve_fallback():
    states, prior, _ = _scenario(1, 12)
    owner = native.pf_device_create(1)
    try:
        native.pf_device_set_particle_states(owner, states)
        native.pf_device_set_log_weights(owner, prior)
        observations = _scenario(1, 0)[2]
        _update(owner, observations)
        _assert_bits(_read(owner, 1), (states, prior))
        observations = _scenario(1, 768)[2]
        _update(owner, observations)
        native.pf_device_sync(owner)
        assert np.isfinite(_read(owner, 1)[1]).all()
    finally:
        native.pf_device_destroy(owner)
