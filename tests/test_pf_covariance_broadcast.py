"""Covariance broadcast preserves bits and the original synchronous setter contract."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'benchmarks'))
from pf_covariance_broadcast_cases import COUNTS, PREDICTIONS, covariance, initial_arrays, invalid_covariances, representation

from gnss_gpu.particle_filter_device import ParticleFilterDevice

native = pytest.importorskip('gnss_gpu._gnss_gpu_pf_device')
pytestmark = [pytest.mark.gpu, pytest.mark.cuda]


def _new(count):
    pf = ParticleFilterDevice(n_particles=count, sigma_pos=1., sigma_cb=3., seed=42)
    pf.initialize([6378137., 0., 0.], clock_bias=3000., spread_pos=10., spread_cb=30.,
                  velocity=[1., 2., 3.], spread_vel=.5, velocity_init_sigma=2.)
    return pf


def _close(*filters):
    for pf in filters:
        native.pf_device_destroy(pf._state)
        pf._state = None


def _snapshot(pf):
    return pf.get_particle_states(), pf.get_log_weights()


def _bits_pair(left, right):
    for a, b in zip(left, right, strict=True):
        np.testing.assert_array_equal(a.view(np.uint64), b.view(np.uint64))


def _restore(pf, states, weights):
    pf.set_particle_states(states)
    pf.set_log_weights(weights)


@pytest.mark.parametrize('count', COUNTS)
@pytest.mark.parametrize('case', range(8))
@pytest.mark.parametrize('form', range(3))
def test_setter_and_prediction_match_retained_roundtrip(count, case, form):
    actual, original = _new(count), _new(count)
    original._pf_device_set_velocity_covariance = None
    try:
        states, weights = initial_arrays(actual.get_particle_states())
        matrix = covariance(case)
        arg = representation(matrix, form)
        for pf in (actual, original):
            _restore(pf, states, weights)
            pf.set_velocity_covariance(arg)
        arg.fill(-999.)
        a, b = _snapshot(actual), _snapshot(original)
        _bits_pair(a, b)
        np.testing.assert_array_equal(a[0][:, :7].view(np.uint64), states[:, :7].view(np.uint64))
        np.testing.assert_array_equal(a[0][:, 7:].view(np.uint64), np.broadcast_to(matrix.ravel(), (count, 9)).view(np.uint64))
        np.testing.assert_array_equal(a[1].view(np.uint64), weights.view(np.uint64))
        for alpha, dt, sp, sc, q, sv in PREDICTIONS:
            for pf in (actual, original):
                _restore(pf, states, weights)
                pf.set_velocity_covariance(representation(matrix, form))
                pf.sigma_cb = sc
                pf.predict([1., 2., 3.], dt=dt, sigma_pos=sp, sigma_vel=sv,
                    velocity_guide_alpha=alpha, rbpf_velocity_kf=True, velocity_process_noise=q)
            _bits_pair(_snapshot(actual), _snapshot(original))
    finally:
        _close(actual, original)


@pytest.mark.parametrize('count', COUNTS)
def test_repeated_broadcast_after_kf_and_resampling(count):
    actual, original = _new(count), _new(count)
    original._pf_device_set_velocity_covariance = None
    try:
        for step, satellites in enumerate((8, 12, 33, 64, 65, 16), 1):
            matrix = covariance(step % 4)
            for pf in (actual, original):
                pf.set_velocity_covariance(matrix)
                pf.predict([1.+step, 2.-step, 3./step], dt=.17, velocity_guide_alpha=.3,
                    rbpf_velocity_kf=True, velocity_process_noise=.13)
                directions = np.random.default_rng(satellites).normal(size=(satellites, 3))
                directions /= np.linalg.norm(directions, axis=1)[:, None]
                positions = np.ascontiguousarray([6378137., 0., 0.]+directions*2.5e7)
                velocities = np.zeros((satellites, 3))
                doppler = np.linspace(-10., 10., satellites)
                weights = np.ones(satellites)
                pf.update_doppler_kf(positions, velocities, doppler, weights=weights, resample=False)
                positions.fill(-999.)
                if step % 2 == 0:
                    native.pf_device_resample_megopolis_coalesced(pf._state, 7, 42+step)
            _bits_pair(_snapshot(actual), _snapshot(original))
    finally:
        _close(actual, original)


@pytest.mark.parametrize('count', COUNTS)
@pytest.mark.parametrize('arg', invalid_covariances())
def test_invalid_covariance_rejected_without_state_changes(count, arg):
    pf = _new(count)
    try:
        before = _snapshot(pf)
        with pytest.raises((ValueError, TypeError)):
            pf.set_velocity_covariance(arg)
        _bits_pair(_snapshot(pf), before)
    finally:
        _close(pf)


@pytest.mark.parametrize('count', COUNTS)
def test_uninitialized_setter_keeps_original_error(count):
    pf = ParticleFilterDevice(n_particles=count)
    try:
        with pytest.raises(RuntimeError, match='not initialized'):
            pf.set_velocity_covariance(np.eye(3))
    finally:
        _close(pf)


def test_injected_legacy_backend_retains_roundtrip_fallback():
    from types import SimpleNamespace
    from gnss_gpu.pf_device_config import _BINDING_ATTRS, init_pf_device_config

    states = np.arange(48, dtype=np.float64).reshape(3, 16)
    weights = np.array([-0., -np.inf, np.nan])
    before_states, before_weights = states.copy(), weights.copy()
    bindings = SimpleNamespace(**{name: (lambda *args: None) for name in _BINDING_ATTRS})
    bindings.pf_device_create = lambda count: object()
    bindings.pf_device_get_particle_states = lambda state: states.copy()
    bindings.pf_device_get_log_weights = lambda state, out: out.__setitem__(slice(None), weights)

    def upload_states(state, arr):
        states[:] = arr
        weights[:] = 0.

    bindings.pf_device_set_particle_states = upload_states
    bindings.pf_device_set_log_weights = lambda state, arr: weights.__setitem__(slice(None), arr)
    pf = ParticleFilterDevice.__new__(ParticleFilterDevice)
    init_pf_device_config(pf, n_particles=3, bindings=bindings)
    pf._initialized = True
    assert getattr(pf, '_pf_device_set_velocity_covariance', None) is None
    matrix = covariance(2)
    pf.set_velocity_covariance(matrix)
    np.testing.assert_array_equal(states[:, :7].view(np.uint64), before_states[:, :7].view(np.uint64))
    np.testing.assert_array_equal(states[:, 7:].view(np.uint64), np.broadcast_to(matrix.ravel(), (3, 9)).view(np.uint64))
    np.testing.assert_array_equal(weights.view(np.uint64), before_weights.view(np.uint64))
    pf._state = None


@pytest.mark.parametrize('bad', [np.zeros(9), np.zeros((3, 2)), np.zeros((1, 9)),
    np.full((3, 3), np.nan), np.full((3, 3), np.inf), np.full((3, 3), -np.inf)])
def test_native_input_validation_preserves_state(bad):
    setter = getattr(native, 'pf_device_set_velocity_covariance', None)
    if setter is None:
        pytest.skip('Preceding native module has no dedicated setter')
    pf = _new(129)
    try:
        before = _snapshot(pf)
        with pytest.raises(RuntimeError):
            setter(pf._state, bad)
        _bits_pair(_snapshot(pf), before)
    finally:
        _close(pf)


@pytest.mark.parametrize('destroyed', [False, True])
def test_native_rejects_null_or_destroyed_state(destroyed):
    setter = getattr(native, 'pf_device_set_velocity_covariance', None)
    if setter is None:
        pytest.skip('Preceding native module has no dedicated setter')
    state = native.pf_device_create(1) if destroyed else None
    if destroyed:
        native.pf_device_destroy(state)
    with pytest.raises(RuntimeError, match='state'):
        setter(state, np.eye(3))


@pytest.mark.parametrize('form', range(5))
def test_native_accepts_conversions_and_preserves_weight_bits(form):
    setter = getattr(native, 'pf_device_set_velocity_covariance', None)
    if setter is None:
        pytest.skip('Preceding native module has no dedicated setter')
    pf = _new(257)
    try:
        states, weights = initial_arrays(pf.get_particle_states())
        _restore(pf, states, weights)
        matrix = covariance(0)
        arg = representation(matrix, form) if form < 3 else matrix.astype(np.float32) if form == 3 else matrix.tolist()
        expected = np.asarray(arg, dtype=np.float64).ravel().copy()
        setter(pf._state, arg)
        out, actual_weights = _snapshot(pf)
        np.testing.assert_array_equal(out[:, :7].view(np.uint64), states[:, :7].view(np.uint64))
        np.testing.assert_array_equal(out[:, 7:].view(np.uint64), np.broadcast_to(expected, (257, 9)).view(np.uint64))
        np.testing.assert_array_equal(actual_weights.view(np.uint64), weights.view(np.uint64))
    finally:
        _close(pf)
