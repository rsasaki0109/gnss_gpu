"""Private native readout reuse preserves every CDF/state bit and input owner."""

import math
from pathlib import Path
import sys
import types
import warnings

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'benchmarks'))
from bench_pf_covariance_broadcast_compare import _load_python, _new_pf
from pf_particle_readout_cases import bits_equal
from pf_pool_normalize_cases import original_cdf

pytestmark = [pytest.mark.cuda, pytest.mark.gpu]


@pytest.fixture(scope='module')
def backend():
    native = pytest.importorskip('gnss_gpu._gnss_gpu_pf_device')
    from gnss_gpu.particle_filter_device import ParticleFilterDevice
    return native, ParticleFilterDevice


def _values(count, variant):
    rng = np.random.default_rng(count+214)
    data = rng.normal(0., 3., count)
    if variant == 1:
        data -= 1e6
    elif variant == 2:
        data = rng.uniform(-745., 5., count)
    elif variant == 3:
        data[:] = np.resize([-1e308, 1e308], count)
    elif variant == 4:
        data[:] = np.resize([-0., 0., 5e-324, -5e-324, 1e-154], count)
    elif variant == 5:
        data[:] = np.resize([np.nan, np.inf, -np.inf, -1e308, 0.], count)
    elif variant == 6:
        data.view(np.uint64)[:] = rng.integers(0, 2**64-1, count, dtype=np.uint64)
    return data


def _check(backend, monkeypatch, count, mode='active', variant=0):
    from gnss_gpu.pf_device_runtime import ParticleFilterDeviceRuntime
    from gnss_gpu.realtime_runtime import PersistentParticleRuntimePool

    native, klass = backend
    source, target = _new_pf(klass, count), _new_pf(klass, count//2)
    prior_profile = sys.getprofile()
    returned, captured = [], []
    try:
        states = np.arange(count*16, dtype=np.float64).reshape(count, 16)
        states[::17, 8] = -0.
        data = _values(count, variant)
        source.set_particle_states(states)
        source.set_log_weights(data)
        caller_owned = source.get_log_weights()
        with np.errstate(all='ignore'):
            expected_cdf = original_cdf(data)
        thresholds = (np.arange(count//2, dtype=np.float64)+.5)/(count//2)
        indices = np.minimum(np.searchsorted(expected_cdf, thresholds, side='left'), count-1)
        actual_gather = target._pf_device_gather_from_cdf

        def cdf_gather(a, b, cdf):
            captured.append(cdf.copy())
            return actual_gather(a, b, cdf)

        for pf in (source, target):
            pf._pf_device_gather_from_cdf = cdf_gather
            pf._pf_device_uniform_cdf_checked_gather = cdf_gather
        if mode == 'python_getter':
            getter = source.get_log_weights
            source.get_log_weights = lambda: getter()
        elif mode == 'native_getter':
            getter = source._pf_device_get_log_weights
            source._pf_device_get_log_weights = lambda *args: getter(*args)
        elif mode in ('allocator', 'preimport_allocator'):
            allocate = np.empty
            retained = allocate(count, dtype=np.float64)

            def allocate_retained(shape, *args, **kwargs):
                return retained if shape == count else allocate(shape, *args, **kwargs)

            monkeypatch.setattr(np, 'empty', allocate_retained)
            if mode == 'preimport_allocator':
                path = Path(PersistentParticleRuntimePool._migrate.__code__.co_filename)
                imported = _load_python(path, 'pf_private_preimport_allocator')
                PersistentParticleRuntimePool = imported.PersistentParticleRuntimePool
        elif mode == 'marker':
            source._pf_device_uniform_scan_getter = None
        elif mode == 'class_getter':
            getter = klass.get_log_weights
            monkeypatch.setattr(klass, 'get_log_weights', lambda self: getter(self))
        elif mode == 'nonfinite':
            data[-1] = np.inf
            source.set_log_weights(data)
            caller_owned = source.get_log_weights()
            with np.errstate(all='ignore'):
                expected_cdf = original_cdf(data)
            indices = np.minimum(np.searchsorted(expected_cdf, thresholds, side='left'), count-1)
        code = ParticleFilterDeviceRuntime._migration_weights_getter.__code__

        def observe(frame, event, value):
            if event == 'return' and frame.f_code is code:
                returned.append(value)
            if prior_profile is not None:
                prior_profile(frame, event, value)

        with warnings.catch_warnings():
            if mode == 'warning_error':
                warnings.simplefilter('error', RuntimeWarning)
            policy = 'raise' if mode == 'strict' else 'warn' if mode == 'warning_error' else 'ignore'
            with np.errstate(all=policy):
                sys.setprofile(observe)
                try:
                    PersistentParticleRuntimePool._migrate(source, target)
                finally:
                    sys.setprofile(prior_profile)
        assert len(returned) == 1
        eligible = mode == 'active' and count >= 65536 and np.isfinite(data).all()
        bits_equal(returned[0], expected_cdf if eligible else data)
        if captured:
            assert len(captured) == 1
            bits_equal(captured[0], expected_cdf)
        bits_equal(caller_owned, data)
        bits_equal(source.get_log_weights(), data)
        bits_equal(source.get_particle_states(), states)
        bits_equal(target.get_particle_states(), states[indices])
        bits_equal(target.get_log_weights(), np.full(count//2, -math.log(count//2)))
    finally:
        sys.setprofile(prior_profile)
        for pf in (source, target):
            native.pf_device_destroy(pf._state)
            pf._state = None


@pytest.mark.parametrize('count', [65535, 65536, 65537])
@pytest.mark.parametrize('mode', ['active', 'python_getter', 'class_getter', 'native_getter', 'allocator', 'preimport_allocator', 'marker', 'strict', 'warning_error', 'nonfinite'])
def test_native_owner_and_injected_getter_allocator_policy_guards(backend, monkeypatch, count, mode):
    _check(backend, monkeypatch, count, mode)


@pytest.mark.parametrize('variant', range(7))
def test_every_CDF_bit_extreme_denormal_nonfinite_and_raw_input(backend, monkeypatch, variant):
    _check(backend, monkeypatch, 65536, variant=variant)
