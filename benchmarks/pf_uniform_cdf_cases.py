"""Exact sequential uniform CDF on GPU, full-state selection and fallback."""

import ctypes
import math

import numpy as np

from bench_pf_covariance_broadcast_compare import _new_pf
from bench_pf_covariance_compare import _digest, _snapshot
from pf_particle_readout_cases import bits_equal
from pf_pool_cdf_cases import select
from pf_pool_gather_cases import state_values
from pf_cdf_validate_cases import validate as validate_previous

COUNTS = (65535, 65536, 65537, 65538, 65539, 65540, 65541, 65542, 65543,
          100000, 131071, 131072, 131073, 500000, 999999, 1000000, 1000001)
TARGETS = (31, 127, 257, 65536)
MODES = ('active', 'missing', 'unpaired', 'cdf_override', 'cdf_mismatch', 'uniform_mismatch', 'decline', 'count_mismatch')


def check_uniform(native, klass, count, target_count, variant):
    source, target = _new_pf(klass, count), _new_pf(klass, target_count)
    try:
        states = state_values(source.get_particle_states(), variant)
        source.set_particle_states(states)
        source.set_log_weights(np.arange(count, dtype=np.float64))
        source_ess = source.get_ess()
        weights = source.get_log_weights()
        expected_cdf = np.cumsum(np.full(count, 1./count))
        expected_states = states[select(expected_cdf, target_count)].copy()
        target_before = _snapshot(native, target._state)
        output = np.full(count, -42.)
        method = getattr(native, 'pf_device_gather_from_uniform_cdf', None)
        accepted = method(target._state, source._state, output) if method else False
        if not accepted:
            for a, b in zip(_snapshot(native, target._state), target_before, strict=True):
                bits_equal(a, b)
            bits_equal(output, np.full(count, -42.))
            assert native.pf_device_gather_from_cdf(target._state, source._state, expected_cdf)
            output = expected_cdf.copy()
        bits_equal(output, expected_cdf)
        bits_equal(target.get_particle_states(), expected_states)
        bits_equal(target.get_log_weights(), np.zeros(target_count))
        bits_equal(source.get_particle_states(), states)
        bits_equal(source.get_log_weights(), weights)
        bits_equal(source.get_ess(), source_ess)
        return dict(CDF_sha256=_digest(output), output_sha256=_digest(expected_states, weights, np.asarray(source_ess)),
                    error=None, full_sequential_CDF_bits_all16_fields_and_source_ownership_exact=True)
    finally:
        for pf in (source, target):
            native.pf_device_destroy(pf._state)
            pf._state = None


def check_dispatch(native, klass, count, mode):
    from gnss_gpu.realtime_runtime import PersistentParticleRuntimePool
    source, target, oracle_source, oracle_target = [_new_pf(klass, n) for n in (count, count//2, count, count//2)]
    calls = []
    try:
        for pf in (source, oracle_source):
            pf.set_log_weights(np.full(count, -math.log(count)))
        method = getattr(target, '_pf_device_gather_from_uniform_cdf', None)
        if callable(method):
            def wrapped(*args):
                calls.append(1)
                return False if mode == 'decline' else method(*args)
            source._pf_device_gather_from_uniform_cdf = target._pf_device_gather_from_uniform_cdf = wrapped
            if mode == 'missing':
                target._pf_device_gather_from_uniform_cdf = None
            elif mode == 'unpaired':
                target._pf_device_uniform_cdf_checked_gather = None
            elif mode == 'cdf_override':
                checked = target._pf_device_gather_from_cdf
                source._pf_device_gather_from_cdf = target._pf_device_gather_from_cdf = lambda *args: checked(*args)
            elif mode == 'cdf_mismatch':
                checked = source._pf_device_gather_from_cdf
                source._pf_device_gather_from_cdf = lambda *args: checked(*args)
            elif mode == 'uniform_mismatch':
                source._pf_device_gather_from_uniform_cdf = lambda *args: method(*args)
        if mode == 'count_mismatch':
            target.n_particles += 1
            oracle_target.n_particles += 1
        oracle_target._pf_device_gather_from_uniform_cdf = None
        errors = []
        for a, b in ((source, target), (oracle_source, oracle_target)):
            try:
                PersistentParticleRuntimePool._migrate(a, b)
                errors.append(None)
            except RuntimeError as exception:
                errors.append((type(exception).__name__, str(exception)))
        assert errors[0] == errors[1]
        if callable(method):
            assert len(calls) == int(mode in ('active', 'decline'))
        history = []
        for a, b in ((source, oracle_source), (target, oracle_target)):
            left, right = _snapshot(native, a._state), _snapshot(native, b._state)
            for x, y in zip(left, right, strict=True):
                bits_equal(x, y)
            history.extend(left)
        return dict(output_sha256=_digest(*history), error=errors[0], old_or_overridden_backend_fallback_exact=True)
    finally:
        for pf in (source, target, oracle_source, oracle_target):
            native.pf_device_destroy(pf._state)
            pf._state = None


def control_rounding(mode):
    library = ctypes.CDLL('ucrtbase')
    function = library._controlfp_s
    function.argtypes = (ctypes.POINTER(ctypes.c_uint), ctypes.c_uint, ctypes.c_uint)
    function.restype = ctypes.c_int
    current = ctypes.c_uint()
    assert function(ctypes.byref(current), 0, 0) == 0
    previous = current.value & 0x300
    assert function(ctypes.byref(current), mode, 0x300) == 0
    return lambda: function(ctypes.byref(current), previous, 0x300)


def check_rounding(native, klass, mode):
    from gnss_gpu.realtime_runtime import PersistentParticleRuntimePool
    count = 100000
    source, target, oracle_source, oracle_target = [_new_pf(klass, n) for n in (count, count//2, count, count//2)]
    restore = control_rounding(mode)
    try:
        output = np.full(count, -42.)
        before = _snapshot(native, target._state)
        method = getattr(native, 'pf_device_gather_from_uniform_cdf', None)
        if mode and method:
            assert not method(target._state, source._state, output)
            bits_equal(output, np.full(count, -42.))
            for a, b in zip(_snapshot(native, target._state), before, strict=True):
                bits_equal(a, b)
        oracle_target._pf_device_gather_from_uniform_cdf = None
        with np.errstate(all='ignore'):
            PersistentParticleRuntimePool._migrate(source, target)
            PersistentParticleRuntimePool._migrate(oracle_source, oracle_target)
        actual = _snapshot(native, target._state)
        for a, b in zip(actual, _snapshot(native, oracle_target._state), strict=True):
            bits_equal(a, b)
        return dict(output_sha256=_digest(*actual), error=None, rounding_mode_fallback_exact=True)
    finally:
        assert restore() == 0
        for pf in (source, target, oracle_source, oracle_target):
            native.pf_device_destroy(pf._state)
            pf._state = None



EXTRA_COUNTS = tuple(sorted(set(
    [2**k+d for k in range(16, 21) for d in (-1, 0, 1)] +
    np.random.default_rng(199).integers(65536, 1000001, size=32).tolist())))
QUEUE_COUNTS = (65536, 100000, 1000000)


def check_extra_cdf(native, klass, count):
    source, target = _new_pf(klass, count), _new_pf(klass, 257)
    try:
        expected = np.cumsum(np.full(count, 1./count))
        output = np.full(count, -42.)
        method = getattr(native, 'pf_device_gather_from_uniform_cdf', None)
        if method and count >= 65536:
            assert method(target._state, source._state, output)
        else:
            assert native.pf_device_gather_from_cdf(target._state, source._state, expected)
            output = expected.copy()
        bits_equal(output, expected)
        return dict(CDF_sha256=_digest(output), error=None, every_CDF_bit_exact=True)
    finally:
        for pf in (source, target):
            native.pf_device_destroy(pf._state)
            pf._state = None


def check_uniform_queue(native, klass, count):
    from bench_pf_doppler_compare import _kf, _observations
    source, target, oracle, source_oracle = [_new_pf(klass, n) for n in (count, count//2, count//2, count)]
    history = []
    try:
        for pf in (source, source_oracle):
            pf.set_particle_states(state_values(pf.get_particle_states(), 0))
        cdf = np.cumsum(np.full(count, 1./count))
        for step in range(3):
            for pf in (source, source_oracle, target, oracle):
                pf.predict([1., 2., 3.], dt=.17, rbpf_velocity_kf=True, velocity_process_noise=.13)
                native.pf_device_resample_megopolis_coalesced(pf._state, 4, 199+step)
            method = getattr(native, 'pf_device_gather_from_uniform_cdf', None)
            if method:
                assert method(target._state, source._state)
            else:
                assert native.pf_device_gather_from_cdf(target._state, source._state, cdf)
            assert native.pf_device_gather_from_cdf(oracle._state, source_oracle._state, cdf)
            for pf in (target, oracle):
                _kf(native, pf._state, _observations(12))
                pf.predict([3., 2., 1.], dt=.1, rbpf_velocity_kf=True, velocity_process_noise=.1)
                native.pf_device_resample_megopolis_coalesced(pf._state, 60, 219+step)
            for a, b in ((source, source_oracle), (target, oracle)):
                actual, expected = _snapshot(native, a._state), _snapshot(native, b._state)
                for x, y in zip(actual, expected, strict=True):
                    bits_equal(x, y)
                history.extend(actual)
            # The next checked CDF gather overwrites the source CDF scratch;
            # then the next iteration reconstructs the uniform CDF again.
            other = np.cumsum(np.linspace(1., 2., count)/np.linspace(1., 2., count).sum())
            assert native.pf_device_gather_from_cdf(target._state, source._state, other)
            assert native.pf_device_gather_from_cdf(oracle._state, source_oracle._state, other)
        return dict(output_sha256=_digest(*history), error=None,
                    queued_producers_CDF_ancestor_B_reuse_and_next_predict_exact=True)
    finally:
        for pf in (source, target, oracle, source_oracle):
            native.pf_device_destroy(pf._state)
            pf._state = None


def validate(native, klass):
    records = validate_previous(native, klass)
    for count in COUNTS:
        for target in TARGETS:
            for variant in range(3):
                records[f'uniform_cdf/{count}/target{target}/v{variant}'] = check_uniform(native, klass, count, target, variant)
    for count in (65536, 100000, 1000000):
        for mode in MODES:
            records[f'uniform_cdf/dispatch/{count}/{mode}'] = check_dispatch(native, klass, count, mode)
    for mode in (0, 0x100, 0x200, 0x300):
        records[f'uniform_cdf/rounding/{mode}'] = check_rounding(native, klass, mode)
    for count in EXTRA_COUNTS:
        records[f'uniform_cdf/extra/{count}'] = check_extra_cdf(native, klass, count)
    for count in QUEUE_COUNTS:
        records[f'uniform_cdf/queue/{count}'] = check_uniform_queue(native, klass, count)
    return records
