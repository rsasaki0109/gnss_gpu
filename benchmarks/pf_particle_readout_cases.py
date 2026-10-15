"""Particle export bits, ownership, and scratch liveness across PF histories."""

import numpy as np

from bench_pf_covariance_broadcast_compare import _new_pf
from bench_pf_covariance_compare import _digest, _snapshot
from bench_pf_doppler_compare import _kf, _observations
from pf_mixture_geometry_cases import validate as validate_previous

COUNTS = (1, 31, 32, 33, 255, 256, 257, 8193, 32769, 100000)
METHODS = ('coalesced_B4', 'coalesced_B60', 'legacy')
COLUMNS = (0, 1, 2, 3)


def bits_equal(left, right):
    np.testing.assert_array_equal(np.asarray(left).view(np.uint64), np.asarray(right).view(np.uint64))


def check_export(native, klass, count, variant):
    pf = _new_pf(klass, count)
    try:
        states = pf.get_particle_states()
        values = ([-0., 0.], [5e-324, -5e-324, 1e-154], [1e308, -1e308, 1e154],
                  [0x7ff8000000000051, 0xfff8000000000052],
                  [0x7ff0000000000053, 0xfff0000000000054], [np.inf, -np.inf])
        if variant in (1, 2, 3, 6):
            states[:] = np.resize(values[variant-1], states.shape)
        elif variant in (4, 5):
            states.view(np.uint64)[:] = np.resize(np.array(values[variant-1], dtype=np.uint64), states.shape)
        elif variant == 7:
            states.view(np.uint64)[:] = np.random.default_rng(794+count).integers(0, 2**64-1, states.shape, dtype=np.uint64)
        prior = np.random.default_rng(795+count).normal(size=count)
        pf.set_particle_states(states)
        pf.set_log_weights(prior)
        expected = states[:, COLUMNS].copy()
        cached = pf.get_ess()
        result = pf.get_particles()
        bits_equal(result, expected)
        result.fill(-999.)
        result = pf.get_particles()
        bits_equal(result, expected)
        snapshot = _snapshot(native, pf._state)
        for left, right in zip(snapshot, (states, prior), strict=True):
            bits_equal(left, right)
        bits_equal(pf.get_ess(), cached)
        pf.set_log_weights(prior)
        bits_equal(pf.get_ess(), cached)
        return dict(input_sha256=_digest(states, prior), output_sha256=_digest(result, *snapshot, np.asarray(cached)),
                    all16_states_weights_and_ess_preserved=True, independent_owned_result=True)
    finally:
        native.pf_device_destroy(pf._state)
        pf._state = None


def check_history(native, klass, count, common, method):
    actual, oracle = _new_pf(klass, count), _new_pf(klass, count)
    history = []
    try:
        if not common:
            states = actual.get_particle_states()
            states[:, 7:] = (np.eye(3)[None, :, :]*(1.+np.arange(count)[:, None, None]/count)).reshape(count, 9)
            for pf in (actual, oracle):
                pf.set_particle_states(states)
        else:
            for pf in (actual, oracle):
                pf.set_velocity_covariance(np.eye(3)*2.)
        prior = np.random.default_rng(count+811).normal(0., 2., count)
        for step in range(1, 4):
            for pf in (actual, oracle):
                pf.predict([1., 2., 3.], dt=.17, rbpf_velocity_kf=True, velocity_process_noise=.13)
            # Only actual executes this getter; the oracle's full-state getter
            # uses an independent allocation. Queued work must finish on return.
            result = actual.get_particles()
            bits_equal(result, oracle.get_particle_states()[:, COLUMNS])
            result.fill(-999.)
            for pf in (actual, oracle):
                pf.set_log_weights(prior)
            cached = actual.get_ess()
            actual.get_particles()
            bits_equal(actual.get_ess(), cached)
            for pf in (actual, oracle):
                if method == 'legacy':
                    native.pf_device_resample_megopolis(pf._state, 3, 81+step)
                elif method == 'systematic':
                    native.pf_device_resample_systematic(pf._state, 81+step)
                else:
                    native.pf_device_resample_megopolis_coalesced(pf._state, 60 if method.endswith('B60') else 4, 81+step)
            # Gather may swap covariance pointers while its work is still queued.
            bits_equal(actual.get_particles(), oracle.get_particle_states()[:, COLUMNS])
            table = native.pf_device_position_mixture_components(actual._state, np.zeros((2, 3)), np.zeros(2), 6.25)
            actual.get_particles()  # Overwrite the same scratch after the table copy.
            from pf_mixture_geometry_cases import reference
            bits_equal(table, reference(oracle.get_particle_states(), np.zeros((2, 3)), np.zeros(2), 6.25))
            for pf in (actual, oracle):
                _kf(native, pf._state, _observations(12))
                pf.predict([1., 2., 3.], dt=.1, rbpf_velocity_kf=True, velocity_process_noise=.1)
            bits_equal(actual.get_particles(), oracle.get_particle_states()[:, COLUMNS])
            a, b = _snapshot(native, actual._state), _snapshot(native, oracle._state)
            for left, right in zip(a, b, strict=True):
                bits_equal(left, right)
            history.extend(a)
        return dict(output_sha256=_digest(*history), independent_no_export_oracle_identical=True,
                    queued_prediction_resampling_KF_and_geometry_preserved=True)
    finally:
        for pf in (actual, oracle):
            native.pf_device_destroy(pf._state)
            pf._state = None


def validate(native, klass):
    records = validate_previous(native, klass)
    for count in COUNTS:
        for variant in range(8):
            records[f'export/N{count}/v{variant}'] = check_export(native, klass, count, variant)
        for common in (False, True):
            for method in METHODS + (('systematic',) if count <= 257 else ()):
                records[f'export/N{count}/common{common}/{method}'] = check_history(native, klass, count, common, method)
    return records
