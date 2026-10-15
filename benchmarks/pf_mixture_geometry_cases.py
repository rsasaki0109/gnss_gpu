"""Explicit-rounding geometry tables and scratch reuse against the 3D oracle."""

import numpy as np

from bench_pf_covariance_broadcast_compare import _new_pf
from bench_pf_covariance_compare import _digest, _snapshot
from bench_pf_coalesced_compare import _scenario
from bench_pf_doppler_compare import _kf, _observations
from pf_mixture_coordinates_cases import validate as validate_previous


COUNTS = (1, 31, 32, 33, 255, 256, 257, 8193, 32769)
COMPONENTS = (2, 3, 4)


def arrays(initial, components, variant):
    count = len(initial)
    states = initial.copy()
    refs = np.zeros((components, 3))
    sigma2 = (6.25, 1e-300, 1e-300, 1e308, 1., 6.25, 6.25, 6.25)[variant]
    if variant == 0:
        refs[:] = [6378137., 0., 0.]
        refs[:, 1] = np.arange(components)
    elif variant == 1:
        states[:, :3] = np.resize([-0., 0.], (count, 3))
    elif variant == 2:
        states[:, :3] = np.resize([5e-324, 1e-154, -1e-154], (count, 3))
    elif variant == 3:
        states[:, :3] = np.resize([1e154, -1e154, 1e153], (count, 3))
        refs[:] = 1e153
    elif variant == 4:
        states[:, :3] = np.resize([1e308, -1e308, 0.], (count, 3))
        refs[:] = -1e308
    elif variant == 5:
        states.view(np.uint64)[count//2, 0] = 0x7ff8000000000051
    elif variant == 6:
        states.view(np.uint64)[count//2, 1] = 0x7ff0000000000053
    else:
        states[count//2, 2] = np.inf
    prior = np.random.default_rng(count+variant).normal(size=count)
    states.view(np.uint64)[::7, 3] = 0x7ff8000000000043
    states.view(np.uint64)[::11, 5] = 0xfff8000000000045
    prior.view(np.uint64)[::13] = 0x7ff8000000000063
    mix = np.resize([-0., -1., -np.inf, -2.], components)
    return states, prior, refs, mix, sigma2


def reference(states, refs, log_mix, sigma2):
    if not np.all(np.isfinite(states[:, :3])):
        return None
    with np.errstate(all='ignore'):
        diff = states[:, None, :3] - refs[None, :, :]
        return log_mix[None, :]-.5*np.sum(diff*diff, axis=2)/sigma2


def check_table(native, klass, count, components, variant):
    pf = _new_pf(klass, count)
    try:
        states, prior, refs, mix, sigma2 = arrays(pf.get_particle_states(), components, variant)
        pf.set_particle_states(states)
        pf.set_log_weights(prior)
        cached = pf.get_ess()
        expected = reference(states, refs, mix, sigma2)
        geometry = getattr(native, 'pf_device_position_mixture_components', None)
        actual = expected if geometry is None else geometry(pf._state, refs, mix, sigma2)
        if expected is None:
            assert actual is None
            output = np.empty(0)
        else:
            assert actual is not None
            np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
            output = actual
        after = _snapshot(native, pf._state)
        for a, b in zip(after, (states, prior), strict=True):
            np.testing.assert_array_equal(a.view(np.uint64), b.view(np.uint64))
        np.testing.assert_array_equal(np.asarray(pf.get_ess()).view(np.uint64), np.asarray(cached).view(np.uint64))
        pf.set_log_weights(prior)
        np.testing.assert_array_equal(np.asarray(pf.get_ess()).view(np.uint64), np.asarray(cached).view(np.uint64))
        return dict(input_sha256=_digest(states, prior, refs, mix, np.array([sigma2])),
            output_sha256=_digest(output, *after, np.asarray(cached)),
            original_3d_numpy_identical=True, nonfinite_fallback=expected is None,
            all16_states_weights_and_cached_ess_preserved=True)
    finally:
        native.pf_device_destroy(pf._state)
        pf._state = None


def check_reuse(native, klass, count, components):
    actual, oracle = _new_pf(klass, count), _new_pf(klass, count)
    refs = np.repeat(np.array([[6378137., 0., 0.]]), components, axis=0)
    mix = np.log(np.arange(1, components+1)/sum(range(1, components+1)))
    geometry = getattr(native, 'pf_device_position_mixture_components', None)
    histories = []
    try:
        for step, operation in enumerate(('common_predict', 'common_B4', 'imported_B4', 'KF', 'imported_B60', 'legacy'), 1):
            for pf in (actual, oracle):
                if operation == 'common_predict':
                    pf.set_velocity_covariance(np.eye(3)*2.)
                    pf.predict([1., 2., 3.], dt=.1, rbpf_velocity_kf=True, velocity_process_noise=.1)
                elif operation == 'KF':
                    _kf(native, pf._state, _observations(12))
                elif operation == 'legacy':
                    native.pf_device_resample_megopolis(pf._state, 3, 37+step)
                else:
                    if operation.startswith('imported'):
                        pf.set_particle_states(pf.get_particle_states())
                    native.pf_device_resample_megopolis_coalesced(pf._state, 60 if operation.endswith('B60') else 4, 37+step)
            if geometry:
                result = geometry(actual._state, refs, mix, 6.25)
                expected = reference(oracle.get_particle_states(), refs, mix, 6.25)
                np.testing.assert_array_equal(result.view(np.uint64), expected.view(np.uint64))
                result.fill(-999.)
            # Keep subsequent reads/prediction on the same state/stream so an
            # incorrect covariance scratch alias affects this independent PF.
            for pf in (actual, oracle):
                pf.predict([1., 2., 3.], dt=.17, rbpf_velocity_kf=True, velocity_process_noise=.13)
            a, b = _snapshot(native, actual._state), _snapshot(native, oracle._state)
            for left, right in zip(a, b, strict=True):
                np.testing.assert_array_equal(left.view(np.uint64), right.view(np.uint64))
            histories.extend(a)
        return dict(output_sha256=_digest(*histories), queued_predict_KF_resample_scratch_preserved=True)
    finally:
        for pf in (actual, oracle):
            native.pf_device_destroy(pf._state)
            pf._state = None


def validate(native, klass):
    records = validate_previous(native, klass)
    for count in COUNTS:
        for components in COMPONENTS:
            for variant in range(8):
                records[f'geometry/N{count}/K{components}/v{variant}'] = check_table(native, klass, count, components, variant)
            records[f'geometry/N{count}/K{components}/reuse'] = check_reuse(native, klass, count, components)
    return records
