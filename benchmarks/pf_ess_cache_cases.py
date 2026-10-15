"""ESS replay cases with an independent cache-invalidating GPU recomputation."""

import numpy as np

from bench_pf_coalesced_compare import _initialize, _scenario
from bench_pf_covariance_compare import _digest, _snapshot
from bench_pf_dd_reference_compare import _cp, _joint, _observations as dd_observations
from bench_pf_doppler_compare import _kf, _observations


COUNTS = (1, 31, 32, 33, 255, 256, 257, 8193)
MUTATORS = ('initialize', 'PR', 'DD_PR', 'GMM', 'position', 'reset_position',
            'systematic', 'megopolis', 'coalesced', 'import', 'weights',
            'carrier', 'DD_carrier', 'DD_joint', 'doppler', 'KF')
PRESERVERS = ('predict', 'shift_position', 'shift_clock', 'covariance', 'queries', 'downloads')
DISTRIBUTIONS = ('normal', 'uniform', 'dominant', 'extreme', 'nan', 'inf', 'neg_inf', 'mixed')


def weights_for(count, kind):
    weights = np.random.default_rng(403+count).normal(size=count)
    if kind == 'uniform':
        weights.fill(-0.)
    elif kind == 'dominant':
        weights.fill(-1000.)
        weights[count//2] = 0.
    elif kind == 'extreme':
        weights = np.resize([0., 1e300, -1e300, 1e-310], count)
    elif kind in ('nan', 'inf', 'neg_inf'):
        weights.fill({'nan': np.nan, 'inf': np.inf, 'neg_inf': -np.inf}[kind])
    elif kind == 'mixed':
        weights = np.resize([0., -np.inf, np.nan, np.inf, -0.], count)
    return weights


def apply(native, state, count, kind):
    position, sat, pr, weights = _scenario()
    dd = dd_observations(12)
    obs = _observations(12)
    if kind == 'initialize':
        _initialize(native, state)
    elif kind == 'PR':
        native.pf_device_weight(state, sat, pr, weights, 12, 5.)
    elif kind == 'DD_PR':
        native.pf_device_weight_dd_pseudorange(state, dd.sat_ecef_k, dd.sat_ecef_ref,
            dd.dd_pseudorange_m, dd.base_range_k, dd.base_range_ref, dd.dd_weights, 12, .75)
    elif kind == 'GMM':
        native.pf_device_weight_gmm(state, sat, pr, weights, 12, 5., .7, 15., 30.)
    elif kind == 'position':
        native.pf_device_position_update(state, *position, 2.)
    elif kind == 'reset_position':
        native.pf_device_reset_position(state, *position, 2., 37, 8)
    elif kind == 'systematic':
        native.pf_device_resample_systematic(state, 37)
    elif kind == 'megopolis':
        native.pf_device_resample_megopolis(state, 7, 37)
    elif kind == 'coalesced':
        native.pf_device_resample_megopolis_coalesced(state, 7, 37)
    elif kind == 'import':
        native.pf_device_set_particle_states(state, native.pf_device_get_particle_states(state))
    elif kind == 'weights':
        native.pf_device_set_log_weights(state, weights_for(count, 'dominant'))
    elif kind == 'carrier':
        native.pf_device_weight_carrier_afv(state, sat, pr/.19, weights, 12, .19, .05)
    elif kind == 'DD_carrier':
        _cp(native, state, dd)
    elif kind == 'DD_joint':
        _joint(native, state, dd)
    elif kind == 'doppler':
        native.pf_device_weight_doppler(state, *obs, 12, .19, .5, .25, 10.)
    elif kind == 'KF':
        _kf(native, state, obs)
    elif kind == 'predict':
        native.pf_device_predict(state, 1., 2., 3., .17, 1., 3., 37, 8, .2, .3, True, .13)
    elif kind == 'shift_position':
        native.pf_device_shift_position(state, 5., -3., 2.)
    elif kind == 'shift_clock':
        native.pf_device_shift_clock_bias(state, 17.)
    elif kind == 'covariance':
        native.pf_device_set_velocity_covariance(state, np.eye(3)*2.)
    elif kind == 'queries':
        native.pf_device_estimate(state)
        native.pf_device_position_spread(state, *position)
    elif kind == 'downloads':
        native.pf_device_get_particles(state)
        native.pf_device_get_particle_states(state)
    else:
        raise ValueError(kind)


def check(native, count, kind, distribution='normal'):
    state = native.pf_device_create(count)
    try:
        _initialize(native, state)
        prior = weights_for(count, distribution)
        native.pf_device_set_log_weights(state, prior)
        first = native.pf_device_ess(state)
        if kind != 'repeat':
            apply(native, state, count, kind)
        before = _snapshot(native, state)
        actual = [native.pf_device_ess(state)]
        # Overwrite both shared reduction buffers and pinned result slots.
        mean = native.pf_device_estimate(state)
        spread = native.pf_device_position_spread(state, *_scenario()[0])
        actual.append(native.pf_device_ess(state))
        # Upload the exact current weights to force the original GPU reduction.
        native.pf_device_set_log_weights(state, before[1])
        expected = native.pf_device_ess(state)
        actual.append(native.pf_device_ess(state))
        np.testing.assert_array_equal(np.asarray(actual).view(np.uint64),
            np.full(3, expected).view(np.uint64))
        if kind in PRESERVERS or kind == 'repeat':
            np.testing.assert_array_equal(np.asarray(first).view(np.uint64),
                np.asarray(expected).view(np.uint64))
        after = _snapshot(native, state)
        for a, b in zip(before, after, strict=True):
            np.testing.assert_array_equal(a.view(np.uint64), b.view(np.uint64))
        # Compare ESS after one actual systematic resample; existing independent
        # systematic draws can differ, so do not use them as a state oracle.
        outputs = (before[1], np.asarray(actual)) if kind == 'systematic' else (
            *before, np.asarray(actual), mean, np.asarray(spread))
        return dict(input_sha256=_digest(prior), output_sha256=_digest(*outputs),
            forced_original_gpu_reduction_identical=True, queries_preserve_all_state_bits=True)
    finally:
        native.pf_device_destroy(state)


def validate(native):
    result = {}
    for count in COUNTS:
        for kind in (*MUTATORS, *PRESERVERS):
            result[f'ESS_cache/N{count}/{kind}'] = check(native, count, kind)
        for distribution in DISTRIBUTIONS:
            result[f'ESS_cache/N{count}/repeat/{distribution}'] = check(native, count, 'repeat', distribution)
    return result
