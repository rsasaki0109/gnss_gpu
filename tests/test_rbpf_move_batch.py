"""Whole frozen-step synthetic integration checks for explicit move batching."""

from copy import deepcopy
import os
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'benchmarks'))
from rbpf_likelihood_batch import BatchScorer  # noqa: E402
from rbpf_move_batch import MoveBatchStep  # noqa: E402
from rbpf_move_cases import comparison, make_step_case, state  # noqa: E402


@pytest.fixture(scope='module')
def scorer():
    path = os.environ.get('GNSS_GPU_RBPF_PROBE_DLL')
    if not path:
        pytest.skip('Explicit experimental DLL is required')
    with BatchScorer(path, capacity=512) as instance:
        yield instance


@pytest.mark.parametrize('seed', [20260710, 20260711, 20260712])
@pytest.mark.parametrize('options', [
    {}, dict(cluster=True), dict(uniform=False), dict(huber=False),
    dict(shadow=True, spawn=True), dict(resample=False),
])
@pytest.mark.parametrize('mode', ['scalar', 'native_cpu', 'gpu'])
def test_complete_step_state_and_rng(scorer, seed, options, mode):
    reference, rtk0, ctx0 = make_step_case(seed=seed, **options)
    candidate, rtk1, ctx1 = make_step_case(seed=seed, **options)
    staged = MoveBatchStep(candidate, scorer, mode)
    candidate.step = staged
    for ep in range(4):
        rtk0.epoch2 = rtk1.epoch2 = ep
        ctx0['obs'].t = ctx1['obs'].t = 100. + ep
        # Repeated calls change observations and priors, checking no stale
        # likelihood or cluster state is reused across epochs.
        for ctx in (ctx0, ctx1):
            ctx['problem'].rows_bmeas[32:] += .001 * ep
            ctx['problem'].rows_v0[:32] += .01 * ep
        out0 = reference.step(rtk0, ctx0)
        out1 = candidate.step(rtk1, ctx1)
        result = comparison(state(reference, rtk0, out0), state(candidate, rtk1, out1))
        assert result['discrete_equal'], result
        assert result['position_bits_equal'], result
        assert result['max_abs_error'] < 1e-10, result
        assert reference.rng.bit_generator.state == candidate.rng.bit_generator.state
        if mode == 'scalar':
            assert result['numeric_bits_equal'], result
        assert staged.last['extend_calls'] == staged.last['held_particles']
        assert staged.last['shrink_calls'] == staged.last['held_particles']


def test_generation_calls_are_not_duplicated(scorer):
    reference, rtk, ctx = make_step_case()
    candidate, rtk1, ctx1 = make_step_case()
    staged = MoveBatchStep(candidate, scorer, 'gpu')
    expected = sum(bool(p.H) for p in reference.particles)
    for runtime, rr, cc, call in ((reference, rtk, ctx, reference.step), (candidate, rtk1, ctx1, staged)):
        counts = dict(_extend=0, _shrink_worst=0)
        codes = {getattr(runtime, name).__func__.__code__: name for name in counts}
        previous = sys.getprofile()

        def profile(frame, event, _arg):
            if event == 'call' and frame.f_code in codes:
                counts[codes[frame.f_code]] += 1

        try:
            sys.setprofile(profile)
            call(rr, cc)
        finally:
            sys.setprofile(previous)
        assert counts == dict(_extend=expected, _shrink_worst=expected)


@pytest.mark.parametrize('event', ['skip', 'reseed', 'release', 'fresh'])
def test_maintenance_and_unusable_epoch(scorer, event):
    a, ra, ca = make_step_case()
    b, rb, cb = make_step_case()
    staged = MoveBatchStep(b, scorer, 'gpu')
    for rtk, ctx in ((ra, ca), (rb, cb)):
        group, ref, sat = ctx['problem'].pairs[0]
        if event == 'skip':
            ctx['problem'].ok = False
            ctx['problem'].reason = 'synthetic outage'
        elif event == 'reseed':
            rtk._rbpf_graph_reseeded = True
        elif event == 'release':
            rtk._rbpf_released = {(sat, group[1])}
        else:
            rtk._amb_init_epoch[(ref, group[1])] = rtk.epoch2
    out0, out1 = a.step(ra, ca), staged(rb, cb)
    result = comparison(state(a, ra, out0), state(b, rb, out1))
    assert result['discrete_equal'] and result['position_bits_equal'], result
    assert result['max_abs_error'] < 1e-10, result
    if event == 'skip':
        assert staged.last is None


def test_strict_numeric_policy_calls_original_step(scorer):
    runtime, rtk, ctx = make_step_case()
    staged = MoveBatchStep(runtime, scorer, 'gpu')
    # The original scalar may raise before or after some particle choices.
    # Compare to its exact exception and partial state, not a relaxed result.
    other, ro, co = make_step_case()
    with np.errstate(all='raise'):
        with pytest.raises(FloatingPointError):
            staged(rtk, ctx)
        with pytest.raises(FloatingPointError):
            other.step(ro, co)
    assert staged.last == dict(original_step_fallback='numpy_error_policy')
    assert runtime.rng.bit_generator.state == other.rng.bit_generator.state
    result = comparison(state(runtime, rtk, None), state(other, ro, None))
    assert result['numeric_bits_equal'] and result['discrete_equal'], result


def test_method_changes_are_rejected(scorer):
    runtime, rtk, ctx = make_step_case()
    staged = MoveBatchStep(runtime, scorer, 'gpu')
    runtime._extend = lambda _h, _p: None
    with pytest.raises(ValueError, match='changed after binding'):
        staged(rtk, ctx)
    with pytest.raises(ValueError, match='Unsupported RBPF method'):
        MoveBatchStep(runtime, scorer, 'gpu')


def test_constructor_requires_explicit_backend():
    runtime, _rtk, _ctx = make_step_case()
    with pytest.raises(ValueError, match='owning scorer'):
        MoveBatchStep(runtime, mode='gpu')
    with pytest.raises(ValueError, match='Unknown'):
        MoveBatchStep(runtime, mode='auto')


@pytest.mark.parametrize('mode', ['scalar', 'native_cpu', 'gpu'])
def test_feedback_commits_and_releases(scorer, mode):
    a, ra, ca = make_step_case(feedback=True)
    b, rb, cb = make_step_case(feedback=True)
    for runtime, ctx in ((a, ca), (b, cb)):
        # A consistent population exercises actual commit/release branches.
        # Tiny positive unused priors avoid introducing log(0) warnings.
        runtime.cfg.p_survive = 1.
        for key in ('p_extend', 'p_shrink', 'p_release', 'p_adopt', 'p_adopt_cons', 'p_spawn', 'p_burst'):
            setattr(runtime.cfg, key, 1e-100)
        full = runtime._adopt(ctx['problem'], 0)
        for p in runtime.particles:
            p.H, p.logw = deepcopy(full), 0.
    staged = MoveBatchStep(b, scorer, mode)
    for ep in range(4):
        ra.epoch2 = rb.epoch2 = ep
        if ep == 2:
            for rr, cc in ((ra, ca), (rb, cb)):
                group, _ref, sat = cc['problem'].pairs[0]
                rr._committed[(sat, group[1])] += 100.
        out0, out1 = a.step(ra, ca), staged(rb, cb)
        result = comparison(state(a, ra, out0), state(b, rb, out1))
        assert result['discrete_equal'] and result['position_bits_equal'], result
        assert result['max_abs_error'] < 1e-10, result
    assert any(event[0] == 'inject' for event in ra.events)
    assert any(event[0] == 'release' for event in ra.events)
