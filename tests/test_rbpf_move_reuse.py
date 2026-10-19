"""Full-state and actual call-count checks for exact move reuse."""

from copy import deepcopy
import os
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'benchmarks'))
from rbpf_likelihood_batch import BatchScorer  # noqa: E402
from rbpf_move_batch import MoveBatchStep  # noqa: E402
from rbpf_move_cases import comparison, make_step_case, state  # noqa: E402
from rbpf_move_reuse import MoveReuseStep, _ordered_integer_key  # noqa: E402


@pytest.fixture(scope='module')
def scorer():
    path = os.environ.get('GNSS_GPU_RBPF_PROBE_DLL')
    if not path:
        pytest.skip('Explicit experimental DLL is required')
    with BatchScorer(path, capacity=512) as value:
        yield value


def assert_exact(a, b):
    result = comparison(a, b)
    assert result['discrete_equal'] and result['numeric_bits_equal'], result
    assert result['max_abs_error'] == 0. and result['position_bits_equal'], result


@pytest.mark.parametrize('seed', [20260710, 20260711, 20260712])
@pytest.mark.parametrize('options', [
    {}, dict(cluster=True), dict(uniform=False), dict(huber=False),
    dict(shadow=True, spawn=True), dict(resample=False),
])
@pytest.mark.parametrize('mode', ['scalar', 'native_cpu', 'gpu'])
def test_same_backend_complete_step_bits(scorer, seed, options, mode):
    a, ra, ca = make_step_case(seed=seed, **options)
    b, rb, cb = make_step_case(seed=seed, **options)
    baseline = a.step if mode == 'scalar' else MoveBatchStep(a, scorer, mode)
    candidate = MoveReuseStep(b, scorer, mode)
    b.step = candidate
    for ep in range(4):
        ra.epoch2 = rb.epoch2 = ep
        ca['obs'].t = cb['obs'].t = 100. + ep
        for ctx in (ca, cb):
            prob = ctx['problem']
            prob.rows_bmeas[32:] += .001 * ep
            prob.rows_v0[:32] += .01 * ep
            prob.Qb *= 1. + ep * .001
        out0, out1 = baseline(ra, ca), candidate(rb, cb)
        assert_exact(state(a, ra, out0), state(b, rb, out1))
        meta = candidate.last
        assert meta['held_particles'] == meta['move_reuse']['requests']['extend']
        assert meta['held_particles'] == meta['move_reuse']['requests']['shrink']


def profile_calls(runtime, call, rtk, ctx):
    counts = dict(_extend=0, _shrink_worst=0)
    codes = {getattr(runtime, name).__func__.__code__: name for name in counts}
    previous = sys.getprofile()

    def profile(frame, event, _arg):
        if event == 'call' and frame.f_code in codes:
            counts[codes[frame.f_code]] += 1

    try:
        sys.setprofile(profile)
        output = call(rtk, ctx)
    finally:
        sys.setprofile(previous)
    return counts, output


def test_exact_dictionary_identity_includes_gauge_and_order(scorer):
    a, ra, ca = make_step_case()
    b, rb, cb = make_step_case()
    original = a._adopt(ca['problem'], 0)
    # Leave unheld support so extend actually solves and constructs a result.
    for integers in original.values():
        integers.pop(next(reversed(integers)))
    gauge = {g: {sat: z + 7 for sat, z in integers.items()} for g, integers in original.items()}
    outer = dict(reversed(list(original.items())))
    inner = {g: dict(reversed(list(integers.items()))) for g, integers in original.items()}
    variants = [original, deepcopy(original), gauge, outer, inner]
    assert len({_ordered_integer_key(h) for h in variants}) == 4
    assert len({a.step.__func__.__globals__['_canon_basin_id'](h) for h in variants}) == 1
    for runtime in (a, b):
        for i, particle in enumerate(runtime.particles):
            particle.H = deepcopy(variants[i % len(variants)])
    old = MoveBatchStep(a, scorer, 'gpu')
    new = MoveReuseStep(b, scorer, 'gpu')
    old_counts, out0 = profile_calls(a, old, ra, ca)
    new_counts, out1 = profile_calls(b, new, rb, cb)
    assert old_counts == dict(_extend=64, _shrink_worst=64)
    assert new_counts == dict(_extend=4, _shrink_worst=4)
    assert new.last['move_reuse']['hits'] == dict(extend=60, shrink=60)
    assert_exact(state(a, ra, out0), state(b, rb, out1))
    assert len({id(p.H) for p in b.particles}) == len(b.particles)
    for group in original:
        maps = [p.H[group] for p in b.particles if group in p.H]
        assert len({id(value) for value in maps}) == len(maps)


def test_none_moves_are_reused_and_expire_each_epoch(scorer):
    a, ra, ca = make_step_case()
    for particle in a.particles:
        particle.H = deepcopy(a._adopt(ca['problem'], 0))
    candidate = MoveReuseStep(a, scorer, 'gpu')
    for ep in range(2):
        # Full held support makes original extend return None.
        for particle in a.particles:
            particle.H = deepcopy(a._adopt(ca['problem'], 0))
        ra.epoch2 = ep
        counts, _out = profile_calls(a, candidate, ra, ca)
        assert counts == dict(_extend=1, _shrink_worst=1)
        assert candidate.last['move_reuse']['hits'] == dict(extend=63, shrink=63)
        ca['problem'].rows_bmeas[32] += 1.


def test_unusual_numeric_types_bypass_reuse(scorer):
    a, ra, ca = make_step_case()
    b, rb, cb = make_step_case()
    full = a._adopt(ca['problem'], 0)
    unusual = {g: {sat: np.int64(z) for sat, z in zm.items()} for g, zm in full.items()}
    assert _ordered_integer_key(unusual) is None
    for runtime in (a, b):
        for particle in runtime.particles:
            particle.H = deepcopy(unusual)
    old, new = MoveBatchStep(a, scorer, 'gpu'), MoveReuseStep(b, scorer, 'gpu')
    out0 = old(ra, ca)
    counts, out1 = profile_calls(b, new, rb, cb)
    assert counts == dict(_extend=64, _shrink_worst=64)
    assert new.last['move_reuse']['uncacheable'] == dict(extend=64, shrink=64)
    assert_exact(state(a, ra, out0), state(b, rb, out1))


@pytest.mark.parametrize('event', ['skip', 'reseed', 'release', 'fresh'])
def test_maintenance_boundary(scorer, event):
    a, ra, ca = make_step_case()
    b, rb, cb = make_step_case()
    for rtk, ctx in ((ra, ca), (rb, cb)):
        group, ref, sat = ctx['problem'].pairs[0]
        if event == 'skip':
            ctx['problem'].ok = False
        elif event == 'reseed':
            rtk._rbpf_graph_reseeded = True
        elif event == 'release':
            rtk._rbpf_released = {(sat, group[1])}
        else:
            rtk._amb_init_epoch[(ref, group[1])] = rtk.epoch2
    old, new = MoveBatchStep(a, scorer, 'gpu'), MoveReuseStep(b, scorer, 'gpu')
    assert_exact(state(a, ra, old(ra, ca)), state(b, rb, new(rb, cb)))


def test_strict_policy_keeps_original_partial_state(scorer):
    a, ra, ca = make_step_case()
    b, rb, cb = make_step_case()
    new = MoveReuseStep(b, scorer, 'gpu')
    with np.errstate(all='raise'):
        with pytest.raises(FloatingPointError):
            a.step(ra, ca)
        with pytest.raises(FloatingPointError):
            new(rb, cb)
    assert new.last == dict(original_step_fallback='numpy_error_policy')
    assert_exact(state(a, ra, None), state(b, rb, None))


@pytest.mark.parametrize('mode', ['scalar', 'native_cpu', 'gpu'])
def test_feedback_commit_release_bits(scorer, mode):
    a, ra, ca = make_step_case(feedback=True)
    b, rb, cb = make_step_case(feedback=True)
    for runtime, ctx in ((a, ca), (b, cb)):
        runtime.cfg.p_survive = 1.
        for key in ('p_extend', 'p_shrink', 'p_release', 'p_adopt', 'p_adopt_cons', 'p_spawn', 'p_burst'):
            setattr(runtime.cfg, key, 1e-100)
        for particle in runtime.particles:
            particle.H, particle.logw = deepcopy(runtime._adopt(ctx['problem'], 0)), 0.
    old, new = MoveBatchStep(a, scorer, mode), MoveReuseStep(b, scorer, mode)
    for ep in range(4):
        ra.epoch2 = rb.epoch2 = ep
        if ep == 2:
            for rtk, ctx in ((ra, ca), (rb, cb)):
                g, _ref, sat = ctx['problem'].pairs[0]
                rtk._committed[(sat, g[1])] += 100.
        assert_exact(state(a, ra, old(ra, ca)), state(b, rb, new(rb, cb)))
    assert any(v[0] == 'inject' for v in ra.events)
    assert any(v[0] == 'release' for v in ra.events)
