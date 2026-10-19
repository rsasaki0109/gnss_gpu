"""Exact whole-state checks for canonical/DD reuse over a complete epoch."""

from copy import deepcopy
import gc
import os
from pathlib import Path
import sys
import warnings
import weakref

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'benchmarks'))
from rbpf_epoch_memo import EpochMemoStep, _EpochMemo  # noqa: E402
from rbpf_likelihood_batch import BatchScorer  # noqa: E402
from rbpf_move_cases import comparison, make_step_case, state  # noqa: E402
from rbpf_move_reuse import MoveReuseStep  # noqa: E402


@pytest.fixture(scope='module')
def scorer():
    dll = os.environ.get('GNSS_GPU_RBPF_PROBE_DLL')
    if not dll:
        pytest.skip('Explicit experimental DLL is required')
    with BatchScorer(dll) as value:
        yield value


def exact(a, b):
    check = comparison(a, b)
    assert check['discrete_equal'] and check['numeric_bits_equal'] and check['position_bits_equal'], check
    assert check['max_abs_error'] == 0., check


def released(caller):
    assert caller._epoch is caller._view is caller._oracle is None
    assert caller._scope['_canon_basin_id'] is caller.canonical


@pytest.mark.parametrize('seed', [20260710, 20260711, 20260712])
@pytest.mark.parametrize('options', [
    {}, dict(cluster=True), dict(uniform=False), dict(huber=False),
    dict(shadow=True, spawn=True), dict(resample=False),
])
@pytest.mark.parametrize('mode', ['scalar', 'native_cpu', 'gpu'])
def test_whole_epoch_same_backend_bits(scorer, seed, options, mode):
    a, ra, ca = make_step_case(seed=seed, **options)
    b, rb, cb = make_step_case(seed=seed, **options)
    old = a.step if mode == 'scalar' else MoveReuseStep(a, scorer, mode)
    new = EpochMemoStep(b, scorer, mode)
    b.step = new
    source_canonical = new._source_globals['_canon_basin_id']
    for ep in range(4):
        ra.epoch2 = rb.epoch2 = ep
        ca['obs'].t = cb['obs'].t = 100. + ep
        for ctx in (ca, cb):
            ctx['problem'].rows_bmeas[32:] += .001 * ep
            ctx['problem'].rows_v0[:32] += .01 * ep
            ctx['problem'].Qb *= 1. + ep * .001
        exact(state(a, ra, old(ra, ca)), state(b, rb, new(rb, cb)))
        released(new)
        assert new._source_globals['_canon_basin_id'] is source_canonical
        memo = new.last['epoch_memo']
        assert memo['canonical_source_calls'] < memo['canonical_requests']
        assert memo['DD_source_calls'] < memo['DD_requests']
    for name, function in new._functions.items():
        assert function.__code__ is new._source_functions[name].__code__
        assert function.__defaults__ == new._source_functions[name].__defaults__


@pytest.mark.parametrize('cache_dd', [False, True])
def test_gauge_and_order_preserve_particle_dictionaries(scorer, cache_dd):
    a, ra, ca = make_step_case()
    b, rb, cb = make_step_case()
    h = a._adopt(ca['problem'], 0)
    for integers in h.values():
        integers.pop(next(reversed(integers)))
    variants = [h, {g: {sat: z + 2**100 for sat,z in zm.items()} for g,zm in h.items()},
        dict(reversed(list(h.items()))), {g: dict(reversed(list(zm.items()))) for g,zm in h.items()}]
    for runtime in (a,b):
        for i,p in enumerate(runtime.particles):
            p.H = deepcopy(variants[i % 4])
    old, new = MoveReuseStep(a, scorer, 'gpu'), EpochMemoStep(b, scorer, 'gpu', cache_dd=cache_dd)
    exact(state(a, ra, old(ra, ca)), state(b, rb, new(rb, cb)))
    released(new)
    assert len({id(p.H) for p in b.particles}) == len(b.particles)


def test_dd_memo_separates_problems_and_bypasses_unusual_integers():
    runtime, _rtk, ctx = make_step_case()
    problem = ctx['problem']
    h = runtime._adopt(problem, 0)
    source = runtime.step.__func__.__globals__['_canon_basin_id']
    memo = _EpochMemo(source, runtime._pair_dd)
    value = memo.pair_dd(h, problem)
    gauge = {g: {sat: z + 2**100 for sat,z in zm.items()} for g,zm in h.items()}
    assert memo.pair_dd(gauge, problem) is value
    other = deepcopy(problem)
    other.pairs.reverse()
    result = memo.pair_dd(h, other)
    assert result is not value and result.tobytes() == runtime._pair_dd(h, other).tobytes()
    unusual = {g: {sat: np.int64(z) for sat,z in zm.items()} for g,zm in h.items()}
    first, second = memo.pair_dd(unusual, problem), memo.pair_dd(unusual, problem)
    assert first is not second and first.tobytes() == second.tobytes()
    assert memo.counts['DD_bypass'] == 2
    assert memo.summary()['problem_entries'] == 2


@pytest.mark.parametrize('event', ['skip', 'reseed', 'release', 'fresh'])
def test_maintenance_and_skip_boundaries(scorer, event):
    a, ra, ca = make_step_case()
    b, rb, cb = make_step_case()
    for rtk,ctx in ((ra,ca),(rb,cb)):
        group,ref,sat = ctx['problem'].pairs[0]
        if event == 'skip':
            ctx['problem'].ok = False
        elif event == 'reseed':
            rtk._rbpf_graph_reseeded = True
        elif event == 'release':
            rtk._rbpf_released = {(sat,group[1])}
        else:
            rtk._amb_init_epoch[(ref,group[1])] = rtk.epoch2
    old, new = MoveReuseStep(a, scorer, 'gpu'), EpochMemoStep(b, scorer, 'gpu')
    exact(state(a, ra, old(ra, ca)), state(b, rb, new(rb, cb)))
    released(new)


@pytest.mark.parametrize('policy', ['raise', 'warn'])
def test_original_error_warning_policy_and_partial_state(scorer, policy):
    a, ra, ca = make_step_case()
    b, rb, cb = make_step_case()
    new = EpochMemoStep(b, scorer, 'gpu')
    messages = []
    for call,rr,cc in ((a.step,ra,ca),(new,rb,cb)):
        with warnings.catch_warnings(record=True) as caught, np.errstate(under=policy):
            warnings.simplefilter('always')
            if policy == 'raise':
                with pytest.raises(FloatingPointError):
                    call(rr,cc)
            else:
                call(rr,cc)
            messages.append([(str(w.message), w.category) for w in caught])
    assert messages[0] == messages[1]
    exact(state(a, ra, None), state(b, rb, None))
    released(new)


def test_downstream_exception_releases_epoch_memo():
    a, ra, ca = make_step_case()
    b, rb, cb = make_step_case()
    retained = []

    class Failure:
        def score(self, *_args, **_kwargs):
            if new._epoch is not None:
                retained.append(weakref.ref(new._epoch))
            raise RuntimeError('injected score failure')

    failing = Failure()
    old, new = MoveReuseStep(a, failing, 'native_cpu'), EpochMemoStep(b, failing, 'native_cpu')
    for call,rr,cc in ((old,ra,ca),(new,rb,cb)):
        with pytest.raises(RuntimeError, match='injected score failure'):
            call(rr,cc)
    exact(state(a, ra, None), state(b, rb, None))
    released(new)
    gc.collect()
    assert retained and all(ref() is None for ref in retained)


def test_changed_consumer_method_is_rejected(scorer):
    runtime,rtk,ctx = make_step_case()
    new = EpochMemoStep(runtime, scorer, 'gpu')
    runtime._conditional_mean = lambda *_args: None
    with pytest.raises(ValueError, match='changed after binding'):
        new(rtk,ctx)


@pytest.mark.parametrize('mode', ['scalar', 'native_cpu', 'gpu'])
def test_feedback_commit_release_same_bits(scorer, mode):
    a,ra,ca = make_step_case(feedback=True)
    b,rb,cb = make_step_case(feedback=True)
    for runtime,ctx in ((a,ca),(b,cb)):
        runtime.cfg.p_survive = 1.
        for key in ('p_extend','p_shrink','p_release','p_adopt','p_adopt_cons','p_spawn','p_burst'):
            setattr(runtime.cfg,key,1e-100)
        for p in runtime.particles:
            p.H,p.logw = deepcopy(runtime._adopt(ctx['problem'],0)),0.
    old,new = MoveReuseStep(a,scorer,mode),EpochMemoStep(b,scorer,mode)
    for ep in range(4):
        ra.epoch2 = rb.epoch2 = ep
        if ep == 2:
            for rr,cc in ((ra,ca),(rb,cb)):
                g,_ref,sat = cc['problem'].pairs[0]
                rr._committed[(sat,g[1])] += 100.
        exact(state(a,ra,old(ra,ca)),state(b,rb,new(rb,cb)))
        released(new)
    assert any(e[0] == 'inject' for e in ra.events)
    assert any(e[0] == 'release' for e in ra.events)
