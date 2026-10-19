"""Proposal sharing must preserve complete frozen RBPF state and draw order."""

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
from rbpf_likelihood_batch import BatchScorer  # noqa: E402
from rbpf_move_cases import comparison, make_step_case, state  # noqa: E402
from rbpf_move_key_memo import MoveKeyMemoStep  # noqa: E402
from rbpf_proposal_reuse import ProposalReuseStep, _ProposalMemo, _raw_statistics  # noqa: E402


@pytest.fixture(scope='module')
def scorer():
    dll = os.environ.get('GNSS_GPU_RBPF_PROBE_DLL')
    if not dll:
        pytest.skip('Explicit experimental DLL is required')
    with BatchScorer(dll) as value:
        yield value


def exact(a, b):
    result = comparison(a, b)
    assert result['discrete_equal'] and result['numeric_bits_equal'] and result['position_bits_equal'], result
    assert result['max_abs_error'] == 0., result


@pytest.mark.parametrize('seed', [20260710, 20260711, 20260712])
@pytest.mark.parametrize('options', [{}, dict(cluster=True), dict(uniform=False), dict(huber=False),
    dict(shadow=True, spawn=True), dict(resample=False)])
@pytest.mark.parametrize('mode', ['scalar', 'native_cpu', 'gpu'])
def test_complete_state_and_all_draws(scorer, seed, options, mode):
    a, ra, ca = make_step_case(seed=seed, **options)
    b, rb, cb = make_step_case(seed=seed, **options)
    old = a.step if mode == 'scalar' else MoveKeyMemoStep(a, scorer, mode)
    new = ProposalReuseStep(b, scorer, mode)
    b.step = new
    for ep in range(4):
        ra.epoch2 = rb.epoch2 = ep
        ca['obs'].t = cb['obs'].t = 100. + ep
        for ctx in (ca, cb):
            ctx['problem'].rows_bmeas[32:] += .001*ep
            ctx['problem'].Qb *= 1. + .001*ep
        exact(state(a, ra, old(ra, ca)), state(b, rb, new(rb, cb)))
        stats = new.last['proposal_reuse']
        assert stats['requests'] == b.cfg.n_particles
        assert stats['evaluations'] + stats['hits'] == stats['requests']
        assert stats['hits'] > 0 and stats['entries'] > 0
        assert len({id(p.H) for p in b.particles}) == b.cfg.n_particles


@pytest.mark.parametrize('mode', ['scalar', 'native_cpu', 'gpu'])
def test_dictionary_gauge_and_order_do_not_change_selected_h(scorer, mode):
    a, ra, ca = make_step_case()
    b, rb, cb = make_step_case()
    full = a._adopt(ca['problem'], 0)
    variants = [full, {g:{s:z+2**100 for s,z in zm.items()} for g,zm in full.items()},
        dict(reversed(list(full.items()))), {g:dict(reversed(list(zm.items()))) for g,zm in full.items()}]
    for runtime in (a, b):
        for i, particle in enumerate(runtime.particles):
            particle.H = deepcopy(variants[i % len(variants)])
    old, new = MoveKeyMemoStep(a, scorer, mode), ProposalReuseStep(b, scorer, mode)
    exact(state(a, ra, old(ra, ca)), state(b, rb, new(rb, cb)))
    assert new.last['proposal_reuse']['hits'] > 0


@pytest.mark.parametrize('event', ['skip', 'release', 'reseed', 'fresh'])
def test_maintenance(scorer, event):
    a, ra, ca = make_step_case()
    b, rb, cb = make_step_case()
    for rtk, ctx in ((ra, ca), (rb, cb)):
        group, ref, sat = ctx['problem'].pairs[0]
        if event == 'skip':
            ctx['problem'].ok = False
        elif event == 'release':
            rtk._rbpf_released = {(sat, group[1])}
        elif event == 'reseed':
            rtk._rbpf_graph_reseeded = True
        else:
            rtk._amb_init_epoch[(ref, group[1])] = rtk.epoch2
    old, new = MoveKeyMemoStep(a, scorer, 'gpu'), ProposalReuseStep(b, scorer, 'gpu')
    exact(state(a, ra, old(ra, ca)), state(b, rb, new(rb, cb)))


@pytest.mark.parametrize('mode', ['scalar', 'native_cpu', 'gpu'])
def test_feedback_commit_and_release(scorer, mode):
    a, ra, ca = make_step_case(feedback=True)
    b, rb, cb = make_step_case(feedback=True)
    for runtime, ctx in ((a, ca), (b, cb)):
        runtime.cfg.p_survive = 1.
        for name in ('p_extend', 'p_shrink', 'p_release', 'p_adopt', 'p_adopt_cons', 'p_spawn', 'p_burst'):
            setattr(runtime.cfg, name, 1e-100)
        for particle in runtime.particles:
            particle.H, particle.logw = deepcopy(runtime._adopt(ctx['problem'], 0)), 0.
    old, new = MoveKeyMemoStep(a, scorer, mode), ProposalReuseStep(b, scorer, mode)
    for ep in range(4):
        ra.epoch2 = rb.epoch2 = ep
        if ep == 2:
            for rtk, ctx in ((ra, ca), (rb, cb)):
                group, _ref, sat = ctx['problem'].pairs[0]
                rtk._committed[(sat, group[1])] += 100.
        exact(state(a, ra, old(ra, ca)), state(b, rb, new(rb, cb)))
    assert any(item[0] == 'inject' for item in ra.events)
    assert any(item[0] == 'release' for item in ra.events)


@pytest.mark.parametrize('prior', [0., -0., -.1, 1e308])
def test_warning_counts_and_partial_error_state(prior):
    a, ra, ca = make_step_case()
    b, rb, cb = make_step_case()
    for runtime in (a, b):
        runtime.cfg.p_survive = prior
    old, new = MoveKeyMemoStep(a), ProposalReuseStep(b)
    messages, errors = [], []
    for call, rtk, ctx in ((old, ra, ca), (new, rb, cb)):
        with warnings.catch_warnings(record=True) as caught, np.errstate(under='ignore', over='warn', divide='warn', invalid='warn'):
            warnings.simplefilter('always')
            try:
                call(rtk, ctx)
                errors.append(None)
            except (ValueError, OverflowError, FloatingPointError) as error:
                errors.append((type(error), str(error)))
            messages.append([(type(w.message), str(w.message)) for w in caught])
    assert errors[0] == errors[1]
    assert messages[0] == messages[1]
    exact(state(a, ra, None), state(b, rb, None))


@pytest.mark.parametrize('policy', ['raise', 'warn'])
def test_strict_original_step_fallback(scorer, policy):
    a, ra, ca = make_step_case()
    b, rb, cb = make_step_case()
    new = ProposalReuseStep(b, scorer, 'gpu')
    messages = []
    for call, rtk, ctx in ((a.step, ra, ca), (new, rb, cb)):
        with warnings.catch_warnings(record=True) as caught, np.errstate(under=policy):
            warnings.simplefilter('always')
            if policy == 'raise':
                with pytest.raises(FloatingPointError):
                    call(rtk, ctx)
            else:
                call(rtk, ctx)
            messages.append([(type(w.message), str(w.message)) for w in caught])
    assert messages[0] == messages[1]
    exact(state(a, ra, None), state(b, rb, None))
    assert new.last == dict(original_step_fallback='numpy_error_policy')


def test_nonstandard_prior_uses_previous_move_block():
    a, ra, ca = make_step_case()
    b, rb, cb = make_step_case()
    a.cfg.p_adopt = b.cfg.p_adopt = np.float64(.08)
    old, new = MoveKeyMemoStep(a), ProposalReuseStep(b)
    exact(state(a, ra, old(ra, ca)), state(b, rb, new(rb, cb)))
    assert new.last['proposal_reuse'] == dict(original_moves_fallback='nonstandard_common_input')


def test_nonstandard_common_integers_keep_previous_staging():
    a, ra, ca = make_step_case()
    b, rb, cb = make_step_case()
    full = a._adopt(ca['problem'], 0)
    uncommon = {g:{s:np.int64(z) for s,z in integers.items()} for g,integers in full.items()}
    old, new = MoveKeyMemoStep(a), ProposalReuseStep(b)
    out_a = old._moves(ca['problem'], None, [deepcopy(uncommon)], [], {}, -1000.)
    out_b = new._moves(cb['problem'], None, [deepcopy(uncommon)], [], {}, -1000.)
    exact(state(a, ra, out_a), state(b, rb, out_b))
    assert new.last['proposal_reuse'] == dict(original_moves_fallback='nonstandard_common_input')


def test_no_common_candidate_or_held_particle_still_draws_for_every_particle():
    a, ra, ca = make_step_case()
    b, rb, cb = make_step_case()
    for runtime in (a, b):
        for particle in runtime.particles:
            particle.H = {}
    old, new = MoveKeyMemoStep(a), ProposalReuseStep(b)
    out_a = old._moves(ca['problem'], None, [], [], {}, -1000.)
    out_b = new._moves(cb['problem'], None, [], [], {}, -1000.)
    exact(state(a, ra, out_a), state(b, rb, out_b))
    assert new.last['proposal_reuse']['evaluations'] == 1
    assert new.last['proposal_reuse']['hits'] == 63
    assert new.last['requested_moves'] == old.last['requested_moves'] == 64


class ChoiceObserver:
    def __init__(self, rng, fail_at=None):
        self.inner, self.fail_at, self.calls = rng, fail_at, 0

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def choice(self, *args, **kwargs):
        self.calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError('injected draw failure')
        result = self.inner.choice(*args, **kwargs)
        if kwargs.get('p') is not None:
            kwargs['p'][:] = 1./len(kwargs['p'])
        return result


def test_custom_rng_probability_mutation_is_not_shared():
    a, ra, ca = make_step_case()
    b, rb, cb = make_step_case()
    a.rng, b.rng = ChoiceObserver(a.rng), ChoiceObserver(b.rng)
    old, new = MoveKeyMemoStep(a), ProposalReuseStep(b)
    exact(state(a, ra, old(ra, ca)), state(b, rb, new(rb, cb)))
    assert a.rng.calls == b.rng.calls
    assert not new.last['proposal_reuse']['standard_generator']
    assert new.last['proposal_reuse']['hits'] == 0


def test_draw_exception_drops_probability_cache(monkeypatch):
    a, ra, ca = make_step_case()
    b, rb, cb = make_step_case()
    a.rng, b.rng = ChoiceObserver(a.rng, 2), ChoiceObserver(b.rng, 2)
    retained = []
    original = _ProposalMemo.__init__

    def observe(self, *args):
        original(self, *args)
        retained.append(weakref.ref(self))

    monkeypatch.setattr(_ProposalMemo, '__init__', observe)
    old, new = MoveKeyMemoStep(a), ProposalReuseStep(b)
    for call, rtk, ctx in ((old, ra, ca), (new, rb, cb)):
        with pytest.raises(RuntimeError, match='injected draw failure'):
            call(rtk, ctx)
    exact(state(a, ra, None), state(b, rb, None))
    gc.collect()
    assert retained and all(ref() is None for ref in retained)


def test_key_keeps_prior_index_order_and_nonfinite_bypass():
    rng = np.random.default_rng(20260710)
    common = ((.3, 2, {'common': 1}, 'adopt'),)
    likelihood = [-2., -5., -1.]
    memo = _ProposalMemo(likelihood, common, rng)
    records = [(.7, 0, {'first': 1}, 'survive'), *common]
    mass, p = memo.statistics(records)
    alternate_h = [(.7, 0, {'different': 2}, 'extend'), *common]
    assert memo.statistics(alternate_h)[1] is p
    assert memo.statistics(alternate_h)[0] == mass
    for other in ([ (.7, 1, {}, 'survive'), *common ], [(.6, 0, {}, 'survive'), *common]):
        actual = memo.statistics(other)
        reference = _raw_statistics(other, likelihood)
        assert np.float64(actual[0]).tobytes() == np.float64(reference[0]).tobytes()
        assert actual[1].tobytes() == reference[1].tobytes()
        assert actual[1] is not p
    overflow = _ProposalMemo(likelihood, (), rng)
    bad = [(1e308, 0, {}, 'a'), (1e308, 1, {}, 'b')]
    with np.errstate(all='ignore'):
        overflow.statistics(bad)
        overflow.statistics(bad)
    assert overflow.evaluations == 2 and not overflow.cache
