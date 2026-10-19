"""Exact batched selection, guarded fallbacks and complete score/PF state bits."""

from contextlib import ExitStack
from copy import deepcopy
import gc
import os
from pathlib import Path
import sys
from types import MethodType
import warnings

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'benchmarks'))
from rbpf_cluster_batch import ClusterBatchEngine, ClusterBatchScorer, restored_source_ast  # noqa: E402
from rbpf_irls_replay import oracle_from_snapshot  # noqa: E402
from rbpf_likelihood_cases import make_case, make_oracle  # noqa: E402
from rbpf_register_lu import RegisterLUScorer  # noqa: E402
from rbpf_feedback_cases import make_feedback_case  # noqa: E402
from rbpf_feedback_mass import NativeFeedbackMassStep  # noqa: E402
from rbpf_move_cases import state, comparison  # noqa: E402


@pytest.fixture(scope='module')
def paths():
    names = ('DLL', 'EXT', 'CANONICAL_EXT', 'PARENT_EXT', 'MASS_EXT')
    env = {name: os.environ.get('GNSS_GPU_RBPF_CLUSTER_' + name) for name in names}
    if not all(env.values()):
        pytest.skip('Explicit frozen DLL and standalone cluster/canonical/parent/mass extensions required')
    return env


@pytest.fixture(scope='module')
def engine(paths):
    return ClusterBatchEngine(paths['EXT'])


def replay(raw, paths):
    return oracle_from_snapshot(dict(cfg=vars(raw.cfg), cluster_states=raw.cluster_states),
        paths['CANONICAL_EXT'], paths['PARENT_EXT'])


def bits(a, b):
    check = comparison(a, b)
    assert check['discrete_equal'] and check['numeric_bits_equal'] and check['max_abs_error'] == 0., check


def result_bits(a, b):
    av, ad = a
    bv, bd = b
    assert av.dtype == bv.dtype and av.shape == bv.shape and av.tobytes() == bv.tobytes()
    assert list(ad) == list(bd)
    for name in ad:
        if isinstance(ad[name], np.ndarray):
            assert ad[name].dtype == bd[name].dtype and ad[name].shape == bd[name].shape
            assert ad[name].tobytes() == bd[name].tobytes()
        else:
            assert ad[name] == bd[name]


@pytest.mark.parametrize('seed', [20260710, 20260711, 20260712])
@pytest.mark.parametrize('n,groups,sats', [(4, 1, 8), (16, 4, 16), (32, 8, 32)])
def test_random_batch_selection_identity_gauge_and_order(engine, seed, n, groups, sats):
    rng = np.random.default_rng(seed + n + groups)
    for sample in range(30):
        minimum = int(rng.integers(1, 10))
        fraction = [0., .5, .8, 1., np.nextafter(.8, -np.inf), np.nextafter(.8, np.inf)][sample % 6]
        raw = make_oracle(cluster_relin=1, cluster_parent_min=minimum, cluster_parent_frac=float(fraction))
        hypotheses = []
        for _ in range(n):
            h = {}
            for gi in rng.permutation(groups):
                group = (int(gi) - 2, int(gi) % 3)
                gauge = (1 << (80 + sample * 7)) * (int(rng.integers(0, 2)) * 2 - 1)
                if rng.random() < .1:
                    continue
                h[group] = {int(s): gauge + int(rng.integers(-2, 3))
                    for s in rng.permutation(sats) if rng.random() > .2}
            hypotheses.append(h)
        states = {(i,): dict(H=deepcopy(hypotheses[i]), pos=np.ones(3) * i, P=np.eye(3))
            for i in range(min(5, n))}
        if sample % 3 == 0:
            states[raw.basin_id(hypotheses[-1])] = next(iter(states.values()))
        raw.cluster_states = states
        expected = [raw._cluster_state(h) for h in hypotheses]
        before = deepcopy((hypotheses, states))
        actual, canonical, parents, exact, selected, ties = engine.module.select_states(hypotheses, states, minimum, float(fraction))
        assert len(actual) == n and all(a is b for a, b in zip(actual, expected, strict=True))
        eligible = [h for h in hypotheses if h and sum(max(len(v) - 1, 0) for v in h.values()) >= minimum]
        hits = sum(raw.basin_id(h) in states for h in eligible)
        assert canonical == len(eligible) and exact == hits and parents == (len(eligible) - hits) * len(states)
        assert selected + exact == sum(x is not None for x in actual) and 0 <= ties <= len(eligible) - hits
        bits(before, (hypotheses, states))
        shifted = [{g: {s: z + (1 << 16000) for s, z in reversed(list(zs.items()))}
            for g, zs in reversed(list(h.items()))} for h in hypotheses]
        reordered = dict(reversed(list(states.items())))
        other = engine.module.select_states(shifted, reordered, minimum, float(fraction))[0]
        assert all(a is b for a, b in zip(other, expected, strict=True))


@pytest.mark.parametrize('minimum', [1, 2, 3, 4, 2**200])
@pytest.mark.parametrize('fraction', [0., .8, 1., float('nan'), float('inf'), -float('inf')])
def test_exact_hit_low_support_ties_and_thresholds(engine, minimum, fraction):
    raw = make_oracle(cluster_relin=1, cluster_parent_min=minimum, cluster_parent_frac=fraction)
    h = {(1, 0): {1: 0, 2: 1, 3: 2, 4: 3}}
    a = dict(H={(1, 0): {0: 99, 1: 100, 2: 101, 3: 102, 4: 103}})
    b = dict(H={(1, 0): {1: -100, 2: -99, 3: -98, 4: -96}})
    for parents in ([], [a], [b], [a, b], [b, a], [a, a]):
        raw.cluster_states = {(i,): v for i, v in enumerate(parents)}
        expected = raw._cluster_state(h)
        assert engine.module.select_states([h], raw.cluster_states, minimum, fraction)[0][0] is expected
    raw.cluster_states[raw.basin_id(h)] = b
    assert engine.module.select_states([h, {}, {(1, 0): {1: 0}}], raw.cluster_states, minimum, fraction)[0][0] is raw._cluster_state(h)


def test_preflight_custom_types_and_errors_do_not_call_custom_arithmetic(engine):
    events = []
    class Integer(int):
        def __sub__(self, other):
            events.append('subtract')
            raise ArithmeticError('custom integer')
    class Dict(dict):
        pass
    h = {(1, 0): {1: 0, 2: 1}}
    state = {(0,): dict(H=deepcopy(h))}
    bad = [None, Dict(h), {(1, 0): {1: Integer(3), 2: 4}}, {(1, 0): {1: np.int64(3), 2: 4}},
        {(1, 0): {1: True, 2: 4}}, {(1, 0): None}]
    for child in bad:
        assert engine.module.select_states([child], state, 1, .8) is None
        assert engine.module.select_states([h], {(0,): dict(H=child)}, 1, .8) is None
    assert events == []
    for minimum in (0, -1, True, 1.):
        assert engine.module.select_states([h], state, minimum, .8) is None
    assert engine.module.select_states([h], state, 1, 1) is None
    assert engine.module.select_states([h], {('custom',): dict(H=h)}, 1, .8) is None


def test_no_retained_inputs_and_updates_between_calls(engine):
    h = {(1, 0): {1001: 0, 1002: 1, 1003: 2}}
    p = {(1, 0): {1001: 100, 1002: 101, 1003: 102}}
    states = {(0,): dict(H=p)}
    hs = [h, {}, h]
    objects = [h, p, states, hs, *states.values(), *h.values(), *p.values()]
    refs = [sys.getrefcount(o) for o in objects]
    for _ in range(10000):
        result = engine.module.select_states(hs, states, 2, .8)
        assert result[0][0] is states[(0,)] and result[0][1] is None
        del result
    gc.collect()
    assert refs == [sys.getrefcount(o) for o in objects]
    p[(1, 0)][1003] += 1
    assert engine.module.select_states(hs, states, 2, .8)[0][0] is None
    h[(1, 0)][1003] += 1
    assert engine.module.select_states(hs, states, 2, .8)[0][0] is states[(0,)]


@pytest.fixture(scope='module')
def scorers(paths):
    with ExitStack() as stack:
        yield (stack.enter_context(RegisterLUScorer(paths['DLL'])),
            stack.enter_context(ClusterBatchScorer(paths['DLL'], paths['EXT'], enabled=False)),
            stack.enter_context(ClusterBatchScorer(paths['DLL'], paths['EXT'])))


@pytest.mark.parametrize('seed', [20260710, 20260711, 20260712])
@pytest.mark.parametrize('rows,groups', [(16, 1), (64, 4), (128, 8)])
@pytest.mark.parametrize('cluster,uniform,huber', [(True, True, True), (True, False, True), (True, True, False),
    (False, True, True), (False, False, True), (False, True, False)])
def test_score_all_diagnostics_status_input_and_logical_counters(scorers, paths, seed, rows, groups, cluster, uniform, huber):
    raw, problem, hs = make_case(8, rows=rows, groups=groups, seed=seed, cluster=cluster, marg_uniform=uniform, huber=huber)
    before = deepcopy((problem, hs))
    native_before = scorers[-1].cluster_engine.batch_calls
    for mode in ('native_cpu', 'gpu'):
        results, counts = [], []
        for scorer in scorers:
            oracle = replay(raw, paths)
            results.append(scorer.score(hs, problem, oracle, mode, diagnostics=True))
            view = oracle._cluster_state.__self__
            ce = oracle.basin_id.__self__
            pe = view._functions['_basin_parent_score'].__closure__[0].cell_contents
            counts.append((ce.counts(), pe.counts()))
        for actual in results[1:]:
            result_bits(results[0], actual)
        assert counts[0] == counts[1] == counts[2]
    assert scorers[-1].cluster_engine.batch_calls - native_before == (2 if cluster else 0)
    bits(before, (problem, hs))


@pytest.mark.parametrize('kind', ['custom_cluster', 'custom_parent', 'custom_native_parent', 'custom_canonical',
    'custom_integer', 'minimum_zero', 'nonfinite_centre', 'wide_centre', 'dtype', 'strict', 'under_warn'])
def test_guarded_fallback_preserves_callbacks_results_errors_warnings(scorers, paths, kind):
    outcomes, events_all = [], []
    for scorer in (scorers[0], scorers[-1]):
        raw, problem, hs = make_case(4, rows=64, groups=4, cluster=True)
        oracle = replay(raw, paths)
        view = oracle._cluster_state.__self__
        events, restore = [], None
        if kind == 'custom_cluster':
            old = oracle._cluster_state
            def custom(h):
                events.append(('state', len(h)))
                return old(h)
            oracle._cluster_state = custom
        elif kind == 'custom_parent':
            old = view._functions['_basin_parent_score']
            def custom(v, h, p):
                events.append(('parent', len(h), len(p)))
                return old(v, h, p)
            view._functions['_basin_parent_score'] = custom
        elif kind == 'custom_native_parent':
            pe = view._functions['_basin_parent_score'].__closure__[0].cell_contents
            old = pe.module.score
            def custom(h, p):
                events.append(('native_parent', len(h), len(p)))
                return old(h, p)
            pe.module.score = custom
            def restore():
                pe.module.score = old
        elif kind == 'custom_canonical':
            ce = oracle.basin_id.__self__
            old = ce.module.canonical
            def custom(h):
                events.append(('canonical', len(h)))
                return old(h)
            ce.module.canonical = custom
            def restore():
                ce.module.canonical = old
        elif kind == 'custom_integer':
            class Integer(int):
                pass
            next(iter(next(h for h in hs if h).values()))[9999] = Integer(12)
        elif kind == 'minimum_zero':
            oracle.cfg.cluster_parent_min = 0
        elif kind in ('nonfinite_centre', 'wide_centre'):
            next(iter(view.cluster_states.values()))['pos'][:] = np.nan if kind == 'nonfinite_centre' else 1e200
        elif kind == 'dtype':
            next(iter(view.cluster_states.values()))['P'] = np.eye(3, dtype=np.float32)
        try:
            with np.errstate(all='raise' if kind == 'strict' else 'warn', under='warn' if kind == 'under_warn' else 'ignore'):
                with warnings.catch_warnings(record=True) as seen:
                    warnings.simplefilter('always')
                    try:
                        result, error = scorer.score(hs, problem, oracle, 'gpu', diagnostics=True), None
                    except (ArithmeticError, ValueError) as exc:
                        result, error = None, (type(exc), str(exc))
                    outcomes.append((result, error, [(w.category, str(w.message)) for w in seen]))
            events_all.append(events)
        finally:
            if restore is not None:
                restore()
    assert outcomes[0][1:] == outcomes[1][1:] and events_all[0] == events_all[1]
    if outcomes[0][0] is not None:
        result_bits(outcomes[0][0], outcomes[1][0])


@pytest.mark.parametrize('seed', [20260710, 20260711, 20260712])
@pytest.mark.parametrize('options', [(16, 6, 16, 1), (64, 6, 64, 4), (64, 32, 64, 4), (128, 32, 128, 8)])
def test_complete_pf_four_epochs_state_rng_graph_and_prior_metadata(scorers, paths, seed, options):
    n, unique, rows, groups = options
    traces, metadata = [], []
    before = scorers[-1].cluster_engine.batch_calls
    for scorer in (scorers[0], scorers[-1]):
        runtime, rtk, ctx = make_feedback_case(n, unique, rows=rows, groups=groups, cluster=True,
            shadow=True, spawn=True, feedback=True, preload=True, seed=seed)
        call = NativeFeedbackMassStep(runtime, scorer, 'gpu', extension=paths['CANONICAL_EXT'],
            parent_extension=paths['PARENT_EXT'], mass_extension=paths['MASS_EXT'])
        runtime.step = call
        trace, meta = [], []
        for _ in range(4):
            output = call(rtk, ctx)
            trace.append(deepcopy(state(runtime, rtk, output)))
            meta.append(deepcopy(call.last))
        traces.append(trace)
        metadata.append(meta)
    for a, b in zip(traces[0], traces[1], strict=True):
        bits(a, b)
    for a, b in zip(metadata[0], metadata[1], strict=True):
        bits(a, b)
    assert scorers[-1].cluster_engine.batch_calls - before == 4


def test_source_AST_live_globals_and_no_module_injection(scorers):
    import rbpf_likelihood_batch as original
    assert restored_source_ast()[0] == restored_source_ast()[1]
    assert scorers[-1].score.__func__.__globals__ is original.BatchScorer.score.__globals__
    assert scorers[1].score.__func__.__code__ is original.BatchScorer.score.__code__
    assert not any(name.startswith('__rbpf_cluster_') for name in vars(original))
    assert len(scorers[-1].score.__func__.__closure__) == 1


def test_repeated_context_lifetime_close_and_returned_array_ownership(paths):
    raw, problem, hs = make_case(3, cluster=True)
    for _ in range(12):
        with ClusterBatchScorer(paths['DLL'], paths['EXT']) as scorer:
            values, _ = scorer.score(hs, problem, replay(raw, paths))
            saved = values.copy()
            scorer.score(hs[:1], problem, replay(raw, paths))
        scorer.close()
        assert values.tobytes() == saved.tobytes()
        with pytest.raises(AssertionError):
            scorer.score(hs, problem, replay(raw, paths))
