"""Live bindings and custom callback fallbacks for batched parameter packing."""

import builtins
from copy import deepcopy
import os
from pathlib import Path
import sys
from types import MethodType

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'benchmarks'))
from rbpf_cluster_fast_guard import ClusterBatchEngine, _PLAIN  # noqa: E402
from rbpf_irls_replay import oracle_from_snapshot  # noqa: E402
from rbpf_likelihood_cases import make_case  # noqa: E402


def setup():
    keys = ['GNSS_GPU_RBPF_CLUSTER_' + suffix for suffix in ('EXT', 'CANONICAL_EXT', 'PARENT_EXT')]
    if not all(os.environ.get(key) for key in keys):
        pytest.skip('Explicit standalone extensions required')
    raw, problem, hs = make_case(8, cluster=True)
    # Force real parent searches rather than all exact basin keys.
    raw.cluster_states = {(i,): s for i, s in enumerate(raw.cluster_states.values())}
    oracle = oracle_from_snapshot(dict(cfg=vars(raw.cfg), cluster_states=raw.cluster_states),
        os.environ[keys[1]], os.environ[keys[2]])
    return ClusterBatchEngine(os.environ[keys[0]]), oracle, problem, hs


@pytest.mark.parametrize('target,name', [('cluster', 'sum'), ('cluster', 'max'), ('cluster', 'len'),
    ('canonical', 'sorted'), ('canonical', 'min'), ('canonical', 'tuple'),
    ('parent', 'set'), ('parent', 'sorted'), ('parent', 'len'),
    ('packing', 'len'), ('packing', 'enumerate'), ('packing', 'float')])
def test_changed_global_builtin_runs_literal_callbacks_once(target, name):
    engine, oracle, problem, hs = setup()
    view = oracle._cluster_state.__self__
    ce = oracle.basin_id.__self__
    pe = view._functions['_basin_parent_score'].__closure__[0].cell_contents
    fn = {'cluster': oracle._cluster_state.__func__, 'canonical': ce.original, 'parent': pe.original, 'packing': _PLAIN}[target]
    scope, events = fn.__globals__, []
    had, old = name in scope, scope.get(name)
    literal = getattr(builtins, name)
    def custom(*args, **kwargs):
        events.append(name)
        return literal(*args, **kwargs)
    scope[name] = custom
    try:
        def observe(call):
            try:
                return call(hs, problem, oracle).tobytes(), None
            except TypeError as exc:
                return None, (type(exc), str(exc))
        expected = observe(_PLAIN)
        saved_events = list(events)
        events.clear()
        actual = observe(engine.prepare)
        assert actual == expected and events == saved_events
        assert events or target == 'packing' and name == 'float' and actual[1] is not None
        assert engine.batch_calls == 0 and engine.fallback_calls == 1
    finally:
        if had:
            scope[name] = old
        else:
            del scope[name]


@pytest.mark.parametrize('kind', ['code', 'bound_parent', 'missing_H', 'custom_key', 'custom_state', 'cfg_type'])
def test_mutated_bindings_and_unsupported_objects_preserve_results_or_errors(kind):
    engine, oracle, problem, hs = setup()
    view = oracle._cluster_state.__self__
    if kind == 'code':
        oracle._cluster_state.__func__.__code__ = (lambda self, h: None).__code__
    elif kind == 'bound_parent':
        view._bound['_basin_parent_score'] = MethodType(lambda self, a, b: (0, 3), view)
    elif kind == 'missing_H':
        next(iter(view.cluster_states.values())).pop('H')
    elif kind == 'custom_key':
        view.cluster_states['custom'] = next(iter(view.cluster_states.values()))
    elif kind == 'custom_state':
        class Dict(dict):
            pass
        view.cluster_states[(0,)] = Dict(view.cluster_states[(0,)])
    elif kind == 'cfg_type':
        class Custom:
            def __getattribute__(self, name):
                return object.__getattribute__(self, name)
        cfg = Custom()
        cfg.__dict__.update(vars(oracle.cfg))
        oracle.cfg = view.cfg = cfg
    outcomes = []
    for call in (_PLAIN, engine.prepare):
        try:
            outcomes.append((call(hs, problem, oracle).tobytes(), None))
        except (KeyError, TypeError, ValueError) as exc:
            outcomes.append((None, (type(exc), str(exc))))
    assert outcomes[0] == outcomes[1]
    assert engine.batch_calls == 0 and engine.fallback_calls == 1


def test_original_and_packed_parameter_arithmetic_signed_zero_and_subnormal():
    engine, oracle, problem, hs = setup()
    problem.pos[:] = 0.
    problem.Ppos[:] = -0.
    for st in oracle._cluster_state.__self__.cluster_states.values():
        st['pos'][:] = [-0., np.nextafter(0., 1.), np.nextafter(0., -1.)]
        st['P'] = np.array([[1., -0., 0.], [-0., 2., 0.], [0., -0., 3.]])
    before = deepcopy((problem, hs))
    with np.errstate(under='ignore'):
        expected = _PLAIN(hs, problem, oracle)
        actual = engine.prepare(hs, problem, oracle)
    assert expected.tobytes() == actual.tobytes() and engine.batch_calls == 1
    assert problem.pos.tobytes() == before[0].pos.tobytes() and hs == before[1]
