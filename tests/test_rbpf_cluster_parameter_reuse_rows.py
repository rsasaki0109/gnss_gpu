"""Identity row reuse: literal arithmetic, array bits, lifetimes and live inputs."""

from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'benchmarks'))
import rbpf_cluster_batch as initial  # noqa: E402
import rbpf_cluster_fast_guard as previous  # noqa: E402
import rbpf_cluster_parameter_reuse as selected  # noqa: E402


def test_literal_arithmetic_and_unchanged_selection_guard_and_live_globals():
    assert selected.packing_AST_exact()
    assert selected.ClusterBatchEngine._eligible is previous.ClusterBatchEngine._eligible
    assert selected._PLAIN is initial._PLAIN
    assert selected._PACKED.__globals__ is initial._PACKED.__globals__
    fn = selected.ClusterBatchEngine.prepare
    assert fn.__globals__ is initial.ClusterBatchEngine.prepare.__globals__
    assert fn.__code__.co_freevars == ('_PACKED',)
    assert fn.__closure__[0].cell_contents is selected._PACKED
    assert selected._PACKED.__code__.co_freevars == ('_identity',)
    assert selected._PACKED.__closure__[0].cell_contents is selected._identity


def fixture(seed, layout):
    rng = np.random.default_rng(seed)
    pos = np.array([0., -0., np.nextafter(0., 1.)])
    raw = rng.integers(0, 2**64, size=(3, 3), dtype=np.uint64)
    # Include signed zero, quiet and signalling NaN payloads, subnormals, infinity.
    raw.flat[:6] = [0x8000000000000000, 0x7ff8000000001234, 0x7ff0000000000001,
        0x0000000000000001, 0xfff0000000000000, 0x8000000000000001]
    prior = raw.view(np.float64)
    if layout == 'F':
        prior = np.asfortranarray(prior)
    elif layout == 'reverse':
        prior = prior[::-1, ::-1]
    problem = SimpleNamespace(pos=pos, Ppos=prior)
    a = dict(pos=np.array([-0., 0., np.nextafter(0., 1.)]), P=prior)
    b = deepcopy(a)
    assert a is not b
    states = [None, a, b, a, None, b] * 11
    rng.shuffle(states)
    return problem, states


@pytest.mark.parametrize('seed', [0, 7, 20260710])
@pytest.mark.parametrize('layout', ['C', 'F', 'reverse'])
def test_complete_rows_match_all_float_bits_and_distinct_equal_states(seed, layout):
    problem, states = fixture(seed, layout)
    hs = [{}] * len(states)
    expected = initial._PACKED(hs, problem, None, states)
    actual = selected._PACKED(hs, problem, None, states)
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    assert actual.tobytes() == expected.tobytes()
    assert actual.flags.c_contiguous and actual.flags.owndata


def test_first_use_only_executes_original_numpy_expressions(monkeypatch):
    problem, states = fixture(1, 'reverse')
    counts = dict.fromkeys(('zeros', 'asarray', 'all', 'isfinite'), 0)
    for name in counts:
        literal = getattr(np, name)
        def counted(*args, _name=name, _literal=literal, **kwargs):
            counts[_name] += 1
            return _literal(*args, **kwargs)
        monkeypatch.setattr(np, name, counted)
    selected._PACKED([{}] * len(states), problem, None, states)
    # None and two distinct, equal-valued state dictionaries are three misses.
    assert counts == dict(zeros=3, asarray=7, all=2, isfinite=2)


def test_empty_parameters():
    problem, _ = fixture(2, 'C')
    result = selected._PACKED([], problem, None, [])
    assert result.shape == (0, 12) and result.dtype == np.float64


def test_no_references_retained_and_mutations_visible_on_next_call():
    problem, states = fixture(3, 'F')
    hs = [{}] * len(states)
    objects = [problem, states, hs, *[s for s in states if s is not None]]
    refs = [sys.getrefcount(value) for value in objects]
    for _ in range(1000):
        result = selected._PACKED(hs, problem, None, states)
        del result
    assert [sys.getrefcount(value) for value in objects] == refs
    first = selected._PACKED(hs, problem, None, states)
    for s in states:
        if s is not None:
            s['pos'][:] = 13.
            s['P'] = np.eye(3) * 7.
    problem.pos[:] = 11.
    problem.Ppos = np.eye(3) * -0.
    actual = selected._PACKED(hs, problem, None, states)
    expected = initial._PACKED(hs, problem, None, states)
    assert actual.tobytes() == expected.tobytes() != first.tobytes()


def test_new_identity_primitive_is_not_a_live_original_global():
    problem, states = fixture(4, 'C')
    scope = selected._PACKED.__globals__
    sentinel = object()
    saved = {name: scope.get(name, sentinel) for name in ('id', '_identity')}
    def forbidden(*args):
        raise AssertionError('The original score never reads these names')
    try:
        scope.update(id=forbidden, _identity=forbidden)
        expected = initial._PACKED([{}] * len(states), problem, None, states)
        actual = selected._PACKED([{}] * len(states), problem, None, states)
        assert actual.tobytes() == expected.tobytes()
    finally:
        for name, value in saved.items():
            if value is sentinel:
                del scope[name]
            else:
                scope[name] = value
