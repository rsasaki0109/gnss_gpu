"""Exact insertion-ordered keys and allocation-free memo eligibility."""

from copy import deepcopy
import gc
import importlib.util
import os
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
from bench_rbpf_feedback_mass import _bind_fixture_callbacks  # noqa: E402
from rbpf_epoch_memo import _EpochMemo  # noqa: E402
from rbpf_feedback_cases import make_feedback_case  # noqa: E402
from rbpf_feedback_mass import NativeFeedbackMassStep  # noqa: E402
from rbpf_likelihood_batch import BatchScorer  # noqa: E402
from rbpf_likelihood_cases import make_case  # noqa: E402
from rbpf_move_cases import comparison, state  # noqa: E402
from rbpf_move_reuse import MoveReuseStep, _ordered_integer_key  # noqa: E402
from rbpf_ordered_keys import OrderedKeyStep, _BINDINGS, _KEY_CODE  # noqa: E402


@pytest.fixture(scope="module")
def extensions():
    names = dict(
        extension="GNSS_GPU_RBPF_CANONICAL_EXT",
        parent_extension="GNSS_GPU_RBPF_PARENT_EXT",
        mass_extension="GNSS_GPU_RBPF_MASS_EXT",
        ordered_extension="GNSS_GPU_RBPF_ORDERED_EXT",
    )
    if any(not os.environ.get(n) for n in names.values()):
        pytest.skip("Explicit unchanged extensions and ordered-key extension required")
    return {k: os.environ[v] for k, v in names.items()}


@pytest.fixture(scope="module")
def module(extensions):
    spec = importlib.util.spec_from_file_location(
        "_rbpf_ordered_keys_probe", extensions["ordered_extension"]
    )
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.fixture
def engine(module):
    return module.OrderedKeys(_ordered_integer_key, _KEY_CODE, _BINDINGS)


@pytest.fixture(scope="module")
def scorer():
    dll = os.environ.get("GNSS_GPU_RBPF_PROBE_DLL")
    if not dll:
        pytest.skip("Explicit unchanged numerical DLL required")
    with BatchScorer(dll) as result:
        yield result


def exact(a, b):
    r = comparison(a, b)
    assert (
        r["discrete_equal"] and r["numeric_bits_equal"] and r["max_abs_error"] == 0.0
    ), r


@pytest.mark.parametrize("seed", [20260710, 20260711, 20260712])
@pytest.mark.parametrize("rows,groups", [(16, 1), (64, 4), (128, 8)])
def test_random_ordered_keys_and_type_support(engine, seed, rows, groups):
    _, _, hs = make_case(16, rows=rows, groups=groups, seed=seed, duplicates=False)
    rng = np.random.default_rng(seed)
    for _ in range(1000):
        h = hs[int(rng.integers(len(hs)))]
        expected = _ordered_integer_key(h)
        actual = engine.key(h)
        assert actual == expected and engine.supported(h) is (expected is not None)
        for (g, items), (gg, vv) in zip(actual, h.items(), strict=True):
            assert g is gg
            for (sat, z), (ss, zz) in zip(items, vv.items(), strict=True):
                assert sat is ss and z is zz
    assert engine.summary() == dict(
        key_requests=1000,
        support_requests=1000,
        native_keys=1000,
        native_support=1000,
        fallback_keys=0,
        fallback_support=0,
    )


def test_huge_gauges_insertion_order_and_empty_groups(engine):
    big = 2**16000
    h = {(3, 2): {9: big, 2: big + 2**90, 7: -big}, (1, 0): {}, (2, 1): {8: -5}}
    expected = _ordered_integer_key(h)
    assert engine.key(h) == expected and engine.supported(h)
    assert engine.key({}) == () and engine.supported({})
    reordered = dict(reversed(list(h.items())))
    assert engine.key(reordered) != expected
    reordered = deepcopy(h)
    reordered[(3, 2)] = dict(reversed(list(h[(3, 2)].items())))
    assert engine.key(reordered) != expected
    shifted = {g: {s: v + 7 for s, v in d.items()} for g, d in h.items()}
    assert engine.key(shifted) != expected
    # Fresh fingerprints must observe subsequent mutation, with no hidden cache.
    h[(3, 2)][9] += 1
    assert engine.key(h) == _ordered_integer_key(h) != expected


@pytest.mark.parametrize("kind", range(13))
def test_unsupported_types_do_not_invoke_callbacks(engine, kind):
    events = []

    class Dict(dict):
        def items(self):
            events.append("items")
            raise AssertionError("No subclass callback")

    class Int(int):
        def __int__(self):
            events.append("int")
            raise AssertionError("No conversion")

    class Tuple(tuple):
        def __iter__(self):
            events.append("iter")
            raise AssertionError("No subclass iteration")

    values = [
        None,
        [],
        Dict(),
        {(1,): {}},
        {(1, 0, 2): {}},
        {Tuple((1, 0)): {}},
        {(Int(1), 0): {}},
        {(True, 0): {}},
        {(1, 0): Dict()},
        {(1, 0): {np.int64(2): 3}},
        {(1, 0): {2: np.int64(3)}},
        {(1, 0): {2: Int(3)}},
        {(1, 0): {2: 3.0}},
    ]
    h = values[kind]
    assert _ordered_integer_key(h) is None and engine.key(h) is None
    assert engine.supported(h) is False and events == []


@pytest.mark.parametrize("name", ["type", "dict", "tuple", "int", "any", "len"])
@pytest.mark.parametrize("raises", [False, True])
def test_custom_binding_callbacks_and_exceptions(engine, monkeypatch, name, raises):
    events = []
    original = dict(
        zip(("type", "dict", "tuple", "int", "any", "len"), _BINDINGS, strict=True)
    )[name]
    h = {(1, 0): {1: 0, 2: 1}}

    def custom(*args):
        events.append(name)
        if raises:
            raise ValueError("original binding error")
        return original(*args)

    monkeypatch.setitem(_ordered_integer_key.__globals__, name, custom)

    def run(fn):
        events.clear()
        try:
            result = fn(h)
        except ValueError as e:
            result = (type(e), str(e))
        return result, list(events)

    expected = run(_ordered_integer_key)
    actual = run(engine.key)
    assert actual == expected
    eligible = run(engine.supported)
    if isinstance(expected[0], tuple) and expected[0] and expected[0][0] is ValueError:
        assert eligible == expected
    else:
        assert eligible == (expected[0] is not None, expected[1])
    assert (
        engine.summary()["fallback_keys"] == engine.summary()["fallback_support"] == 1
    )


def test_mutated_helper_code_falls_back(engine, monkeypatch):
    def replacement(h):
        return ("custom",)

    monkeypatch.setattr(_ordered_integer_key, "__code__", replacement.__code__)
    assert engine.key({}) == ("custom",) and engine.supported({}) is True
    assert engine.summary()["native_keys"] == engine.summary()["native_support"] == 0


def test_refcounts_gc_and_reinitialize(module, engine):
    g, sat, value = (2**70, 0), 2**80, 2**16000
    h = {g: {sat: value}}
    tracked = [
        g,
        sat,
        value,
        h,
        _ordered_integer_key,
        _KEY_CODE,
        _BINDINGS,
        module.OrderedKeys,
    ]
    gc.collect()
    before = [sys.getrefcount(v) for v in tracked]
    for _ in range(10000):
        fresh = module.OrderedKeys(_ordered_integer_key, _KEY_CODE, _BINDINGS)
        a = fresh.key(h)
        assert a[0][0] is g and a[0][1][0][0] is sat and a[0][1][0][1] is value
        assert fresh.supported(h)
        del a, fresh
    gc.collect()
    assert [sys.getrefcount(v) for v in tracked] == before
    engine.key(h)
    engine.__init__(_ordered_integer_key, _KEY_CODE, _BINDINGS)
    assert not any(engine.summary().values())
    done = []

    class Payload:
        def __del__(self):
            done.append(True)

    # Native source -> private Python globals -> native engine cycle.
    from types import FunctionType

    scope = dict(_ordered_integer_key.__globals__)
    function = FunctionType(_KEY_CODE, scope)
    cycle = module.OrderedKeys(function, _KEY_CODE, _BINDINGS)
    assert gc.is_tracked(cycle)
    scope.update(cycle=cycle, payload=Payload())
    del cycle, function, scope
    gc.collect()
    assert done == [True]


def test_canonical_callback_mutating_helper_bindings_is_observed(engine, monkeypatch):
    a, b = [], []
    original = _ordered_integer_key.__builtins__["any"]

    def custom(values):
        b.append("any")
        return original(values)

    def canonical(h):
        a.append(h)
        monkeypatch.setitem(_ordered_integer_key.__globals__, "any", custom)
        return (42,)

    h = {(1, 0): {1: 2}}
    expected = _EpochMemo(canonical, None, cache_dd=False)
    assert expected.canonical(h) == (42,)
    assert b == ["any", "any"]
    b.clear()
    canonical(h)
    assert engine.supported(h) and b == ["any", "any"]


@pytest.mark.parametrize("mode", ["scalar", "native_cpu", "gpu"])
@pytest.mark.parametrize("seed", [20260710, 20260711, 20260712])
@pytest.mark.parametrize(
    "options",
    [
        {},
        dict(cluster=True),
        dict(shadow=True, spawn=True),
        dict(feedback=False, preload=False, resample=False),
    ],
)
def test_four_epoch_all_state_metadata_rng_and_graph_events(
    extensions, scorer, mode, seed, options
):
    a, ra, ca = make_feedback_case(seed=seed, **options)
    b, rb, cb = make_feedback_case(seed=seed, **options)
    old = NativeFeedbackMassStep(
        a,
        scorer,
        mode,
        **{k: v for k, v in extensions.items() if k != "ordered_extension"},
    )
    call = OrderedKeyStep(b, scorer, mode, **extensions)
    a.step, b.step = old, call
    for ep in range(4):
        ra.epoch2 = rb.epoch2 = ep
        if options.get("feedback", True):
            _bind_fixture_callbacks(ra)
            _bind_fixture_callbacks(rb)
        oa, ob = old(ra, deepcopy(ca)), call(rb, deepcopy(cb))
        exact(state(a, ra, oa), state(b, rb, ob))
        exact(old.last, {k: v for k, v in call.last.items() if k != "ordered_keys"})
        s = call.last["ordered_keys"]
        assert s["key_requests"] == s["native_keys"] > 0
        assert s["support_requests"] == s["native_support"] > 0
        assert s["fallback_keys"] == s["fallback_support"] == 0


def test_strict_policy_and_helper_mutation_guard(extensions, monkeypatch):
    a, ra, ca = make_feedback_case()
    b, rb, cb = make_feedback_case()
    call = OrderedKeyStep(b, mode="scalar", **extensions)
    with np.errstate(under="raise"):
        for fn, rtk, ctx in ((a.step, ra, ca), (call, rb, cb)):
            with pytest.raises(FloatingPointError):
                fn(rtk, ctx)
    exact(state(a, ra, None), state(b, rb, None))
    assert call.last == dict(original_step_fallback="numpy_error_policy")
    monkeypatch.setattr(MoveReuseStep, "_moves", lambda *a: None)
    with pytest.raises(ValueError, match="helper changed"):
        call(rb, cb)
