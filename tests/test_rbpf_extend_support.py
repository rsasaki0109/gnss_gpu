"""Exact extend support, integer assembly, error order and full PF states."""

from copy import deepcopy
import os
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
import rbpf_extend_support as selected  # noqa: E402
from rbpf_feedback_mass import NativeFeedbackMassStep  # noqa: E402
from rbpf_feedback_cases import make_feedback_case  # noqa: E402
from rbpf_move_cases import comparison, state  # noqa: E402
from rbpf_register_lu import RegisterLUScorer  # noqa: E402


@pytest.fixture(scope="module")
def paths():
    names = ("DLL", "EXTEND_EXT", "CANONICAL_EXT", "PARENT_EXT", "MASS_EXT")
    found = {n: os.environ.get("GNSS_GPU_RBPF_EXTEND_" + n) for n in names}
    if not all(found.values()):
        pytest.skip("Explicit frozen numeric DLL and standalone extensions required")
    return found


@pytest.fixture(scope="module")
def scorer(paths):
    with RegisterLUScorer(paths["DLL"]) as owned:
        yield owned


def exact(a, b):
    c = comparison(a, b)
    assert (
        c["discrete_equal"] and c["numeric_bits_equal"] and c["max_abs_error"] == 0.0
    ), c


def make(
    paths,
    scorer=None,
    *,
    candidate=True,
    backend="scalar",
    seed=20260710,
    options=(64, 6, 64, 4, False),
):
    n, u, m, g, cluster = options
    runtime, rtk, ctx = make_feedback_case(
        n,
        u,
        rows=m,
        groups=g,
        cluster=cluster,
        seed=seed,
        shadow=True,
        spawn=True,
        feedback=True,
        preload=True,
    )
    extensions = dict(
        extension=paths["CANONICAL_EXT"],
        parent_extension=paths["PARENT_EXT"],
        mass_extension=paths["MASS_EXT"],
    )
    cls = selected.NativeExtendSupportStep if candidate else NativeFeedbackMassStep
    call = cls(
        runtime,
        scorer,
        backend,
        **extensions,
        **(dict(extend_extension=paths["EXTEND_EXT"]) if candidate else {}),
    )
    runtime.step = call
    return runtime, rtk, ctx, call


def test_ast_and_original_solver_block(paths):
    runtime, _, ctx, call = make(paths)
    assert selected.AST_PROOF["restored_AST_equal"]
    assert call.extend_engine.copy_literal
    assert call._functions["_extend"].__globals__ is call._scope
    engine = call.extend_engine
    for h in [p.H for p in runtime.particles]:
        exact(
            engine.original_support(runtime, h, ctx["problem"]),
            engine.support(runtime, h, ctx["problem"]),
        )
    assert engine.support_native == 64 and engine.support_fallback == 0


@pytest.mark.parametrize("seed", range(8))
def test_random_gauge_order_support_and_independent_assembly(paths, seed):
    runtime, _, ctx, call = make(
        paths, seed=20260710 + seed, options=(64, 32, 128, 8, True)
    )
    engine, prob = call.extend_engine, ctx["problem"]
    for particle in runtime.particles:
        h = deepcopy(particle.H)
        for integers in h.values():
            for sat in integers:
                integers[sat] += 2**300
        expected = engine.original_support(runtime, h, prob)
        actual = engine.support(runtime, h, prob)
        exact(expected, actual)
        dd, idxH, idxC = actual
        assert idxH.dtype == np.dtype("intp")
        assert all(type(i) is int for i in idxC)
        if len(idxH) and idxC:
            conditional = np.random.default_rng(seed).normal(0.0, 10.0, len(idxC))
            left = engine.original_finish(h, prob, idxC, conditional)
            right = engine.finish(h, prob, idxC, conditional)
            exact(left, right)
            assert list(left) == list(right)
            assert all(list(left[g]) == list(right[g]) for g in left)
            assert right is not h and all(right[g] is not h[g] for g in h)
    assert engine.support_native == 64 and engine.support_fallback == 0
    assert engine.finish_native > 0 and engine.finish_fallback == 0


@pytest.mark.parametrize(
    "value",
    [
        0.0,
        -0.0,
        0.5,
        -0.5,
        1.5,
        2.5,
        -2.5,
        np.nextafter(0.5, 0.0),
        np.nextafter(0.5, 1.0),
        np.nextafter(-0.5, -1.0),
        np.nextafter(0.0, 1.0),
        1e100,
        -1e100,
        np.finfo(float).max,
    ],
)
def test_finite_rint_integer_bits_and_huge_gauge(paths, value):
    _, _, ctx, call = make(paths)
    prob = ctx["problem"]
    g, ref, j = prob.pairs[0]
    h = {g: {ref: 2**4000, j: 2**4000 + 7}}
    conditional = np.array([value], dtype=float)
    engine = call.extend_engine
    exact(
        engine.original_finish(h, prob, [0], conditional),
        engine.finish(h, prob, [0], conditional),
    )
    assert engine.finish_native == 1 and engine.finish_fallback == 0


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_scalar_conversion_errors_fallback(paths, value):
    _, _, ctx, call = make(paths)
    prob, engine = ctx["problem"], call.extend_engine
    g, ref, j = prob.pairs[0]
    h = {g: {ref: 0, j: 1}}
    conditional = np.array([value])
    errors = []
    for fn in (engine.original_finish, engine.finish):
        with pytest.raises((ValueError, OverflowError)) as caught:
            fn(h, prob, [0], conditional)
        errors.append((type(caught.value), str(caught.value)))
    assert errors[0] == errors[1]
    assert engine.finish_native == 0 and engine.finish_fallback == 1


def test_all_pair_conversion_overflow_before_empty_move(paths):
    runtime, _, ctx, call = make(paths)
    prob, engine = ctx["problem"], call.extend_engine
    g, ref, j = prob.pairs[-1]
    h = {g: {ref: 0, j: 10**400}}
    errors = []
    for fn in (engine.original_support, engine.support):
        with pytest.raises(OverflowError) as caught:
            fn(runtime, h, prob)
        errors.append(str(caught.value))
    assert errors[0] == errors[1]
    assert engine.support_native == engine.support_fallback == 0


@pytest.mark.parametrize(
    "kind", ["dict_subclass", "numpy_integer", "none", "bool", "pair_tuple"]
)
def test_unsupported_support_executes_original_once(paths, kind):
    runtime, _, ctx, call = make(paths)
    prob, engine = ctx["problem"], call.extend_engine
    g, ref, j = prob.pairs[0]
    h = {g: {ref: 0, j: 1}}
    if kind == "dict_subclass":

        class Mapping(dict):
            pass

        h = Mapping(h)
    elif kind == "numpy_integer":
        h[g][j] = np.int64(1)
    elif kind == "none":
        h[g][j] = None
    elif kind == "bool":
        h[g][j] = True
    elif kind == "pair_tuple":
        prob.pairs = tuple(prob.pairs)
    exact(engine.original_support(runtime, h, prob), engine.support(runtime, h, prob))
    assert engine.support_native == 0 and engine.support_fallback == 1


def test_rint_callback_and_copy_callback_order(paths, monkeypatch):
    _, _, ctx, call = make(paths)
    prob, engine = ctx["problem"], call.extend_engine
    g, ref, j = prob.pairs[0]
    h, indices, conditional = {g: {ref: 0, j: 1}}, [0, 0], np.array([0.5, 1.5])
    events = []
    original = np.rint

    def counted(value):
        events.append(("rint", float(value)))
        return original(value)

    monkeypatch.setattr(np, "rint", counted)
    engine.original_finish(h, prob, indices, conditional)
    expected = list(events)
    events.clear()
    engine.finish(h, prob, indices, conditional)
    assert events == expected and engine.finish_fallback == 1
    monkeypatch.setattr(np, "rint", original)
    copy = engine.scope["_copy_H"]

    def copied(value):
        events.append(("copy", len(value)))
        return copy(value)

    monkeypatch.setitem(engine.scope, "_copy_H", copied)
    events.clear()
    engine.original_finish(h, prob, indices, conditional)
    expected = list(events)
    events.clear()
    engine.finish(h, prob, indices, conditional)
    assert events == expected and engine.finish_fallback == 2


@pytest.mark.parametrize("layout", ["strided", "unaligned", "float32", "subclass"])
def test_finish_unsupported_buffer_fallback(paths, layout):
    _, _, ctx, call = make(paths)
    prob, engine = ctx["problem"], call.extend_engine
    g, ref, j = prob.pairs[0]
    h = {g: {ref: 0, j: 1}}
    if layout == "strided":
        conditional = np.arange(4, dtype=float)[::2]
    elif layout == "unaligned":
        conditional = np.ndarray((2,), dtype=float, buffer=bytearray(17), offset=1)
    elif layout == "float32":
        conditional = np.ones(2, dtype=np.float32)
    else:

        class Array(np.ndarray):
            pass

        conditional = np.ones(2).view(Array)
    exact(
        engine.original_finish(h, prob, [0, 0], conditional),
        engine.finish(h, prob, [0, 0], conditional),
    )
    assert engine.finish_native == 0 and engine.finish_fallback == 1


def test_native_preflight_and_repeated_calls_release_inputs(paths):
    runtime, _, ctx, call = make(paths)
    prob, engine = ctx["problem"], call.extend_engine
    h = deepcopy(runtime.particles[2].H)
    start = sys.getrefcount(h), sys.getrefcount(prob.pairs)
    for _ in range(1000):
        dd, held, missing = engine.support(runtime, h, prob)
        if missing:
            engine.finish(h, prob, missing, np.zeros(len(missing)))
    assert (sys.getrefcount(h), sys.getrefcount(prob.pairs)) == start
    h.clear()
    dd, held, missing = engine.support(runtime, h, prob)
    assert not len(held) and not missing and np.isnan(dd).all()
    output = np.full(len(prob.pairs), 123.0)
    assert engine.module.support([], prob.pairs, output) is None
    assert np.all(output == 123.0)


@pytest.mark.parametrize("backend", ["native_cpu", "gpu"])
@pytest.mark.parametrize("seed", [20260710, 20260711])
@pytest.mark.parametrize(
    "options", [(64, 6, 64, 4, False), (64, 6, 64, 4, True), (128, 32, 128, 8, True)]
)
def test_four_epochs_full_state_metadata_rng_graph(
    paths, scorer, backend, seed, options
):
    saved, metadata = [], []
    for candidate in (False, True):
        runtime, rtk, ctx, call = make(
            paths,
            scorer,
            candidate=candidate,
            backend=backend,
            seed=seed,
            options=options,
        )
        states, details = [], []
        for _ in range(4):
            output = call(rtk, ctx)
            states.append(state(runtime, rtk, output))
            detail = deepcopy(call.last)
            if candidate:
                counts = detail.pop("extend_support")
                assert counts["support_fallback"] == counts["finish_fallback"] == 0
                assert counts["support_native"] == detail["extend_calls"]
            details.append(detail)
        saved.append(states)
        metadata.append(details)
    exact(saved[0], saved[1])
    exact(metadata[0], metadata[1])


def test_strict_policy_original_partial_state_and_cleanup(paths, scorer):
    saved = []
    for candidate in (False, True):
        runtime, rtk, ctx, call = make(
            paths, scorer, candidate=candidate, backend="gpu"
        )
        with np.errstate(all="raise"):
            try:
                output = call(rtk, ctx)
                error = None
            except Exception as exc:
                output, error = None, (type(exc).__name__, str(exc))
        saved.append((state(runtime, rtk, output), error, deepcopy(call.last)))
        assert "_rbpf_extend_support" not in call._scope
        assert "_rbpf_extend_finish" not in call._scope
    exact(saved[0], saved[1])
