"""Exact PF states, function identities, guarded private view promotion."""

from copy import deepcopy
import gc
import json
from pathlib import Path
import sys
from types import MethodType, SimpleNamespace
import weakref

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
from bench_rbpf_feedback_mass import _bind_fixture_callbacks  # noqa: E402
from rbpf_direct_view import DirectViewEngine, DirectViewStep  # noqa: E402
from rbpf_feedback_cases import make_feedback_case  # noqa: E402
from rbpf_feedback_mass import NativeFeedbackMassStep  # noqa: E402
from rbpf_move_cases import comparison, state  # noqa: E402
from rbpf_native_canonical import _CanonicalView  # noqa: E402
from rbpf_register_lu import RegisterLUScorer  # noqa: E402


@pytest.fixture(scope="module")
def dependencies():
    return json.loads((ROOT / "dist/rbpf_gc_clocks_tests_20261018.plan.json").read_text(encoding="utf8"))["extensions"]


@pytest.fixture(scope="module")
def scorer(dependencies):
    with RegisterLUScorer(ROOT / dependencies["DLL"]) as owned:
        yield owned


def exact(a, b):
    result = comparison(a, b)
    assert result["discrete_equal"] and result["numeric_bits_equal"] and result["max_abs_error"] == 0, result


def make(inputs, dependencies, scorer=None, backend="scalar", candidate=True):
    runtime, rtk, _ = inputs
    _bind_fixture_callbacks(rtk)
    call = (DirectViewStep if candidate else NativeFeedbackMassStep)(runtime, scorer, backend,
        extension=ROOT / dependencies["CANONICAL_EXT"], parent_extension=ROOT / dependencies["PARENT_EXT"],
        mass_extension=ROOT / dependencies["MASS_EXT"])
    runtime.step = call
    return call


@pytest.mark.parametrize("backend", ("scalar", "native_cpu", "gpu"))
@pytest.mark.parametrize("seed", (20260710, 20260713))
@pytest.mark.parametrize("cluster", (False, True))
def test_four_full_epochs_exact(dependencies, scorer, backend, seed, cluster):
    original = make_feedback_case(64, 32, rows=128 if cluster else 64,
        groups=8 if cluster else 4, seed=seed, cluster=cluster, shadow=True, spawn=True)
    candidate = deepcopy(original)
    calls = [make(original, dependencies, scorer, backend, False), make(candidate, dependencies, scorer, backend)]
    enabled, threshold = gc.isenabled(), gc.get_threshold()
    identities = calls[1]._functions.copy()
    for epoch in range(4):
        values, metadata = [], []
        for inputs, call in zip((original, candidate), calls):
            inputs[1].epoch2 = epoch
            out = call(inputs[1], inputs[2])
            values.append(state(*inputs[:2], out))
            detail = deepcopy(call.last)
            extra = detail.pop("direct_view", None)
            if call is calls[1]:
                assert extra == dict(calls=1, promoted=1, declined=0, initial_cached_methods=3)
            else:
                assert extra is None
            metadata.append(detail)
        exact(*values)
        exact(*metadata)
    assert gc.isenabled() == enabled and gc.get_threshold() == threshold
    assert all(calls[1]._functions[k] is v for k, v in identities.items())
    assert calls[1]._view is calls[1]._oracle is None


def test_same_view_identity_fields_retained_method_and_no_new_cache():
    runtime = SimpleNamespace(value=object())
    def method(self):
        return self.value
    table = dict(method=method)
    engine = DirectViewEngine(runtime, table)
    view = _CanonicalView(runtime, table)
    saved = view.method
    original_id = id(view)
    assert engine.promote(view) and id(view) == original_id
    assert saved() is view.method() is runtime.value
    assert type(view).__getattribute__ is object.__getattribute__
    assert not hasattr(type(view), "__getattr__")
    assert type(view).method is method and not object.__getattribute__(view, "_bound")
    view.value = None
    assert runtime.value is saved() is view.method() is None
    with pytest.raises(AttributeError):
        _ = view.cfg


@pytest.mark.parametrize("kind", ("value", "key", "foreign", "mapping", "functions", "class", "field"))
def test_nonordinary_private_inputs_decline_unchanged(kind):
    runtime = SimpleNamespace(value=1)
    def method(self):
        return self.value
    table = dict(method=method)
    engine = DirectViewEngine(runtime, table)
    view = _CanonicalView(runtime, table)
    cache = object.__getattribute__(view, "_bound")
    if kind == "value":
        cache["method"] = object()
    elif kind == "key":
        cache[7] = view.method
    elif kind == "foreign":
        cache["method"] = MethodType(method, runtime)
    elif kind == "mapping":
        class Custom(dict):
            def clear(self):
                raise AssertionError("Custom clear must not run")
        object.__setattr__(view, "_bound", Custom())
    elif kind == "functions":
        table["other"] = method
    elif kind == "class":
        engine.view_type.method = None
    else:
        engine.fields["value"].name = "other"
    before = list(object.__getattribute__(view, "_bound").items())
    assert not engine.promote(view)
    assert type(view) is _CanonicalView
    assert list(object.__getattribute__(view, "_bound").items()) == before
    assert engine.counts == dict(calls=1, promoted=0, declined=1, initial_cached_methods=0)


def test_original_cached_method_dies_without_collection(dependencies):
    inputs = make_feedback_case(32, 6)
    call = make(inputs, dependencies)
    refs, sizes, hits = [], [], []
    base = call.compiled
    def observed(view, rtk, ctx):
        refs.append(weakref.ref(view._pair_dd))
        result = base(view, rtk, ctx)
        sizes.append(len(object.__getattribute__(view, "_bound")))
        hits.append(type(view) is call.direct_engine.view_type)
        return result
    call.compiled = observed
    enabled = gc.isenabled()
    gc.disable()
    try:
        for epoch in range(8):
            inputs[1].epoch2 = epoch
            call(inputs[1], inputs[2])
        assert all(ref() is None for ref in refs) and sizes == [0] * 8 and all(hits)
    finally:
        if enabled:
            gc.enable()
        gc.collect()


@pytest.mark.parametrize("strict", (False, True))
def test_skip_and_strict_fallback(dependencies, strict):
    original = make_feedback_case(32, 8)
    original[2]["problem"].ok = False
    values = []
    for inputs, candidate in ((original, False), (deepcopy(original), True)):
        call = make(inputs, dependencies, candidate=candidate)
        with np.errstate(all="raise" if strict else "ignore"):
            out = call(inputs[1], inputs[2])
        detail = deepcopy(call.last)
        if detail is not None:
            detail.pop("direct_view", None)
        values.append((state(*inputs[:2], out), detail))
        if candidate:
            assert call.direct_engine.counts["promoted"] == int(not strict)
    exact(*values)


def test_partial_state_and_original_rng_exception(dependencies):
    initial = make_feedback_case(64, 32)
    values = []
    class Interrupted:
        def __init__(self):
            self.source = np.random.default_rng(20261018)
            self.bit_generator = self.source.bit_generator
        def __getattr__(self, name):
            return getattr(self.source, name)
        def choice(self, *args, **kwargs):
            self.source.choice(*args, **kwargs)
            raise RuntimeError("original choice interrupted")
    for candidate in (False, True):
        inputs = deepcopy(initial)
        inputs[0].rng = Interrupted()
        call = make(inputs, dependencies, candidate=candidate)
        with pytest.raises(RuntimeError, match="original choice interrupted"):
            call(inputs[1], inputs[2])
        detail = deepcopy(call.last)
        if detail is not None:
            detail.pop("direct_view", None)
        values.append((state(*inputs[:2], None), detail))
        assert call._view is call._oracle is None
    exact(*values)
