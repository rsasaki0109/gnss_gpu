"""Full PF equivalence and observable release of completed method cycles."""

from copy import deepcopy
import gc
import json
from pathlib import Path
import sys
from types import MethodType
import weakref

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
from bench_rbpf_feedback_mass import _bind_fixture_callbacks  # noqa: E402
from rbpf_feedback_cases import make_feedback_case  # noqa: E402
from rbpf_feedback_mass import NativeFeedbackMassStep  # noqa: E402
from rbpf_move_cases import comparison, state  # noqa: E402
from rbpf_native_canonical import _CanonicalView  # noqa: E402
from rbpf_register_lu import RegisterLUScorer  # noqa: E402
from rbpf_view_release import ViewReleaseStep  # noqa: E402


@pytest.fixture(scope="module")
def dependencies():
    return json.loads((ROOT / "dist/rbpf_extend_factor_tests_20261018.plan.json").read_text(encoding="utf8"))["extensions"]


@pytest.fixture(scope="module")
def scorer(dependencies):
    with RegisterLUScorer(ROOT / dependencies["DLL"]) as owned:
        yield owned


def make(inputs, dependencies, scorer=None, backend="scalar", candidate=True):
    runtime, rtk, _ = inputs
    _bind_fixture_callbacks(rtk)
    cls = ViewReleaseStep if candidate else NativeFeedbackMassStep
    call = cls(runtime, scorer, backend, extension=ROOT / dependencies["CANONICAL_EXT"], parent_extension=ROOT / dependencies["PARENT_EXT"], mass_extension=ROOT / dependencies["MASS_EXT"])
    runtime.step = call
    return call


def observe(call):
    refs, caches = [], []
    base = call.compiled
    def observed(view, rtk, ctx):
        refs.append(weakref.ref(view._pair_dd))
        caches.append(object.__getattribute__(view, "_bound"))
        return base(view, rtk, ctx)
    call.compiled = observed
    return refs, caches


def exact(a, b):
    result = comparison(a, b)
    assert result["discrete_equal"] and result["numeric_bits_equal"] and result["max_abs_error"] == 0, result


@pytest.mark.parametrize("backend", ["scalar", "native_cpu", "gpu"])
@pytest.mark.parametrize("seed", [20260710, 20260713])
@pytest.mark.parametrize("cluster", [False, True])
def test_full_pf_four_epochs(dependencies, scorer, backend, seed, cluster):
    original = make_feedback_case(64, 32, rows=128 if cluster else 64, groups=8 if cluster else 4, seed=seed, cluster=cluster, shadow=True, spawn=True)
    modified = deepcopy(original)
    calls = [make(original, dependencies, scorer, backend, False), make(modified, dependencies, scorer, backend, True)]
    refs, caches = observe(calls[1])
    enabled, threshold = gc.isenabled(), gc.get_threshold()
    for epoch in range(4):
        states, metadata = [], []
        for (runtime, rtk, ctx), call in zip((original, modified), calls):
            rtk.epoch2 = epoch
            out = call(rtk, ctx)
            states.append(state(runtime, rtk, out))
            metadata.append(deepcopy(call.last))
        extra = metadata[1].pop("view_release")
        assert extra["calls"] == 1 and extra["released_methods"] >= 3 and extra["declined"] == 0
        assert refs[-1]() is None and not caches[-1]
        exact(states[0], states[1])
        exact(metadata[0], metadata[1])
    assert gc.isenabled() == enabled and gc.get_threshold() == threshold


def test_method_lifetime_without_automatic_gc(dependencies):
    enabled = gc.isenabled()
    gc.disable()
    try:
        for candidate in (False, True):
            inputs = make_feedback_case(64, 6)
            call = make(inputs, dependencies, candidate=candidate)
            refs, caches = observe(call)
            for epoch in range(12):
                inputs[1].epoch2 = epoch
                call(inputs[1], inputs[2])
            assert len(refs) == 12
            if candidate:
                assert all(ref() is None for ref in refs) and all(not cache for cache in caches)
            else:
                assert all(ref() is not None for ref in refs) and all(cache for cache in caches)
    finally:
        if enabled:
            gc.enable()
        gc.collect()


def test_retained_external_method_remains_callable(dependencies):
    inputs = make_feedback_case(64, 8)
    runtime, rtk, ctx = inputs
    call = make(inputs, dependencies)
    external = []
    base = call.compiled
    def retained(view, rtk, ctx):
        external.append(view._pair_dd)
        return base(view, rtk, ctx)
    call.compiled = retained
    call(rtk, ctx)
    method = external[0]
    assert type(method) is MethodType
    assert not object.__getattribute__(method.__self__, "_bound")
    h = runtime.particles[0].H
    exact(method(h, ctx["problem"]), runtime._pair_dd(h, ctx["problem"]))


@pytest.mark.parametrize("kind", ["value", "key", "foreign_method", "mapping"])
def test_unusual_cache_declines_without_callbacks(dependencies, kind):
    inputs = make_feedback_case(8, 4)
    call = make(inputs, dependencies)
    view = _CanonicalView(inputs[0], call._functions)
    cache = object.__getattribute__(view, "_bound")
    if kind == "value":
        cache["custom"] = object()
    elif kind == "key":
        cache[7] = view._pair_dd
    elif kind == "foreign_method":
        cache["custom"] = inputs[0]._pair_dd
    else:
        class CustomDict(dict):
            def clear(self):
                raise AssertionError("custom clear must not execute")
        object.__setattr__(view, "_bound", CustomDict(custom=view._pair_dd))
    call.release_view(view)
    assert call.release_counts == dict(calls=1, released_methods=0, declined=1)
    assert object.__getattribute__(view, "_bound")


@pytest.mark.parametrize("strict", [False, True])
def test_skipped_and_strict_epoch_lifecycle(dependencies, strict):
    original = make_feedback_case(32, 8)
    original[2]["problem"].ok = False
    modified = deepcopy(original)
    results = []
    for inputs, candidate in [(original, False), (modified, True)]:
        call = make(inputs, dependencies, candidate=candidate)
        refs, caches = observe(call)
        with np.errstate(all="raise" if strict else "ignore"):
            out = call(inputs[1], inputs[2])
        extra = call.last.pop("view_release", None) if call.last is not None else None
        results.append((state(*inputs[:2], out), deepcopy(call.last)))
        if candidate:
            assert call.release_counts["calls"] == int(not strict)
            assert extra is None or extra["calls"] == int(not strict)
            assert all(not cache for cache in caches) and all(ref() is None for ref in refs)
    exact(*results)


def test_rng_exception_clears_completed_method_cache(dependencies):
    original = make_feedback_case(64, 32)
    modified = deepcopy(original)
    results = []
    class InterruptedRng:
        def __init__(self):
            self.source = np.random.default_rng(20261018)
            self.bit_generator = self.source.bit_generator
        def __getattr__(self, name):
            return getattr(self.source, name)
        def choice(self, *args, **kwargs):
            self.source.choice(*args, **kwargs)
            raise RuntimeError("original choice interrupted")
    for inputs, candidate in [(original, False), (modified, True)]:
        inputs[0].rng = InterruptedRng()
        call = make(inputs, dependencies, candidate=candidate)
        _, caches = observe(call)
        with pytest.raises(RuntimeError, match="original choice interrupted"):
            call(inputs[1], inputs[2])
        if candidate:
            assert call.release_counts["calls"] == 1 and call.release_counts["released_methods"] >= 3
            assert all(not cache for cache in caches)
        results.append((state(*inputs[:2], None), deepcopy(call.last)))
    exact(*results)
