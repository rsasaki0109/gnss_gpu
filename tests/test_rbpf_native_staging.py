"""Native staging preserves every candidate, callback, lookup and RNG draw."""

from copy import deepcopy
import gc
import importlib.util
import os
from pathlib import Path
import sys
from types import FunctionType, SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
from bench_rbpf_feedback_mass import _bind_fixture_callbacks  # noqa: E402
from rbpf_feedback_cases import make_feedback_case  # noqa: E402
from rbpf_feedback_mass import NativeFeedbackMassStep  # noqa: E402
from rbpf_likelihood_batch import BatchScorer  # noqa: E402
from rbpf_likelihood_cases import make_case  # noqa: E402
from rbpf_move_batch import MoveBatchStep  # noqa: E402
from rbpf_move_cases import comparison, state  # noqa: E402
from rbpf_move_reuse import MoveReuseStep  # noqa: E402
from rbpf_native_staging import NativeStagingStep, _RAW_STAGE  # noqa: E402


@pytest.fixture(scope="module")
def extensions():
    names = dict(
        extension="GNSS_GPU_RBPF_CANONICAL_EXT",
        parent_extension="GNSS_GPU_RBPF_PARENT_EXT",
        mass_extension="GNSS_GPU_RBPF_MASS_EXT",
        staging_extension="GNSS_GPU_RBPF_STAGING_EXT",
    )
    if any(not os.environ.get(n) for n in names.values()):
        pytest.skip(
            "Explicit unchanged extensions and native-staging extension required"
        )
    return {k: os.environ[v] for k, v in names.items()}


@pytest.fixture(scope="module")
def module(extensions):
    spec = importlib.util.spec_from_file_location(
        "_rbpf_native_staging_probe", extensions["staging_extension"]
    )
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


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


ORIGINAL = FunctionType(_RAW_STAGE, MoveBatchStep._moves.__globals__)


def trial(
    function,
    seed=20260710,
    rows=64,
    groups=4,
    custom=False,
    failure=None,
    dynamic=False,
    priors=None,
):
    oracle, prob, hs = make_case(
        12, rows=rows, groups=groups, seed=seed, duplicates=False
    )
    events = []
    candidates = hs[1:8] + [deepcopy(hs[1])]
    spawn = hs[8:12]
    huge = {(2, 0): {1: 2**16000, 9: 2**16000 + 2**90}}
    hypotheses = [{}, *hs[:6], hs[0], deepcopy(hs[0]), huge] * 2
    cfg = SimpleNamespace(
        p_survive=0.62,
        p_extend=0.12,
        p_shrink=0.04,
        p_release=0.02,
        p_adopt_cons=0.12,
        p_adopt=0.08,
        p_spawn=0.10,
    )
    if priors:
        vars(cfg).update(priors)

    def event(name, value):
        events.append((name, value))
        if failure == name and sum(e[0] == name for e in events) == 3:
            raise ValueError("original " + name + " error")

    def extend(h, p):
        assert p is prob
        event("extend", oracle.basin_id(h))
        return None if len(h) % 2 else deepcopy(h)

    def shrink(h, p):
        assert p is prob
        event("shrink", oracle.basin_id(h))
        return deepcopy(h) if len(h) % 2 else None

    class Key:
        def __init__(self, value):
            self.value = value

        def __hash__(self):
            event("hash", self.value)
            return 13

        def __eq__(self, other):
            event("eq", (self.value, other.value))
            return self.value == other.value

    active = False

    def canonical(h):
        key = oracle.basin_id(h)
        event("canonical", key)
        if dynamic and active and len(candidates) == 8:
            candidates.append(hs[0])
        return Key(key) if custom else key

    runtime = SimpleNamespace(
        particles=[SimpleNamespace(H=h) for h in hypotheses],
        _extend=extend,
        _shrink_worst=shrink,
    )
    context = SimpleNamespace(canonical=canonical)
    keys, out_hs, lookup, pending, staged = [], [], {}, [], []
    # Both stored None and a real value must use the original pending rule.
    cache = {canonical(hs[0]): None, canonical(hs[1]): -0.0}
    events.clear()
    active = True
    error = None
    try:
        held = function(
            context,
            runtime,
            cfg,
            prob,
            hs[0],
            candidates,
            spawn,
            cache,
            keys,
            out_hs,
            lookup,
            pending,
            staged,
        )
    except ValueError as e:
        error, held = str(e), None
    labels = [k.value if custom else k for k in keys]
    return dict(
        held=held,
        keys=labels,
        hypotheses=out_hs,
        pending=pending,
        staged=staged,
        events=events,
        error=error,
    )


@pytest.mark.parametrize("seed", [20260710, 20260711, 20260712])
@pytest.mark.parametrize("rows,groups", [(16, 1), (64, 4), (128, 8)])
def test_all_candidate_values_order_priors_and_key_callbacks(
    module, seed, rows, groups
):
    a, b = trial(ORIGINAL, seed, rows, groups), trial(module.stage, seed, rows, groups)
    exact(a, b)
    assert a["held"] > 0 and a["error"] is None
    assert sum(len(v) for v in a["staged"]) == sum(
        e[0] == "canonical" for e in a["events"]
    )


@pytest.mark.parametrize(
    "failure", [None, "canonical", "hash", "eq", "extend", "shrink"]
)
def test_custom_key_hash_equality_and_partial_exception_order(module, failure):
    a = trial(ORIGINAL, custom=True, failure=failure)
    b = trial(module.stage, custom=True, failure=failure)
    exact(a, b)
    assert (
        a["error"] is None
        if failure is None
        else a["error"] == "original " + failure + " error"
    )


@pytest.mark.parametrize(
    "priors",
    [
        dict(p_survive=-0.0, p_extend=0.0, p_shrink=1e-320, p_release=1.0),
        dict(p_survive=1e16, p_extend=1.0, p_shrink=-1e16, p_release=1e-320),
        dict(p_adopt=-0.0, p_spawn=1e-320),
    ],
)
def test_python_prior_arithmetic_order_and_subnormal_bits(module, priors):
    exact(trial(ORIGINAL, priors=priors), trial(module.stage, priors=priors))


def test_original_list_iteration_observes_callback_append(module):
    exact(trial(ORIGINAL, dynamic=True), trial(module.stage, dynamic=True))


def test_fresh_release_dicts_and_no_cross_call_references(module):
    g, big = (2**70, 0), 2**16000
    h = {g: {1: big, 2: big + 7}}
    cfg = SimpleNamespace(
        p_survive=0.62,
        p_extend=0.12,
        p_shrink=0.04,
        p_release=0.02,
        p_adopt_cons=0.12,
        p_adopt=0.08,
        p_spawn=0.1,
    )
    runtime = SimpleNamespace(
        particles=[SimpleNamespace(H=h), SimpleNamespace(H=h)],
        _extend=lambda h, p: None,
        _shrink_worst=lambda h, p: None,
    )
    context = SimpleNamespace(
        canonical=lambda h: tuple((g, tuple(d.items())) for g, d in h.items())
    )
    candidates = [h]
    objects = [h, g, big, cfg, runtime, context, candidates, module.stage]
    gc.collect()
    before = [sys.getrefcount(o) for o in objects]
    for _ in range(5000):
        keys, hs, lookup, pending, staged = [], [], {}, [], []
        assert (
            module.stage(
                context,
                runtime,
                cfg,
                None,
                None,
                candidates,
                [],
                {},
                keys,
                hs,
                lookup,
                pending,
                staged,
            )
            == 2
        )
        first, second = staged[0][1][2], staged[1][1][2]
        assert first == second == {} and first is not second
        assert staged[0][0] is not staged[1][0]
        assert staged[0][0][2] is staged[1][0][2] is h
        del keys, hs, lookup, pending, staged, first, second
    gc.collect()
    assert [sys.getrefcount(o) for o in objects] == before


def test_empty_particles_and_empty_shared_lists(module):
    args = (
        SimpleNamespace(),
        SimpleNamespace(particles=[]),
        None,
        None,
        None,
        [],
        [],
        {},
        [],
        [],
        {},
        [],
        [],
    )
    assert ORIGINAL(*args) == module.stage(*args) == 0
    assert args[-1] == []


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
        **{k: v for k, v in extensions.items() if k != "staging_extension"},
    )
    call = NativeStagingStep(b, scorer, mode, **extensions)
    a.step, b.step = old, call
    for ep in range(4):
        ra.epoch2 = rb.epoch2 = ep
        if options.get("feedback", True):
            _bind_fixture_callbacks(ra)
            _bind_fixture_callbacks(rb)
        oa, ob = old(ra, deepcopy(ca)), call(rb, deepcopy(cb))
        exact(state(a, ra, oa), state(b, rb, ob))
        exact(old.last, {k: v for k, v in call.last.items() if k != "native_staging"})
        s = call.last["native_staging"]
        assert s == dict(
            native_calls=1,
            fallback_calls=0,
            native_particles=old.last["particles"],
            native_moves=old.last["requested_moves"],
        )


@pytest.mark.parametrize("kind", ["len_binding", "integer_prior"])
def test_unsupported_bindings_use_original_staging_once(extensions, monkeypatch, kind):
    events = []
    if kind == "len_binding":

        def custom_len(x):
            events.append("len")
            return len(x)

        monkeypatch.setitem(MoveBatchStep._moves.__globals__, "len", custom_len)
    a, ra, ca = make_feedback_case(feedback=False)
    b, rb, cb = make_feedback_case(feedback=False)
    if kind == "integer_prior":
        a.cfg.p_adopt = b.cfg.p_adopt = 1
    old = NativeFeedbackMassStep(
        a,
        mode="scalar",
        **{k: v for k, v in extensions.items() if k != "staging_extension"},
    )
    call = NativeStagingStep(b, mode="scalar", **extensions)
    a.step, b.step = old, call
    events.clear()  # Exclude fixture/controller construction on both sides.
    oa = old(ra, ca)
    expected_events = list(events)
    events.clear()
    ob = call(rb, cb)
    exact(state(a, ra, oa), state(b, rb, ob))
    exact(old.last, {k: v for k, v in call.last.items() if k != "native_staging"})
    assert events == expected_events
    assert call.last["native_staging"] == dict(
        native_calls=0, fallback_calls=1, native_particles=0, native_moves=0
    )


def test_strict_policy_and_helper_mutation_guard(extensions, monkeypatch):
    a, ra, ca = make_feedback_case()
    b, rb, cb = make_feedback_case()
    call = NativeStagingStep(b, mode="scalar", **extensions)
    with np.errstate(under="raise"):
        for fn, rtk, ctx in ((a.step, ra, ca), (call, rb, cb)):
            with pytest.raises(FloatingPointError):
                fn(rtk, ctx)
    exact(state(a, ra, None), state(b, rb, None))
    assert call.last == dict(original_step_fallback="numpy_error_policy")
    monkeypatch.setattr(MoveReuseStep, "_moves", lambda *a: None)
    with pytest.raises(ValueError, match="helper changed"):
        call(rb, cb)
