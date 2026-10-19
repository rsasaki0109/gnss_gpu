"""Guarded memo/index fusion keeps old candidate semantics and full PF state."""

import ast
from copy import deepcopy
import gc
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
from bench_rbpf_feedback_mass import _bind_fixture_callbacks  # noqa: E402
from rbpf_epoch_memo import _EpochMemo  # noqa: E402
from rbpf_feedback_cases import make_feedback_case  # noqa: E402
from rbpf_feedback_mass import NativeFeedbackMassStep  # noqa: E402
from rbpf_fused_staging import FusedStagingStep, _CODE, _MEMO  # noqa: E402
from rbpf_likelihood_batch import BatchScorer  # noqa: E402
from rbpf_move_cases import state  # noqa: E402
from rbpf_move_reuse import MoveReuseStep  # noqa: E402
from test_rbpf_native_staging import ORIGINAL, exact, trial  # noqa: E402


@pytest.fixture(scope="module")
def extensions():
    names = dict(
        extension="GNSS_GPU_RBPF_CANONICAL_EXT",
        parent_extension="GNSS_GPU_RBPF_PARENT_EXT",
        mass_extension="GNSS_GPU_RBPF_MASS_EXT",
        staging_extension="GNSS_GPU_RBPF_FUSED_STAGING_EXT",
    )
    if any(not os.environ.get(n) for n in names.values()):
        pytest.skip(
            "Explicit independent fused extension and unchanged extensions required"
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
    if not os.environ.get("GNSS_GPU_RBPF_PROBE_DLL"):
        pytest.skip("Explicit unchanged numerical DLL required")
    with BatchScorer(os.environ["GNSS_GPU_RBPF_PROBE_DLL"]) as value:
        yield value


def _label_safe_trial():
    # The original bare-staging fixture assumes every custom key has .value.
    # The real memo intentionally returns builtin () for an empty hypothesis.
    # Change only that fixture observation, preserving all calls and values.
    path = ROOT / "tests/test_rbpf_native_staging.py"
    node = next(
        n
        for n in ast.parse(path.read_bytes()).body
        if isinstance(n, ast.FunctionDef) and n.name == "trial"
    )
    baseline = deepcopy(node)
    labels = [
        n
        for n in ast.walk(node)
        if isinstance(n, ast.IfExp)
        and ast.unparse(n.body) == "k.value"
        and ast.unparse(n.test) == "custom"
    ]
    assert len(labels) == 1
    labels[0].body = ast.parse("getattr(k, 'value', k)", mode="eval").body
    restored = deepcopy(node)
    target = next(
        n
        for n in ast.walk(restored)
        if isinstance(n, ast.IfExp)
        and ast.unparse(n.test) == "custom"
        and ast.unparse(n.body).startswith("getattr(k,")
    )
    target.body = ast.parse("k.value", mode="eval").body
    assert ast.dump(restored, include_attributes=False) == ast.dump(
        baseline, include_attributes=False
    )
    scope = dict(trial.__globals__)
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
            str(path),
            "exec",
        ),
        scope,
    )
    return scope["trial"]


_TRIAL = _label_safe_trial()


def memo_trial(module, variant, **options):
    saved = {}

    def wrapped(context, *args):
        memo = _EpochMemo(context.canonical, None, cache_dd=False)
        context.canonical = memo.canonical
        try:
            if variant == "original":
                return ORIGINAL(context, *args)
            if variant == "off":
                return module.stage(context, *args)
            held, stats = module.stage_fused(context, *args, _MEMO, _CODE, _EpochMemo)
            saved["fusion"] = stats
            return held
        finally:
            saved["memo"] = memo.summary()

    result = _TRIAL(wrapped, **options)
    result["memo"] = saved["memo"]
    return result, saved.get("fusion")


@pytest.mark.parametrize("seed", [20260710, 20260711, 20260712])
@pytest.mark.parametrize("rows,groups", [(16, 1), (64, 4), (128, 8)])
def test_all_candidate_values_order_priors_callbacks_and_counters(
    module, seed, rows, groups
):
    original, _ = memo_trial(module, "original", seed=seed, rows=rows, groups=groups)
    off, _ = memo_trial(module, "off", seed=seed, rows=rows, groups=groups)
    on, stats = memo_trial(module, "on", seed=seed, rows=rows, groups=groups)
    exact(original, off)
    exact(original, on)
    assert stats["fused_hits"] > 0 and stats["original_index_calls"] > 0
    assert stats["fused_hits"] + stats["original_index_calls"] == sum(
        len(v) for v in on["staged"]
    )


@pytest.mark.parametrize(
    "failure", [None, "canonical", "hash", "eq", "extend", "shrink"]
)
def test_custom_keys_and_partial_exception_order(module, failure):
    a, _ = memo_trial(module, "original", custom=True, failure=failure)
    b, stats = memo_trial(module, "on", custom=True, failure=failure)
    exact(a, b)
    if stats is not None:
        assert stats["fused_hits"] == 0


@pytest.mark.parametrize(
    "priors",
    [
        dict(p_survive=-0.0, p_extend=0.0, p_shrink=1e-320, p_release=1.0),
        dict(p_survive=1e16, p_extend=1.0, p_shrink=-1e16, p_release=1e-320),
        dict(p_adopt=-0.0, p_spawn=1e-320),
    ],
)
def test_original_python_prior_arithmetic_bits(module, priors):
    exact(
        memo_trial(module, "original", priors=priors)[0],
        memo_trial(module, "on", priors=priors)[0],
    )


def test_callback_appended_candidates_preserve_iteration(module):
    exact(
        memo_trial(module, "original", dynamic=True)[0],
        memo_trial(module, "on", dynamic=True)[0],
    )


@pytest.mark.parametrize(
    "change", ["lookup", "identities", "counts", "bound_method", "class"]
)
def test_callback_mutation_invalidates_or_bypasses_aliases(module, change):
    def run(fused):
        a = {(1, 0): {1: 0, 2: 2}}
        b = {(1, 0): {1: 0, 2: 3}}
        lookup = {}
        keys = []
        hs = []
        pending = []
        staged = []
        events = []
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
            particles=[SimpleNamespace(H={}), SimpleNamespace(H={})]
        )
        context = SimpleNamespace()

        def key(h):
            return tuple((g, tuple(z.items())) for g, z in h.items())

        mutated = False

        def canonical(h):
            nonlocal mutated
            events.append(("source", key(h)))
            if h is b and not mutated:
                mutated = True
                if change == "lookup":
                    lookup[key(a)] = 27
                elif change == "identities":
                    memo.identities = {}
                elif change == "counts":
                    memo.counts = dict(memo.counts)
                elif change == "bound_method":
                    context.canonical = lambda h: memo.canonical(h)
                else:
                    _EpochMemo.test_marker = 17
            return key(h)

        memo = _EpochMemo(canonical, None, cache_dd=False)
        context.canonical = memo.canonical
        args = (
            context,
            runtime,
            cfg,
            None,
            None,
            [a, a, b, a, a],
            [],
            {},
            keys,
            hs,
            lookup,
            pending,
            staged,
        )
        try:
            if fused:
                held, stats = module.stage_fused(*args, _MEMO, _CODE, _EpochMemo)
            else:
                held, stats = ORIGINAL(*args), None
            return (
                held,
                keys,
                hs,
                lookup,
                pending,
                staged,
                events,
                memo.summary(),
            ), stats
        finally:
            if hasattr(_EpochMemo, "test_marker"):
                delattr(_EpochMemo, "test_marker")

    a, _ = run(False)
    b, stats = run(True)
    exact(a, b)
    assert stats["fused_hits"] > 0
    if change == "lookup":
        assert stats["invalidations"] > 0


def test_call_local_native_references_are_released(module):
    watched = [_EpochMemo, _MEMO, _CODE, module]
    gc.collect()
    before = [sys.getrefcount(v) for v in watched]
    for _ in range(80):
        result, stats = memo_trial(module, "on")
        assert stats["fused_hits"] > 0
        del result, stats
    gc.collect()
    assert before == [sys.getrefcount(v) for v in watched]


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
@pytest.mark.parametrize("enabled", [False, True])
def test_four_epoch_all_state_metadata_rng_and_graph(
    extensions, scorer, mode, seed, options, enabled
):
    a, ra, ca = make_feedback_case(seed=seed, **options)
    b, rb, cb = make_feedback_case(seed=seed, **options)
    old = NativeFeedbackMassStep(
        a,
        scorer,
        mode,
        **{k: v for k, v in extensions.items() if k != "staging_extension"},
    )
    new = FusedStagingStep(b, scorer, mode, fusion=enabled, **extensions)
    a.step, b.step = old, new
    for epoch in range(4):
        ra.epoch2 = rb.epoch2 = epoch
        if options.get("feedback", True):
            _bind_fixture_callbacks(ra)
            _bind_fixture_callbacks(rb)
        oa, ob = old(ra, deepcopy(ca)), new(rb, deepcopy(cb))
        exact(state(a, ra, oa), state(b, rb, ob))
        exact(
            old.last,
            {
                k: v
                for k, v in new.last.items()
                if k not in ("native_staging", "fused_staging")
            },
        )
        if enabled:
            assert new.last_fusion["enabled"] and new.last_fusion["fused_hits"] > 0
        else:
            assert not new.last_fusion["enabled"]


@pytest.mark.parametrize("kind", ["len_binding", "integer_prior", "memo_type_binding"])
def test_unsupported_bindings_fallback(extensions, monkeypatch, kind):
    from rbpf_move_batch import MoveBatchStep

    events = []
    if kind == "len_binding":

        def custom_len(value):
            events.append("len")
            return len(value)

        monkeypatch.setitem(MoveBatchStep._moves.__globals__, "len", custom_len)
    if kind == "memo_type_binding":

        def custom_type(value):
            events.append("type")
            return type(value)

        monkeypatch.setitem(_MEMO.__globals__, "type", custom_type)
    a, ra, ca = make_feedback_case(feedback=False)
    b, rb, cb = make_feedback_case(feedback=False)
    if kind == "integer_prior":
        a.cfg.p_adopt = b.cfg.p_adopt = 1
    old = NativeFeedbackMassStep(
        a,
        mode="scalar",
        **{k: v for k, v in extensions.items() if k != "staging_extension"},
    )
    new = FusedStagingStep(b, mode="scalar", **extensions)
    a.step, b.step = old, new
    events.clear()
    oa = old(ra, ca)
    expected = list(events)
    events.clear()
    ob = new(rb, cb)
    exact(state(a, ra, oa), state(b, rb, ob))
    exact(
        old.last,
        {
            k: v
            for k, v in new.last.items()
            if k not in ("native_staging", "fused_staging")
        },
    )
    assert events == expected and (
        new.last_fusion is None or not new.last_fusion["enabled"]
    )


def test_strict_policy_and_helper_guard(extensions, monkeypatch):
    a, ra, ca = make_feedback_case()
    b, rb, cb = make_feedback_case()
    new = FusedStagingStep(b, mode="scalar", **extensions)
    with np.errstate(under="raise"):
        for fn, r, c in ((a.step, ra, ca), (new, rb, cb)):
            with pytest.raises(FloatingPointError):
                fn(r, c)
    exact(state(a, ra, None), state(b, rb, None))
    assert new.last == dict(original_step_fallback="numpy_error_policy")
    monkeypatch.setattr(MoveReuseStep, "_moves", lambda *args: None)
    with pytest.raises(ValueError, match="helper changed"):
        new(rb, cb)
