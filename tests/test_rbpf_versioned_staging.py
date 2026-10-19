"""Versioned attribute/ID access preserves original callbacks and full PF bits."""

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
from rbpf_epoch_memo import _EpochMemo  # noqa: E402
from rbpf_feedback_cases import make_feedback_case  # noqa: E402
from rbpf_feedback_mass import NativeFeedbackMassStep  # noqa: E402
from rbpf_fused_staging import _MEMO, _CODE  # noqa: E402
from rbpf_move_cases import state  # noqa: E402
from rbpf_register_lu import RegisterLUScorer  # noqa: E402
from rbpf_versioned_staging import VersionedStagingStep  # noqa: E402
from test_rbpf_native_staging import ORIGINAL, exact  # noqa: E402
from test_rbpf_fused_staging import (  # noqa: E402,F401
    memo_trial,
    test_all_candidate_values_order_priors_callbacks_and_counters,
    test_custom_keys_and_partial_exception_order,
    test_original_python_prior_arithmetic_bits,
    test_callback_appended_candidates_preserve_iteration,
    test_callback_mutation_invalidates_or_bypasses_aliases,
    test_call_local_native_references_are_released,
)


@pytest.fixture(scope="module")
def paths():
    names = ("DLL", "STAGING_EXT", "CANONICAL_EXT", "PARENT_EXT", "MASS_EXT")
    result = {n: os.environ.get("GNSS_GPU_RBPF_VERSIONED_" + n) for n in names}
    if not all(result.values()):
        pytest.skip("Explicit frozen DLL and independent CPU extensions required")
    return result


@pytest.fixture(scope="module")
def native(paths):
    spec = importlib.util.spec_from_file_location(
        "_rbpf_native_staging_probe", paths["STAGING_EXT"]
    )
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.fixture(scope="module")
def module(native):
    return SimpleNamespace(stage=native.stage, stage_fused=native.stage_versioned)


@pytest.fixture(scope="module")
def scorer(paths):
    with RegisterLUScorer(paths["DLL"]) as owned:
        yield owned


def options(paths):
    return dict(
        extension=paths["CANONICAL_EXT"],
        parent_extension=paths["PARENT_EXT"],
        mass_extension=paths["MASS_EXT"],
    )


def legacy(details):
    return {
        k: v
        for k, v in details.items()
        if k not in ("native_staging", "versioned_staging")
    }


def test_same_extension_controls_and_fast_access_are_exercised(native, module):
    original, _ = memo_trial(module, "original")
    off, _ = memo_trial(
        SimpleNamespace(stage=native.stage, stage_fused=native.stage_fused), "on"
    )
    on, stats = memo_trial(module, "on")
    exact(original, off)
    exact(original, on)
    assert stats["attribute_cache_reads"] > 0 and stats["identity_cache_hits"] > 0
    assert stats["identity_rechecks"] > 0 and stats["cache_deoptimizations"] == 0


@pytest.mark.parametrize(
    "change",
    [
        "context_dict",
        "memo_dict",
        "identity_dict",
        "identity_entry",
        "counts_dict",
        "counts_value",
        "counts_key",
        "memo_class",
    ],
)
def test_dictionary_slots_versions_and_partial_callback_order(native, change):
    def run(versioned):
        a = {(1, 0): {1: 0, 2: 2}}
        b = {(1, 0): {1: 0, 2: 3}}
        keys, hs, lookup, pending, staged = [], [], {}, [], []
        events = []

        class Count(int):
            def __add__(self, other):
                events.append(("counter_add", int(self), other))
                return int(self) + other

        class Foreign:
            def __hash__(self):
                events.append(("foreign_hash",))
                return 17

        class Other(_EpochMemo):
            def __getattribute__(self, name):
                if name in ("counts", "identities"):
                    events.append(("memo_attr", name))
                return object.__getattribute__(self, name)

        foreign = Foreign()
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

        def source(h):
            nonlocal mutated
            events.append(("source", key(h)))
            if h is b and not mutated:
                mutated = True
                if change == "context_dict":
                    context.__dict__ = dict(vars(context))
                elif change == "memo_dict":
                    memo.__dict__ = dict(vars(memo))
                elif change == "identity_dict":
                    memo.identities = {}
                elif change == "identity_entry":
                    memo.identities[id(a)] = (a, ((99, 99), ((1, 0), (2, 99))))
                elif change == "counts_dict":
                    memo.counts = dict(memo.counts)
                elif change == "counts_value":
                    memo.counts["canonical_requests"] = Count(
                        memo.counts["canonical_requests"]
                    )
                elif change == "counts_key":
                    memo.counts[foreign] = 0
                elif change == "memo_class":
                    memo.__class__ = Other
            return key(h)

        memo = _EpochMemo(source, None, cache_dd=False)
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
        stats = None
        error = None
        held = None
        try:
            if versioned:
                held, stats = native.stage_versioned(*args, _MEMO, _CODE, _EpochMemo)
            else:
                held = ORIGINAL(*args)
        except Exception as exc:
            error = type(exc).__name__, str(exc)
        if change == "counts_key" and mutated:
            del memo.counts[foreign]
        return (
            held,
            keys,
            hs,
            lookup,
            pending,
            staged,
            events,
            memo.summary(),
            error,
        ), stats

    expected, _ = run(False)
    actual, stats = run(True)
    exact(expected, actual)
    if stats is not None:
        assert stats["fused_hits"] > 0
        if change in (
            "context_dict",
            "memo_dict",
            "identity_dict",
            "counts_dict",
            "memo_class",
        ):
            assert stats["cache_deoptimizations"] > 0


@pytest.mark.parametrize("backend", ["scalar", "native_cpu", "gpu"])
@pytest.mark.parametrize("seed", [20260710, 20260711])
@pytest.mark.parametrize(
    "config", [(64, 6, 64, 4, False), (64, 6, 64, 4, True), (128, 32, 128, 8, True)]
)
def test_four_epochs_full_state_rng_graph_and_old_metadata(
    paths, scorer, backend, seed, config
):
    n, u, m, g, c = config
    saved, metadata = [], []
    for variant in ("original", "off", "on"):
        runtime, rtk, ctx = make_feedback_case(
            n,
            u,
            rows=m,
            groups=g,
            cluster=c,
            seed=seed,
            shadow=True,
            spawn=True,
            feedback=True,
            preload=True,
        )
        if variant == "original":
            call = NativeFeedbackMassStep(runtime, scorer, backend, **options(paths))
        else:
            call = VersionedStagingStep(
                runtime,
                scorer,
                backend,
                staging_extension=paths["STAGING_EXT"],
                cached=variant == "on",
                **options(paths),
            )
        runtime.step = call
        values, details = [], []
        for _ in range(4):
            output = call(rtk, ctx)
            values.append(state(runtime, rtk, output))
            details.append(deepcopy(legacy(call.last)))
            if variant != "original":
                stats = call.last_versioned
                assert stats["enabled"] and stats["fused_hits"] > 0
                assert (
                    stats["fused_hits"] + stats["original_index_calls"]
                    == call.last["requested_moves"]
                )
                if variant == "on":
                    assert stats["attribute_cache_reads"] > 0
                    assert stats["cache_deoptimizations"] == stats["invalidations"] == 0
        saved.append(values)
        metadata.append(details)
    for value in saved[1:]:
        exact(saved[0], value)
    for value in metadata[1:]:
        exact(metadata[0], value)


@pytest.mark.parametrize("binding", ["len", "type", "integer_prior"])
def test_literal_fallback_callback_trace(paths, monkeypatch, binding):
    from rbpf_move_batch import MoveBatchStep

    events = []
    if binding == "len":

        def custom(value):
            events.append("len")
            return len(value)

        monkeypatch.setitem(MoveBatchStep._moves.__globals__, "len", custom)
    elif binding == "type":

        def custom(value):
            events.append("type")
            return type(value)

        monkeypatch.setitem(_MEMO.__globals__, "type", custom)
    saved = []
    for variant in (False, True):
        runtime, rtk, ctx = make_feedback_case(feedback=False)
        if binding == "integer_prior":
            runtime.cfg.p_adopt = 1
        cls = VersionedStagingStep if variant else NativeFeedbackMassStep
        call = cls(
            runtime,
            mode="scalar",
            **options(paths),
            **(dict(staging_extension=paths["STAGING_EXT"]) if variant else {}),
        )
        runtime.step = call
        events.clear()
        result = call(rtk, ctx)
        saved.append((state(runtime, rtk, result), legacy(call.last), list(events)))
        if variant:
            assert call.last_versioned is None or not call.last_versioned["enabled"]
    exact(saved[0], saved[1])


def test_references_gc_next_call_and_strict_cleanup(paths, scorer):
    runtime, rtk, ctx = make_feedback_case()
    call = VersionedStagingStep(
        runtime, scorer, "gpu", staging_extension=paths["STAGING_EXT"], **options(paths)
    )
    runtime.step = call
    watched = [_EpochMemo, _MEMO, _CODE]
    gc.collect()
    before = [sys.getrefcount(x) for x in watched]
    for _ in range(12):
        call(rtk, ctx)
    gc.collect()
    assert before == [sys.getrefcount(x) for x in watched]
    assert call._view is call._oracle is None
    assert all(not n.startswith("_rbpf_component_mass") for n in call._scope)
    a, ra, ca = make_feedback_case()
    b, rb, cb = make_feedback_case()
    selected = VersionedStagingStep(
        b, mode="scalar", staging_extension=paths["STAGING_EXT"], **options(paths)
    )
    with np.errstate(under="raise"):
        for fn, r, c in ((a.step, ra, ca), (selected, rb, cb)):
            with pytest.raises(FloatingPointError):
                fn(r, c)
    exact(state(a, ra, None), state(b, rb, None))
    assert selected.last == dict(original_step_fallback="numpy_error_policy")
