"""Sparse complete PF boundaries preserve full state, live math, RNG and errors."""

from contextlib import ExitStack
from copy import deepcopy
import os
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
import rbpf_component_mass as component  # noqa: E402
import rbpf_pf_boundaries as selected  # noqa: E402
from rbpf_cluster_parameter_reuse import ClusterBatchScorer  # noqa: E402
from rbpf_feedback_mass import NativeFeedbackMassStep  # noqa: E402
from rbpf_feedback_cases import make_feedback_case  # noqa: E402
from rbpf_move_batch import MoveBatchStep  # noqa: E402
from rbpf_move_cases import comparison, state  # noqa: E402


@pytest.fixture(scope="module")
def paths():
    names = ("DLL", "EXT", "CANONICAL_EXT", "PARENT_EXT", "MASS_EXT")
    found = {n: os.environ.get("GNSS_GPU_RBPF_CLUSTER_" + n) for n in names}
    if not all(found.values()):
        pytest.skip(
            "Explicit frozen numeric DLL and all standalone extensions required"
        )
    return found


@pytest.fixture(scope="module")
def scorer(paths):
    with ClusterBatchScorer(paths["DLL"], paths["EXT"]) as owned:
        yield owned


def exact(a, b):
    c = comparison(a, b)
    assert (
        c["discrete_equal"] and c["numeric_bits_equal"] and c["max_abs_error"] == 0.0
    ), c


def make(
    paths, scorer, kind, backend="gpu", seed=20260710, options=(64, 6, 64, 4, True)
):
    n, u, m, g, cluster = options
    runtime, rtk, ctx = make_feedback_case(
        n,
        u,
        rows=m,
        groups=g,
        cluster=cluster,
        shadow=True,
        spawn=True,
        feedback=True,
        preload=True,
        seed=seed,
    )
    ext = dict(
        extension=paths["CANONICAL_EXT"],
        parent_extension=paths["PARENT_EXT"],
        mass_extension=paths["MASS_EXT"],
    )
    cls = NativeFeedbackMassStep if kind == "original" else selected.BoundaryProfileStep
    call = cls(
        runtime,
        scorer,
        backend,
        **({} if kind == "original" else dict(trace=kind == "profile")),
        **ext,
    )
    runtime.step = call
    return runtime, rtk, ctx, call


def trace_ok(call):
    assert call._events is None
    parts = selected.partition(call.last_trace)
    assert sum(parts.values()) == call.last_trace[-1][1] - call.last_trace[0][1]
    assert all(type(v) is int and v >= 0 for v in parts.values())
    assert "move_staging" in parts and "move_scores" in parts and "move_choice" in parts
    assert len(call.last_trace) == 21


@pytest.mark.parametrize("backend", ["native_cpu", "gpu"])
@pytest.mark.parametrize("seed", [20260710, 20260711])
@pytest.mark.parametrize(
    "options", [(64, 6, 64, 4, False), (64, 6, 64, 4, True), (128, 32, 128, 8, True)]
)
def test_four_epoch_complete_state_rng_graph_and_metadata(
    paths, scorer, backend, seed, options
):
    traces = []
    metadata = []
    counts = []
    for kind in ("original", "clone", "profile"):
        runtime, rtk, ctx, call = make(paths, scorer, kind, backend, seed, options)
        before = scorer.cluster_engine.counts()
        saved = []
        details = []
        for _ in range(4):
            out = call(rtk, ctx)
            saved.append(state(runtime, rtk, out))
            details.append(deepcopy(call.last))
            if kind == "profile":
                trace_ok(call)
            elif kind == "clone":
                assert call.last_trace is None and call._events is None
        traces.append(saved)
        metadata.append(details)
        counts.append(
            {k: v - before[k] for k, v in scorer.cluster_engine.counts().items()}
        )
    for actual in traces[1:]:
        exact(traces[0], actual)
    for actual in metadata[1:]:
        exact(metadata[0], actual)
    assert counts[0] == counts[1] == counts[2]


def test_ast_restoration_live_global_objects_and_wrapper_code(paths, scorer):
    assert len(selected.AST_PROOFS) == 5 and all(
        p["restored_AST_equal"] for p in selected.AST_PROOFS
    )
    _, _, _, clone = make(paths, scorer, "clone")
    _, _, _, profile = make(paths, scorer, "profile")
    assert clone.inner_step.__code__ is component._STEP_CODE
    assert clone.inner_step.__globals__ is clone._scope
    assert profile.inner_step.__globals__ is profile._scope
    assert profile.inner_step.__code__.co_freevars == ("_mark",)
    assert profile.compiled.__code__ is profile.original_compiled.__code__

    def base(fn):
        return dict(zip(fn.__code__.co_freevars, fn.__closure__, strict=True))[
            "base"
        ].cell_contents

    assert base(profile.compiled).__code__ is base(profile.original_compiled).__code__
    assert profile.compiled.__globals__ is profile.original_compiled.__globals__
    fn = profile.boundary_moves
    from rbpf_native_canonical import NativeCanonicalStep
    from rbpf_move_key_memo import MoveKeyMemoStep
    from rbpf_move_reuse import MoveReuseStep

    for cls in (NativeCanonicalStep, MoveKeyMemoStep, MoveReuseStep):
        assert fn.__globals__ is cls._moves.__globals__ and fn.__code__.co_freevars == (
            "_inner",
        )
        fn = fn.__closure__[0].cell_contents
    assert (
        fn.__globals__ is MoveBatchStep._moves.__globals__
        and fn.__code__.co_freevars == ("_mark",)
    )
    assert not any(n.startswith("_mark") for n in profile._scope)


@pytest.mark.parametrize(
    "kind", ["bad_extract", "unsafe_policy", "skip", "feedback_error", "rng_error"]
)
def test_exceptions_skips_and_partial_state_match_original(paths, scorer, kind):
    results = []
    objects = []
    for mode in ("original", "profile"):
        runtime, rtk, ctx, call = make(paths, scorer, mode)
        events = []
        if kind == "bad_extract":

            def bad(*args):
                events.append("extract")
                raise ValueError("sentinel extract")

            runtime._extract = bad
        elif kind == "skip":
            ctx["problem"].ok = False
        elif kind == "feedback_error":

            def bad(*args, **kwargs):
                events.append("release")
                raise ValueError("sentinel release")

            rtk._release_ambiguity = bad
            # Force a previously held ambiguity to disappear at feedback prelude.
            rtk._committed[(99999, 0)] = 1.0
        elif kind == "rng_error":
            literal = runtime.rng

            class RNG:
                bit_generator = literal.bit_generator

                def choice(self, *args, **kwargs):
                    events.append("choice")
                    raise ValueError("sentinel choice")

            runtime.rng = RNG()
        with np.errstate(
            all="raise" if kind == "unsafe_policy" else "warn", under="ignore"
        ):
            try:
                output, error = call(rtk, ctx), None
            except (ValueError, ArithmeticError) as exc:
                output, error = None, (type(exc), str(exc))
        results.append(
            (error, events, state(runtime, rtk, output), deepcopy(call.last))
        )
        objects.append(call)
    assert results[0][:2] == results[1][:2]
    exact(results[0][2:], results[1][2:])
    call = objects[-1]
    assert call._events is None
    parts = selected.partition(call.last_trace)
    if kind == "unsafe_policy":
        assert list(parts) == ["call_entry"]
    if kind in ("skip", "bad_extract"):
        assert "move_staging" not in parts and "body_exit" in parts


def test_cleanup_then_next_epoch_replaces_trace(paths, scorer):
    runtime, rtk, ctx, call = make(paths, scorer, "profile")
    old = runtime._extract

    def bad(*args):
        raise ValueError("first epoch")

    runtime._extract = bad
    with pytest.raises(ValueError, match="first epoch"):
        call(rtk, ctx)
    first = call.last_trace
    runtime._extract = old
    call(rtk, ctx)
    trace_ok(call)
    assert call.last_trace is not first and len(first) == 5
    assert (
        "_rbpf_feedback_mass" not in call._scope
        and "_rbpf_component_mass" not in call._scope
    )


def test_mutated_helper_rejected_without_stale_trace(paths, scorer, monkeypatch):
    _, rtk, ctx, call = make(paths, scorer, "profile")
    call(rtk, ctx)
    monkeypatch.setattr(MoveBatchStep, "_moves", lambda *a: [])
    with pytest.raises(ValueError, match="Frozen move helper"):
        call(rtk, ctx)
    assert call.last_trace is None and call._events is None


def test_original_live_numpy_binding_is_evaluated_once_per_original_expression(
    paths, scorer
):
    scope = MoveBatchStep._moves.__globals__
    original = scope["math"]
    from types import SimpleNamespace

    events = []

    def log(value):
        events.append(float(value))
        return original.log(value)

    outcomes = []
    try:
        scope["math"] = SimpleNamespace(log=log)
        for kind in ("original", "profile"):
            runtime, rtk, ctx, call = make(paths, scorer, kind)
            # Setup can use source math; compare only execution callbacks.
            events.clear()
            out = call(rtk, ctx)
            outcomes.append(
                (state(runtime, rtk, out), deepcopy(call.last), list(events))
            )
    finally:
        scope["math"] = original
    exact(outcomes[0][:2], outcomes[1][:2])
    assert outcomes[0][2] == outcomes[1][2] and outcomes[0][2]


def test_nonreentrant_guard_preserves_outer_recorder(paths, scorer):
    _, rtk, ctx, call = make(paths, scorer, "profile")
    events = call._events = []
    with pytest.raises(RuntimeError, match="concurrent"):
        call(rtk, ctx)
    assert call._events is events
    call._events = None


def test_partition_rejects_duplicate_or_backwards_boundaries():
    for events in (
        (("call_entry", 1), ("body_entry", 0), ("call_return", 2)),
        (("call_entry", 1), ("body_entry", 2), ("body_entry", 3), ("call_return", 4)),
    ):
        with pytest.raises(AssertionError):
            selected.partition(events)
