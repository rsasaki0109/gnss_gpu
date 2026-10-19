"""Profiler spans preserve D236 state, metadata, RNG and error behavior."""

from copy import deepcopy
import os
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
from bench_rbpf_feedback_mass import _bind_fixture_callbacks  # noqa: E402
from rbpf_feedback_cases import make_feedback_case  # noqa: E402
from rbpf_feedback_mass import NativeFeedbackMassStep  # noqa: E402
from rbpf_likelihood_batch import BatchScorer  # noqa: E402
from rbpf_move_cases import comparison, state  # noqa: E402
from rbpf_move_phase_profile import MovePhaseProfileStep, AST_PROOFS  # noqa: E402
from rbpf_phase_profile import _Spans  # noqa: E402
from rbpf_move_reuse import MoveReuseStep  # noqa: E402


@pytest.fixture(scope="module")
def extensions():
    names = dict(
        extension="GNSS_GPU_RBPF_CANONICAL_EXT",
        parent_extension="GNSS_GPU_RBPF_PARENT_EXT",
        mass_extension="GNSS_GPU_RBPF_MASS_EXT",
    )
    if any(not os.environ.get(n) for n in names.values()):
        pytest.skip("Explicit unchanged extensions required")
    return {k: os.environ[n] for k, n in names.items()}


@pytest.fixture(scope="module")
def scorer():
    dll = os.environ.get("GNSS_GPU_RBPF_PROBE_DLL")
    if not dll:
        pytest.skip("Explicit unchanged DLL required")
    with BatchScorer(dll) as result:
        yield result


def exact(a, b):
    result = comparison(a, b)
    assert (
        result["discrete_equal"]
        and result["numeric_bits_equal"]
        and result["max_abs_error"] == 0.0
    ), result


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
def test_four_epoch_all_state_metadata_and_span_partition(
    extensions, scorer, mode, seed, options
):
    a, ra, ca = make_feedback_case(seed=seed, **options)
    b, rb, cb = make_feedback_case(seed=seed, **options)
    old = NativeFeedbackMassStep(a, scorer, mode, **extensions)
    profiled = MovePhaseProfileStep(b, scorer, mode, **extensions)
    a.step, b.step = old, profiled
    snapshots = []
    for ep in range(4):
        ra.epoch2 = rb.epoch2 = ep
        _bind_fixture_callbacks(ra)
        _bind_fixture_callbacks(rb)
        oa, ob = old(ra, deepcopy(ca)), profiled(rb, deepcopy(cb))
        exact(state(a, ra, oa), state(b, rb, ob))
        exact(old.last, profiled.last)
        spans = profiled.last_profile
        assert spans["step"]["calls"] == 1 and spans["_moves"]["calls"] == 1
        assert spans["_conditional_mean"]["inclusive_ns"] >= spans.get(
            "_ls_offset", {}
        ).get("inclusive_ns", 0)
        assert (
            sum(v["exclusive_ns"] for v in spans.values())
            == spans["step"]["inclusive_ns"]
        )
        assert all(
            v["exclusive_ns"] >= 0 and v["inclusive_ns"] >= v["exclusive_ns"]
            for v in spans.values()
        )
        for name in ("_move_staging", "_move_scores", "_move_choice"):
            assert spans[name]["calls"] == 1
        if mode != "scalar":
            for name in ("_score_pack", "_score_native", "_score_finish"):
                assert spans[name]["calls"] == 1
        assert (
            sum(
                spans[n]["inclusive_ns"]
                for n in ("_move_staging", "_move_scores", "_move_choice")
            )
            <= spans["_moves"]["inclusive_ns"]
        )
        assert all(spans is not p for p in snapshots)
        snapshots.append(spans)
        assert profiled._spans is None


def test_strict_policy_error_preserves_partial_state_and_cleans_stack(extensions):
    a, ra, ca = make_feedback_case()
    b, rb, cb = make_feedback_case()
    profiled = MovePhaseProfileStep(b, mode="scalar", **extensions)
    with np.errstate(under="raise"):
        for call, rtk, ctx in ((a.step, ra, ca), (profiled, rb, cb)):
            with pytest.raises(FloatingPointError):
                call(rtk, ctx)
    exact(state(a, ra, None), state(b, rb, None))
    assert profiled._spans is None and list(profiled.last_profile) == ["step"]
    assert profiled.last == dict(original_step_fallback="numpy_error_policy")


def test_all_literal_source_restoration_proofs():
    assert len(AST_PROOFS) == 5 and all(p["restored_AST_equal"] for p in AST_PROOFS)


def test_helper_mutation_guard(extensions, monkeypatch):
    runtime, rtk, ctx = make_feedback_case()
    call = MovePhaseProfileStep(runtime, mode="scalar", **extensions)
    monkeypatch.setattr(MoveReuseStep, "_moves", lambda *a: None)
    with pytest.raises(ValueError, match="helper changed"):
        call(rtk, ctx)
    assert call._spans is None


def test_empty_batch_keeps_original_early_return_and_closes_span(extensions, scorer):
    runtime, rtk, ctx = make_feedback_case()
    call = MovePhaseProfileStep(runtime, scorer, "native_cpu", **extensions)
    runtime.step = call
    call(rtk, ctx)  # refresh private literal helper scopes
    spans = call._spans = _Spans()
    root = spans.start("step")
    try:
        a = scorer.score([], None, None, mode="native_cpu")
        b = call.scorer.score([], None, None, mode="native_cpu")
        exact(a, b)
    finally:
        spans.stop(root)
        call._spans = None
    assert (
        spans.values["_score_pack"]["calls"] == 1
        and "_score_native" not in spans.values
    )
    assert not spans.stack


def test_capacity_scalar_fallback_preserves_state_and_metadata(extensions):
    dll = os.environ.get("GNSS_GPU_RBPF_PROBE_DLL")
    if not dll:
        pytest.skip("Explicit DLL required")
    a, ra, ca = make_feedback_case()
    b, rb, cb = make_feedback_case()
    with BatchScorer(dll, capacity=1) as scorer:
        old = NativeFeedbackMassStep(a, scorer, "gpu", **extensions)
        call = MovePhaseProfileStep(b, scorer, "gpu", **extensions)
        a.step, b.step = old, call
        oa, ob = old(ra, deepcopy(ca)), call(rb, deepcopy(cb))
        exact(state(a, ra, oa), state(b, rb, ob))
        exact(old.last, call.last)
        assert call.last["batch"]["scalar_fallbacks"] > 0
        assert (
            call.last_profile["_score_pack"]["calls"] == 1
            and "_score_native" not in call.last_profile
        )
