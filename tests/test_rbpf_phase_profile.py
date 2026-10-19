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
from rbpf_phase_profile import PhaseProfileStep  # noqa: E402


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
@pytest.mark.parametrize("seed", [20260710, 20260711])
@pytest.mark.parametrize(
    "options", [{}, dict(cluster=True), dict(shadow=True, spawn=True)]
)
def test_four_epoch_all_state_metadata_and_span_partition(
    extensions, scorer, mode, seed, options
):
    a, ra, ca = make_feedback_case(seed=seed, **options)
    b, rb, cb = make_feedback_case(seed=seed, **options)
    old = NativeFeedbackMassStep(a, scorer, mode, **extensions)
    profiled = PhaseProfileStep(b, scorer, mode, **extensions)
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
        assert all(spans is not p for p in snapshots)
        snapshots.append(spans)
        assert profiled._spans is None


def test_strict_policy_error_preserves_partial_state_and_cleans_stack(extensions):
    a, ra, ca = make_feedback_case()
    b, rb, cb = make_feedback_case()
    profiled = PhaseProfileStep(b, mode="scalar", **extensions)
    with np.errstate(under="raise"):
        for call, rtk, ctx in ((a.step, ra, ca), (profiled, rb, cb)):
            with pytest.raises(FloatingPointError):
                call(rtk, ctx)
    exact(state(a, ra, None), state(b, rb, None))
    assert profiled._spans is None and list(profiled.last_profile) == ["step"]
    assert profiled.last == dict(original_step_fallback="numpy_error_policy")
