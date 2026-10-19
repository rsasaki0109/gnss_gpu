"""Check the diagnostic wrapper before using it with receiver data."""

import gc
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
from bench_rbpf_feedback_mass import _bind_fixture_callbacks  # noqa: E402
from bench_rbpf_view_release_field import BaselineObservedStep, CandidateObservedStep  # noqa: E402
from rbpf_feedback_cases import make_feedback_case  # noqa: E402


def make(release, *, skipped=False, interrupt=False):
    runtime, rtk, ctx = make_feedback_case(32, 8)
    ctx["problem"].ok = not skipped
    original = runtime._extract
    def extract(rtk, context):
        gc.collect(0)
        if interrupt:
            raise RuntimeError("diagnostic cleanup")
        return original(rtk, context)
    runtime._extract = extract
    _bind_fixture_callbacks(rtk)
    paths = json.loads((ROOT / "dist/rbpf_view_release_tests_20261018.plan.json").read_text(encoding="utf8"))["extensions"]
    cls = CandidateObservedStep if release else BaselineObservedStep
    call = cls(runtime, None, "scalar", extension=ROOT / paths["CANONICAL_EXT"], parent_extension=ROOT / paths["PARENT_EXT"], mass_extension=ROOT / paths["MASS_EXT"])
    runtime.step = call
    return call, rtk, ctx


@pytest.mark.parametrize("release", [False, True])
def test_real_gc_event_and_weak_observation(release):
    call, rtk, ctx = make(release)
    callbacks, policy = list(gc.callbacks), (gc.isenabled(), gc.get_threshold())
    call(rtk, ctx)
    assert gc.callbacks == callbacks and policy == (gc.isenabled(), gc.get_threshold())
    row = call.records[0]
    assert row["valid"] and row["profiled_ns"] > 0
    assert row["GC_events"] and all(e["ns"] >= 0 for e in row["GC_events"])
    assert (row["method_cache_at_compiled_exit"] == 0) == release
    assert row["watched_method_live_after_step"] != release
    assert "view_release" not in call.last
    assert call._events is None


@pytest.mark.parametrize("strict", [False, True])
def test_observer_skipped_epoch_and_strict_fallback(strict):
    call, rtk, ctx = make(True, skipped=True)
    callbacks = list(gc.callbacks)
    with np.errstate(all="raise" if strict else "ignore"):
        call(rtk, ctx)
    assert gc.callbacks == callbacks and len(call.records) == 1
    row = call.records[0]
    assert not row["valid"] and not row["watched_method_live_after_step"]
    assert row["method_cache_at_compiled_exit"] == (None if strict else 0)


def test_observer_exception_removes_callback():
    call, rtk, ctx = make(True, interrupt=True)
    callbacks = list(gc.callbacks)
    with pytest.raises(RuntimeError, match="diagnostic cleanup"):
        call(rtk, ctx)
    assert gc.callbacks == callbacks and call._events is None
    assert len(call.records) == 1 and call.records[0]["method_cache_at_compiled_exit"] == 0
