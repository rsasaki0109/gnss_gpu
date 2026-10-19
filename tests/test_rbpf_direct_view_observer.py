"""Observe lookup work without changing numeric metadata or profiler owners."""

from copy import deepcopy
import sys

import pytest

from test_rbpf_direct_view import ROOT, dependencies, scorer, exact  # noqa: F401
from bench_rbpf_direct_view_field import ObservedDirectStep
from bench_rbpf_feedback_mass import _bind_fixture_callbacks
from rbpf_feedback_cases import make_feedback_case
from rbpf_feedback_mass import NativeFeedbackMassStep
from rbpf_move_cases import state


def make(inputs, deps, cls, scorer=None, mode="scalar"):
    runtime, rtk, _ = inputs
    _bind_fixture_callbacks(rtk)
    call = cls(runtime, scorer, mode, extension=ROOT / deps["CANONICAL_EXT"],
        parent_extension=ROOT / deps["PARENT_EXT"], mass_extension=ROOT / deps["MASS_EXT"])
    runtime.step = call
    return call


@pytest.mark.parametrize("mode", ("scalar", "native_cpu", "gpu"))
def test_lookup_count_and_legacy_metadata_exact(dependencies, scorer, mode):
    initial = make_feedback_case(32, 6)
    left, right = deepcopy(initial), deepcopy(initial)
    original = make(left, dependencies, NativeFeedbackMassStep, scorer, mode)
    candidate = make(right, dependencies, ObservedDirectStep, scorer, mode)
    for epoch in range(2):
        values = []
        for inputs, call in ((left, original), (right, candidate)):
            inputs[1].epoch2 = epoch
            out = call(inputs[1], inputs[2])
            values.append((state(*inputs[:2], out), deepcopy(call.last)))
        exact(*values)
        row = candidate.records[-1]
        assert row["original_getattr_calls"] == 3 and row["cache_at_compiled_exit"] == 0
        assert row["profiler_restored"] and not row["watched_method_live_after_return"]
        assert sys.getprofile() is None


def test_existing_profiler_is_retained(dependencies):
    inputs = make_feedback_case(8, 4)
    call = make(inputs, dependencies, ObservedDirectStep)
    def existing(frame, event, arg):
        return None
    before = sys.getprofile()
    sys.setprofile(existing)
    try:
        with pytest.raises(RuntimeError, match="existing profiler"):
            call(inputs[1], inputs[2])
        assert sys.getprofile() is existing and not call.records
    finally:
        sys.setprofile(before)


def test_exception_restores_profiler_and_view(dependencies):
    inputs = make_feedback_case(8, 4)
    call = make(inputs, dependencies, ObservedDirectStep)
    base = call.compiled
    def interrupted(view, rtk, ctx):
        base(view, rtk, ctx)
        raise ValueError("observer original exception")
    call.compiled = interrupted
    with pytest.raises(ValueError, match="observer original exception"):
        call(inputs[1], inputs[2])
    assert sys.getprofile() is None and call._view is call._oracle is None
    assert len(call.records) == 1 and call.records[0]["profiler_restored"]
