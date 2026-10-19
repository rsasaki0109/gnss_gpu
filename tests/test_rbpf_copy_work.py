"""Exact copy counts/ownership and unchanged full PF observer behavior."""

from copy import deepcopy
import gc
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
from bench_rbpf_copy_work_field import ObservedCopyStep  # noqa: E402
from bench_rbpf_feedback_mass import _bind_fixture_callbacks  # noqa: E402
from rbpf_copy_work import CopyWork, inventory  # noqa: E402
from rbpf_feedback_cases import make_feedback_case  # noqa: E402
from rbpf_feedback_mass import NativeFeedbackMassStep  # noqa: E402
from rbpf_move_cases import comparison, state  # noqa: E402
from rbpf_register_lu import RegisterLUScorer  # noqa: E402


@pytest.fixture(scope="module")
def deps():
    return json.loads((ROOT / "dist/rbpf_direct_view_schedule_tests_20261018.plan.json").read_text(encoding="utf8"))["extensions"]


@pytest.fixture(scope="module")
def owned_scorer(deps):
    with RegisterLUScorer(ROOT / deps["DLL"]) as owned:
        yield owned


def _copy(H):
    return {g: dict(z) for g, z in H.items() if z}


def test_order_empty_groups_fresh_dicts_and_distinct_callers():
    h = {(2, 1): {7: 1 << 100, 3: -9}, (1, 0): {}, (2, 0): {9: 0}}
    def first():
        return _copy(h)
    def second():
        return _copy(h)
    observer = CopyWork(_copy)
    with observer:
        a, b = first(), second()
        with pytest.raises(RuntimeError, match="Stop observing"):
            observer.records()
    rows = observer.records()
    assert observer.restored and not observer.pending and sys.getprofile() is None
    assert {r["caller"] for r in rows} == {"first", "second"}
    for row in rows:
        assert row["calls"] == row["completed"] == 1
        assert row["source_groups"] == row["copied_entries"] == 3
        assert row["copied_groups"] == 2 and row["dropped_empty_groups"] == 1
        assert not any(row[k] for k in ("unsupported", "result_mismatches", "outer_aliases", "inner_aliases", "temporary_unchanged_groups"))
    assert a == b == {(2, 1): {7: 1 << 100, 3: -9}, (2, 0): {9: 0}}
    a[(2, 1)][7] = 2
    assert h[(2, 1)][7] == b[(2, 1)][7] == 1 << 100


def test_inventory_detects_cross_particle_and_internal_aliases():
    shared = {3: 2}
    h = {(1, 0): shared, (2, 0): shared}
    result = inventory([("a", h), ("b", h), ("c", {(3, 0): shared})])
    assert result == dict(hypotheses=3, unsupported=0, repeated_outer=1,
        repeated_inner=4, groups=5, entries=5)


def test_custom_mapping_is_not_inspected_by_observer():
    class Custom(dict):
        reads = 0
        def items(self):
            self.reads += 1
            return super().items()
    h = Custom({(1, 0): {3: 2}})
    with CopyWork(_copy) as observer:
        result = _copy(h)
    assert result == dict(h) and h.reads == 1
    assert observer.records()[0]["unsupported"] == 1
    assert inventory([("custom", h)])["unsupported"] == 1 and h.reads == 1


def test_aliases_and_wrong_result_are_reported_without_repair():
    def wrong(H):
        return H
    h = {(1, 0): {2: 1}, (2, 0): {}}
    with CopyWork(wrong) as observer:
        result = wrong(h)
    row = observer.records()[0]
    assert result is h and row["outer_aliases"] == 1
    assert row["inner_aliases"] == 2 and row["result_mismatches"] == 1


def test_unregistered_same_name_does_not_read_temporary_locals():
    def _extend():
        return _copy({(1, 0): {3: 0}})
    with CopyWork(_copy) as observer:
        _extend()
    assert observer.records()[0]["temporary_unchanged_groups"] == 0


def test_registered_temporary_allocation_counts():
    h = {(1, 0): {2: 0, 3: 1}, (2, 0): {4: 0, 5: 2}}
    def shrink():
        g = (1, 0)
        return _copy(h), g
    with CopyWork(_copy, {shrink.__code__: "shrink"}) as observer:
        shrink()
    row = observer.records()[0]
    assert row["copied_groups"] == 2 and row["temporary_unchanged_groups"] == 1
    assert row["temporary_unchanged_entries"] == 2


def test_exception_restores_callback_and_does_not_suppress_error():
    def interrupted(H):
        raise ValueError("original copy error")
    observer = CopyWork(interrupted)
    with pytest.raises(ValueError, match="original copy error"):
        with observer:
            interrupted({})
    assert observer.restored and sys.getprofile() is None and not observer.pending
    row = observer.records()[0]
    assert row["calls"] == 1 and row["completed"] == 0


def test_existing_profiler_is_retained():
    def existing(frame, event, arg):
        pass
    prior = sys.getprofile()
    sys.setprofile(existing)
    try:
        with pytest.raises(RuntimeError, match="existing profiler"):
            with CopyWork(_copy):
                pytest.fail("Observer body must not execute")
        assert sys.getprofile() is existing
    finally:
        sys.setprofile(prior)


def _make(inputs, deps, cls, scorer=None, backend="scalar"):
    runtime, rtk, _ = inputs
    _bind_fixture_callbacks(rtk)
    call = cls(runtime, scorer, backend, extension=ROOT / deps["CANONICAL_EXT"],
        parent_extension=ROOT / deps["PARENT_EXT"], mass_extension=ROOT / deps["MASS_EXT"])
    runtime.step = call
    return call


@pytest.mark.parametrize("backend", ("scalar", "native_cpu", "gpu"))
@pytest.mark.parametrize("seed", (20260710, 20260713))
@pytest.mark.parametrize("cluster", (False, True))
def test_four_full_epochs_exact(deps, owned_scorer, backend, seed, cluster):
    left = make_feedback_case(32, 6, seed=seed, cluster=cluster, shadow=True, spawn=True)
    right = deepcopy(left)
    calls = (_make(left, deps, NativeFeedbackMassStep, owned_scorer, backend),
        _make(right, deps, ObservedCopyStep, owned_scorer, backend))
    enabled, threshold = gc.isenabled(), gc.get_threshold()
    original_codes = {k: f.__code__ for k, f in calls[1]._functions.items()}
    for epoch in range(4):
        values = []
        for inputs, call in zip((left, right), calls):
            inputs[1].epoch2 = epoch
            out = call(inputs[1], inputs[2])
            values.append((state(*inputs[:2], out), deepcopy(call.last)))
        result = comparison(*values)
        assert result["discrete_equal"] and result["numeric_bits_equal"] and result["max_abs_error"] == 0
        row = calls[1].records[-1]
        assert row["completed"] and row["profiler_restored"] and row["valid"]
        assert not any(r[k] for r in row["sites"] for k in ("unsupported", "result_mismatches", "outer_aliases", "inner_aliases"))
        for point in ("before", "after"):
            assert not any(row[point][k] for k in ("unsupported", "repeated_outer", "repeated_inner"))
    assert all(calls[1]._functions[k].__code__ is code for k, code in original_codes.items())
    assert gc.isenabled() == enabled and gc.get_threshold() == threshold
    assert sys.getprofile() is None and calls[1]._view is calls[1]._oracle is None


def test_full_step_exception_cleanup(deps):
    inputs = make_feedback_case(8, 4)
    call = _make(inputs, deps, ObservedCopyStep)
    compiled = call.compiled
    def interrupted(view, rtk, ctx):
        compiled(view, rtk, ctx)
        raise ValueError("original PF error")
    call.compiled = interrupted
    with pytest.raises(ValueError, match="original PF error"):
        call(inputs[1], inputs[2])
    assert sys.getprofile() is None and call._view is call._oracle is None
    assert not call.records[-1]["completed"] and call.records[-1]["profiler_restored"]
