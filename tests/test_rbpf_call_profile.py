"""Original state/RNG, nested clocks, call counts and profiler cleanup."""

from copy import deepcopy
import os
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
from bench_rbpf_feedback_mass import _bind_fixture_callbacks  # noqa: E402
from bench_rbpf_call_profile_field import adapt_field  # noqa: E402
from rbpf_call_profile import CallProfileStep, ProcessMonitor  # noqa: E402
from rbpf_feedback_cases import make_feedback_case  # noqa: E402
from rbpf_feedback_mass import NativeFeedbackMassStep  # noqa: E402
from rbpf_likelihood_batch import BatchScorer  # noqa: E402
from rbpf_move_cases import state  # noqa: E402
from test_rbpf_native_staging import exact  # noqa: E402


@pytest.fixture(scope="module")
def extensions():
    names = dict(
        extension="GNSS_GPU_RBPF_CANONICAL_EXT",
        parent_extension="GNSS_GPU_RBPF_PARENT_EXT",
        mass_extension="GNSS_GPU_RBPF_MASS_EXT",
    )
    if any(not os.environ.get(n) for n in names.values()):
        pytest.skip("Explicit unchanged extensions required")
    return {k: os.environ[v] for k, v in names.items()}


@pytest.fixture(scope="module")
def scorer():
    if not os.environ.get("GNSS_GPU_RBPF_PROBE_DLL"):
        pytest.skip("Explicit unchanged numerical DLL required")
    with BatchScorer(os.environ["GNSS_GPU_RBPF_PROBE_DLL"]) as value:
        yield value


@pytest.mark.parametrize("mode", ["scalar", "native_cpu", "gpu"])
@pytest.mark.parametrize("seed", [20260710, 20260711])
@pytest.mark.parametrize(
    "options",
    [
        {},
        dict(cluster=True),
        dict(shadow=True, spawn=True),
        dict(feedback=False, preload=False, resample=False),
    ],
)
def test_four_epoch_all_state_rng_graph_metadata_and_call_counts(
    extensions, scorer, mode, seed, options
):
    a, ra, ca = make_feedback_case(seed=seed, **options)
    b, rb, cb = make_feedback_case(seed=seed, **options)
    old = NativeFeedbackMassStep(a, scorer, mode, **extensions)
    new = CallProfileStep(b, scorer, mode, **extensions)
    a.step, b.step = old, new
    for epoch in range(4):
        ra.epoch2 = rb.epoch2 = epoch
        if options.get("feedback", True):
            _bind_fixture_callbacks(ra)
            _bind_fixture_callbacks(rb)
        oa, ob = old(ra, deepcopy(ca)), new(rb, deepcopy(cb))
        exact(state(a, ra, oa), state(b, rb, ob))
        exact(old.last, new.last)
        assert (
            sys.getprofile() is None
            and not new._profiling
            and new.monitor.current is None
        )
    assert len(new.monitor.records) == 4
    rows = new.export()
    root = [
        r
        for r in rows
        if r["code"]["file"] == "benchmarks/rbpf_feedback_mass.py"
        and r["code"]["function"] == "__call__"
    ]
    assert len(root) == 1 and root[0]["calls"] == 4
    assert all(
        r["calls"] >= r["recursive_calls"] >= 0 and r["self_seconds"] >= 0 for r in rows
    )


class _Covariance:
    def jointMarginalCovariance(self, keys):
        return np.eye(len(keys))


mg = _Covariance()


class _Base:
    phase = 2

    def process(self, value):
        return self._write_back_tc(value)

    def _write_back_tc(self, value):
        return mg.jointMarginalCovariance(list(range(value)))


class _Derived(_Base):
    def _write_back_tc(self, value):
        return super()._write_back_tc(value) + mg.jointMarginalCovariance(
            list(range(value))
        )


def test_actual_method_adapter_order_super_closure_and_integer_span_partition():
    monitor = ProcessMonitor()
    measured = monitor.instrument_type(_Base)
    exact(_Base().process(3), measured().process(3))
    assert len(monitor.records) == 1
    record = monitor.records[0]
    assert record["context"]["phase_before"] == 2
    for key in ("wall_ns", "cpu_ns"):
        assert (
            sum(v["exclusive_" + key] for v in record["spans"].values())
            == record["spans"]["rtk.process"][key]
        )
    assert record["spans"]["gtsam.jointMarginalCovariance"]["calls"] == 1
    assert all(p["restored_AST_equal"] for p in monitor.proofs)
    derived_monitor = ProcessMonitor()
    derived = derived_monitor.instrument_type(_Derived)
    exact(_Derived().process(4), derived().process(4))
    assert (
        derived_monitor.records[0]["spans"]["gtsam.jointMarginalCovariance"]["calls"]
        == 1
    )
    proof = next(
        p for p in derived_monitor.proofs if p["method"] == "_Derived._write_back_tc"
    )
    assert proof["replaced_C_calls"] == ["mg.jointMarginalCovariance"]


def test_field_adapter_keeps_original_function_and_literal_snapshot_statements():
    import bench_rbpf_move_field as field

    original = field.main
    adapted, proof = adapt_field(ProcessMonitor(), None, None)
    assert field.main is original and adapted is not original
    assert proof["restored_main_AST_equal"]


def test_snapshot_is_an_explicit_child_and_error_unwinds():
    monitor = ProcessMonitor()
    with pytest.raises(ValueError, match="original"):
        with monitor.span("process"):
            with monitor.span("pf"):
                pass
            with monitor.span("diagnostic_snapshot"):
                raise ValueError("original")
    assert monitor.current is None and len(monitor.records) == 1
    assert set(monitor.records[0]["spans"]) == {"pf", "diagnostic_snapshot", "process"}


def test_strict_policy_preserves_partial_state_and_hook(extensions):
    a, ra, ca = make_feedback_case()
    b, rb, cb = make_feedback_case()
    new = CallProfileStep(b, mode="scalar", **extensions)
    with np.errstate(under="raise"):
        for fn, rtk, ctx in ((a.step, ra, ca), (new, rb, cb)):
            with pytest.raises(FloatingPointError):
                fn(rtk, ctx)
    exact(state(a, ra, None), state(b, rb, None))
    assert (
        sys.getprofile() is None and not new._profiling and new.monitor.current is None
    )


def test_existing_profiling_hook_is_not_overwritten(extensions):
    runtime, rtk, ctx = make_feedback_case()
    call = CallProfileStep(runtime, mode="scalar", **extensions)

    def hook(*args):
        pass

    sys.setprofile(hook)
    try:
        with pytest.raises(RuntimeError, match="idle profiling hook"):
            call(rtk, ctx)
        assert sys.getprofile() is hook
    finally:
        sys.setprofile(None)
