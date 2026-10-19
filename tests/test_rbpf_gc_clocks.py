"""GC phase attribution, immutable estimator forwarding and barrier timing."""

from copy import deepcopy
import gc
import io
import json
from pathlib import Path
import subprocess
import sys
import threading

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
from rbpf_gc_clocks import BodyClocks, GCPhases, PhaseLockstepClocks, thread_cycle_clock  # noqa: E402
from rbpf_epoch_lockstep import LABELS, PREFIX, PipeEndpoint, coordinate  # noqa: E402


def counter(values):
    iterator = iter(values)
    return lambda: next(iterator)


def monitor(**options):
    return BodyClocks(counter([0, 100]), counter([0, 70]), counter([0, 60]), phases=GCPhases(), cycles=counter([200, 250]), **options)


def test_exclusive_phases_restore_and_keep_gc_policy():
    phases = GCPhases(counter([10, 20, 30, 40]))
    before = gc.isenabled(), gc.get_threshold()
    with phases.phase("process"):
        with phases.phase("pf"):
            phases.event("start", dict(generation=0))
            phases.event("stop", dict(generation=0, collected=3, uncollectable=0))
        assert phases.current == "process"
    with phases.phase("serialize"):
        phases.event("start", dict(generation=1))
        phases.event("stop", dict(generation=1, collected=7, uncollectable=0))
    assert phases.current == "outside" and before == (gc.isenabled(), gc.get_threshold())
    assert [e["phase"] for e in phases.events] == ["pf", "serialize"]
    assert [e["ns"] for e in phases.events] == [10, 10]
    assert all(not e["crossed_boundary"] and e["owner_thread"] for e in phases.events)


@pytest.mark.parametrize("problem", ["stop", "overlap", "backwards", "generation", "unknown"])
def test_invalid_probe_events_retained_as_errors(problem):
    p = GCPhases(counter([20, 10, 30]))
    if problem == "stop":
        p.event("stop", dict(generation=0))
    elif problem == "unknown":
        p.event("other", {})
    else:
        p.event("start", dict(generation=0))
        if problem == "overlap":
            p.event("start", dict(generation=0))
        p.event("stop", dict(generation=1 if problem == "generation" else 0))
    assert p.errors


def test_cross_phase_and_foreign_thread_stay_explicit():
    p = GCPhases(counter([0, 10]))
    p.owner = -1
    p.current, p.epoch = "process", 3
    p.event("start", dict(generation=0))
    p.current, p.epoch = "serialize", 4
    p.event("stop", dict(generation=0))
    assert p.events[0]["crossed_boundary"] and not p.events[0]["owner_thread"]


def test_owned_callback_cleanup_preserves_other_observers():
    original = list(gc.callbacks)
    other = lambda phase, info: None
    gc.callbacks.append(other)
    p = GCPhases()
    try:
        p.install()
        assert gc.callbacks[-1] is p._callback
        with pytest.raises(RuntimeError):
            p.install()
        p.close()
        assert gc.callbacks == original + [other]
        with pytest.raises(RuntimeError):
            p.close()
    finally:
        if p._callback is not None:
            p.close()
        gc.callbacks.remove(other)
    assert gc.callbacks == original


@pytest.mark.parametrize("name", ["process", "pf", "snapshot", "serialize", "barrier", "outside"])
def test_actual_gc_recording_without_enable_or_threshold_change(name):
    p = GCPhases()
    policy = gc.isenabled(), gc.get_threshold()
    p.install()
    try:
        with p.phase(name):
            gc.collect(0)
    finally:
        p.close()
    assert not p.errors and p.events
    assert any(e["phase"] == name and e["generation"] == 0 for e in p.events)
    assert policy == (gc.isenabled(), gc.get_threshold())


def test_body_forwarding_after_hook_outside_clocks():
    value, token, order = [], object(), []
    def after(sample):
        assert not call.active and call.phases.current == "outside"
        assert sample["completed"] and sample["thread_cycles"] == 50
        order.append("snapshot")
    call = monitor(after=after)
    def original(v, *, result):
        assert call.active and call.phases.current == "process"
        v.append(result)
        order.append("body")
        return result
    assert call.call(2, original, value, result=token) is token
    assert value == [token] and order == ["body", "snapshot"]
    assert call.records == [dict(epoch=0, phase_before=2, wall_ns=100, process_cpu_ns=70, thread_cpu_ns=60, completed=True, thread_cycles=50, GC_observed=True)]


def test_original_exception_identity_and_no_snapshot_on_failure():
    after, value, error = [], [], ValueError("literal")
    call = monitor(after=after.append)
    def original():
        value.append(1)
        raise error
    with pytest.raises(ValueError) as caught:
        call.call(1, original)
    assert caught.value is error and value == [1] and after == []
    assert not call.active and not call.records[0]["completed"]
    assert call.records[0]["thread_cycles"] == 50 and call.phases.current == "outside"


def test_unobserved_control_never_reads_cycles_or_changes_phase():
    def forbidden():
        raise AssertionError("cycle probe must be disabled")
    call = monitor(observed=False)
    call.cycles = forbidden
    assert call.call(1, lambda: call.phases.current) == "outside"
    assert call.records[0]["thread_cycles"] is None and not call.records[0]["GC_observed"]


def test_cycle_backwards_is_rejected_and_clock_state_recovers():
    call = monitor()
    call.cycles = counter([10, 5])
    with pytest.raises(ValueError, match="backwards"):
        call.call(1, lambda: None)
    assert not call.active and not call.records[0]["completed"]


def test_windows_cycle_clock_monotonic_integer_samples():
    read = thread_cycle_clock()
    a, b = read(), read()
    assert type(a) is type(b) is int and 0 <= a <= b


def test_barrier_io_and_after_hook_are_outside_body_clock():
    trace = []
    call = None
    class Incoming:
        def __init__(self):
            self.commands = iter(["go 0\n", "advance 0\n"])
        def readline(self):
            assert not call.active
            trace.append("command")
            return next(self.commands)
    class Outgoing(io.StringIO):
        def write(self, text):
            assert not call.active
            trace.append(json.loads(text[len(PREFIX):])["event"])
            return super().write(text)
    def after(sample):
        assert not call.active and call.phases.current == "barrier"
        trace.append("snapshot")
    call = PhaseLockstepClocks(counter([0, 100]), counter([0, 80]), counter([0, 70]), phases=GCPhases(), cycles=counter([0, 50]), after=after, label="A", incoming=Incoming(), outgoing=Outgoing())
    def body():
        assert call.active and call.phases.current == "process"
        trace.append("body")
        return 7
    assert call.call(2, body) == 7
    assert trace == ["ready", "command", "body", "snapshot", "done", "command"]
    assert call.phases.current == "outside" and call.phases.epoch == 0


def test_real_independent_children_keep_extended_samples():
    endpoints, logs = {}, {}
    program = "import sys;sys.path.insert(0," + repr(str(ROOT / "benchmarks")) + ");from rbpf_gc_clocks import PhaseLockstepClocks,GCPhases;p=GCPhases();p.install();c=PhaseLockstepClocks(phases=p,label=sys.argv[1]);[c.call(1 if e<2 else 2,lambda:sum(range(20))) for e in range(4)];p.close();print('complete')"
    try:
        for label in LABELS:
            logs[label] = io.StringIO()
            child = subprocess.Popen([sys.executable, "-B", "-u", "-c", program, label], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf8", bufsize=1)
            endpoints[label] = PipeEndpoint(child, logs[label], timeout=30)
        rows = coordinate(endpoints, 4)
        assert all(e.finish() == 0 for e in endpoints.values())
        assert len(rows) == 4
        assert all(row["samples"][label]["thread_cycles"] >= 0 and row["samples"][label]["GC_observed"] for row in rows for label in LABELS)
    finally:
        for endpoint in endpoints.values():
            endpoint.close()
    assert all("complete" in log.getvalue() for log in logs.values())
