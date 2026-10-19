"""Cross-child barriers, exact forwarding, and bounded protocol failures."""

from copy import deepcopy
import io
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
from rbpf_epoch_lockstep import (  # noqa: E402
    LABELS,
    PREFIX,
    LockstepClocks,
    PipeEndpoint,
    ProtocolError,
    coordinate,
    validate_event,
)


def counter(values):
    iterator = iter(values)
    return lambda: next(iterator)


def event(kind, label="A", epoch=0, phase=1):
    data = dict(event=kind, label=label, epoch=epoch, phase=phase)
    if kind == "done":
        data["sample"] = dict(
            epoch=epoch,
            phase_before=phase,
            completed=True,
            wall_ns=10,
            process_cpu_ns=7,
            thread_cpu_ns=5,
        )
    return data


def monitor(commands="go 0\nadvance 0\n", *, outgoing=None):
    output = io.StringIO() if outgoing is None else outgoing
    return LockstepClocks(
        counter([0, 10]),
        counter([0, 7]),
        counter([0, 5]),
        incoming=io.StringIO(commands),
        outgoing=output,
        label="A",
    ), output


def test_all_protocol_IO_is_outside_original_clock_interval():
    trace = []

    class Incoming:
        def readline(self):
            assert not call.active
            trace.append("command")
            return next(commands)

    class Outgoing(io.StringIO):
        def write(self, value):
            assert not call.active
            trace.append(json.loads(value[len(PREFIX) :])["event"])
            return super().write(value)

    commands = iter(["go 0\n", "advance 0\n"])
    call, _ = monitor(outgoing=Outgoing())
    call.incoming = Incoming()
    payload, token = [], object()

    def literal(value, *, result):
        assert call.active
        value.append(result)
        trace.append("compute")
        return result

    assert call.call(1, literal, payload, result=token) is token
    assert payload == [token] and trace == [
        "ready",
        "command",
        "compute",
        "done",
        "command",
    ]
    assert call.records[0] == event("done")["sample"] and not call.waiting


def test_original_exception_identity_and_partial_state():
    call, output = monitor("go 0\n")
    value, error = [], ValueError("original")

    def literal():
        value.append(1)
        raise error

    with pytest.raises(ValueError) as caught:
        call.call(1, literal)
    assert caught.value is error and value == [1]
    assert not call.waiting and not call.active and not call.records[0]["completed"]
    messages = [json.loads(v[len(PREFIX) :]) for v in output.getvalue().splitlines()]
    assert [v["event"] for v in messages] == ["ready", "failed"]


@pytest.mark.parametrize("command", ["", "go 1\n", "advance 0\n", "go 0\ngo 1\n"])
def test_invalid_commands_rejected_without_unapproved_call(command):
    call, _ = monitor(command)
    calls = []
    with pytest.raises(ProtocolError):
        call.call(1, lambda: calls.append(1))
    assert len(calls) == (1 if command.startswith("go 0\n") else 0)
    assert not call.waiting and not call.active


def test_reentry_does_not_consume_commands_or_rng():
    call, _ = monitor("go 0\n")
    with pytest.raises(ProtocolError, match="reenter"):
        call.call(1, lambda: call.call(1, lambda: None))
    assert not call.waiting and not call.active and len(call.records) == 1


@pytest.mark.parametrize("label", ["", "B", None])
def test_explicit_child_label_required(label, monkeypatch):
    monkeypatch.delenv("GNSS_GPU_LOCKSTEP_LABEL", raising=False)
    with pytest.raises(ProtocolError):
        LockstepClocks(label=label)


class Endpoint:
    def __init__(self, label, epochs, trace):
        self.label, self.trace = label, trace
        self.events = iter(
            [
                event(kind, label, epoch, 1 if epoch < epochs // 2 else 2)
                for epoch in range(epochs)
                for kind in ("ready", "done")
            ]
        )

    def event(self):
        value = next(self.events)
        self.trace.append((self.label, value["event"], value["epoch"]))
        return value

    def command(self, text):
        command, epoch = text.split()
        self.trace.append((self.label, command, int(epoch)))


@pytest.mark.parametrize("pair_index", [0, 1, 2])
def test_ready_and_advance_barriers_prevent_timing_and_preparation_overlap(pair_index):
    trace, progress = [], []
    endpoints = {label: Endpoint(label, 4, trace) for label in LABELS}
    rows = coordinate(endpoints, 4, pair_index=pair_index, progress=progress.append)
    assert rows == progress and len(rows) == 4
    for epoch, row in enumerate(rows):
        first, second = LABELS if (epoch + pair_index) % 2 == 0 else LABELS[::-1]
        assert row["order"] == [first, second]
        assert trace[epoch * 8 : (epoch + 1) * 8] == [
            ("A", "ready", epoch),
            ("AA", "ready", epoch),
            (first, "go", epoch),
            (first, "done", epoch),
            (second, "go", epoch),
            (second, "done", epoch),
            ("A", "advance", epoch),
            ("AA", "advance", epoch),
        ]
    assert sum(r["order"][0] == "A" for r in rows) == 2


@pytest.mark.parametrize(
    "mutation",
    [
        "kind",
        "label",
        "epoch",
        "phase",
        "negative",
        "float_tick",
        "incomplete",
        "sample_epoch",
        "sample_phase",
    ],
)
def test_bad_events_and_samples_are_rejected(mutation):
    data = event("done")
    if mutation == "kind":
        data["event"] = "failed"
    if mutation == "label":
        data["label"] = "AA"
    if mutation == "epoch":
        data["epoch"] = True
    if mutation == "phase":
        data["phase"] = 3
    if mutation == "negative":
        data["sample"]["wall_ns"] = -1
    if mutation == "float_tick":
        data["sample"]["process_cpu_ns"] = 1.0
    if mutation == "incomplete":
        data["sample"]["completed"] = False
    if mutation == "sample_epoch":
        data["sample"]["epoch"] = 1
    if mutation == "sample_phase":
        data["sample"]["phase_before"] = 2
    with pytest.raises(ProtocolError):
        validate_event(data, "done", "A", 0, 1)


def test_phase_divergence_stops_both_before_any_compute():
    trace = []
    endpoints = {label: Endpoint(label, 2, trace) for label in LABELS}
    endpoints["AA"].events = iter([event("ready", "AA", phase=2)])
    with pytest.raises(ProtocolError, match="diverged"):
        coordinate(endpoints, 2)
    assert all(v[1] == "ready" for v in trace)


def child(program, log):
    process = subprocess.Popen(
        [sys.executable, "-B", "-u", "-c", program],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf8",
        bufsize=1,
    )
    return PipeEndpoint(process, log, timeout=20)


def test_real_independent_processes_complete_serial_protocol():
    endpoints, logs = {}, {}
    program = (
        "import sys;sys.path.insert(0,"
        + repr(str(ROOT / "benchmarks"))
        + ");from rbpf_epoch_lockstep import LockstepClocks;c=LockstepClocks(label=sys.argv[1]);v=[];[c.call(1 if e<2 else 2,lambda:v.append(e)) for e in range(4)];assert v==list(range(4));print('complete')"
    )
    try:
        for label in LABELS:
            logs[label] = io.StringIO()
            endpoints[label] = child(
                program.replace("sys.argv[1]", repr(label)), logs[label]
            )
        rows = coordinate(endpoints, 4)
        assert len(rows) == 4 and all(e.finish() == 0 for e in endpoints.values())
    finally:
        for endpoint in endpoints.values():
            endpoint.close()
    assert all("complete" in log.getvalue() for log in logs.values())


@pytest.mark.parametrize(
    "program,match",
    [
        ("print('ordinary log')", "closed stdout"),
        ("print('" + PREFIX + "bad')", "Malformed"),
    ],
)
def test_child_early_exit_and_bad_json_preserve_logs(program, match):
    log = io.StringIO()
    endpoint = child(program, log)
    try:
        with pytest.raises(ProtocolError, match=match):
            endpoint.event()
    finally:
        endpoint.close()
    assert (
        log.getvalue()
        and endpoint.process.poll() is not None
        and not endpoint.reader.is_alive()
    )


def test_child_timeout_terminates_only_owned_child():
    endpoint = child("import sys;sys.stdin.read()", io.StringIO())
    endpoint.timeout = 0.05
    try:
        with pytest.raises(ProtocolError, match="Timed out"):
            endpoint.event()
    finally:
        endpoint.close()
    assert endpoint.process.poll() is not None and not endpoint.reader.is_alive()


def test_finish_rejects_extra_protocol_event():
    endpoint = child("print('" + PREFIX + "{}')", io.StringIO())
    try:
        with pytest.raises(ProtocolError, match="extra event"):
            endpoint.finish()
    finally:
        endpoint.close()
