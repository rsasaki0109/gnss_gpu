"""Ownership,unaltered driver forwarding and predeclared A/A gate controls."""

from copy import deepcopy
import hashlib
import inspect
from itertools import count
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
from rbpf_epoch_clocks import EpochClocks  # noqa: E402
from rbpf_inprocess_pair import LABELS, Offerer, Owner, PairError, drive, evaluate, paired_driver  # noqa: E402


def clock():
    return EpochClocks(*(count(step=10).__next__ for _ in range(3)))


def setup_streams(epochs=4, pair=0, fail=None, after=None):
    owner = Owner()
    offers = {label: Offerer(owner, label) for label in LABELS}
    clocks = {label: clock() for label in LABELS}
    trace, calls, closed = [], [], []

    def stream(label):
        try:
            for epoch in range(epochs):
                assert not owner.active and not any(c.active for c in clocks.values())
                trace.append(("prepare", label, epoch))

                def process(epoch_=epoch):
                    assert owner.active and clocks[label].active
                    calls.append((label, epoch_))
                    if fail == (label, epoch_):
                        raise LookupError("original failure")
                    return (label, epoch_)

                ticket = offers[label].make(epoch, 1 if epoch < epochs // 2 else 2, process)
                yield ticket
                ticket.require_completed()
                assert ticket.function is ticket.context is ticket.result is None
                trace.append(("tail", label, epoch))
            return 0
        finally:
            closed.append(label)

    streams = {label: stream(label) for label in LABELS}
    return owner, offers, clocks, trace, calls, closed, streams


@pytest.mark.parametrize("pair", [0, 1])
def test_prepare_both_then_alternating_calls_and_tail(pair):
    owner, offers, clocks, trace, calls, closed, streams = setup_streams()
    captures = []

    def capture(epoch, tickets, monitors):
        assert not owner.active and all(not c.active for c in monitors.values())
        assert all(t.state == "done" for t in tickets.values())
        assert all(len(c.records) == epoch + 1 for c in monitors.values())
        captures.append(epoch)

    result = drive(streams, owner, pair_index=pair, epochs=4, clocks=clocks, after=capture)
    expected = [(label, epoch) for epoch in range(4)
        for label in (LABELS[::-1] if (epoch + pair) % 2 else LABELS)]
    assert calls == expected and captures == list(range(4))
    assert sorted(closed) == sorted(LABELS) and result["completed"] == {"A": 0, "AA": 0}
    assert len(result["events"]) == 32
    for epoch in range(4):
        rows = result["events"][epoch * 8:(epoch + 1) * 8]
        assert [r["event"] for r in rows] == ["ready", "ready", "run", "done", "run", "done", "advance", "advance"]
        assert {r["label"] for r in rows[:2]} == set(LABELS)
    assert all(c.records == [dict(epoch=e, phase_before=1 if e < 2 else 2,
        wall_ns=10, process_cpu_ns=10, thread_cpu_ns=10, completed=True) for e in range(4)] for c in clocks.values())


def test_identity_forwarding_and_reference_release():
    owner = Owner()
    offer = Offerer(owner, "A")
    value, keyword, answer = object(), object(), object()

    def compute(arg, *, token):
        assert arg is value and token is keyword and owner.active
        return answer

    ticket = offer.make(0, 1, compute, (value,), {"token": keyword}, value)
    assert ticket.execute(owner, clock()) is answer and ticket.result is answer
    ticket.release()
    assert ticket.state == "done" and ticket.args is ticket.kwargs is ticket.context is ticket.result is None
    ticket.require_completed()


def test_original_exception_identity_and_cleanup():
    owner, error = Owner(), LookupError("kept")

    def fail():
        raise error

    ticket, monitor = Offerer(owner, "A").make(0, 1, fail), clock()
    with pytest.raises(LookupError) as raised:
        ticket.execute(owner, monitor)
    assert raised.value is error and ticket.state == "failed"
    assert not owner.active and not monitor.active and monitor.records[0]["completed"] is False
    with pytest.raises(PairError):
        ticket.require_completed()


@pytest.mark.parametrize("kind", ["foreign", "active", "clock_active", "clock_epoch", "repeat"])
def test_ticket_ownership_and_recursion_guards(kind):
    owner, monitor = Owner(), clock()
    ticket = Offerer(owner, "A").make(0, 1, lambda: None)
    other = Owner() if kind == "foreign" else owner
    owner.active = kind == "active"
    monitor.active = kind == "clock_active"
    if kind == "clock_epoch":
        monitor.records.append({})
    if kind == "repeat":
        ticket.execute(owner, monitor)
    with pytest.raises(PairError):
        ticket.execute(other, monitor)


@pytest.mark.parametrize("epoch,phase", [(1, 1), (-1, 1), (True, 1), (0, True), (0, 0), (0, 3)])
def test_offer_rejects_invalid_epoch_and_phase(epoch, phase):
    with pytest.raises(PairError):
        Offerer(Owner(), "A").make(epoch, phase, lambda: None)


def test_offer_labels_and_unfinished_input_guard():
    owner = Owner()
    offer = Offerer(owner, "A")
    for label in ("A", "other"):
        with pytest.raises(PairError):
            Offerer(owner, label)
    ticket = offer.make(0, 1, lambda: None)
    with pytest.raises(PairError):
        offer.make(1, 1, lambda: None)
    ticket.execute(owner, clock())
    assert offer.make(1, 2, lambda: None).epoch == 1


def test_failure_closes_both_streams():
    owner, _, clocks, _, calls, closed, streams = setup_streams(fail=("A", 0))
    with pytest.raises(LookupError, match="original failure"):
        drive(streams, owner, pair_index=0, epochs=4, clocks=clocks)
    assert calls == [("A", 0)] and set(closed) == set(LABELS)
    assert not owner.active and not any(c.active for c in clocks.values())


@pytest.mark.parametrize("kind", ["early", "extra", "return", "phase", "foreign"])
def test_invalid_driver_protocol_closes_streams(kind):
    owner, offers, clocks, _, _, closed, streams = setup_streams()
    original = streams["AA"]

    def changed():
        try:
            if kind == "early":
                return 0
            for epoch in range(5 if kind == "extra" else 4):
                ticket = offers["AA"].make(epoch, 2 if kind == "phase" else (1 if epoch < 2 else 2), lambda: None)
                if kind == "foreign":
                    ticket.owner = Owner()
                yield ticket
                ticket.require_completed()
            return 1 if kind == "return" else 0
        finally:
            closed.append("AA")

    original.close()
    streams["AA"] = changed()
    with pytest.raises(PairError):
        drive(streams, owner, pair_index=0, epochs=4, clocks=clocks)
    assert set(closed) == set(LABELS)


def test_shared_generators_and_clocks_rejected():
    owner, _, clocks, _, _, _, streams = setup_streams()
    try:
        with pytest.raises(PairError):
            drive({"A": streams["A"], "AA": streams["A"]}, owner, pair_index=0, epochs=4)
        with pytest.raises(PairError):
            drive(streams, owner, pair_index=0, epochs=4, clocks={"A": clocks["A"], "AA": clocks["A"]})
    finally:
        for stream in streams.values():
            stream.close()


SOURCE = b'''def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    args = ap.parse_args()
    from rbpf_fgo import RbpfGtsamRtkTc
    rtk = RbpfGtsamRtkTc(args.label)
    obsb = object()
    for ne in range(4):
        obs = object()
        rtk.process(obs, obsb=obsb)
    saved.append(rtk)
    return 0
'''


def build(source=SOURCE, **kwargs):
    import argparse
    return paired_driver(source, {"argparse": argparse, "saved": []}, lambda cls: cls,
        lambda *args: None, ["driver", "--label", "A"], expected_sha256=hashlib.sha256(source).hexdigest(),
        filename="synthetic_driver.py", **kwargs)


def test_original_parser_and_body_forwarding(monkeypatch):
    import argparse
    saved, instances = [], []

    class RTK:
        def __init__(self, label):
            self.label, self.phase, self.inputs = label, 1, []
            instances.append(self)

        def process(self, obs, *, obsb):
            self.inputs.append((obs, obsb))
            self.phase = 2 if len(self.inputs) >= 2 else 1

    monkeypatch.setitem(sys.modules, "rbpf_fgo", SimpleNamespace(RbpfGtsamRtkTc=RTK))
    owner = Owner()
    offers = {label: Offerer(owner, label) for label in LABELS}
    streams = {}
    for label in LABELS:
        def offer(epoch, rtk, obs, obsb, label_=label):
            return offers[label_].make(epoch, rtk.phase, rtk.process, (obs,), {"obsb": obsb})
        function, proof = paired_driver(SOURCE, {"argparse": argparse, "saved": saved}, lambda cls: cls,
            offer, ["driver", "--label", label], expected_sha256=hashlib.sha256(SOURCE).hexdigest(), filename="synthetic.py")
        assert inspect.isgeneratorfunction(function) and all(v is True for k, v in proof.items() if k != "original_driver_sha256")
        streams[label] = function()
    result = drive(streams, owner, pair_index=0, epochs=4)
    assert result["completed"] == {"A": 0, "AA": 0}
    assert {rtk.label for rtk in saved} == set(LABELS) and len(saved) == len(instances) == 2
    assert all(len(rtk.inputs) == 4 and len({id(a) for a, _ in rtk.inputs}) == 4
        and len({id(b) for _, b in rtk.inputs}) == 1 for rtk in saved)
    assert saved[0].inputs[0][1] is not saved[1].inputs[0][1]


def test_frozen_actual_driver_AST_restoration():
    import json
    path = ROOT / "experiments/rbpf_fgo/wp16_run_rbpf.py"
    plan = json.loads((ROOT / "dist/rbpf_discovery_pipeline_field_20261018/plan.json").read_text(encoding="utf8"))
    function, proof = paired_driver(path.read_bytes(), {}, lambda cls: cls, lambda *args: None,
        [str(path), "3"], expected_sha256=plan["source_sha256"][path.relative_to(ROOT).as_posix()], filename=str(path))
    assert inspect.isgeneratorfunction(function) and proof["original_driver_AST_restored_exact"]


@pytest.mark.parametrize("change", [
    (b"ap.parse_args()", b"ap.parse_args([])"),
    (b"rtk.process(obs, obsb=obsb)", b"rtk.process(obs, obsb=None)"),
    (b"rtk.process(obs, obsb=obsb)", b"pass"),
    (b"rtk.process(obs, obsb=obsb)", b"rtk.process(obs, obsb=obsb)\n        rtk.process(obs, obsb=obsb)"),
    (b"from rbpf_fgo import RbpfGtsamRtkTc", b"from rbpf_fgo import Other"),
    (b"saved.append(rtk)", b"_pair_ticket = None"),
])
def test_changed_driver_shape_rejected(change):
    with pytest.raises(ValueError):
        build(SOURCE.replace(*change))


def test_driver_hash_rejected_before_compile():
    with pytest.raises(ValueError, match="Frozen driver changed"):
        paired_driver(SOURCE, {}, None, None, [], expected_sha256="0" * 64, filename="unused")


def pairs():
    def rows():
        return [dict(epoch=e, phase_before=1 if e < 200 else 2, wall_ns=100,
            process_cpu_ns=80, thread_cpu_ns=60, completed=True) for e in range(400)]
    return [dict(pair=i, warm=i == 0, records={label: rows() for label in LABELS}) for i in range(6)]


def test_exact_gate_warm_retained_and_not_used():
    data = pairs()
    data[0]["records"]["AA"][0]["wall_ns"] = 10**9
    result = evaluate(data)
    assert result["calibration_pass"] and all(v["paired_percent"] == [0.0] * 5 for v in result["metrics"].values())
    assert not result["optimization_speedup_established"] and not result["whole_continuous_RTK_throughput_established"]


def test_signed_cancellation_cannot_pass():
    data = pairs()
    for pair, difference in zip(data[1:], [-10, 10, -10, 10, 0], strict=True):
        for row in pair["records"]["AA"]:
            row["wall_ns"] = 100 - difference
    assert not evaluate(data)["calibration_pass"]


def test_four_of_five_stable_required_in_every_phase():
    data = pairs()
    for p in data[1:3]:
        for row in p["records"]["AA"][:200]:
            row["wall_ns"] = 110
    result = evaluate(data)
    assert result["metrics"]["1"]["absolute_median_percent"] == 0
    assert result["metrics"]["1"]["absolute_rounds_under5"] == 3 and not result["calibration_pass"]


@pytest.mark.parametrize("kind", ["missing_pair", "pair_order", "warm", "missing_epoch", "epoch_order", "phase", "incomplete", "negative", "float", "zero"])
def test_incomplete_and_invalid_calibration_rejected(kind):
    data = pairs()
    rows = data[1]["records"]["A"]
    if kind == "missing_pair":
        data.pop()
    elif kind == "pair_order":
        data.reverse()
    elif kind == "warm":
        data[1]["warm"] = True
    elif kind == "missing_epoch":
        rows.pop()
    elif kind == "epoch_order":
        rows[0]["epoch"] = 1
    elif kind == "phase":
        rows[0]["phase_before"] = 2
    elif kind == "incomplete":
        rows[0]["completed"] = False
    elif kind in ("negative", "float"):
        rows[0]["wall_ns"] = -1 if kind == "negative" else 1.0
    else:
        for row in rows:
            row["wall_ns"] = 0
    with pytest.raises((PairError, ValueError)):
        evaluate(data)
