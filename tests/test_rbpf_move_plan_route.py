"""Gate regressions and common clock/return instrumentation for real RTK."""

from copy import deepcopy
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
from rbpf_move_plan_route import (  # noqa: E402
    ORDERS, VARIANTS, RouteClocks, evaluate, metric_totals, retaining_type, schedule,
)


def performance(factor=1):
    return dict(records=[dict(epoch=i, phase_before=1 if i < 200 else 2,
        wall_ns=int(1000 * factor), process_cpu_ns=0, thread_cpu_ns=0, completed=True)
        for i in range(400)], driver_wall_ns=int(500000 * factor))


def cases():
    return [dict(case=c, performance=performance(0.8 if c["variant"] == "candidate" else 1))
        for c in schedule()]


def test_schedule_preserves_all_four_warm_and_twenty_timed_prefixes():
    order = schedule()
    assert len(order) == 24 and sum(c["warm"] for c in order) == 4
    assert all(set(row) == set(VARIANTS) for row in ORDERS)
    for position in range(4):
        assert set(row[position] for row in ORDERS[:4]) == set(VARIANTS)


def test_all_three_metrics_and_both_controls_are_required():
    result = evaluate(cases())
    assert result["speed_gate"] and set(result["metrics"]) == {"all", "2", "driver"}
    assert all(m["contrasts"]["reference"]["median_percent"] == pytest.approx(20)
        for m in result["metrics"].values())


def test_warm_startup_is_retained_but_never_used_to_pass_a_gate():
    rows = cases()
    for row in rows:
        if row["case"]["warm"]:
            row["performance"] = performance(0.001 if row["case"]["variant"] == "reference" else 100)
    assert evaluate(rows)["speed_gate"]


def test_AA_absolute_values_cannot_cancel_signed_instability():
    rows = cases()
    for row in rows:
        c = row["case"]
        if not c["warm"] and c["variant"] == "AA":
            row["performance"] = performance((0.8, 1.2, 1, 0.8, 1.2)[c["round"]])
    result = evaluate(rows)
    assert result["metrics"]["all"]["contrasts"]["AA"]["median_percent"] == 0
    assert not result["speed_gate"]


def test_three_stable_AA_rounds_do_not_suffice():
    rows = cases()
    for row in rows:
        c = row["case"]
        if c["variant"] == "AA" and c["round"] >= 3:
            row["performance"] = performance(1.1)
    result = evaluate(rows)
    assert result["metrics"]["all"]["contrasts"]["AA"]["absolute_median_percent"] == 0
    assert not result["speed_gate"]


def test_same_adapter_off_control_must_also_improve():
    rows = cases()
    for row in rows:
        if row["case"]["variant"] == "off":
            row["performance"] = performance(0.8)
    assert not evaluate(rows)["speed_gate"]


def test_process_gain_cannot_substitute_for_driver_gain():
    rows = cases()
    for row in rows:
        if row["case"]["variant"] == "candidate":
            row["performance"]["driver_wall_ns"] = 500000
    result = evaluate(rows)
    assert result["metrics"]["all"]["pass_gate"] and not result["metrics"]["driver"]["pass_gate"]
    assert not result["speed_gate"]


def test_four_positive_rounds_are_required_even_if_median_gain_is_large():
    rows = cases()
    for row in rows:
        if row["case"]["variant"] == "candidate" and row["case"]["round"] >= 3:
            row["performance"] = performance(1.1)
    assert not evaluate(rows)["speed_gate"]


@pytest.mark.parametrize("mutation", ("missing", "duplicate", "reorder"))
def test_schedule_cannot_be_filtered_or_relabelled(mutation):
    rows = cases()
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows[-1] = deepcopy(rows[-2])
    else:
        rows[-1], rows[-2] = rows[-2], rows[-1]
    with pytest.raises(ValueError, match="scheduled prefix"):
        evaluate(rows)


@pytest.mark.parametrize("mutation", ("dropped", "phase", "fraction", "negative", "failed", "driver"))
def test_raw_clocks_reject_invalid_or_omitted_epochs(mutation):
    value = performance()
    if mutation == "dropped":
        value["records"].pop()
    elif mutation == "phase":
        value["records"][0]["phase_before"] = 2
    elif mutation == "fraction":
        value["records"][0]["wall_ns"] = 0.5
    elif mutation == "negative":
        value["records"][0]["wall_ns"] = -1
    elif mutation == "failed":
        value["records"][0]["completed"] = False
    else:
        value["driver_wall_ns"] = 1
    with pytest.raises(ValueError):
        metric_totals(value)


def test_complete_function_return_exception_and_clock_boundaries_are_preserved():
    events = []
    def wall():
        events.append("wall")
        return len(events)
    monitor = RouteClocks(wall=wall, process=lambda: 0, thread=lambda: 0)
    token = object()
    def original():
        events.append("body")
        return token
    assert monitor.call(1, original) is token
    assert events == ["wall", "body", "wall"]
    error = RuntimeError("same")
    def fail():
        raise error
    with pytest.raises(RuntimeError) as caught:
        monitor.call(2, fail)
    assert caught.value is error and not monitor.active
    assert monitor.records[-1]["completed"] is False


def test_latest_PF_references_and_exception_do_not_copy_or_reinvoke():
    class Original:
        def __init__(self):
            self.calls = []
        def __call__(self, rtk, ctx):
            self.calls.append((rtk, ctx))
            if isinstance(ctx, Exception):
                raise ctx
            return ctx
    step = retaining_type(Original)()
    rtk, first, second = object(), object(), object()
    assert step(rtk, first) is first and step(rtk, second) is second
    assert step.route_calls == 2 and step.route_last[0] is rtk and step.route_last[1] is second
    error = ValueError("original")
    with pytest.raises(ValueError) as caught:
        step(rtk, error)
    assert caught.value is error and step.route_calls == 2 and len(step.calls) == 3
