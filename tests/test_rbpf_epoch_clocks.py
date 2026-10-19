"""Clock accounting and literal estimator forwarding negative controls."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from rbpf_epoch_clocks import EpochClocks, summarize  # noqa: E402


def counter(values):
    iterator = iter(values)
    return lambda: next(iterator)


def sample(epoch=0, phase=1, wall=100, process=70, thread=50, completed=True):
    return dict(
        epoch=epoch,
        phase_before=phase,
        wall_ns=wall,
        process_cpu_ns=process,
        thread_cpu_ns=thread,
        completed=completed,
    )


def test_clock_order_args_result_and_mutation():
    trace = []

    def clock(name, values):
        def read():
            trace.append(name)
            return values.pop(0)

        return read

    monitor = EpochClocks(
        clock("wall", [100, 900]),
        clock("process", [200, 500]),
        clock("thread", [40, 240]),
    )
    payload, result = [], object()

    def compute(value, *, token):
        trace.append("estimator")
        assert value is payload and token is result
        value.append(token)
        return token

    assert monitor.call(2, compute, payload, token=result) is result
    assert payload == [result]
    assert trace == [
        "wall",
        "process",
        "thread",
        "estimator",
        "thread",
        "process",
        "wall",
    ]
    assert monitor.records == [sample(phase=2, wall=800, process=300, thread=200)]


def test_exception_partial_state_and_monitor_recovery():
    monitor = EpochClocks(
        counter([0, 10, 20, 30]), counter([0, 10, 20, 30]), counter([0, 10, 20, 30])
    )
    error, value = ValueError("original"), []

    def compute():
        value.append(1)
        raise error

    with pytest.raises(ValueError) as caught:
        monitor.call(1, compute)
    assert caught.value is error and value == [1] and not monitor.active
    assert monitor.call(2, lambda: value) is value
    assert [v["completed"] for v in monitor.records] == [False, True]
    with pytest.raises(ValueError, match="did not complete"):
        summarize(monitor.records)


def test_subclass_keeps_super_closure_and_original_methods():
    class Parent:
        def process(self, value):
            self.phase = 2
            self.state.append(value)
            return self.state

    class Original(Parent):
        def __init__(self):
            self.phase, self.state = 1, []

        def process(self, value):
            return super().process(value)

    function, code, globals_before = (
        Original.process,
        Original.process.__code__,
        dict(Original.process.__globals__),
    )
    monitor = EpochClocks(counter([0, 10]), counter([0, 7]), counter([0, 5]))
    chosen = monitor.instrument_type(Original)
    value = chosen()
    token = object()
    assert (
        value.process(token) is value.state
        and value.state == [token]
        and value.phase == 2
    )
    assert monitor.records == [sample(wall=10, process=7, thread=5)]
    assert Original.process is function and function.__code__ is code
    assert Original.process.__globals__ == globals_before


def test_zero_ticks_aggregate_before_division():
    rows = [sample(0, 1, 10, 0, 0), sample(1, 2, 10, 16, 16), sample(2, 2, 10, 0, 0)]
    result = summarize(rows)
    assert result["all"]["thread_cpu_over_wall"] == 16 / 30
    assert result["2"]["thread_cpu_over_wall"] == 0.8
    assert result["1"]["thread_cpu_over_wall"] == 0
    assert result["all"]["observed_positive_CPU_delta_gcd_ns"] == dict(
        process_cpu_ns=16, thread_cpu_ns=16
    )
    assert result["all"]["zero_CPU_epochs"] == dict(process_cpu_ns=2, thread_cpu_ns=2)


def test_parallel_cpu_and_signed_clock_quantization_not_clipped():
    result = summarize([sample(wall=10, process=30, thread=11)])["all"]
    assert (
        result["process_cpu_over_wall"] == 3 and result["thread_cpu_over_wall"] == 1.1
    )
    assert (
        result["wall_minus_process_cpu_ns"] == -20
        and result["wall_minus_thread_cpu_ns"] == -1
    )
    assert result["process_minus_thread_cpu_ns"] == 19


def test_zero_wall_is_not_divided():
    assert summarize([sample(wall=0)])["all"]["thread_cpu_over_wall"] is None


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [sample(1)],
        [sample(0), sample(0)],
        [sample(0), sample(2)],
        [sample(completed=False)],
        [sample(wall=-1)],
        [sample(process=1.0)],
        [sample(phase=3)],
        [sample(phase=True)],
    ],
)
def test_reject_invalid_raw_records(rows):
    with pytest.raises(ValueError):
        summarize(rows)


def test_recursive_calls_rejected_and_original_error_recorded():
    monitor = EpochClocks(counter([0, 10]), counter([0, 7]), counter([0, 5]))
    with pytest.raises(RuntimeError, match="nonrecursive"):
        monitor.call(1, lambda: monitor.call(1, lambda: None))
    assert (
        not monitor.active
        and len(monitor.records) == 1
        and not monitor.records[0]["completed"]
    )


def test_backwards_clock_rejected_and_monitor_recovered():
    monitor = EpochClocks(counter([10, 0]), counter([0, 7]), counter([0, 5]))
    with pytest.raises(ValueError, match="backwards"):
        monitor.call(1, lambda: None)
    assert not monitor.active and not monitor.records


def test_thread_boundary_rejected(monkeypatch):
    identities = iter([1, 2])
    monkeypatch.setattr(
        "rbpf_epoch_clocks.threading.get_ident", lambda: next(identities)
    )
    monitor = EpochClocks(counter([0, 10]), counter([0, 7]), counter([0, 5]))
    with pytest.raises(RuntimeError, match="thread boundary"):
        monitor.call(1, lambda: None)
    assert not monitor.active and not monitor.records
