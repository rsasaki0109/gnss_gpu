"""Original function forwarding, nested partition, closure and AST checks."""

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
import bench_rbpf_process_profile_field as field  # noqa: E402
from bench_rbpf_rtk_work_profile_field import build_main  # noqa: E402
import rbpf_call_profile as base  # noqa: E402
from rbpf_rtk_work_profile import RtkWorkMonitor  # noqa: E402


events = []


def build_pim(value):
    events.append(("imu", value))
    return value + 1


def satposs(value, *, step=2):
    events.append(("sat", value))
    return value * step


class _Original:
    phase = 2

    def process(self, value):
        return self._process_phase2(value)

    def _process_phase2(self, value):
        return self.prepare_double_difference_measurements(build_pim(value))

    def prepare_double_difference_measurements(self, value):
        return satposs(value, step=3)


class _Derived(_Original):
    def _process_phase2(self, value):
        if value == 99:
            raise ValueError("original error")
        return super()._process_phase2(build_pim(value))


@pytest.mark.parametrize("cls", [_Original, _Derived])
@pytest.mark.parametrize("value", [0, -7, 99])
def test_order_output_error_and_partition(cls, value):
    original_methods, original_calls = base._METHODS, dict(base._C_CALLS)
    monitor = RtkWorkMonitor()
    instrumented = monitor.instrument_type(cls)
    events.clear()
    try:
        expected = cls().process(value)
    except ValueError:
        expected = None
    before = list(events)
    events.clear()
    if cls is _Derived and value == 99:
        with pytest.raises(ValueError, match="original error"):
            instrumented().process(value)
    else:
        assert instrumented().process(value) == expected
    assert events == before
    assert base._METHODS is original_methods and base._C_CALLS == original_calls
    assert monitor.current is None and len(monitor.records) == 1
    spans = monitor.records[0]["spans"]
    for key in ("wall_ns", "cpu_ns"):
        assert (
            sum(v["exclusive_" + key] for v in spans.values())
            == spans["rtk.process"][key]
        )
    assert all(p["restored_AST_equal"] for p in monitor.proofs)
    assert sys.getprofile() is None
    if expected is not None:
        assert spans["rtk.satposs"]["calls"] == 1
        assert spans["rtk.build_pim"]["calls"] == 1


def test_parent_main_and_module_globals_unchanged():
    before = field.main
    run, proof = build_main()
    assert field.main is before and run is not before
    assert proof["restored_parent_main_AST_equal"]
    assert field.ProcessMonitor is base.ProcessMonitor
