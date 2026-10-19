"""Validate nested clock recovery,raw integer sums,and diagnostic boundaries."""

from copy import deepcopy
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
from rbpf_rtk_saved_partition import ARMS, ORDERS, analyze, paired, partition, recover_ns, source_boundaries  # noqa: E402


def fixture(factor=1):
    rows = [dict(epoch=i, phase_before=1 if i < 200 else 2, wall_ns=10000 * factor,
        process_cpu_ns=9000 * factor, thread_cpu_ns=8000 * factor, completed=True) for i in range(400)]
    totals = {p: {k: sum(r[k] for r in rows if p == "all" or r["phase_before"] == int(p))
        for k in ("wall_ns", "process_cpu_ns", "thread_cpu_ns")} for p in ("all", "1", "2")}
    return dict(records=rows, summary=totals, driver_wall_ns=5_000_000 * factor, PF_calls=200), dict(
        calls=200, call_seconds=[1000 * factor / 1e9] * 200)


def cases():
    schedule = [dict(warm=True, round=-1, variant=v) for v in ARMS]
    schedule += [dict(warm=False, round=r, variant=v) for r, order in enumerate(ORDERS) for v in order]
    return [dict(case=c, performance=fixture()[0], record=fixture()[1]) for c in schedule]


def test_source_boundaries_from_frozen_estimator_and_caller():
    result = source_boundaries((ROOT / "benchmarks/bench_rbpf_move_field.py").read_bytes(),
        (ROOT / "experiments/rbpf_fgo/rbpf_fgo.py").read_bytes())
    assert result["original_timer_prefix_AST_exact"] and result["PF_dispatch_guarded_to_phase2"]
    assert not result["per_epoch_pairing_inferred"]


@pytest.mark.parametrize("kind", ("timer", "call", "append", "guard", "extra"))
def test_changed_source_boundary_is_rejected(kind):
    field = (ROOT / "benchmarks/bench_rbpf_move_field.py").read_text(encoding="utf8")
    model = (ROOT / "experiments/rbpf_fgo/rbpf_fgo.py").read_text(encoding="utf8")
    if kind == "timer":
        field = field.replace("/ 1e9", "/ 1e6")
    elif kind == "call":
        field = field.replace("output = selected(rtk, ctx)", "output = other(rtk, ctx)")
    elif kind == "append":
        field = field.replace("calls.append(elapsed)", "calls.append(elapsed + 1)")
    elif kind == "guard":
        model = model.replace("getattr(self, 'phase', 1) != 2", "getattr(self, 'phase', 1) != 1")
    else:
        model = model.replace("self._rbpf_last = self.rbpf.step(self, self._rbpf_ctx)",
            "self.rbpf.step(self, self._rbpf_ctx)\n        self._rbpf_last = self.rbpf.step(self, self._rbpf_ctx)")
    with pytest.raises(ValueError):
        source_boundaries(field, model)


def test_original_integer_ns_recovered_without_rounding_away_measurements():
    samples = [0, 1, 9, 999999999, 1000000000, 1000000001, 3600 * 10**9 - 1, 3600 * 10**9]
    samples += [(i * 9999999967) % (3600 * 10**9) for i in range(4096)]
    for n in samples:
        assert recover_ns(n / 1e9) == n


@pytest.mark.parametrize("value", (1, True, None, float("nan"), float("inf"), -1e-9, -0.0, 3600.000000001, 0.5e-9))
def test_invalid_or_noncanonical_clock_seconds_rejected(value):
    with pytest.raises(ValueError):
        recover_ns(value)


def test_nested_partition_conserves_every_original_clock():
    performance, record = fixture()
    result = partition(performance, record)
    assert result["PF_wall_ns"] == 200000 and result["PF_calls"] == 200
    assert result["phase2_outside_PF_wall_ns"] + result["PF_wall_ns"] == result["RTK"]["2"]["wall_ns"]
    assert result["driver_outside_RTK_wall_ns"] + result["RTK"]["all"]["wall_ns"] == result["driver_wall_ns"]
    assert result["PF_fraction_of_phase2_wall"] == pytest.approx(.1)
    assert result["PF_call_ns"] == [1000] * 200


def test_CPU_clock_quantization_keeps_signed_residuals():
    performance, record = fixture()
    for r in performance["records"]:
        r["process_cpu_ns"] = 20000
        r["thread_cpu_ns"] = 15000
    for p in ("all", "1", "2"):
        n = 400 if p == "all" else 200
        performance["summary"][p]["process_cpu_ns"] = 20000 * n
        performance["summary"][p]["thread_cpu_ns"] = 15000 * n
    result = partition(performance, record)
    assert result["wall_minus_thread_cpu_ns"]["all"] == -2000000
    assert result["process_minus_thread_cpu_ns"]["all"] == 2000000


@pytest.mark.parametrize("kind", ("drop_epoch", "phase", "failed", "negative", "fraction", "summary", "PF_count", "PF_seconds", "nesting", "driver"))
def test_incomplete_or_invalid_clock_records_fail_closed(kind):
    p, r = fixture()
    if kind == "drop_epoch":
        p["records"].pop()
    elif kind == "phase":
        p["records"][0]["phase_before"] = 2
    elif kind == "failed":
        p["records"][0]["completed"] = False
    elif kind == "negative":
        p["records"][0]["wall_ns"] = -1
    elif kind == "fraction":
        p["records"][0]["wall_ns"] = 1.5
    elif kind == "summary":
        p["summary"]["2"]["wall_ns"] += 1
    elif kind == "PF_count":
        r["calls"] = 199
    elif kind == "PF_seconds":
        r["call_seconds"].pop()
    elif kind == "nesting":
        r["call_seconds"] = [1.0] * 200
    else:
        p["driver_wall_ns"] = 1
    with pytest.raises(ValueError):
        partition(p, r)


def test_complete_schedule_keeps_warm_and_primary_data_without_adoption():
    values = cases()
    for c in values:
        if c["case"]["warm"]:
            c["performance"], c["record"] = fixture(100)
    before = deepcopy(values)
    result = analyze(values)
    assert values == before and len(result["records"]) == 24
    assert len(result["series"]) == len(result["contrasts"]) == 13
    assert all(v["median_percent"] == 0 for contrasts in result["contrasts"].values() for v in contrasts.values())
    assert result["new_estimator_runs"] == result["new_GPU_launches"] == result["new_timings"] == 0
    assert result["diagnostic_only"] and not result["previous_speed_gate_recomputed_or_replaced"]


@pytest.mark.parametrize("kind", ("missing", "reorder", "duplicate"))
def test_no_prefix_can_be_omitted_or_relabelled(kind):
    values = cases()
    if kind == "missing":
        values.pop()
    elif kind == "reorder":
        values[-1], values[-2] = values[-2], values[-1]
    else:
        values[-1] = deepcopy(values[-2])
    with pytest.raises(ValueError, match="scheduled"):
        analyze(values)


def test_absolute_diagnostic_differences_do_not_cancel():
    result = paired([100] * 5, [80, 120, 100, 80, 120])
    assert result["median_percent"] == 0
    assert result["absolute_median_percent"] == 20 and result["absolute_rounds_under5"] == 1
    with pytest.raises(ValueError):
        paired([1, 2, 3, 4], [1] * 5)
