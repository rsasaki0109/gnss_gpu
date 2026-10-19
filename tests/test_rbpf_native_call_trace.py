"""Exact host-clock parsing and original/traced native numeric/owner controls."""

from contextlib import ExitStack
import ctypes
from pathlib import Path
import os
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
from rbpf_native_call_trace import NativeCallScorer, parse_trace  # noqa: E402
from rbpf_register_lu import RegisterLUScorer  # noqa: E402
from rbpf_likelihood_cases import make_case  # noqa: E402


def words(ids=tuple(range(8)), frequency=10_000_000):
    rows = [[phase, 1000 + i * 7, 300 + i * 3, 500 + i * 2] for i, phase in enumerate(ids)]
    return [1, 1, frequency, len(rows), 0, 0, *[v for row in rows for v in row]]


@pytest.mark.parametrize("ids", [(0, 7), (0, 1, 2, 7), tuple(range(8))])
def test_partial_CPU_error_and_complete_GPU_paths(ids):
    raw = words(ids)
    trace = parse_trace(np.array(raw, dtype=np.uint64))
    assert trace["raw_words"] == raw and trace["full_GPU_path"] is (len(ids) == 8)
    assert trace["totals"] == dict(wall_ticks=(len(ids)-1)*7, user_100ns=(len(ids)-1)*3,
        kernel_100ns=(len(ids)-1)*2, thread_100ns=(len(ids)-1)*5)
    assert all(sum(row[key] for row in trace["segments"]) == value for key, value in trace["totals"].items())


@pytest.mark.parametrize("frequency", [19, 10_000_000, 1_000_000_000])
def test_raw_ticks_retained_without_lossy_ns_rounding(frequency):
    result = parse_trace(words(frequency=frequency))
    assert result["frequency"] == frequency and result["totals"]["wall_ticks"] == 49


@pytest.mark.parametrize("kind", ["version", "generation", "frequency", "flags", "active", "count", "length", "negative", "bool", "stage_first", "stage_last", "stage_repeat", "stage_range", "wall", "user", "kernel"])
def test_bad_clock_header_or_topology_rejected(kind):
    raw = words()
    header = {"version": (0, 2), "generation": (1, 0), "frequency": (2, 0), "flags": (4, 1),
        "active": (5, 1), "count": (3, 9), "negative": (6, -1), "bool": (0, True)}
    if kind in header:
        index, value = header[kind]
        raw[index] = value
    elif kind == "length":
        raw.pop()
    elif kind.startswith("stage"):
        index, value = {"stage_first": (6, 1), "stage_last": (34, 6), "stage_repeat": (10, 0), "stage_range": (14, 8)}[kind]
        raw[index] = value
    else:
        raw[{"wall": 11, "user": 12, "kernel": 13}[kind]] = 0
    with pytest.raises(ValueError):
        parse_trace(raw)


@pytest.mark.parametrize("raw", [[], [1, 1], np.zeros((2, 19), dtype=np.uint64), np.zeros(38, dtype=np.float64)])
def test_invalid_trace_storage_rejected(raw):
    with pytest.raises(ValueError):
        parse_trace(raw)


@pytest.fixture(scope="module")
def scorers():
    baseline, traced = os.environ.get("GNSS_RBPF_BASELINE_DLL"), os.environ.get("GNSS_RBPF_NATIVE_TRACE_DLL")
    if not baseline or not traced:
        pytest.skip("Explicit corrected baseline/new host-trace DLLs required")
    with ExitStack() as stack:
        yield (stack.enter_context(RegisterLUScorer(baseline)),
            stack.enter_context(NativeCallScorer(traced, enabled=False)),
            stack.enter_context(NativeCallScorer(traced)))


def exact(left, right):
    a, ad = left
    b, bd = right
    assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
    assert ad["unique"] == bd["unique"] and ad["scalar_fallbacks"] == bd["scalar_fallbacks"]
    for key in ("statuses", "diagnostic"):
        assert (key in ad) == (key in bd)
        if key in ad:
            assert ad[key].dtype == bd[key].dtype and ad[key].shape == bd[key].shape and ad[key].tobytes() == bd[key].tobytes()


@pytest.mark.parametrize("seed,rows,groups,cluster", [(20260710, 16, 1, False), (20260711, 64, 4, False),
    (20260712, 128, 8, False), (20260710, 16, 1, True), (20260711, 64, 4, True), (20260712, 128, 8, True)])
def test_numeric_bits_CPU_GPU_and_trace_path(scorers, seed, rows, groups, cluster):
    oracle, problem, hypotheses = make_case(8, rows=rows, groups=groups, seed=seed, cluster=cluster)
    for mode in ("native_cpu", "gpu"):
        expected = scorers[0].score(hypotheses, problem, oracle, mode, diagnostics=True)
        for scorer in scorers[1:]:
            exact(expected, scorer.score(hypotheses, problem, oracle, mode, diagnostics=True))
        trace = scorers[2].last_native_trace
        assert trace is not None and trace["full_GPU_path"] is (mode == "gpu")
        assert scorers[2].native_records[-1]["probe_error"] is None


@pytest.mark.parametrize("cluster", [0, 1])
def test_all_actual_saved_scores_match_corrected_baseline(scorers, cluster):
    from bench_rbpf_irls_replay import prepare
    from rbpf_irls_replay import load_corpus, oracle_from_snapshot
    records = load_corpus(ROOT / f"dist/rbpf_irls_replay_capture_20261016/r3_seed0_capture_cpu_c{cluster}.corpus.jsonl.gz")
    assert len(records) == 197
    oracle = oracle_from_snapshot(records[0]["inputs"], ROOT / os.environ["GNSS_RBPF_CANONICAL_EXT"], ROOT / os.environ["GNSS_RBPF_PARENT_EXT"])
    for entry in records:
        hypotheses, problem = prepare(oracle, entry["inputs"])
        expected = scorers[0].score(hypotheses, problem, oracle, "gpu", diagnostics=True)
        for scorer in scorers[1:]:
            exact(expected, scorer.score(hypotheses, problem, oracle, "gpu", diagnostics=True))
        assert scorers[2].last_native_trace["full_GPU_path"]
        assert scorers[2].native_records[-1]["probe_error"] is None


def test_trace_owner_and_capacity_never_publish_foreign_or_partial_words(scorers):
    oracle, problem, hypotheses = make_case(4, rows=16, groups=1)
    scorers[2].score(hypotheses, problem, oracle, "gpu")
    reader = scorers[2].dll.rbpf_trace_read
    data = np.full(38, 123, dtype=np.uint64)
    assert reader(scorers[1].handle, data, len(data)) == 0 and np.all(data == 123)
    assert reader(scorers[2].handle, data, 1) == -2 and np.all(data == 123)
    count = reader(scorers[2].handle, data, len(data))
    assert count == 38 and parse_trace(data[:count])["full_GPU_path"]


def test_stale_trace_not_reused_when_original_caller_returns_without_native_work(scorers):
    oracle, problem, hypotheses = make_case(4, rows=16, groups=1)
    scorers[2].score(hypotheses, problem, oracle, "gpu")
    exact(scorers[0].score([], problem, oracle, "gpu"), scorers[2].score([], problem, oracle, "gpu"))
    assert scorers[2].last_native_trace is None and scorers[2].native_records[-1]["trace"] is None
    assert scorers[2].native_records[-1]["probe_error"] is None


def test_reader_cannot_observe_another_threads_trace(scorers):
    from concurrent.futures import ThreadPoolExecutor
    oracle, problem, hypotheses = make_case(4, rows=16, groups=1)
    scorers[2].score(hypotheses, problem, oracle, "gpu")

    def read_on_other_thread():
        data = np.full(38, 123, dtype=np.uint64)
        count = scorers[2].dll.rbpf_trace_read(scorers[2].handle, data, len(data))
        return count, data

    with ThreadPoolExecutor(max_workers=1) as pool:
        count, data = pool.submit(read_on_other_thread).result()
    assert count == 0 and np.all(data == 123)
    assert scorers[2].last_native_trace["full_GPU_path"]


def test_invalid_native_call_preserves_return_and_records_exit(scorers):
    scorer = scorers[2]
    rows, meta = np.empty((0, 7)), np.empty((0, 3), dtype=np.int32)
    pairs, dd, parameters, cfg = np.zeros((1, 2)), np.zeros((1, 1)), np.zeros((1, 12)), np.ones(7)
    out, status = np.full(1, 123.0), np.full(1, 456, dtype=np.int32)
    code = scorer.dll.rbpf_score_trace(scorer.handle, rows, meta, pairs, dd, parameters, cfg,
        0, 0, 1, 1, out, status, None)
    assert code == -1 and out[0] == 123 and status[0] == 456
    storage = np.zeros(38, dtype=np.uint64)
    count = scorer.dll.rbpf_trace_read(scorer.handle, storage, len(storage))
    trace = parse_trace(storage[:count])
    assert count == 14 and not trace["full_GPU_path"]
