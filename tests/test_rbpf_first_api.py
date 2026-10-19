"""First-operation observation and exact score restoration after staging writes."""

from contextlib import ExitStack
import os
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
from rbpf_first_api import API_COUNTS, BYTES, OPERATIONS, FirstApiScorer, parse_probe  # noqa: E402
from rbpf_likelihood_cases import make_case  # noqa: E402
from rbpf_register_lu import RegisterLUScorer  # noqa: E402


def words(op=0):
    return [1, op, 19, BYTES[op], 0, 0, 0, API_COUNTS[op],
        0, 100, 40, 60, 1, 107, 43, 62, 2, 114, 46, 64]


@pytest.mark.parametrize("op", range(6))
def test_raw_probe_ticks_and_CPU_integer_conservation(op):
    raw = words(op)
    result = parse_probe(np.array(raw, dtype=np.uint64))
    assert result["raw_words"] == raw and result["frequency"] == 19
    assert result["operation"] == OPERATIONS[op] and result["bytes"] == BYTES[op]
    assert result["CUDA_API_count"] == API_COUNTS[op]
    assert result["total"] == dict(wall_ticks=14, user_100ns=6, kernel_100ns=4, thread_100ns=10)


@pytest.mark.parametrize("kind", ["version", "op", "frequency", "bytes", "API_status", "completion_status",
    "flags", "API_count", "negative", "bool", "first", "last", "wall", "user", "kernel", "length"])
def test_invalid_or_failed_probe_rejected(kind):
    raw = words()
    changes = {"version":(0,2), "op":(1,6), "frequency":(2,0), "bytes":(3,1),
        "API_status":(4,1), "completion_status":(5,1), "flags":(6,1), "API_count":(7,1),
        "negative":(8,-1), "bool":(0,True), "first":(8,1), "last":(16,0),
        "wall":(13,0), "user":(14,0), "kernel":(15,0)}
    if kind == "length":
        raw.pop()
    else:
        index, value = changes[kind]
        raw[index] = value
    with pytest.raises(ValueError):
        parse_probe(raw)


@pytest.mark.parametrize("raw", [[], (), np.zeros(20, dtype=np.float64), np.zeros((2,10), dtype=np.uint64)])
def test_invalid_probe_storage(raw):
    with pytest.raises(ValueError):
        parse_probe(raw)


@pytest.fixture(scope="module")
def paths():
    old, new = os.environ.get("GNSS_RBPF_BASELINE_DLL"), os.environ.get("GNSS_RBPF_FIRST_API_DLL")
    if not old or not new:
        pytest.skip("Explicit corrected original/first-API diagnostic DLL required")
    return old, new


@pytest.fixture(scope="module")
def scorers(paths):
    with ExitStack() as stack:
        yield stack.enter_context(RegisterLUScorer(paths[0])), stack.enter_context(FirstApiScorer(paths[1]))


def exact(left, right):
    a, ad = left
    b, bd = right
    assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
    assert ad.keys() == bd.keys()
    for key in ad:
        if isinstance(ad[key], np.ndarray):
            assert ad[key].dtype == bd[key].dtype and ad[key].shape == bd[key].shape and ad[key].tobytes() == bd[key].tobytes()
        else:
            assert ad[key] == bd[key]


@pytest.mark.parametrize("operation", OPERATIONS)
def test_native_operations_then_fresh_CPU_GPU_scores_exact(scorers, operation):
    old, new = scorers
    oracle, problem, hypotheses = make_case(8, rows=64, groups=4, cluster=True)
    expected = {mode:old.score(hypotheses, problem, oracle, mode, diagnostics=True) for mode in ("gpu", "native_cpu")}
    for mode in ("gpu", "native_cpu"):
        new.prepare_probe()
        result = parse_probe(new.probe_raw(operation))
        assert result["operation"] == operation
        exact(expected[mode], new.score(hypotheses, problem, oracle, mode, diagnostics=True))


@pytest.mark.parametrize("cluster", [0, 1])
def test_all_saved_inputs_fresh_score_exact_after_every_operation(scorers, cluster):
    from bench_rbpf_irls_replay import digest, prepare
    from rbpf_irls_replay import load_corpus, oracle_from_snapshot
    corpus = load_corpus(ROOT / f"dist/rbpf_irls_replay_capture_20261016/r3_seed0_capture_cpu_c{cluster}.corpus.jsonl.gz")
    assert len(corpus) == 197
    oracle = oracle_from_snapshot(corpus[0]["inputs"], ROOT / os.environ["GNSS_RBPF_CANONICAL_EXT"], ROOT / os.environ["GNSS_RBPF_PARENT_EXT"])
    for entry in corpus:
        hs, problem = prepare(oracle, entry["inputs"])
        data = (hs, problem, oracle.cfg, oracle._cluster_state.__self__.cluster_states)
        before = digest(data)
        expected = scorers[0].score(hs, problem, oracle, "gpu", diagnostics=True)
        for operation in OPERATIONS:
            scorers[1].prepare_probe()
            assert parse_probe(scorers[1].probe_raw(operation))["operation"] == operation
            exact(expected, scorers[1].score(hs, problem, oracle, "gpu", diagnostics=True))
        assert before == digest(data)


@pytest.mark.parametrize("operation", OPERATIONS)
def test_diagnostic_does_not_replace_native_score_TLS_trace(scorers, operation):
    old, new = scorers
    oracle, problem, hs = make_case(4, rows=16, groups=1)
    new.score(hs, problem, oracle, "gpu")
    before = np.empty(38, dtype=np.uint64)
    reader = new.dll.rbpf_trace_read
    assert reader(new.handle, before, len(before)) == 38
    new.prepare_probe()
    new.probe_raw(operation)
    after = np.empty(38, dtype=np.uint64)
    assert reader(new.handle, after, len(after)) == 38 and before.tobytes() == after.tobytes()


def test_invalid_raw_args_leave_output_untouched(scorers):
    new = scorers[1]
    raw = np.full(20, 123, dtype=np.uint64)
    api = new.dll.rbpf_first_api_probe
    for handle, op, size in ((None,0,20), (new.handle,-1,20), (new.handle,6,20), (new.handle,0,19)):
        assert api(handle,op,raw,size) == -1 and np.all(raw == 123)
    assert new.dll.rbpf_first_api_prepare(None,1) == -1
    assert new.dll.rbpf_first_api_prepare(new.handle,0) == -1
    assert new.dll.rbpf_first_api_prepare(new.handle,2**30) == -1


@pytest.mark.parametrize("value", [0, -1, True, None, 65537])
def test_prepare_size_validation(scorers, value):
    with pytest.raises(ValueError):
        scorers[1].prepare_probe(value)


def test_small_native_capacity_rejects_64KiB_probe_without_output_write(paths):
    with FirstApiScorer(paths[1], capacity=2) as scorer:
        with pytest.raises(ValueError):
            scorer.prepare_probe()
        raw = np.full(20, 123, dtype=np.uint64)
        assert scorer.dll.rbpf_first_api_probe(scorer.handle,4,raw,20) == -1 and np.all(raw == 123)


def test_owned_return_survives_probe_and_close(paths):
    oracle, problem, hs = make_case(4, rows=16, groups=1)
    scorer = FirstApiScorer(paths[1])
    result = scorer.score(hs, problem, oracle, "gpu")
    before = result[0].tobytes()
    scorer.prepare_probe()
    scorer.probe_raw("memset1")
    scorer.close()
    scorer.close()
    assert result[0].tobytes() == before
    with pytest.raises(AssertionError):
        scorer.probe_raw("none")
