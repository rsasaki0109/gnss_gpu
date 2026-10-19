"""One-shot ownership, mandatory completion, and complete numeric equivalence."""

from contextlib import ExitStack
import os
from pathlib import Path
import sys
import threading

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
from rbpf_likelihood_cases import make_case  # noqa: E402
from rbpf_register_lu import RegisterLUScorer  # noqa: E402
from rbpf_transition_wake import OneShot, TransitionWakeScorer, transition_type  # noqa: E402
from rbpf_transition_wake import ORDERS, evaluate  # noqa: E402


def test_operation_once_and_integer_timeline():
    calls = []
    owner = threading.get_ident()
    job = OneShot(lambda: calls.append("prepare"), lambda: calls.append("worker") or 7)
    assert job.start() and not job.start()
    job.finish()
    job.finish()
    assert not job.start() and job.state == "done" and not job.thread.is_alive()
    assert calls == ["prepare", "worker"] and job.record["result"] == 7
    assert job.record["worker_ident"] != owner
    r = job.record
    assert r["start_ns"] <= r["worker_start_ns"] <= r["worker_end_ns"] <= r["join_end_ns"]
    assert r["start_ns"] <= r["start_return_ns"] <= r["join_start_ns"] <= r["join_end_ns"]


def test_join_precedes_workspace_reuse():
    entered, release = threading.Event(), threading.Event()
    events = []

    def work():
        events.append("operation")
        entered.set()
        assert release.wait(5)
        events.append("complete")

    job = OneShot(lambda: None, work)
    assert job.start() and entered.wait(5)
    timer = threading.Timer(0.01, release.set)
    timer.start()
    try:
        job.finish()
        events.append("reuse")
    finally:
        release.set()
        timer.join()
        job.finish()
    assert events == ["operation", "complete", "reuse"]


@pytest.mark.parametrize("method", ["start", "finish"])
def test_foreign_owner_rejected_before_work(method):
    calls, errors = [], []
    job = OneShot(lambda: calls.append(1), lambda: calls.append(2))

    def foreign():
        try:
            getattr(job, method)()
        except RuntimeError as error:
            errors.append(str(error))

    thread = threading.Thread(target=foreign)
    thread.start()
    thread.join()
    assert len(errors) == 1 and calls == [] and job.state == "idle"


def test_worker_error_propagates_and_is_drained_once():
    error = ValueError("injected operation failure")

    def fail():
        raise error

    job = OneShot(lambda: None, fail)
    job.start()
    for _ in range(2):
        with pytest.raises(RuntimeError) as got:
            job.finish()
        assert got.value.__cause__ is error and not job.thread.is_alive()
    assert job.state == "failed" and not job.start()


def test_prepare_failure_has_no_worker_or_retry():
    def fail():
        raise ValueError("prepare")

    job = OneShot(fail, lambda: pytest.fail("Unexpected worker"))
    with pytest.raises(ValueError):
        job.start()
    assert job.thread is None and job.state == "failed" and not job.start()
    with pytest.raises(RuntimeError):
        job.finish()


def test_start_failure_is_terminal(monkeypatch):
    class Unstartable:
        def __init__(self, **kwargs):
            pass

        def start(self):
            raise OSError("injected thread start failure")

    monkeypatch.setattr(threading, "Thread", Unstartable)
    job = OneShot(lambda: None, lambda: pytest.fail("Unexpected work"))
    with pytest.raises(OSError):
        job.start()
    job.finish()
    assert job.state == "failed" and job.thread is None and not job.start()


@pytest.mark.parametrize("raises", [False, True])
def test_transition_forwards_original_once_and_preserves_process(raises):
    events, expected = [], object()

    class Original:
        def process(self):
            return expected

        def _transition_to_phase2(self, *args, **kwargs):
            events.append((args, kwargs))
            if raises:
                raise ValueError("original failure")
            return expected

    Selected = transition_type(Original, lambda: events.append("begin"))
    assert Selected.process is Original.process
    obj = Selected()
    if raises:
        with pytest.raises(ValueError, match="original failure"):
            obj._transition_to_phase2(expected, x=3)
    else:
        assert obj._transition_to_phase2(expected, x=3) is expected
    assert events == ["begin", ((expected,), {"x": 3})]


@pytest.fixture(scope="module")
def paths():
    names = ("GNSS_RBPF_BASELINE_DLL", "GNSS_RBPF_FIRST_API_DLL", "GNSS_RBPF_CUDA_RUNTIME")
    values = tuple(os.environ.get(n) for n in names)
    if not all(values):
        pytest.skip("Explicit original/probe DLL and shared runtime required")
    return values


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


@pytest.mark.parametrize("mode", ["gpu", "native_cpu"])
@pytest.mark.parametrize("cluster", [False, True])
def test_native_background_probe_then_original_complete_score(paths, mode, cluster):
    oracle, problem, hs = make_case(8, rows=64, groups=4, cluster=cluster)
    with ExitStack() as stack:
        old = stack.enter_context(RegisterLUScorer(paths[0]))
        new = stack.enter_context(TransitionWakeScorer(paths[1], paths[2], mode="overlap"))
        expected = old.score(hs, problem, oracle, mode, diagnostics=True)
        assert new.transition() and not new.transition()
        actual = new.score(hs, problem, oracle, mode, diagnostics=True)
        exact(expected, actual)
        assert new.wake.state == "done" and new.wake.record["result"]["probe"]["operation"] == "h2d1"
        assert new.wake.record["result"]["device"] == new.device
        assert new.wake.record["join_end_ns"] <= new.score_records[0]["drained_ns"]
        exact(expected, new.score(hs, problem, oracle, mode, diagnostics=True))


@pytest.mark.parametrize("cluster", [0, 1])
def test_all_actual_inputs_original_bits_and_input_ownership(paths, cluster):
    from bench_rbpf_irls_replay import digest, prepare
    from rbpf_irls_replay import load_corpus, oracle_from_snapshot
    corpus = load_corpus(ROOT / f"dist/rbpf_irls_replay_capture_20261016/r3_seed0_capture_cpu_c{cluster}.corpus.jsonl.gz")
    assert len(corpus) == 197
    oracle = oracle_from_snapshot(corpus[0]["inputs"], ROOT / os.environ["GNSS_RBPF_CANONICAL_EXT"], ROOT / os.environ["GNSS_RBPF_PARENT_EXT"])
    with ExitStack() as stack:
        old = stack.enter_context(RegisterLUScorer(paths[0]))
        new = stack.enter_context(TransitionWakeScorer(paths[1], paths[2], mode="overlap"))
        new.transition()
        for entry in corpus:
            hs, problem = prepare(oracle, entry["inputs"])
            data = hs, problem, oracle.cfg, oracle._cluster_state.__self__.cluster_states
            before = digest(data)
            exact(old.score(hs, problem, oracle, "gpu", diagnostics=True),
                new.score(hs, problem, oracle, "gpu", diagnostics=True))
            assert digest(data) == before
        assert len(new.score_records) == 197 and new.wake.state == "done"


def test_none_never_starts_extra_operation(paths):
    oracle, problem, hs = make_case(4, rows=16, groups=1)
    with TransitionWakeScorer(paths[1], paths[2], mode="none") as new:
        assert not new.transition() and not new.transition()
        new.score(hs, problem, oracle, "gpu")
        assert new.wake.state == "idle" and new.wake.thread is None and new.wake.record == {}


def test_close_drains_before_native_destroy_and_retained_output_is_owned(paths):
    oracle, problem, hs = make_case(4, rows=16, groups=1)
    new = TransitionWakeScorer(paths[1], paths[2], mode="overlap")
    result = new.score(hs, problem, oracle, "gpu")
    original = result[0].tobytes()
    entered, release = threading.Event(), threading.Event()
    operation = new.wake.operation

    def blocked():
        entered.set()
        assert release.wait(5)
        assert new.handle
        return operation()

    new.wake.operation = blocked
    new.transition()
    assert entered.wait(5)
    timer = threading.Timer(0.01, release.set)
    timer.start()
    try:
        new.close()
    finally:
        release.set()
        timer.join()
        new.close()
    assert new.handle is None and not new.wake.thread.is_alive() and result[0].tobytes() == original
    with pytest.raises(AssertionError):
        new.score(hs, problem, oracle, "gpu")


def test_close_frees_workspace_after_propagating_worker_error(paths):
    def fail():
        raise ValueError("injected worker error")

    new = TransitionWakeScorer(paths[1], paths[2], mode="overlap")
    new.wake.operation = fail
    new.transition()
    with pytest.raises(RuntimeError):
        new.close()
    assert new.handle is None and not new.wake.thread.is_alive()


def test_invalid_mode_rejected_before_allocating():
    with pytest.raises(ValueError):
        TransitionWakeScorer("missing", "missing", mode="unknown")


def screen_rows():
    result = []
    for rd, order in enumerate(ORDERS):
        for variant in order:
            durations = [1000]*400
            for i in range(199, 204):
                durations[i] = 70 if variant == "overlap" else 100
            bounds, cursor = [], 0
            for duration in durations:
                bounds.append([cursor, cursor+duration])
                cursor += duration
            records = [dict(epoch=i, phase_before=1 if i < 200 else 2, completed=True,
                wall_ns=n) for i, n in enumerate(durations)]
            result.append(dict(trial=dict(round=rd, variant=variant, accuracy=False),
                verified=True, all_final_and_trajectory_bits_exact=True,
                records=records, call_bounds_ns=bounds, driver_wall_ns=cursor,
                transition=dict(transition_epoch=199, first_score_epoch=203),
                transition_through_first_score_wall_ns=sum(durations[199:204]),
                transition_through_first_score_RTK_ns=sum(durations[199:204])))
    return result


def test_latency_gate_requires_both_controls_without_whole_run_claim():
    result = evaluate(screen_rows())
    assert result["first_transition_latency_screen"] is True
    assert result["continuous_RTK_throughput_established"] is False
    assert result["all400_epochs_retained"] and result["original_failed_gates_unchanged"]
    assert all(v["pass_gate"] is None for k, v in result["metrics"].items() if k != "latency")


@pytest.mark.parametrize("kind", ["missing", "order", "incomplete", "clock", "boundary", "interval"])
def test_screen_rejects_invalid_or_partial_raw_evidence(kind):
    rows = screen_rows()
    if kind == "missing":
        rows.pop()
    elif kind == "order":
        rows[0], rows[1] = rows[1], rows[0]
    elif kind == "incomplete":
        rows[0]["records"][0]["completed"] = False
    elif kind == "clock":
        rows[0]["call_bounds_ns"][1][0] = 0
    elif kind == "boundary":
        rows[0]["transition"]["first_score_epoch"] = 204
    else:
        rows[0]["transition_through_first_score_wall_ns"] += 1
    with pytest.raises(ValueError):
        evaluate(rows)


def test_signed_control_cancellation_cannot_pass_gate():
    rows = screen_rows()
    for row in rows:
        if row["trial"]["variant"] == "AA":
            factor = 2 if row["trial"]["round"] % 2 else 0.5
            row["call_bounds_ns"] = [[int(a*factor), int(b*factor)] for a, b in row["call_bounds_ns"]]
            for record in row["records"]:
                record["wall_ns"] = int(record["wall_ns"]*factor)
            for key in ("transition_through_first_score_wall_ns", "transition_through_first_score_RTK_ns", "driver_wall_ns"):
                row[key] = int(row[key]*factor)
    result = evaluate(rows)
    assert not result["metrics"]["latency"]["AA_valid"] and not result["first_transition_latency_screen"]
