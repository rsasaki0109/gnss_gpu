"""Exact numeric/state comparisons and fresh optional GPU elapsed-cycle markers."""

from contextlib import ExitStack
from copy import deepcopy
import ctypes
import os
from pathlib import Path
import sys
import warnings

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
from rbpf_gpu_stages import StageScorer  # noqa: E402
from rbpf_likelihood_batch import BatchScorer  # noqa: E402
from rbpf_likelihood_cases import make_case  # noqa: E402
from rbpf_feedback_cases import make_feedback_case  # noqa: E402
from rbpf_feedback_mass import NativeFeedbackMassStep  # noqa: E402
from rbpf_move_cases import comparison, state  # noqa: E402


class ProfiledView(StageScorer):
    def __init__(self, dll, capacity=512, control=False):
        super().__init__(dll, capacity)
        self.control = control

    def score(self, hypotheses, problem, oracle, mode="gpu", diagnostics=False):
        # Explicit base binding avoids recursion through the optional wrapper.
        return self.profiled_score(hypotheses, problem, oracle, mode, diagnostics, self.control)[:2]


@pytest.fixture(scope="module")
def scorers():
    old = os.environ.get("GNSS_GPU_RBPF_PROBE_DLL")
    new = os.environ.get("GNSS_GPU_RBPF_STAGES_DLL")
    if not old or not new:
        pytest.skip("Explicit old/new standalone DLLs required")
    with ExitStack() as stack:
        yield (stack.enter_context(BatchScorer(old)), stack.enter_context(StageScorer(new)),
            stack.enter_context(ProfiledView(new, control=True)), stack.enter_context(ProfiledView(new)))


def exact(left, right):
    x, xd = left
    y, yd = right
    assert x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes()
    for name in ("unique", "scalar_fallbacks"):
        assert xd[name] == yd[name]
    for name in ("diagnostic", "statuses"):
        assert (name in xd) == (name in yd)
        if name in xd:
            a, b = xd[name], yd[name]
            assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


@pytest.mark.parametrize("seed", [20260710, 20260711, 20260712])
@pytest.mark.parametrize("rows,groups", [(16, 1), (64, 4), (128, 8)])
@pytest.mark.parametrize("options", [(True, True, False), (False, True, False), (True, False, False),
    (False, False, False), (True, True, True), (False, True, True)])
def test_all_numeric_bits_same_backend(scorers, seed, rows, groups, options):
    oracle, problem, hypotheses = make_case(8, rows=rows, groups=groups, seed=seed,
        marg_uniform=options[0], huber=options[1], cluster=options[2])
    before = deepcopy((problem, hypotheses, oracle.cluster_states))
    for mode in ("native_cpu", "gpu"):
        expected = scorers[0].score(hypotheses, problem, oracle, mode, diagnostics=True)
        for scorer in scorers[1:]:
            exact(expected, scorer.score(hypotheses, problem, oracle, mode, diagnostics=True))
    assert hypotheses == before[1]
    for name, x in vars(problem).items():
        y = getattr(before[0], name)
        assert x.tobytes() == y.tobytes() if isinstance(x, np.ndarray) else x == y


@pytest.mark.parametrize("rows,count", [(0, 1), (1, 3), (63, 7), (127, 21)])
def test_alignment_empty_rows_and_completed_host_readback(scorers, rows, count):
    oracle, problem, hypotheses = make_case(count, rows=128, groups=8)
    for name, x in vars(problem).items():
        if name.startswith("rows_"):
            setattr(problem, name, x[:rows].copy())
    expected = scorers[0].score(hypotheses, problem, oracle, "gpu", diagnostics=True)
    for scorer in scorers[1:]:
        result = scorer.score(hypotheses, problem, oracle, "gpu", diagnostics=True)
        exact(expected, result)
        saved = result[0].copy()
        # The next call can overwrite workspace buffers,never returned arrays.
        scorer.score(hypotheses[:1], problem, oracle, "gpu")
        assert result[0].tobytes() == saved.tobytes()


@pytest.mark.parametrize("kind", ["capacity", "rows", "groups", "dtype", "singular", "strict", "warning", "invalid", "empty"])
def test_original_fallbacks_errors_warnings(scorers, kind):
    oracle, problem, hypotheses = make_case(8, rows=130 if kind == "rows" else 64,
        groups=9 if kind == "groups" else 4)
    if kind == "capacity":
        hypotheses *= 1
    if kind == "dtype":
        problem.rows_J = problem.rows_J.astype(np.float32)
    elif kind == "singular":
        problem.Ppos[:] = 0.0
    elif kind == "invalid":
        oracle.cfg.mix_eps = 0.0
    elif kind == "empty":
        hypotheses = []
    elif kind == "strict":
        hypotheses = hypotheses[:2]
    outcomes = []
    with ExitStack() as stack:
        selected = scorers
        if kind == "capacity":
            selected = (stack.enter_context(BatchScorer(os.environ["GNSS_GPU_RBPF_PROBE_DLL"], capacity=4)),
                stack.enter_context(ProfiledView(os.environ["GNSS_GPU_RBPF_STAGES_DLL"], capacity=4)))
        for scorer in selected:
            with np.errstate(all="raise" if kind == "strict" else "warn", under="warn" if kind == "warning" else "ignore"):
                with warnings.catch_warnings(record=True) as seen:
                    warnings.simplefilter("always")
                    try:
                        result = scorer.score(hypotheses, problem, oracle, "gpu")
                    except (ValueError, FloatingPointError) as exc:
                        result, error = None, (type(exc), str(exc))
                    else:
                        error = None
                    outcomes.append((result, error, [(v.category, str(v.message)) for v in seen]))
    for result, error, seen in outcomes[1:]:
        assert (error, seen) == outcomes[0][1:]
        if result is not None:
            exact(outcomes[0][0], result)


@pytest.mark.parametrize("huber", [False, True])
def test_markers_topology_counts_and_fresh_owned_storage(scorers, huber):
    oracle, problem, hs = make_case(21, cluster=True, huber=huber)
    scorer = scorers[1]
    expected = scorers[0].score(hs, problem, oracle, "gpu", diagnostics=True)
    values, details, trace = scorer.profiled_score(hs, problem, oracle, diagnostics=True)
    exact(expected, (values, details))
    assert trace.dtype == np.uint64 and trace.shape == (details["unique"], 16)
    sentinel = np.iinfo(np.uint64).max
    for row, status in zip(trace, details["statuses"], strict=True):
        assert int(row[15] >> np.uint64(32)) == status == 0
        assert 3 <= int(row[15] & np.uint64(0xffffffff)) <= 11
        assert 0 <= int(row[14]) <= len(problem.rows_v0)
        visited = list(range(14)) if huber else [0, 1, 2, 3, 4, 11, 12, 13]
        assert [i for i, v in enumerate(row[:14]) if v != sentinel] == visited
        assert all(int(row[b]) >= int(row[a]) for a, b in zip(visited, visited[1:]))
        assert int(row[13]) > int(row[0])
    saved = trace.copy()
    assert scorer.profiled_score(hs[:1], problem, oracle)[2].shape == (1, 16)
    assert trace.tobytes() == saved.tobytes()
    assert scorer.profiled_score(hs, problem, oracle, control=True)[2] is None
    assert scorer.profiled_score([], problem, oracle)[2] is None
    assert scorer.profiled_score(hs, problem, oracle, mode="native_cpu")[2] is None
    assert scorer.dll.rbpf_profile_read(scorer.handle, None, 0) == 0
    with np.errstate(all="raise"):
        scorer.profiled_score(hs, problem, oracle)
    assert scorer.dll.rbpf_profile_read(scorer.handle, None, 0) == 0
    problem.rows_J = problem.rows_J.astype(np.float32)
    assert scorer.profiled_score(hs, problem, oracle)[2] is None
    assert scorer.dll.rbpf_profile_select(scorer.handle, -1) == -1
    assert scorer.dll.rbpf_profile_select(scorer.handle, 3) == -1
    assert scorer.dll.rbpf_profile_read(scorer.handle, None, -1) == -1


def test_error_trace_has_only_visited_markers(scorers):
    oracle, problem, hs = make_case(3)
    problem.Ppos[:] = 0.
    expected = scorers[0].score(hs, problem, oracle, "gpu", diagnostics=True)
    values, details, trace = scorers[1].profiled_score(hs, problem, oracle, diagnostics=True)
    exact(expected, (values, details))
    assert trace is not None
    for row in trace:
        assert [i for i, v in enumerate(row[:14]) if v != np.iinfo(np.uint64).max] == [0, 1, 13]
        assert int(row[15] >> np.uint64(32)) == 1


def test_repeated_context_lifetime_and_idempotent_close():
    oracle, problem, hypotheses = make_case(3)
    for _ in range(24):
        scorer = ProfiledView(os.environ["GNSS_GPU_RBPF_STAGES_DLL"], capacity=3)
        result, _ = scorer.score(hypotheses, problem, oracle, "gpu")
        saved = result.copy()
        scorer.close()
        scorer.close()
        assert result.tobytes() == saved.tobytes()
        with pytest.raises(RuntimeError):
            scorer.profiled_score(hypotheses, problem, oracle)


@pytest.mark.parametrize("seed", [20260710, 20260711, 20260712])
@pytest.mark.parametrize("options", [(16, 6, 16, 1), (64, 6, 64, 4), (64, 32, 64, 4), (128, 32, 128, 8)])
def test_complete_pf_four_epochs_all_saved_state_rng_and_graph_events(scorers, seed, options):
    n, u, rows, groups = options
    traces = []
    for scorer in scorers:
        runtime, rtk, ctx = make_feedback_case(n, u, rows=rows, groups=groups,
            cluster=rows == 128, feedback=True, preload=True, seed=seed)
        call = NativeFeedbackMassStep(runtime, scorer, "gpu",
            extension=os.environ["GNSS_GPU_RBPF_CANONICAL_EXT"],
            parent_extension=os.environ["GNSS_GPU_RBPF_PARENT_EXT"],
            mass_extension=os.environ["GNSS_GPU_RBPF_MASS_EXT"])
        runtime.step = call
        trace = []
        for _ in range(4):
            output = call(rtk, ctx)
            trace.append(deepcopy(state(runtime, rtk, output)))
        traces.append(trace)
    for trace in traces[1:]:
        for a, b in zip(traces[0], trace, strict=True):
            check = comparison(a, b)
            assert check["discrete_equal"] and check["numeric_bits_equal"] and check["max_abs_error"] == 0.0

