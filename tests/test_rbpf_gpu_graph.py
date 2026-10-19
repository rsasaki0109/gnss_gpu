"""Same-backend bits for pinned stream/graph,shape changes and lifetime."""

from contextlib import ExitStack
from copy import deepcopy
import os
from pathlib import Path
import sys
import warnings

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
from rbpf_gpu_graph import GraphBatchScorer  # noqa: E402
from rbpf_likelihood_batch import BatchScorer  # noqa: E402
from rbpf_likelihood_cases import make_case  # noqa: E402
from rbpf_feedback_cases import make_feedback_case  # noqa: E402
from rbpf_feedback_mass import NativeFeedbackMassStep  # noqa: E402
from rbpf_move_cases import comparison, state  # noqa: E402


@pytest.fixture(scope="module")
def scorers():
    old = os.environ.get("GNSS_GPU_RBPF_PROBE_DLL")
    new = os.environ.get("GNSS_GPU_RBPF_GRAPH_DLL")
    if not old or not new:
        pytest.skip("Explicit old/new standalone DLLs required")
    with ExitStack() as stack:
        yield (stack.enter_context(BatchScorer(old)),
            stack.enter_context(GraphBatchScorer(new, transport="stream")),
            stack.enter_context(GraphBatchScorer(new, transport="graph")))


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
                stack.enter_context(GraphBatchScorer(os.environ["GNSS_GPU_RBPF_GRAPH_DLL"], capacity=4)))
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


def test_fresh_values_shape_updates_and_no_fallback_launch():
    with GraphBatchScorer(os.environ["GNSS_GPU_RBPF_GRAPH_DLL"]) as graph, \
            BatchScorer(os.environ["GNSS_GPU_RBPF_PROBE_DLL"]) as old:
        assert graph.stats() == dict(instantiations=1, shape_updates=0, graph_launches=0, stream_launches=0)
        prior_shape, updates, launches = None, 0, 0
        for count, rows, groups, source_rows, diagnostic, seed in (
            (3, 64, 4, 64, False, 1), (3, 64, 4, 64, False, 2),
            (3, 64, 4, 128, False, 2), (3, 64, 4, 128, True, 2),
            (21, 128, 8, 128, True, 3), (1, 1, 1, 2, False, 4),
            (7, 63, 4, 64, True, 5), (3, 0, 1, 2, False, 6), (3, 64, 4, 64, False, 1)):
            oracle, problem, hypotheses = make_case(count, rows=source_rows, groups=groups, seed=seed)
            if rows != source_rows:
                for name, x in vars(problem).items():
                    if name.startswith("rows_"):
                        setattr(problem, name, x[:rows].copy())
            expected = old.score(hypotheses, problem, oracle, "gpu", diagnostic)
            result = graph.score(hypotheses, problem, oracle, "gpu", diagnostic)
            exact(expected, result)
            assert result[1]["scalar_fallbacks"] == 0
            launches += 1
            shape = count, rows, len(problem.pairs), diagnostic
            updates += int(shape != prior_shape)
            prior_shape = shape
            assert graph.stats() == dict(instantiations=1, shape_updates=updates,
                graph_launches=launches, stream_launches=0)
        before = graph.stats()
        graph.score([], problem, oracle, "gpu")
        graph.score(hypotheses, problem, oracle, "native_cpu")
        assert graph.stats() == before
        assert graph.dll.rbpf_graph_select(graph.handle, 0) == -1


def test_transport_selection_is_explicit_and_immutable():
    dll = os.environ["GNSS_GPU_RBPF_GRAPH_DLL"]
    with pytest.raises(ValueError):
        GraphBatchScorer(dll, transport="invalid")
    for transport in ("stream", "graph"):
        with GraphBatchScorer(dll, transport=transport) as scorer:
            assert scorer.dll.rbpf_graph_select(scorer.handle, 1) == -1
            oracle, problem, hypotheses = make_case(3)
            scorer.score(hypotheses, problem, oracle, "gpu")
            counts = scorer.stats()
            assert counts["graph_launches"] == int(transport == "graph")
            assert counts["stream_launches"] == int(transport == "stream")


@pytest.mark.parametrize("transport", ["stream", "graph"])
def test_repeated_context_lifetime_and_idempotent_close(transport):
    oracle, problem, hypotheses = make_case(3)
    for _ in range(24):
        scorer = GraphBatchScorer(os.environ["GNSS_GPU_RBPF_GRAPH_DLL"], capacity=3, transport=transport)
        result, _ = scorer.score(hypotheses, problem, oracle, "gpu")
        saved = result.copy()
        scorer.close()
        scorer.close()
        assert result.tobytes() == saved.tobytes()
        with pytest.raises(RuntimeError):
            scorer.stats()


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
