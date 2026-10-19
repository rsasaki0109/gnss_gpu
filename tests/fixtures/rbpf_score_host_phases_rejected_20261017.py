"""Register-row ordered LU and distributed solves;full score/PF state/raw bits."""

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
from rbpf_register_lu import RegisterLUScorer
from rbpf_score_host_phases import HostPhaseScorer, PHASES, spans, restored_ast  # noqa: E402
from rbpf_likelihood_batch import BatchScorer  # noqa: E402
from rbpf_likelihood_cases import make_case  # noqa: E402
from rbpf_feedback_cases import make_feedback_case  # noqa: E402
from rbpf_feedback_mass import NativeFeedbackMassStep  # noqa: E402
from rbpf_move_cases import comparison, state  # noqa: E402


@pytest.fixture(scope="module")
def scorers():
    dll = os.environ.get("GNSS_GPU_RBPF_HOST_PHASES_DLL")
    if not dll:
        pytest.skip("Explicit corrected register DLL required")
    with ExitStack() as stack:
        yield (stack.enter_context(RegisterLUScorer(dll)),
            stack.enter_context(HostPhaseScorer(dll, measure=False)),
            stack.enter_context(HostPhaseScorer(dll)))


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
            selected = (stack.enter_context(RegisterLUScorer(os.environ["GNSS_GPU_RBPF_HOST_PHASES_DLL"], capacity=4)),
                stack.enter_context(HostPhaseScorer(os.environ["GNSS_GPU_RBPF_HOST_PHASES_DLL"], capacity=4)))
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


def test_repeated_context_lifetime_and_idempotent_close():
    oracle, problem, hypotheses = make_case(3)
    for _ in range(24):
        scorer = HostPhaseScorer(os.environ["GNSS_GPU_RBPF_HOST_PHASES_DLL"], capacity=3)
        result, _ = scorer.score(hypotheses, problem, oracle, "gpu")
        saved = result.copy()
        scorer.close()
        scorer.close()
        assert result.tobytes() == saved.tobytes()
        with pytest.raises(AssertionError):
            scorer.score(hypotheses, problem, oracle)


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




def test_AST_restoration_literal_bytecode_live_globals_and_no_module_patch():
    import rbpf_likelihood_batch as original
    from rbpf_score_host_phases import _LITERAL, _PROFILED
    assert restored_ast()[0] == restored_ast()[1]
    assert _LITERAL.__code__.co_code == original.BatchScorer.score.__code__.co_code
    assert _LITERAL.__globals__ is _PROFILED.__globals__ is original.BatchScorer.score.__globals__
    assert not any(k.startswith("__rbpf_host_") for k in vars(original))
    assert len(_PROFILED.__closure__) == 2


@pytest.mark.parametrize("kind", ["normal", "empty", "unsupported", "unsafe", "dd_error"])
def test_partial_trace_exception_cleanup_and_fresh_ownership(scorers, kind):
    oracle, problem, hs = make_case(8, rows=64, groups=4)
    scorer = scorers[-1]
    if kind == "empty":
        hs = []
    elif kind == "unsupported":
        problem.rows_J = problem.rows_J.astype(np.float32)
    elif kind == "unsafe":
        oracle.cfg.mix_eps = 0.0
    elif kind == "dd_error":
        def broken(*args):
            raise ValueError("deliberate DD callback failure")
        oracle._pair_dd = broken
    outcomes = []
    for candidate in (scorers[0], scorer):
        try:
            result = candidate.score(hs, problem, oracle, "gpu", diagnostics=True)
        except Exception as exc:
            outcomes.append((type(exc), str(exc)))
        else:
            outcomes.append(result)
    if kind == "dd_error":
        assert outcomes[0] == outcomes[1] == (ValueError, "deliberate DD callback failure")
    else:
        exact(outcomes[0], outcomes[1])
    trace = scorer.last_profile
    parts = spans(trace)
    assert parts["total"] >= 0
    expected_last = {"normal": "finish", "empty": "entry", "unsupported": "support",
        "unsafe": "validate", "dd_error": "dd"}[kind]
    assert trace[-2][0] == expected_last
    assert len(trace) == PHASES.index(expected_last) + 2
    assert sum(v for k,v in parts.items() if k != "total") == parts["total"]
    other, other_problem, other_hs = make_case(2)
    scorer.score(other_hs, other_problem, other)
    assert scorer.last_profile is not trace
    assert trace[-2][0] == expected_last


def test_clock_validation_and_original_live_global_change(scorers):
    import rbpf_likelihood_batch as original
    with pytest.raises(ValueError):
        spans((("entry", 2), ("end", 1)))
    with pytest.raises(ValueError):
        spans((("entry", 1), ("native", 2), ("end", 3)))
    oracle, problem, hs = make_case(4)
    saved = original.MAX_ROWS
    try:
        original.MAX_ROWS = 1
        expected = scorers[0].score(hs, problem, oracle, "gpu")
        for scorer in scorers[1:]:
            exact(expected, scorer.score(hs, problem, oracle, "gpu"))
    finally:
        original.MAX_ROWS = saved
    assert scorers[-1].last_profile[-2][0] == "support"
