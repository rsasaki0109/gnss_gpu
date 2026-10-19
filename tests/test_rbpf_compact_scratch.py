"""Adaptive row scratch;all numeric bodies retained;full score/PF/raw bits."""

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
from rbpf_compact_scratch import CompactScratchScorer as RegisterLUScorer  # noqa: E402
from rbpf_likelihood_batch import BatchScorer  # noqa: E402
from rbpf_likelihood_cases import make_case  # noqa: E402
from rbpf_feedback_cases import make_feedback_case  # noqa: E402
from rbpf_feedback_mass import NativeFeedbackMassStep  # noqa: E402
from rbpf_move_cases import comparison, state  # noqa: E402


@pytest.fixture(scope="module")
def scorers():
    old = os.environ.get("GNSS_GPU_RBPF_PROBE_DLL")
    new = os.environ.get("GNSS_GPU_RBPF_REGISTER_DLL")
    if not old or not new:
        pytest.skip("Explicit old/new standalone DLLs required")
    with ExitStack() as stack:
        yield (stack.enter_context(BatchScorer(old)),
            stack.enter_context(RegisterLUScorer(new, kernel="original")),
            stack.enter_context(RegisterLUScorer(new, kernel="generic")),
            stack.enter_context(RegisterLUScorer(new, kernel="register")),
            stack.enter_context(RegisterLUScorer(new, kernel="synced")),
            stack.enter_context(RegisterLUScorer(new, kernel="full")))


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
                stack.enter_context(RegisterLUScorer(os.environ["GNSS_GPU_RBPF_REGISTER_DLL"], capacity=4)))
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


def test_explicit_selector_rejects_invalid_modes_without_affecting_score(scorers):
    oracle, problem, hs = make_case(21, rows=128, groups=8, cluster=True)
    expected = scorers[0].score(hs, problem, oracle, "gpu", diagnostics=True)
    for scorer in scorers[1:]:
        assert scorer.dll.rbpf_register_select(scorer.handle, -1) == -1
        assert scorer.dll.rbpf_register_select(scorer.handle, 5) == -1
        exact(expected, scorer.score(hs, problem, oracle, "gpu", diagnostics=True))
    assert scorers[1].dll.rbpf_register_select(None, 0) == -1


def test_repeated_context_lifetime_and_idempotent_close():
    oracle, problem, hypotheses = make_case(3)
    for _ in range(24):
        scorer = RegisterLUScorer(os.environ["GNSS_GPU_RBPF_REGISTER_DLL"], capacity=3)
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


KINDS = ("spd", "asymmetric", "rowpivot", "ties", "singular", "duplicate", "negative", "nan", "inf", "overflow", "subnormal", "signedzero", "mixed_scale", "ill_conditioned", "rhs_nan", "rhs_inf", "rhs_huge", "pivot_ties_later", "near_singular", "rhs_signedzero")


@pytest.mark.parametrize("n", range(3, 12))
@pytest.mark.parametrize("kind", KINDS)
def test_raw_LU_pivots_solution_logdet_status_exact(scorers, n, kind):
    rng = np.random.default_rng(20261017 + n * 100 + KINDS.index(kind))
    data = np.zeros((32, 132))
    for case in data:
        a = rng.normal(size=(n, n))
        matrix = a @ a.T + np.eye(n) * 0.01
        rhs = rng.normal(size=n)
        if kind == "asymmetric":
            matrix += rng.normal(size=(n, n)) * 0.02
        elif kind == "rowpivot":
            matrix = matrix[np.roll(np.arange(n), 2)]
        elif kind == "ties":
            matrix[:2, 0] = [2., -2.]
            matrix[2:, 0] = 0.
        elif kind == "singular":
            matrix[:] = 0.
        elif kind == "duplicate":
            matrix[-1] = matrix[-2]
        elif kind == "negative":
            matrix[0] *= -1.
        elif kind in ("nan", "inf"):
            matrix[int(rng.integers(n)), int(rng.integers(n))] = np.nan if kind == "nan" else np.inf
        elif kind == "overflow":
            matrix = rng.choice([-1., 1.], size=(n, n)) * (np.finfo(float).max / 2)
        elif kind == "subnormal":
            matrix = np.eye(n)
            matrix[0, 0] = np.nextafter(0., 1.)
            rhs[0] = np.nextafter(0., 1.)
        elif kind == "signedzero":
            matrix = np.eye(n)
            zero = rng.choice([0., -0.], size=(n, n))
            matrix[np.eye(n) == 0] = zero[np.eye(n) == 0]
            rhs = rng.choice([0., -0.], size=n)
        elif kind == "mixed_scale":
            scale = np.exp2(rng.integers(-60, 61, size=n).astype(float))
            matrix *= scale[:, None] * scale[None, :]
        elif kind == "ill_conditioned":
            index = np.arange(n, dtype=float)
            matrix = 1. / (index[:, None] + index[None, :] + 1.)
            matrix += np.eye(n) * (2. ** -50)
        elif kind == "rhs_nan":
            rhs[int(rng.integers(n))] = np.nan
        elif kind == "rhs_inf":
            rhs[int(rng.integers(n))] = np.inf
        elif kind == "rhs_huge":
            rhs *= np.finfo(float).max / 16
        elif kind == "pivot_ties_later":
            matrix = np.eye(n)
            matrix[1, 1] = 2.
            matrix[2, 1] = -2.
            matrix[1, 2] = 1.
        elif kind == "near_singular":
            matrix[-1] = matrix[-2] + rng.normal(size=n) * (2. ** -50)
        elif kind == "rhs_signedzero":
            rhs = rng.choice([0., -0.], size=n)
        case[:121].reshape(11, 11)[:n, :n] = matrix
        case[121:121+n] = rhs
    dims = np.full(len(data), n, dtype=np.int32)
    output = np.empty((len(data), 4, 144))
    status = np.empty((len(data), 4), dtype=np.int32)
    f = scorers[-1].dll.rbpf_compact_solve_probe
    doubles = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    integers = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
    f.argtypes = [doubles, integers, ctypes.c_int, doubles, integers]
    f.restype = ctypes.c_int
    assert f(data, dims, len(data), output, status) == 0
    assert output[:, 0].tobytes() == output[:, 1].tobytes()
    assert status[:, 0].tobytes() == status[:, 1].tobytes()
    for variant in (2, 3):
        assert output[:, 0].tobytes() == output[:, variant].tobytes()
        assert status[:, 0].tobytes() == status[:, variant].tobytes()


@pytest.mark.parametrize("rows", [0, 59, 60, 61, 63, 64, 65, 128])
@pytest.mark.parametrize("huber", [False, True])
@pytest.mark.parametrize("cluster", [False, True])
def test_row_capacity_boundaries_preserve_all_input_and_output_bits(scorers, rows, huber, cluster):
    oracle, problem, hypotheses = make_case(21, rows=128, groups=8, huber=huber, cluster=cluster)
    for name, array in vars(problem).items():
        if name.startswith("rows_"):
            setattr(problem, name, array[:rows].copy())
    before = {name: x.tobytes() for name, x in vars(problem).items() if isinstance(x, np.ndarray)}
    expected = scorers[0].score(hypotheses, problem, oracle, "gpu", diagnostics=True)
    for scorer in scorers[1:]:
        exact(expected, scorer.score(hypotheses, problem, oracle, "gpu", diagnostics=True))
    assert before == {name: x.tobytes() for name, x in vars(problem).items() if isinstance(x, np.ndarray)}


def test_compact_function_resources_and_invalid_query(scorers):
    f = scorers[-1].dll.rbpf_compact_resources
    f.argtypes = [ctypes.c_int, ctypes.c_void_p]
    f.restype = ctypes.c_int
    for cap, shared in ((0, 30904), (60, 16216), (64, 17080), (128, 30904)):
        result = np.zeros(8, dtype=np.int64)
        assert f(cap, result.ctypes.data) == 0
        assert result[0] > 0 and result[1] == shared and result[2] >= 0
        assert result[3] >= 128 and result[4] > 0 and result[5] == 128
    assert f(60, None) == -1
    assert f(61, result.ctypes.data) == -1
