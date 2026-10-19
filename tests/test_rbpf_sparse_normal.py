"""Same-backend bit comparisons for ordered structural-zero products and fallback."""

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
from rbpf_sparse_normal import SparseNormalScorer  # noqa: E402
from rbpf_likelihood_batch import BatchScorer  # noqa: E402
from rbpf_likelihood_cases import make_case  # noqa: E402
from rbpf_feedback_cases import make_feedback_case  # noqa: E402
from rbpf_feedback_mass import NativeFeedbackMassStep  # noqa: E402
from rbpf_move_cases import comparison, state  # noqa: E402


@pytest.fixture(scope="module")
def scorers():
    old = os.environ.get("GNSS_GPU_RBPF_PROBE_DLL")
    new = os.environ.get("GNSS_GPU_RBPF_SPARSE_DLL")
    if not old or not new:
        pytest.skip("Explicit old/new standalone DLLs required")
    with ExitStack() as stack:
        yield (stack.enter_context(BatchScorer(old)),
            stack.enter_context(SparseNormalScorer(new, kernel="original")),
            stack.enter_context(SparseNormalScorer(new, kernel="dense")),
            stack.enter_context(SparseNormalScorer(new, kernel="sparse")))


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
                stack.enter_context(SparseNormalScorer(os.environ["GNSS_GPU_RBPF_SPARSE_DLL"], capacity=4)))
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


def test_counter_diagnostics_are_separate_nonstale_and_do_not_change_bits(scorers):
    oracle, problem, hypotheses = make_case(21, cluster=True)
    expected = scorers[0].score(hypotheses, problem, oracle, "gpu", diagnostics=True)
    for scorer in scorers[1:]:
        values, details, counts = scorer.count_score(hypotheses, problem, oracle, diagnostics=True)
        exact(expected, (values, details))
        if scorer.kernel == "sparse":
            assert counts["sparse_stages"] > 0 and counts["skipped_products"] > 0
            assert counts["dense_stages"] == counts["unsafe_stages"] == 0
        elif scorer.kernel == "dense":
            assert counts["dense_stages"] > 0
            assert counts["sparse_stages"] == counts["skipped_products"] == counts["unsafe_stages"] == 0
        else:
            assert counts is None
        assert scorer.count_score([], problem, oracle)[2] is None
        exact(expected, scorer.score(hypotheses, problem, oracle, "gpu", diagnostics=True))
        output = (ctypes.c_ulonglong * 4)()
        assert scorer.dll.rbpf_sparse_profile_get(scorer.handle, output) == 1
        assert scorer.dll.rbpf_sparse_select(scorer.handle, 0) == -1


def test_repeated_context_lifetime_and_idempotent_close():
    oracle, problem, hypotheses = make_case(3)
    for _ in range(24):
        scorer = SparseNormalScorer(os.environ["GNSS_GPU_RBPF_SPARSE_DLL"], capacity=3)
        result, _ = scorer.score(hypotheses, problem, oracle, "gpu")
        saved = result.copy()
        scorer.close()
        scorer.close()
        assert result.tobytes() == saved.tobytes()
        with pytest.raises(RuntimeError):
            scorer.count_score(hypotheses, problem, oracle)


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


def normal_probe(scorer, data, groups, rows, dimension):
    doubles = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    integers = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
    function = scorer.dll.rbpf_sparse_normal_probe
    function.argtypes = [doubles, integers, ctypes.c_int, ctypes.c_int, ctypes.c_int, doubles, integers]
    function.restype = ctypes.c_int
    output = np.empty((len(data), 2, 132))
    safe = np.empty(len(data), dtype=np.int32)
    assert function(data, groups, len(data), rows, dimension, output, safe) == 0
    assert output[:, 0].tobytes() == output[:, 1].tobytes()
    return safe


def normal_inputs(dimension, rows, count=32):
    rng = np.random.default_rng(20261017 + dimension * 200 + rows)
    data = np.zeros((count, 1657))
    groups = np.full((count, 128), -1, dtype=np.int32)
    pool = np.array([0., -0., np.nextafter(0., 1.), -np.nextafter(0., 1.),
        np.finfo(float).tiny, -np.finfo(float).tiny, 1., -1.,
        np.finfo(float).max, -np.finfo(float).max, 2.**-500, 2.**500])
    for h in range(count):
        aw = data[h, :1408].reshape(128, 11)
        bw = data[h, 1408:1536]
        prior = data[h, 1536:].reshape(11, 11)
        aw[:rows, :3] = rng.normal(size=(rows, 3)) if h < 16 else rng.choice(pool, (rows, 3))
        bw[:rows] = rng.normal(size=rows) if h < 16 else rng.choice(pool, rows)
        prior[:] = rng.normal(size=(11, 11)) if h < 16 else rng.choice(pool, (11, 11))
        if dimension > 3:
            groups[h, :rows] = rng.integers(-1, dimension - 3, rows)
            for r in range(rows):
                g = groups[h, r]
                if g >= 0:
                    aw[r, 3 + g] = rng.normal() if h < 16 else rng.choice(pool)
    return data, groups


@pytest.mark.parametrize("dimension", range(3, 12))
@pytest.mark.parametrize("rows", [0, 1, 17, 64, 128])
def test_normal_signed_zero_subnormal_cancellation_overflow_asymmetry(scorers, dimension, rows):
    data, groups = normal_inputs(dimension, rows)
    safe = normal_probe(scorers[-1], data, groups, rows, dimension)
    aw = data[:, :1408].reshape(-1, 128, 11)[:, :rows, :dimension]
    bw = data[:, 1408:1536][:, :rows]
    expected = np.all(np.abs(aw) <= 2.**500, axis=(1, 2)) & np.all(np.abs(bw) <= 2.**500, axis=1)
    assert np.array_equal(safe, expected.astype(np.int32))


@pytest.mark.parametrize("dimension", [4, 7, 11])
@pytest.mark.parametrize("kind", ["nan", "inf", "rhs", "magnitude", "structural", "negative_zero", "group"])
def test_normal_unsafe_values_and_structure_use_original_sum(scorers, dimension, kind):
    data, groups = normal_inputs(dimension, 17, count=4)
    for h in range(4):
        if kind in ("nan", "inf"):
            data[h, 0] = np.nan if kind == "nan" else np.inf
        elif kind == "rhs":
            data[h, 1408] = np.inf
        elif kind == "magnitude":
            data[h, 0] = np.finfo(float).max
        elif kind == "group":
            groups[h, 0] = dimension
        else:
            groups[h, 0] = -1
            data[h, 3] = -0. if kind == "negative_zero" else 1.
    assert np.all(normal_probe(scorers[-1], data, groups, 17, dimension) == 0)
