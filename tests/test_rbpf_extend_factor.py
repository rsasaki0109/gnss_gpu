"""Exact NumPy LAPACK factor reuse and complete frozen PF comparisons."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
import rbpf_extend_factor as selected  # noqa: E402
from bench_rbpf_feedback_mass import _bind_fixture_callbacks  # noqa: E402
from rbpf_feedback_cases import make_feedback_case  # noqa: E402
from rbpf_feedback_mass import NativeFeedbackMassStep  # noqa: E402
from rbpf_move_cases import comparison, state  # noqa: E402
from rbpf_register_lu import RegisterLUScorer  # noqa: E402


def exact(a, b):
    c = comparison(a, b)
    assert c["discrete_equal"] and c["numeric_bits_equal"] and c["max_abs_error"] == 0.0, c


@pytest.fixture(scope="module")
def dependencies():
    return json.loads((ROOT / "dist/rbpf_common_staging_warm_tests_20261018.plan.json").read_text(encoding="utf8"))["extensions"]


@pytest.fixture(scope="module")
def scorer(dependencies):
    with RegisterLUScorer(ROOT / dependencies["DLL"]) as owned:
        yield owned


def extensions(dependencies):
    return dict(extension=ROOT / dependencies["CANONICAL_EXT"], parent_extension=ROOT / dependencies["PARENT_EXT"], mass_extension=ROOT / dependencies["MASS_EXT"])


@pytest.mark.parametrize("size", [1, 2, 3, 7, 11, 32, 64, 128])
@pytest.mark.parametrize("symmetric", [False, True])
def test_same_lapack_all_rhs_bits_and_input_ownership(size, symmetric):
    rng = np.random.default_rng(20261018 + size)
    cache = selected.ExtendFactorCache()
    cache.begin()
    matrix = rng.normal(size=(size, size))
    matrix = matrix @ matrix.T + np.eye(size) if symmetric else matrix + np.eye(size) * size
    saved = matrix.copy()
    for layout in (matrix, np.asfortranarray(matrix), matrix[::-1, ::-1]):
        for _ in range(6):
            rhs = rng.normal(size=size)
            before = rhs.copy()
            expected = np.linalg.solve(layout, rhs)
            actual = cache.solve(layout, rhs)
            assert actual.tobytes() == expected.tobytes()
            assert rhs.tobytes() == before.tobytes()
            actual[:] = 123
            assert cache.solve(layout, rhs).tobytes() == expected.tobytes()
    assert matrix.tobytes() == saved.tobytes()
    assert cache.counts["factor_hits"] > 0 and cache.counts["fallbacks"] == 0
    assert cache.counts["substitutions"] == 36
    cache.end()
    assert cache.cache is None


def test_exact_matrix_key_and_epoch_lifetime():
    cache = selected.ExtendFactorCache()
    cache.begin()
    matrix = np.eye(4)
    rhs = np.arange(4, dtype=float)
    for _ in range(2):
        exact(cache.solve(matrix, rhs), np.linalg.solve(matrix, rhs))
    assert len(cache.cache) == 1
    matrix[0, 1] = -0.0
    exact(cache.solve(matrix, rhs), np.linalg.solve(matrix, rhs))
    matrix[0, 0] = np.nextafter(1.0, 2.0)
    exact(cache.solve(matrix, rhs), np.linalg.solve(matrix, rhs))
    assert len(cache.cache) == 3
    with pytest.raises(RuntimeError):
        cache.begin()
    cache.end()
    cache.begin()
    assert not cache.cache
    exact(cache.solve(matrix, rhs), np.linalg.solve(matrix, rhs))
    assert cache.counts["factorizations"] == 4
    cache.end()


@pytest.mark.parametrize("case", ["singular", "nan", "inf", "huge", "small", "rhs_inf", "subnormal", "overflow_solution"])
def test_exception_nonfinite_and_extreme_inputs(case):
    matrix, rhs = np.eye(3), np.array([1.0, -0.0, 3.0])
    if case == "singular":
        matrix[1, 1] = 0
    elif case in ("nan", "inf"):
        matrix[1, 1] = float(case)
    elif case == "huge":
        matrix *= 2.0**800
    elif case == "small":
        matrix *= 2.0**-800
    elif case == "rhs_inf":
        rhs[0] = np.inf
    elif case == "subnormal":
        rhs[0] = np.nextafter(0.0, 1.0)
    elif case == "overflow_solution":
        matrix[0, 0] = 2.0**-1000
        rhs[0] = 2.0**500
    cache = selected.ExtendFactorCache()
    cache.begin()
    for _ in range(2):
        try:
            expected = np.linalg.solve(matrix, rhs)
        except np.linalg.LinAlgError:
            with pytest.raises(np.linalg.LinAlgError):
                cache.solve(matrix, rhs)
        else:
            assert cache.solve(matrix, rhs).tobytes() == expected.tobytes()
    cache.end()


def test_custom_solve_fallback(monkeypatch):
    original = np.linalg.solve
    calls = []
    def custom(a, b):
        calls.append((a, b))
        return original(a, b)
    cache = selected.ExtendFactorCache()
    cache.begin()
    monkeypatch.setattr(np.linalg, "solve", custom)
    matrix, rhs = np.eye(2), np.ones(2)
    exact(cache.solve(matrix, rhs), original(matrix, rhs))
    assert len(calls) == 1 and calls[0][0] is matrix and calls[0][1] is rhs
    assert cache.counts["fallbacks"] == 1 and not cache.cache
    cache.end()


def test_capacity_and_unsupported_inputs():
    cache = selected.ExtendFactorCache()
    cache.begin()
    for i in range(130):
        matrix, rhs = np.eye(2) * (i + 1), np.ones(2)
        exact(cache.solve(matrix, rhs), np.linalg.solve(matrix, rhs))
    for matrix, rhs in [(np.eye(129), np.ones(129)), (np.eye(2, dtype=np.float32), np.ones(2, dtype=np.float32)), (np.eye(2), np.ones((2, 1))), ([[1.0]], [1.0])]:
        exact(cache.solve(matrix, rhs), np.linalg.solve(matrix, rhs))
    assert len(cache.cache) == 128 and cache.counts["fallbacks"] == 6
    cache.end()


@pytest.mark.parametrize("backend", ["scalar", "native_cpu", "gpu"])
@pytest.mark.parametrize("seed", [20260710, 20260713])
@pytest.mark.parametrize("cluster", [False, True])
def test_full_pf_four_epochs(dependencies, scorer, backend, seed, cluster):
    original = make_feedback_case(64, 32, rows=128 if cluster else 64, groups=8 if cluster else 4, seed=seed, cluster=cluster, shadow=True, spawn=True)
    modified = deepcopy(original)
    calls = []
    for inputs, cls in [(original, NativeFeedbackMassStep), (modified, selected.ExtendFactorStep)]:
        runtime, rtk, _ = inputs
        _bind_fixture_callbacks(rtk)
        runtime.step = cls(runtime, scorer, backend, **extensions(dependencies))
        calls.append(runtime.step)
    total_solves = 0
    for ep in range(4):
        states, metadata = [], []
        for (runtime, rtk, ctx), call in zip((original, modified), calls):
            rtk.epoch2 = ep
            output = runtime.step(rtk, ctx)
            states.append(state(runtime, rtk, output))
            metadata.append(deepcopy(call.last))
        extra = metadata[1].pop("extend_factor")
        total_solves += extra["substitutions"]
        exact(states[0], states[1])
        exact(metadata[0], metadata[1])
        assert calls[1].extend_factor.cache is None
        assert "_rbpf_extend_solve" not in calls[1]._scope
    assert total_solves > 0
    assert selected.AST_PROOF["restored_AST_equal"]


def test_strict_policy_cleanup(dependencies, scorer):
    inputs = make_feedback_case(64, 8)
    other = deepcopy(inputs)
    results = []
    for (runtime, rtk, ctx), cls in [(inputs, NativeFeedbackMassStep), (other, selected.ExtendFactorStep)]:
        _bind_fixture_callbacks(rtk)
        call = cls(runtime, scorer, "native_cpu", **extensions(dependencies))
        runtime.step = call
        with np.errstate(all="raise"):
            try:
                out = call(rtk, ctx)
                error = None
            except FloatingPointError as exc:
                out, error = None, str(exc)
        results.append((state(runtime, rtk, out), error, call.last))
        if cls is selected.ExtendFactorStep:
            assert call.extend_factor.counts["requests"] == 0
            assert call.extend_factor.cache is None
    exact(results[0], results[1])
