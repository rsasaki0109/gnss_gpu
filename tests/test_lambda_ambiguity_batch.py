"""Parity and fail-closed tests for the multi-problem ILS adapter."""

import numpy as np
import pytest

from gnss_gpu.lambda_ambiguity import integer_search_batch


def _problems():
    output = []
    for seed, n in enumerate((3, 5, 8, 12)):
        rng = np.random.default_rng(seed)
        ahat = rng.normal(0.0, 5.0, n)
        a = rng.normal(size=(n, n))
        qahat = 0.03 * (a @ a.T) + np.eye(n) * 0.03
        output.append((ahat, qahat))
    return output


def test_cpu_batch_matches_individual_search_contract():
    problems = _problems()
    results = integer_search_batch(
        [problem[0] for problem in problems],
        [problem[1] for problem in problems],
        n_candidates=6,
        engine="cpu",
    )
    assert len(results) == len(problems)
    assert all(candidates.shape[0] == 6 for candidates, _ in results)
    assert all(candidates.dtype == np.int64 for candidates, _ in results)


def test_batch_rejects_length_mismatch_and_unknown_engine():
    with pytest.raises(ValueError, match="equal length"):
        integer_search_batch([np.zeros(2)], [], engine="cpu")
    with pytest.raises(ValueError, match="engine"):
        integer_search_batch([], [], engine="magic")


def test_gpu_batch_exact_integer_parity_when_available():
    pytest.importorskip("gnss_gpu.lambda_batch")
    from gnss_gpu.lambda_batch import HAS_LAMBDA_BATCH

    if not HAS_LAMBDA_BATCH:
        pytest.skip("CUDA batch LAMBDA is unavailable")
    problems = _problems()
    cpu = integer_search_batch(
        [problem[0] for problem in problems],
        [problem[1] for problem in problems],
        n_candidates=24,
        engine="cpu",
    )
    gpu = integer_search_batch(
        [problem[0] for problem in problems],
        [problem[1] for problem in problems],
        n_candidates=24,
        engine="gpu-batch",
    )
    for (cpu_candidates, cpu_residuals), (gpu_candidates, gpu_residuals) in zip(cpu, gpu):
        np.testing.assert_array_equal(gpu_candidates, cpu_candidates)
        np.testing.assert_allclose(gpu_residuals, cpu_residuals, rtol=1e-12, atol=1e-12)
