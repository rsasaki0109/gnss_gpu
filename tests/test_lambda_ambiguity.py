import numpy as np
import pytest

from gnss_gpu.lambda_ambiguity import (
    decorrelate_ambiguities,
    integer_search,
    integer_search_prepared,
    prepare_integer_search,
    ratio_test,
    solve_lambda,
)


def test_prepared_integer_search_is_identical_for_multiple_means():
    covariance = np.array([[0.8, 0.2], [0.2, 0.5]], dtype=np.float64)
    workspace = prepare_integer_search(covariance)
    for ambiguity in (
        np.array([2.2, -1.4]),
        np.array([8.7, 3.1]),
    ):
        expected = integer_search(ambiguity, covariance, n_candidates=3)
        actual = integer_search_prepared(ambiguity, workspace, n_candidates=3)
        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_allclose(actual[1], expected[1], rtol=0.0, atol=0.0)


def test_lambda_solver_recovers_correlated_integer_vector():
    true_integer = np.array([12, -4, 7], dtype=np.int64)
    float_amb = np.array([12.04, -3.97, 6.93], dtype=np.float64)
    cov = np.array(
        [
            [0.018, 0.010, -0.004],
            [0.010, 0.025, 0.006],
            [-0.004, 0.006, 0.020],
        ],
        dtype=np.float64,
    )

    fixed, ok, solution = solve_lambda(float_amb, cov, ratio_threshold=3.0)

    assert ok
    np.testing.assert_array_equal(fixed, true_integer)
    assert solution.candidates.shape == (2, 3)
    assert solution.ratio >= 3.0


def test_ratio_test_rejects_ambiguous_half_cycle_float():
    float_amb = np.array([4.49], dtype=np.float64)
    cov = np.array([[0.04]], dtype=np.float64)
    candidates, residuals = integer_search(float_amb, cov, n_candidates=2)

    fixed, ok = ratio_test(candidates, residuals, threshold=3.0)

    assert not ok
    assert fixed is None


def test_decorrelate_validates_positive_definite_covariance():
    with pytest.raises(ValueError, match="positive definite"):
        decorrelate_ambiguities(
            np.array([1.0, 2.0], dtype=np.float64),
            np.array([[1.0, 2.0], [2.0, 1.0]], dtype=np.float64),
        )
