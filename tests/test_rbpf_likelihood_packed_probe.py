"""Alignment and fallback checks for the packed-transfer likelihood prototype."""

import warnings

import numpy as np
import pytest

from test_rbpf_likelihood_batch_probe import check_case, scorer  # noqa: F401
from rbpf_likelihood_batch import BatchScorer
from rbpf_likelihood_cases import make_case


@pytest.mark.parametrize('rows,count', [(0, 1), (1, 3), (63, 7), (127, 21)])
def test_partial_row_and_odd_hypothesis_alignment(scorer, rows, count):
    oracle, problem, hypotheses = make_case(count, rows=128, groups=8)
    for key, value in vars(problem).items():
        if key.startswith('rows_'):
            setattr(problem, key, value[:rows].copy())
    check_case(scorer, oracle, problem, hypotheses)


def test_alternate_dtype_keeps_original_scalar_arithmetic(scorer):
    oracle, problem, hypotheses = make_case(8)
    problem.rows_J = problem.rows_J.astype(np.float32)
    original = BatchScorer.scalar(hypotheses, problem, oracle)
    actual, details = scorer.score(hypotheses, problem, oracle)
    assert details['scalar_fallbacks'] == 8
    assert actual.tobytes() == original.tobytes()


def test_invalid_noise_configuration_preserves_original_exception(scorer):
    oracle, problem, hypotheses = make_case(8)
    oracle.cfg.mix_eps = 0.
    with pytest.raises(ValueError) as original:
        BatchScorer.scalar(hypotheses, problem, oracle)
    with pytest.raises(ValueError) as actual:
        scorer.score(hypotheses, problem, oracle)
    assert str(actual.value) == str(original.value)


def test_requested_underflow_warning_keeps_original_warning(scorer):
    oracle, problem, hypotheses = make_case(8)
    with np.errstate(under='warn'), warnings.catch_warnings(record=True) as original:
        warnings.simplefilter('always')
        expected = BatchScorer.scalar(hypotheses, problem, oracle)
    with np.errstate(under='warn'), warnings.catch_warnings(record=True) as actual:
        warnings.simplefilter('always')
        result, details = scorer.score(hypotheses, problem, oracle)
    assert details['scalar_fallbacks'] == 8
    assert [(item.category, str(item.message)) for item in actual] == [(item.category, str(item.message)) for item in original]
    assert result.tobytes() == expected.tobytes()
