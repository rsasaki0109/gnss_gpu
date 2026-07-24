from __future__ import annotations

import numpy as np
import pytest

from experiments.apply_wp29_moving_offset_shadow import selected_offset
from experiments.analyze_wp29_moving_offset_widelane_shadow import (
    evaluate_dynamic_offset,
    fit_dynamic_offset,
)


def test_selected_offset_requires_accepted_matching_candidate() -> None:
    candidates = {
        "segment": [10, 20],
        "candidates": [{"candidate_id": 7, "offset_ecef_m": [1.0, 2.0, 3.0]}],
    }
    selection = {
        "segment": [10, 20],
        "selection_reason": "regularized_widelane_consensus",
        "selected_candidate_id": 7,
    }
    start, end, candidate_id, offset = selected_offset(candidates, selection)
    assert (start, end, candidate_id) == (10, 20, 7)
    np.testing.assert_allclose(offset, [1.0, 2.0, 3.0])


def test_selected_offset_fails_closed() -> None:
    candidates = {"segment": [10, 20], "candidates": []}
    selection = {
        "segment": [10, 20],
        "selection_reason": "insufficient_evidence_epochs",
        "selected_candidate_id": None,
    }
    with pytest.raises(RuntimeError, match="not accepted"):
        selected_offset(candidates, selection)


def test_fit_dynamic_offset_recovers_linear_curve() -> None:
    coefficients = np.asarray([[0.4, -0.2, 0.1], [0.1, 0.05, -0.03]])
    rows = []
    jacobians = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, -1.0, 0.5]]
    )
    for t in np.linspace(-1.0, 1.0, 12):
        offset = evaluate_dynamic_offset(coefficients, float(t))
        for jacobian in jacobians:
            rows.append((float(t), jacobian, float(jacobian @ offset), 1.0))
    fitted, objective = fit_dynamic_offset(
        rows, degree=1, ridge=0.0, huber_k=1.5
    )
    np.testing.assert_allclose(fitted, coefficients, atol=1.0e-8)
    assert objective < 1.0e-16
