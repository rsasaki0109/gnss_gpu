from __future__ import annotations

import pandas as pd
import pytest

from experiments.tune_gsdc2023_fgo_source_proxy import (
    Condition,
    add_derived_features,
    candidate_thresholds,
    dataset_summary,
    evaluate_conditions,
    search_rules,
)


def _chunks() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "trip_slug": ["a", "a", "b", "b"],
            "baseline_score_m": [4.0, 3.0, 5.0, 2.0],
            "fgo_score_m": [3.0, 4.0, 4.0, 3.0],
            "oracle_source": ["fgo", "baseline", "fgo", "baseline"],
            "fgo_candidate_quality_score": [0.5, 0.8, 0.55, 0.9],
            "fgo_candidate_gap_p95_m": [5.0, 8.0, 4.0, 9.0],
            "fgo_candidate_mse_pr": [8.0, 12.0, 9.0, 15.0],
            "baseline_candidate_mse_pr": [10.0, 10.0, 10.0, 10.0],
            "raw_wls_candidate_mse_pr": [7.0, 8.0, 8.0, 9.0],
        },
    )
    add_derived_features(frame)
    return frame


def test_candidate_thresholds_are_midpoints_and_limited() -> None:
    values = pd.Series([1.0, 2.0, 4.0, 4.0])

    assert candidate_thresholds(values, max_cuts=10) == [1.5, 3.0]
    assert len(candidate_thresholds(pd.Series(range(20)), max_cuts=4)) == 4


def test_evaluate_conditions_scores_fgo_selection_gain() -> None:
    frame = _chunks()
    condition = Condition("fgo_candidate_quality_score", "<=", 0.6)

    result = evaluate_conditions(frame, (condition,), group_column="trip_slug")

    assert result.gain_score_m == 2.0
    assert result.selected_chunks == 2
    assert result.true_positive_chunks == 2
    assert result.false_positive_chunks == 0
    assert result.false_negative_chunks == 0
    assert result.loo_min_gain_score_m == pytest.approx(1.0)


def test_search_rules_ranks_positive_gain_rules_first() -> None:
    results = search_rules(
        _chunks(),
        features=("fgo_candidate_quality_score", "fgo_candidate_gap_p95_m"),
        max_cuts_per_feature=8,
        max_conditions=2,
        group_column="trip_slug",
    )

    assert results[0].gain_score_m == 2.0
    assert results[0].false_positive_chunks == 0
    assert "fgo_candidate_quality_score" in results[0].payload()["rule"]


def test_dataset_summary_reports_oracle_gain() -> None:
    summary = dataset_summary(_chunks())

    assert summary["chunks"] == 4
    assert summary["fgo_win_chunks"] == 2
    assert summary["oracle_gain_score_m"] == 2.0
    assert summary["oracle_source_counts"] == {"fgo": 2, "baseline": 2}
