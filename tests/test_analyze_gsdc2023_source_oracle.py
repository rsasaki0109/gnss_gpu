from __future__ import annotations

import pandas as pd
import pytest

from experiments.analyze_gsdc2023_source_oracle import (
    Condition,
    add_derived_features,
    dataset_summary,
    evaluate_source_rule,
    search_source_rules,
)


def _chunks() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "trip_slug": ["a", "a", "b", "b"],
            "baseline_score_m": [4.0, 3.0, 5.0, 2.0],
            "raw_wls_score_m": [3.0, 4.0, 6.0, 1.0],
            "fgo_score_m": [2.0, 5.0, 4.0, 3.0],
            "baseline_candidate_mse_pr": [10.0, 10.0, 20.0, 20.0],
            "raw_wls_candidate_mse_pr": [6.0, 11.0, 22.0, 5.0],
            "fgo_candidate_mse_pr": [7.0, 20.0, 12.0, 24.0],
            "raw_wls_candidate_quality_score": [0.4, 0.9, 0.9, 0.3],
            "fgo_candidate_quality_score": [0.3, 0.9, 0.4, 0.8],
            "raw_wls_candidate_gap_p95_m": [5.0, 20.0, 20.0, 6.0],
            "fgo_candidate_gap_p95_m": [4.0, 30.0, 7.0, 25.0],
        },
    )
    add_derived_features(frame)
    return frame


def test_dataset_summary_reports_3way_oracle_gain() -> None:
    summary = dataset_summary(_chunks(), ("raw_wls", "fgo"))

    assert summary["chunks"] == 4
    assert summary["source_score_sum_m"] == {"baseline": 14.0, "raw_wls": 14.0, "fgo": 14.0}
    assert summary["oracle_source_counts_3way"] == {"fgo": 2, "baseline": 1, "raw_wls": 1}
    assert summary["oracle_vs_baseline_gain_3way_m"] == 4.0


def test_evaluate_allow_rule_scores_source_gain() -> None:
    frame = _chunks()
    condition = Condition("fgo_candidate_quality_score", "<=", 0.5)

    result = evaluate_source_rule(
        frame,
        "fgo",
        (condition,),
        mode="allow",
        group_column="trip_slug",
    )

    assert result.gain_score_m == 3.0
    assert result.selected_chunks == 2
    assert result.true_positive_chunks == 2
    assert result.false_positive_chunks == 0
    assert result.false_negative_chunks == 0
    assert result.oracle_hit_chunks == 2
    assert result.loo_min_gain_score_m == pytest.approx(1.0)


def test_evaluate_guard_rule_scores_blocked_source_losses() -> None:
    frame = _chunks()
    condition = Condition("fgo_candidate_quality_score", ">=", 0.8)

    result = evaluate_source_rule(
        frame,
        "fgo",
        (condition,),
        mode="guard",
        group_column="trip_slug",
    )

    assert result.gain_score_m == 3.0
    assert result.selected_chunks == 2
    assert result.true_positive_chunks == 2
    assert result.false_positive_chunks == 0
    assert result.false_negative_chunks == 0
    assert result.oracle_hit_chunks == 0


def test_search_source_rules_ranks_best_allow_rule_first() -> None:
    results = search_source_rules(
        _chunks(),
        sources=("raw_wls", "fgo"),
        features=("raw_wls_candidate_quality_score", "fgo_candidate_quality_score"),
        max_cuts_per_feature=8,
        max_conditions=2,
        group_column="trip_slug",
        mode="allow",
    )

    assert results["fgo"][0].gain_score_m == 3.0
    assert results["fgo"][0].false_positive_chunks == 0
    assert results["raw_wls"][0].gain_score_m == 2.0
