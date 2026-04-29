from __future__ import annotations

import pandas as pd
import pytest

from experiments.replay_gsdc2023_fulltrip_source_rules import (
    parse_rule_spec,
    replay_inputs,
    score_errors_m,
    summarize,
)


def _write_run(tmp_path) -> None:
    epoch = pd.DataFrame(
        {
            "baseline_error_2d_m": [1.0, 1.0, 1.0, 1.0, 100.0, *([1.0] * 15)],
            "fgo_error_2d_m": [10.0, 10.0, 10.0, 10.0, 10.0, *([1.0] * 15)],
        },
    )
    chunks = pd.DataFrame(
        {
            "start_epoch": [0, 5],
            "end_epoch": [5, 20],
            "baseline_score_m": [40.0, 1.0],
            "fgo_score_m": [10.0, 1.0],
            "baseline_candidate_mse_pr": [10.0, 10.0],
            "fgo_candidate_mse_pr": [5.0, 500.0],
        },
    )
    epoch.to_csv(tmp_path / "epoch_diagnostics.csv", index=False)
    chunks.to_csv(tmp_path / "chunk_diagnostics.csv", index=False)


def test_parse_rule_spec_parses_source_and_conditions() -> None:
    rule = parse_rule_spec(
        "safe_fgo:fgo:fgo_candidate_mse_pr<=400,raw_wls_candidate_quality_score>=0.5",
    )

    assert rule.name == "safe_fgo"
    assert rule.source == "fgo"
    assert [condition.feature for condition in rule.conditions] == [
        "fgo_candidate_mse_pr",
        "raw_wls_candidate_quality_score",
    ]
    assert rule.conditions[0].op == "<="
    assert rule.conditions[1].op == ">="


def test_replay_scores_full_trip_not_chunk_sum(tmp_path) -> None:
    _write_run(tmp_path)
    rule = parse_rule_spec("allow_fgo:fgo:fgo_candidate_mse_pr<=400")

    frame = replay_inputs([("run_a", tmp_path)], [rule])
    row = frame.iloc[0]

    assert row["selected_chunks"] == 1
    assert row["true_positive_chunks"] == 1
    assert row["false_positive_chunks"] == 0
    assert row["base_score_m"] == pytest.approx(
        score_errors_m(pd.read_csv(tmp_path / "epoch_diagnostics.csv")["baseline_error_2d_m"].to_numpy()),
    )
    assert row["replay_score_m"] > row["base_score_m"]
    assert row["gain_score_m"] < 0.0


def test_summarize_reports_rule_totals(tmp_path) -> None:
    _write_run(tmp_path)
    rule = parse_rule_spec("allow_fgo:fgo:fgo_candidate_mse_pr<=400")
    frame = replay_inputs([("run_a", tmp_path)], [rule])

    summary = summarize(frame, [rule])

    rule_summary = summary["rule_summary"]["allow_fgo"]
    assert rule_summary["runs"] == 1
    assert rule_summary["runs_with_selected_chunks"] == 1
    assert rule_summary["selected_chunks"] == 1
    assert rule_summary["true_positive_chunks"] == 1
    assert rule_summary["total_gain_score_m"] < 0.0
