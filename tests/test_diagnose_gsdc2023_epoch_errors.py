from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from experiments.diagnose_gsdc2023_epoch_errors import (
    _finite_corr,
    _score_from_error,
    chunk_diagnostics_frame,
    summary_payload,
)


def _epoch_frame() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "epoch": [0, 1, 2, 3],
            "UnixTimeMillis": [1000, 2000, 3000, 4000],
            "selected_source": ["baseline", "baseline", "fgo", "fgo"],
        },
    )
    values = {
        "baseline": [1.0, 2.0, 3.0, 4.0],
        "raw_wls": [5.0, 6.0, 7.0, 8.0],
        "fgo": [0.5, 1.5, 4.0, 6.0],
        "selected": [1.0, 2.0, 4.0, 6.0],
    }
    for source, errors in values.items():
        frame[f"{source}_error_2d_m"] = errors
        frame[f"{source}_error_3d_m"] = np.asarray(errors, dtype=float) + 0.25
        frame[f"{source}_pr_wmse_m2"] = np.asarray(errors, dtype=float) * 10.0
        frame[f"{source}_baseline_gap_m"] = 0.0 if source == "baseline" else np.asarray(errors, dtype=float)
    frame["fgo_minus_baseline_error_2d_m"] = frame["fgo_error_2d_m"] - frame["baseline_error_2d_m"]
    frame["fgo_minus_baseline_pr_wmse_m2"] = frame["fgo_pr_wmse_m2"] - frame["baseline_pr_wmse_m2"]
    frame["raw_minus_baseline_pr_wmse_m2"] = frame["raw_wls_pr_wmse_m2"] - frame["baseline_pr_wmse_m2"]
    return frame


def _result() -> SimpleNamespace:
    return SimpleNamespace(
        trip="train/course/phone",
        n_epochs=4,
        selected_source_mode="gated",
        chunk_selection_records=[
            {
                "start_epoch": 0,
                "end_epoch": 2,
                "auto_source": "fgo",
                "gated_source": "baseline",
                "candidates": {
                    "fgo": {"mse_pr": 1.5, "quality_score": 0.7, "baseline_gap_p95_m": 2.0},
                },
            },
            {
                "start_epoch": 2,
                "end_epoch": 4,
                "auto_source": "baseline",
                "gated_source": "baseline",
                "candidates": {},
            },
        ],
        metrics_payload=lambda: {
            "selected_source_counts": {"baseline": 2, "fgo": 2},
            "selected_score_m": 3.0,
            "kaggle_wls_score_m": 2.5,
            "raw_wls_score_m": 6.5,
            "fgo_score_m": 4.0,
            "selected_mse_pr": 30.0,
            "baseline_mse_pr": 25.0,
            "raw_wls_mse_pr": 20.0,
            "fgo_mse_pr": 22.0,
        },
    )


def test_score_and_corr_helpers_ignore_non_finite_values() -> None:
    assert _score_from_error(np.array([1.0, 2.0, np.nan, 4.0])) == pytest.approx(2.9)
    assert _finite_corr(np.array([1.0, 2.0, np.nan]), np.array([2.0, 4.0, 6.0])) == pytest.approx(1.0)
    assert _finite_corr(np.array([1.0, 1.0]), np.array([2.0, 3.0])) is None


def test_chunk_diagnostics_frame_reports_oracle_and_candidate_fields() -> None:
    chunks = chunk_diagnostics_frame(_result(), _epoch_frame())

    assert list(chunks["oracle_source"]) == ["fgo", "baseline"]
    assert chunks.loc[0, "fgo_candidate_mse_pr"] == 1.5
    assert chunks.loc[0, "fgo_minus_baseline_score_m"] < 0.0
    assert chunks.loc[1, "fgo_minus_baseline_score_m"] > 0.0


def test_summary_payload_includes_scores_correlations_and_worst_chunks() -> None:
    chunks = chunk_diagnostics_frame(_result(), _epoch_frame())
    payload = summary_payload(_result(), _epoch_frame(), chunks)

    assert payload["scores_m"]["fgo"] == 4.0
    assert payload["mse_pr"]["raw_wls"] == 20.0
    assert payload["oracle_chunk_source_counts"] == {"fgo": 1, "baseline": 1}
    assert payload["worst_fgo_minus_baseline_chunks"][0]["start_epoch"] == 2
