from __future__ import annotations

from experiments.eval_gsdc2023_dd_carrier_fgo import dd_candidate_summary, metrics_row
from experiments.gsdc2023_dd_carrier_bridge import DD_CARRIER_FGO_SOURCE


def test_dd_candidate_summary_extracts_bridge_candidate() -> None:
    payload = {
        "chunk_selection_records": [
            {
                "candidates": {
                    DD_CARRIER_FGO_SOURCE: {
                        "mse_pr": 4.0,
                        "quality_score": 0.7,
                    },
                },
            },
            {
                "candidates": {
                    DD_CARRIER_FGO_SOURCE: {
                        "mse_pr": 2.0,
                        "quality_score": 0.5,
                    },
                },
            },
            {"candidates": {"fgo": {"mse_pr": 1.0, "quality_score": 0.1}}},
        ],
    }

    summary = dd_candidate_summary(payload)

    assert summary["dd_candidate_chunks"] == 2
    assert summary["dd_candidate_mean_mse_pr"] == 3.0
    assert summary["dd_candidate_min_mse_pr"] == 2.0
    assert summary["dd_candidate_mean_quality_score"] == 0.6


def test_metrics_row_includes_dd_carrier_counts() -> None:
    row = metrics_row(
        "dd_carrier",
        "train/course/phone",
        {
            "selected_source_counts": {"baseline": 5, DD_CARRIER_FGO_SOURCE: 7},
            "dd_carrier_accepted_anchor_epochs": 3,
            "dd_carrier_dd_epochs": 4,
            "dd_carrier_base_snapped_epochs": 4,
            "dd_carrier_dd_pairs_mean": 5.5,
        },
    )

    assert row["selected_baseline_epochs"] == 5
    assert row["selected_dd_carrier_epochs"] == 7
    assert row["dd_carrier_accepted_anchor_epochs"] == 3
