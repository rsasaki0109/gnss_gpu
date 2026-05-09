import json

import pandas as pd

from experiments.analyze_gsdc2023_raw_wls_policy import (
    DEFAULT_VARIANTS,
    evaluate_train_variants,
    load_metrics_rows,
)


def test_evaluate_train_variants_tracks_gt_delta():
    train = pd.DataFrame(
        [
            {
                "trip": "train/a/samsunga325g",
                "baseline_mse_pr": 56.7,
                "raw_wls_mse_pr": 15.7,
                "raw_wls_baseline_gap_max_m": 123.4,
                "raw_minus_baseline_score_m": -10.4,
            },
            {
                "trip": "train/a/pixel5",
                "baseline_mse_pr": 40.0,
                "raw_wls_mse_pr": 12.0,
                "raw_wls_baseline_gap_max_m": 80.0,
                "raw_minus_baseline_score_m": 4.0,
            },
        ]
    )

    summary, selected = evaluate_train_variants(train, [DEFAULT_VARIANTS[0]])

    assert summary.loc[0, "variant"] == "current_high_pr"
    assert summary.loc[0, "selected_windows"] == 1
    assert summary.loc[0, "raw_better_count"] == 1
    assert summary.loc[0, "raw_worse_count"] == 0
    assert selected.iloc[0]["trip"] == "train/a/samsunga325g"


def test_load_metrics_rows_classifies_raw_wls_branches(tmp_path):
    metrics_path = tmp_path / "bridge_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "trip": "test/route/pixel5",
                "chunk_selection_records": [
                    {
                        "start_epoch": 0,
                        "end_epoch": 30,
                        "gated_source": "raw_wls",
                        "candidates": {
                            "baseline": {"mse_pr": 600.0, "step_p95_m": 60.0},
                            "raw_wls": {
                                "mse_pr": 300.0,
                                "step_p95_m": 80.0,
                                "baseline_gap_p95_m": 120.0,
                                "baseline_gap_max_m": 150.0,
                                "quality_score": 0.8,
                            },
                        },
                    },
                    {
                        "start_epoch": 30,
                        "end_epoch": 60,
                        "gated_source": "raw_wls",
                        "candidates": {
                            "baseline": {"mse_pr": 56.0, "step_p95_m": 20.0},
                            "raw_wls": {
                                "mse_pr": 15.0,
                                "step_p95_m": 50.0,
                                "baseline_gap_p95_m": 90.0,
                                "baseline_gap_max_m": 120.0,
                                "quality_score": 1.5,
                            },
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    rows = load_metrics_rows([metrics_path])

    assert rows["raw_wls_branch"].tolist() == [
        "high_baseline_fallback",
        "train_backed_high_pr_rescue",
    ]
    assert rows["saved_gated_source"].tolist() == ["raw_wls", "raw_wls"]
    assert rows["current_source"].tolist() == ["baseline", "baseline"]
