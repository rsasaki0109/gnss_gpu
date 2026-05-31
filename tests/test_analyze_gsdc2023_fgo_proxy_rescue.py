from __future__ import annotations

import pandas as pd

from experiments.analyze_gsdc2023_fgo_proxy_rescue import parse_phone_groups, phone_group_mask, rescue_mask, score_thresholds


def test_rescue_mask_requires_raw_wls_block_and_thresholds() -> None:
    frame = pd.DataFrame(
        {
            "fgo_candidate_chunks": [1, 1, 1, 1],
            "fgo_raw_wls_mse_block_chunks": [1, 0, 1, 1],
            "fgo_mean_mse_ratio_vs_raw_wls": [1.10, 1.10, 1.30, 1.10],
            "fgo_mean_baseline_gap_step_p95_ratio": [1.0, 1.0, 1.0, 2.0],
            "fgo_mean_quality_delta_vs_baseline": [-0.40, -0.40, -0.40, -0.40],
            "fgo_mean_mse_delta_vs_baseline": [-2.0, -2.0, -2.0, -2.0],
        },
    )

    mask = rescue_mask(
        frame,
        ratio_max=1.2,
        gap_ratio_max=1.25,
        quality_delta_max=-0.35,
        mse_delta_vs_baseline_max=0.0,
    )

    assert mask.tolist() == [True, False, False, False]


def test_score_thresholds_sorts_best_precision_then_delta() -> None:
    frame = pd.DataFrame(
        {
            "baseline_score_m": [10.0, 10.0, 10.0],
            "fgo_score_m": [8.0, 12.0, 9.0],
            "fgo_candidate_chunks": [1, 1, 1],
            "fgo_raw_wls_mse_block_chunks": [1, 1, 1],
            "fgo_mean_mse_ratio_vs_raw_wls": [1.05, 1.05, 1.15],
            "fgo_mean_baseline_gap_step_p95_ratio": [1.0, 1.0, 1.0],
            "fgo_mean_quality_delta_vs_baseline": [-0.4, -0.2, -0.4],
            "fgo_mean_mse_delta_vs_baseline": [-2.0, -2.0, -2.0],
        },
    )

    result = score_thresholds(
        frame,
        ratio_max_values=[1.10, 1.20],
        gap_ratio_max_values=[1.25],
        quality_delta_max_values=[-0.35],
        mse_delta_vs_baseline_max_values=[0.0],
    )

    assert not result.empty
    assert result.iloc[0]["selected_rows"] == 2
    assert result.iloc[0]["win_rows"] == 2
    assert result.iloc[0]["sum_score_delta_m"] == -3.0


def test_phone_groups_filter_by_trip_phone() -> None:
    frame = pd.DataFrame(
        {
            "trip": ["train/course/pixel4", "train/course/pixel5", "train/course/pixel4xl"],
        },
    )

    assert parse_phone_groups(["pix4:pixel4,pixel4xl;all:*"]) == [
        ("pix4", ("pixel4", "pixel4xl")),
        ("all", None),
    ]
    assert phone_group_mask(frame, ("pixel4", "pixel4xl")).tolist() == [True, False, True]


def test_score_thresholds_records_phone_group() -> None:
    frame = pd.DataFrame(
        {
            "trip": ["train/course/pixel4", "train/course/pixel5"],
            "baseline_score_m": [10.0, 10.0],
            "fgo_score_m": [8.0, 12.0],
            "fgo_candidate_chunks": [1, 1],
            "fgo_raw_wls_mse_block_chunks": [1, 1],
            "fgo_mean_mse_ratio_vs_raw_wls": [1.05, 1.05],
            "fgo_mean_baseline_gap_step_p95_ratio": [1.0, 1.0],
            "fgo_mean_quality_delta_vs_baseline": [-0.4, -0.4],
            "fgo_mean_mse_delta_vs_baseline": [-2.0, -2.0],
        },
    )

    result = score_thresholds(
        frame,
        ratio_max_values=[1.10],
        gap_ratio_max_values=[1.25],
        quality_delta_max_values=[-0.35],
        mse_delta_vs_baseline_max_values=[0.0],
        phone_group_name="pix4",
        phone_allow=("pixel4",),
    )

    assert result.iloc[0]["phone_group"] == "pix4"
    assert result.iloc[0]["phone_allow"] == "pixel4"
    assert result.iloc[0]["selected_rows"] == 1
    assert result.iloc[0]["win_rows"] == 1
