from __future__ import annotations

import argparse

import pytest

from experiments.eval_gsdc2023_tdcp_fgo_sweep import metrics_row, parse_bool_list, tdcp_variant_grid, variant_name
from experiments.gsdc2023_tdcp import DEFAULT_TDCP_CONSISTENCY_THRESHOLD_M, DEFAULT_TDCP_WEIGHT_SCALE


def test_parse_bool_list_accepts_common_values() -> None:
    assert parse_bool_list("true,false,1,0,on,off") == [True, False, True, False, True, False]


def test_parse_bool_list_rejects_unknown_value() -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        parse_bool_list("maybe")


def test_variant_name_is_filename_friendly() -> None:
    assert variant_name(scale=3.0e-7, threshold=1.5, geometry=True) == "tdcp_s3e-07_thr1p5_geom"


def test_tdcp_variant_grid_moves_default_first() -> None:
    variants = tdcp_variant_grid(
        [1.0e-7, DEFAULT_TDCP_WEIGHT_SCALE],
        [DEFAULT_TDCP_CONSISTENCY_THRESHOLD_M],
        [False, True],
    )

    assert variants[0] == (DEFAULT_TDCP_WEIGHT_SCALE, DEFAULT_TDCP_CONSISTENCY_THRESHOLD_M, True)


def test_metrics_row_includes_tdcp_and_source_counts() -> None:
    row = metrics_row(
        "tdcp_default",
        "train/course/phone",
        {
            "selected_source_counts": {"baseline": 2, "fgo": 3, "fgo_no_tdcp": 1},
            "selected_score_m": 1.2,
            "selected_mse_pr": 10.0,
            "tdcp_weight_scale": 3.0e-7,
            "tdcp_scale_candidate_enabled": True,
            "tdcp_scale_candidate_weight_scale": 1.0e-7,
            "fgo_raw_wls_proxy_rescue_enabled": True,
            "fgo_raw_wls_proxy_rescue_mse_ratio_max": 1.20,
            "fgo_raw_wls_proxy_rescue_gap_step_p95_ratio_max": 1.25,
            "fgo_raw_wls_proxy_rescue_quality_delta_max": -0.35,
            "fgo_raw_wls_proxy_rescue_mse_delta_vs_baseline_max": 0.0,
            "tdcp_consistency_mask_count": 4,
            "tdcp_geometry_correction_applied": True,
            "tdcp_geometry_correction_count": 5,
            "chunk_selection_records": [
                {
                    "candidates": {
                        "baseline": {
                            "mse_pr": 20.0,
                            "quality_score": 1.0,
                            "step_p95_m": 10.0,
                        },
                        "raw_wls": {
                            "mse_pr": 14.0,
                            "quality_score": 0.9,
                        },
                        "fgo": {
                            "mse_pr": 15.0,
                            "quality_score": 0.8,
                            "baseline_gap_p95_m": 11.0,
                        },
                        "fgo_tdcp_scale": {
                            "mse_pr": 14.5,
                            "quality_score": 0.79,
                            "baseline_gap_p95_m": 11.5,
                        },
                    },
                },
            ],
        },
        {"selected_score_m": 1.5, "selected_mse_pr": 11.0},
    )

    assert row["selected_baseline_epochs"] == 2
    assert row["selected_fgo_epochs"] == 3
    assert row["selected_fgo_no_tdcp_epochs"] == 1
    assert row["selected_fgo_tdcp_scale_epochs"] == 0
    assert row["tdcp_scale_candidate_enabled"] is True
    assert row["fgo_raw_wls_proxy_rescue_enabled"] is True
    assert row["fgo_raw_wls_proxy_rescue_mse_ratio_max"] == pytest.approx(1.20)
    assert row["delta_selected_score_m_vs_default"] == pytest.approx(-0.3)
    assert row["tdcp_geometry_correction_count"] == 5
    assert row["fgo_candidate_chunks"] == 1
    assert row["fgo_raw_wls_mse_block_chunks"] == 1
    assert row["fgo_baseline_gap_ok_chunks"] == 1
    assert row["fgo_mean_mse_pr"] == pytest.approx(15.0)
    assert row["fgo_mean_quality_score"] == pytest.approx(0.8)
    assert row["fgo_mean_baseline_gap_p95_m"] == pytest.approx(11.0)
    assert row["fgo_mean_mse_delta_vs_raw_wls"] == pytest.approx(1.0)
    assert row["fgo_mean_mse_ratio_vs_raw_wls"] == pytest.approx(15.0 / 14.0)
    assert row["tdcp_scale_candidate_chunks"] == 1
    assert row["tdcp_scale_mean_mse_delta_vs_fgo"] == pytest.approx(-0.5)
    assert row["tdcp_scale_mean_quality_delta_vs_fgo"] == pytest.approx(-0.01)
