from __future__ import annotations

import numpy as np

from experiments.eval_gsdc2023_ct_rbpf_fgo import (
    ct_candidate_summary,
    discover_train_trips,
    parse_float_list,
    score_delta,
)


def test_parse_float_list_rejects_empty_values() -> None:
    assert parse_float_list("0.1, 0.2") == [0.1, 0.2]


def test_score_delta_handles_missing_or_nonfinite_values() -> None:
    assert score_delta(2.0, 3.5) == -1.5
    assert score_delta(np.nan, 3.5) is None
    assert score_delta(None, 3.5) is None


def test_ct_candidate_summary_extracts_fgo_ct_rbpf_candidates() -> None:
    payload = {
        "chunk_selection_records": [
            {
                "candidates": {
                    "fgo_ct_rbpf": {
                        "mse_pr": 4.0,
                        "quality_score": 0.8,
                    },
                },
            },
            {
                "candidates": {
                    "fgo_ct_rbpf": {
                        "mse_pr": 2.0,
                        "quality_score": 0.6,
                    },
                },
            },
            {"candidates": {"fgo": {"mse_pr": 1.0, "quality_score": 0.1}}},
        ],
    }

    summary = ct_candidate_summary(payload)

    assert summary["ct_candidate_chunks"] == 2
    assert summary["ct_candidate_mean_mse_pr"] == 3.0
    assert summary["ct_candidate_min_mse_pr"] == 2.0
    assert summary["ct_candidate_mean_quality_score"] == 0.7


def test_discover_train_trips_finds_device_gnss_and_truth(tmp_path) -> None:
    phone_dir = tmp_path / "train" / "run-a" / "pixel5"
    phone_dir.mkdir(parents=True)
    (phone_dir / "device_gnss.csv").write_text("x\n", encoding="utf-8")
    (phone_dir / "ground_truth.csv").write_text("x\n", encoding="utf-8")
    ignored = tmp_path / "train" / "run-a" / "pixel6"
    ignored.mkdir()
    (ignored / "device_gnss.csv").write_text("x\n", encoding="utf-8")

    assert discover_train_trips(tmp_path) == ["train/run-a/pixel5"]
