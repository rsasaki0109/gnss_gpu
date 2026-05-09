from __future__ import annotations

import json

from experiments.export_gsdc2023_source_chunks_from_summary import (
    dataset_summary,
    expand_summary_paths,
    summary_paths_to_frame,
)


def _quality(mse_pr: float, quality_score: float, gap_p95_m: float, gap_max_m: float) -> dict[str, float]:
    return {
        "mse_pr": mse_pr,
        "step_mean_m": 2.0,
        "step_p95_m": 10.0,
        "accel_mean_m": 1.0,
        "accel_p95_m": 3.0,
        "bridge_jump_m": 0.0,
        "baseline_gap_mean_m": gap_p95_m * 0.5,
        "baseline_gap_p95_m": gap_p95_m,
        "baseline_gap_max_m": gap_max_m,
        "quality_score": quality_score,
    }


def test_summary_paths_to_frame_exports_candidate_and_score_fields(tmp_path):
    run_dir = tmp_path / "probe" / "trip_a" / "run"
    run_dir.mkdir(parents=True)
    summary = {
        "raw_bridge": {
            "trip": "train/course/phone",
            "n_epochs": 20,
            "kaggle_wls_score_m": 3.0,
            "raw_wls_score_m": 5.0,
            "fgo_score_m": 2.0,
            "selected_score_m": 3.0,
            "fgo_metrics": {"rms_2d_m": 1.5},
            "chunk_selection_records": [
                {
                    "start_epoch": 0,
                    "end_epoch": 20,
                    "auto_source": "fgo",
                    "gated_source": "baseline",
                    "candidates": {
                        "baseline": _quality(9.0, 1.0, 0.0, 0.0),
                        "raw_wls": _quality(7.0, 0.85, 10.0, 12.0),
                        "fgo": _quality(8.0, 0.7, 9.5, 11.0),
                    },
                },
            ],
        },
    }
    path = run_dir / "summary.json"
    path.write_text(json.dumps(summary), encoding="utf-8")

    paths = expand_summary_paths([str(tmp_path / "probe" / "*" / "*" / "summary.json")])
    frame = summary_paths_to_frame(paths)

    assert paths == [path]
    assert frame.shape[0] == 1
    row = frame.iloc[0].to_dict()
    assert row["trip_slug"] == "train__course__phone"
    assert row["oracle_source"] == "fgo"
    assert row["current_gated_source"] == "fgo"
    assert row["fgo_candidate_mse_pr"] == 8.0
    assert row["fgo_minus_baseline_score_m"] == -1.0

    summary_payload = dataset_summary(frame, paths)
    assert summary_payload["rows"] == 1
    assert summary_payload["source_vs_baseline_gain_m"]["fgo"] == 1.0
