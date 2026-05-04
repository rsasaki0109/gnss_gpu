from __future__ import annotations

import json
import pandas as pd

from experiments.audit_gsdc2023_pr_proxy_risk import load_guard_rows, load_risky_rows, main, summarize


def _quality(mse_pr: float, quality_score: float, gap_p95_m: float, gap_max_m: float) -> dict[str, float]:
    return {
        "mse_pr": mse_pr,
        "step_mean_m": 4.0,
        "step_p95_m": 16.0,
        "accel_mean_m": 2.0,
        "accel_p95_m": 10.0,
        "bridge_jump_m": 0.0,
        "baseline_gap_mean_m": gap_p95_m * 0.5,
        "baseline_gap_p95_m": gap_p95_m,
        "baseline_gap_max_m": gap_max_m,
        "quality_score": quality_score,
    }


def test_load_risky_rows_flags_low_baseline_pr_proxy_gain(tmp_path):
    metrics_path = tmp_path / "bridge_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "trip": "test/2023-05-23-22-16-us-ca-mtv-ie2/pixel6pro",
                "vd_seed_guard_records": [
                    {
                        "chunk_start_epoch": 0,
                        "chunk_end_epoch": 200,
                        "segment_start_epoch": 0,
                        "segment_end_epoch": 200,
                        "segment_epochs": 200,
                        "doppler_count": 1800,
                        "doppler_rms_mps": 187.712,
                        "tdcp_count": 1700,
                        "tdcp_rms_m": 7.711,
                        "reject_reason": "doppler",
                    },
                ],
                "chunk_selection_records": [
                    {
                        "start_epoch": 0,
                        "end_epoch": 200,
                        "auto_source": "raw_wls",
                        "gated_source": "baseline",
                        "candidates": {
                            "baseline": _quality(17.641993, 1.0, 0.0, 0.0),
                            "raw_wls": _quality(7.798628, 0.933937, 20.0, 35.090),
                            "fgo": _quality(7.798628, 0.933937, 20.0, 35.090),
                        },
                    },
                    {
                        "start_epoch": 200,
                        "end_epoch": 400,
                        "auto_source": "raw_wls",
                        "gated_source": "raw_wls",
                        "candidates": {
                            "baseline": _quality(80.0, 1.0, 0.0, 0.0),
                            "raw_wls": _quality(20.0, 0.8, 40.0, 60.0),
                        },
                    },
                ],
            },
        ),
        encoding="utf-8",
    )

    rows = load_risky_rows([metrics_path])
    guard_rows = load_guard_rows([metrics_path])

    assert rows["candidate_source"].tolist() == ["raw_wls", "fgo"]
    assert rows["phone"].tolist() == ["pixel6pro", "pixel6pro"]
    assert rows["candidate_mse_ratio"].tolist() == [7.798628 / 17.641993, 7.798628 / 17.641993]
    assert all("low_baseline_pr_proxy_gain" in value for value in rows["risk_reasons"])
    assert all("baseline_gap_exceeds_fgo_guard" in value for value in rows["risk_reasons"])
    assert rows["vd_guard_overlap_segments"].tolist() == [1, 1]
    assert rows["vd_guard_reject_reasons"].tolist() == ["doppler", "doppler"]
    assert rows["vd_guard_max_doppler_rms_mps"].tolist() == [187.712, 187.712]
    assert guard_rows.shape[0] == 1
    assert guard_rows.iloc[0]["reject_reason"] == "doppler"

    summary = summarize(rows, input_count=1, guard_rows=guard_rows)
    assert summary["risky_rows"] == 2
    assert summary["risky_chunks"] == 1
    assert summary["by_candidate_source"] == {"raw_wls": 1, "fgo": 1}
    assert summary["vd_guard_rows"] == 1
    assert summary["vd_guard_reject_reasons"] == {"doppler": 1}


def test_load_risky_rows_accepts_nested_raw_bridge_summary(tmp_path):
    metrics_path = tmp_path / "summary.json"
    metrics_path.write_text(
        json.dumps(
            {
                "raw_bridge": {
                    "trip": "test/course/pixel6pro",
                    "chunk_selection_records": [
                        {
                            "start_epoch": 0,
                            "end_epoch": 200,
                            "gated_source": "baseline",
                            "candidates": {
                                "baseline": _quality(20.0, 1.0, 0.0, 0.0),
                                "fgo_no_tdcp": _quality(10.0, 0.7, 18.0, 25.0),
                            },
                        },
                    ],
                },
            },
        ),
        encoding="utf-8",
    )

    rows = load_risky_rows([metrics_path])

    assert rows.shape[0] == 1
    assert rows.iloc[0]["candidate_source"] == "fgo_no_tdcp"


def test_main_fail_on_risk_returns_nonzero_only_when_risks_are_found(tmp_path):
    risky_path = tmp_path / "risky_bridge_metrics.json"
    risky_path.write_text(
        json.dumps(
            {
                "trip": "test/course/pixel6pro",
                "chunk_selection_records": [
                    {
                        "start_epoch": 0,
                        "end_epoch": 200,
                        "gated_source": "baseline",
                        "candidates": {
                            "baseline": _quality(20.0, 1.0, 0.0, 0.0),
                            "raw_wls": _quality(10.0, 0.9, 20.0, 35.0),
                        },
                    },
                ],
            },
        ),
        encoding="utf-8",
    )
    safe_path = tmp_path / "safe_bridge_metrics.json"
    safe_path.write_text(
        json.dumps(
            {
                "trip": "test/course/pixel5",
                "chunk_selection_records": [
                    {
                        "start_epoch": 0,
                        "end_epoch": 200,
                        "gated_source": "raw_wls",
                        "candidates": {
                            "baseline": _quality(80.0, 1.0, 0.0, 0.0),
                            "raw_wls": _quality(20.0, 0.8, 40.0, 60.0),
                        },
                    },
                ],
            },
        ),
        encoding="utf-8",
    )

    assert (
        main(
            [
                "--input",
                str(risky_path),
                "--output-dir",
                str(tmp_path / "risky_out"),
                "--fail-on-risk",
            ],
        )
        == 2
    )
    risky_summary = json.loads((tmp_path / "risky_out" / "summary.json").read_text(encoding="utf-8"))
    assert risky_summary["vd_seed_guard_records_csv"].endswith("vd_seed_guard_records.csv")
    assert pd.read_csv(tmp_path / "risky_out" / "vd_seed_guard_records.csv").empty
    assert (
        main(
            [
                "--input",
                str(safe_path),
                "--output-dir",
                str(tmp_path / "safe_out"),
                "--fail-on-risk",
            ],
        )
        == 0
    )
