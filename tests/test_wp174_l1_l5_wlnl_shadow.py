from __future__ import annotations

import csv
from pathlib import Path

from experiments.analyze_wp174_l1_l5_wlnl_shadow import analyze


def _write(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_analyzer_labels_only_two_stage_candidates(tmp_path: Path) -> None:
    debug = tmp_path / "debug.csv"
    reference = tmp_path / "reference.csv"
    _write(
        debug,
        [
            {
                "tow": "10.000",
                "lambda_l1_l5_wlnl_shadow_attempted": "1",
                "lambda_l1_l5_wlnl_shadow_pair_count": "4",
                "lambda_l1_l5_wlnl_shadow_wl_ffrt_passed": "1",
                "lambda_l1_l5_wlnl_shadow_nl_ffrt_passed": "1",
                "lambda_l1_l5_wlnl_shadow_mw_disagreements": "0",
                "lambda_l1_l5_wlnl_shadow_best_ecef_x": "1.1",
                "lambda_l1_l5_wlnl_shadow_best_ecef_y": "2",
                "lambda_l1_l5_wlnl_shadow_best_ecef_z": "3",
                "lambda_l1_l5_wlnl_shadow_runtime_ms": "0.2",
            },
            {
                "tow": "10.200",
                "lambda_l1_l5_wlnl_shadow_attempted": "1",
                "lambda_l1_l5_wlnl_shadow_pair_count": "4",
                "lambda_l1_l5_wlnl_shadow_wl_ffrt_passed": "1",
                "lambda_l1_l5_wlnl_shadow_nl_ffrt_passed": "0",
                "lambda_l1_l5_wlnl_shadow_mw_disagreements": "0",
                "lambda_l1_l5_wlnl_shadow_best_ecef_x": "100",
                "lambda_l1_l5_wlnl_shadow_best_ecef_y": "2",
                "lambda_l1_l5_wlnl_shadow_best_ecef_z": "3",
                "lambda_l1_l5_wlnl_shadow_runtime_ms": "0.3",
            },
        ],
    )
    _write(
        reference,
        [
            {
                "GPS TOW (s)": "10.0",
                "ECEF X (m)": "1",
                "ECEF Y (m)": "2",
                "ECEF Z (m)": "3",
            },
            {
                "GPS TOW (s)": "10.2",
                "ECEF X (m)": "1",
                "ECEF Y (m)": "2",
                "ECEF Z (m)": "3",
            },
        ],
    )

    epochs, summary = analyze(debug, reference)

    assert len(epochs) == 1
    assert epochs[0]["sub50cm"] == 1
    assert summary["candidate_epochs"] == 1
    assert summary["not_sub50cm_candidate_epochs"] == 0
