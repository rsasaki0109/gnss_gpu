# ruff: noqa: E402
from pathlib import Path
import sys

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_EXPERIMENTS_DIR = _PROJECT_ROOT / "experiments"
if str(_EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_DIR))

import exp_ppc_epoch_failure_diagnostic as diagnostic


def test_bool_value_accepts_csv_bool_spellings():
    assert diagnostic._bool_value(True)
    assert diagnostic._bool_value("true")
    assert diagnostic._bool_value("1")
    assert not diagnostic._bool_value(False)
    assert not diagnostic._bool_value("")
    assert not diagnostic._bool_value("false")


def test_summarize_epochs_separates_horizontal_and_vertical_failures():
    rows = [
        {
            "fused_error_2d_m": "0.2",
            "fused_error_3d_m": "0.2",
            "wls_error_3d_m": "0.5",
            "ppc_segment_distance_m": "0.0",
            "tdcp_used": "false",
            "tdcp_last_velocity_used": "false",
            "dd_pr_anchor_used": "true",
            "dd_anchor_effective_alpha": "1.0",
            "widelane_anchor_used": "false",
            "rsp_correction_used": "false",
            "height_hold_used": "true",
            "height_hold_effective_alpha": "1.0",
            "dd_pr_shift_m": "2.0",
            "dd_pr_robust_rms_m": "0.3",
        },
        {
            "fused_error_2d_m": "0.4",
            "fused_error_3d_m": "0.6",
            "wls_error_3d_m": "0.8",
            "ppc_segment_distance_m": "5.0",
            "tdcp_used": "true",
            "tdcp_last_velocity_used": "false",
            "dd_pr_anchor_used": "true",
            "dd_anchor_effective_alpha": "0.3",
            "widelane_anchor_used": "false",
            "rsp_correction_used": "false",
            "height_hold_used": "true",
            "height_hold_effective_alpha": "1.0",
            "dd_pr_shift_m": "1.0",
            "dd_pr_robust_rms_m": "0.4",
        },
        {
            "fused_error_2d_m": "0.4",
            "fused_error_3d_m": "0.4",
            "wls_error_3d_m": "0.9",
            "ppc_segment_distance_m": "5.0",
            "tdcp_used": "false",
            "tdcp_last_velocity_used": "true",
            "dd_pr_anchor_used": "false",
            "dd_anchor_effective_alpha": "0.0",
            "widelane_anchor_used": "true",
            "rsp_correction_used": "false",
            "height_hold_used": "true",
            "height_hold_effective_alpha": "0.0",
            "dd_pr_shift_m": "",
            "dd_pr_robust_rms_m": "",
        },
        {
            "fused_error_2d_m": "2.0",
            "fused_error_3d_m": "3.5",
            "wls_error_3d_m": "3.0",
            "ppc_segment_distance_m": "10.0",
            "tdcp_used": "false",
            "tdcp_last_velocity_used": "false",
            "dd_pr_anchor_used": "false",
            "dd_anchor_effective_alpha": "0.0",
            "widelane_anchor_used": "false",
            "rsp_correction_used": "true",
            "height_hold_used": "true",
            "height_hold_effective_alpha": "1.0",
            "dd_pr_shift_m": "",
            "dd_pr_robust_rms_m": "",
        },
    ]

    summary = diagnostic.summarize_epochs(rows, label="case")

    assert summary["ppc_pass_epochs"] == 2
    assert summary["ppc_epoch_pass_pct"] == pytest.approx(50.0)
    assert summary["ppc_score_pct"] == pytest.approx(25.0)
    assert summary["horizontal_pass_epochs"] == 3
    assert summary["vertical_limited_epochs"] == 1
    assert summary["near_3d_fail_epochs"] == 1
    assert summary["far_3d_fail_epochs"] == 1
    assert summary["high_dd_blend_epochs"] == 1
    assert summary["height_released_epochs"] == 1
    assert summary["fused_improves_wls_3d_epochs"] == 3


def test_failure_spans_split_contiguous_far_error_blocks():
    rows = [
        {
            "epoch": str(i),
            "tow": str(1000 + i),
            "fused_error_2d_m": str(err_2d),
            "fused_error_3d_m": str(err_3d),
            "ppc_segment_distance_m": "1.0",
            "tdcp_used": "true" if i == 0 else "false",
            "tdcp_last_velocity_used": "false",
            "dd_pr_anchor_used": "true",
            "dd_anchor_effective_alpha": "1.0" if i == 3 else "0.3",
            "widelane_anchor_used": "false",
            "rsp_correction_used": "false",
            "height_hold_used": "true",
            "height_hold_effective_alpha": "1.0",
        }
        for i, (err_2d, err_3d) in enumerate(
            [(3.0, 4.0), (4.0, 5.0), (0.2, 0.3), (2.5, 3.1), (1.0, 2.9)]
        )
    ]

    spans = diagnostic.build_failure_spans(rows, label="case", failure_threshold_m=3.0)

    assert len(spans) == 2
    assert spans[0]["start_epoch"] == 0
    assert spans[0]["end_epoch"] == 1
    assert spans[0]["n_epochs"] == 2
    assert spans[0]["tdcp_used_epochs"] == 1
    assert spans[1]["start_epoch"] == 3
    assert spans[1]["end_epoch"] == 3
    assert spans[1]["high_dd_blend_epochs"] == 1
