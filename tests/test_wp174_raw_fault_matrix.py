from __future__ import annotations

from experiments.summarize_wp174_raw_fault_matrix import summarize


def _audit(fault: str, *, false: int = 0, lost: int = 0) -> dict:
    return {
        "fault": fault,
        "debug_epochs": 100,
        "event_count": 2,
        "truth_labeled_fixed_epochs": 10,
        "false_fixed_epochs": false,
        "false_fixed_epochs_during_fault": false,
        "lost_epochs": lost,
        "lost_epochs_during_fault": lost,
        "reacquisition_p95_s": 1.0,
        "pass_false_fix_zero": false == 0,
        "pass_lost_zero": lost == 0,
        "pass_reacquisition_p95_10s": True,
        "pass_fixed_truth_coverage": True,
    }


def test_matrix_summary_does_not_hide_a_failed_case() -> None:
    result = summarize(
        {
            "tokyo": [_audit("cycle_slip"), _audit("outage", lost=5)],
            "nagoya": [_audit("nlos", false=1)],
        }
    )

    assert result["all_false_fix_zero"] is False
    assert result["all_lost_zero"] is False
    assert result["totals"] == {"false_fix_epochs": 1, "lost_epochs": 5}
