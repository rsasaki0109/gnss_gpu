from experiments.summarize_wp175_library_fix_fault_matrix import (
    CITIES,
    FAULTS,
    summarize,
)


def test_summary_uses_separate_outage_and_expanded_limits() -> None:
    audits = {}
    for city in CITIES:
        for fault in FAULTS:
            audits[(city, fault)] = {
                "event_count": 8,
                "recovered_events": 8,
                "truth_labeled_fixed_epochs": 10,
                "false_fixed_epochs": 0,
                "fixed_epochs_during_fault": 0,
                "lost_epochs": 0,
                "reacquisition_p95_s": 7.9 if fault == "outage" else 9.9,
                "reacquisition_max_s": 12.0,
                "lambda_shadow_runtime_p95_ms": 99.0,
            }
    result = summarize(audits)
    assert result["totals"]["events"] == 64
    assert result["passes_complete_outage_p95_8s"] is True
    assert result["passes_expanded_fault_p95_10s"] is True
    assert result["passes_false_fix_zero"] is True
    assert result["passes_nlos_outage_negative_fix_zero"] is True
    assert result["passes_lost_zero"] is True
