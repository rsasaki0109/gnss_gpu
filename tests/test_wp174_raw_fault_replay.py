from __future__ import annotations

import pytest

from experiments.analyze_wp174_raw_fault_replay import analyze


def test_raw_fault_audit_counts_recovery_and_fails_closed() -> None:
    debug = [
        {"tow": "1.0", "status": "4"},
        {"tow": "1.2", "status": "0"},
        {"tow": "1.4", "status": "0"},
        {"tow": "1.6", "status": "3"},
        {"tow": "1.8", "status": "4"},
    ]
    positions = {
        1.0: {"ecef_x": 1.0, "ecef_y": 2.0, "ecef_z": 3.0, "status": 4},
        1.8: {"ecef_x": 1.1, "ecef_y": 2.0, "ecef_z": 3.0, "status": 4},
    }
    reference = [
        {
            "GPS TOW (s)": "1.0",
            "ECEF X (m)": "1",
            "ECEF Y (m)": "2",
            "ECEF Z (m)": "3",
        },
        {
            "GPS TOW (s)": "1.8",
            "ECEF X (m)": "1",
            "ECEF Y (m)": "2",
            "ECEF Z (m)": "3",
        },
    ]
    manifest = {
        "fault": "outage",
        "events": [{"start_tow": 1.2, "end_tow": 1.4}],
    }
    result = analyze(debug, positions, reference, manifest)
    assert result["fixed_epochs_during_fault"] == 0
    assert result["reacquisition_p95_s"] == pytest.approx(0.4)
    assert result["false_fixed_epochs"] == 0
    assert result["unlabeled_fixed_epochs"] == 0
    assert result["lost_epochs"] == 3
    assert result["solver_status_lost_epochs"] == 2
    assert result["pass_lost_zero"] is False
    assert result["pass_reacquisition_p95_10s"] is True


def test_truth_correct_fix_during_fault_is_not_a_false_fix() -> None:
    debug = [
        {"tow": "1.0", "status": "4"},
        {"tow": "1.2", "status": "4"},
        {"tow": "1.4", "status": "4"},
    ]
    positions = {
        tow: {
            "ecef_x": 1.0,
            "ecef_y": 2.0,
            "ecef_z": 3.0,
            "status": 4,
        }
        for tow in (1.0, 1.2, 1.4)
    }
    reference = [
        {
            "GPS TOW (s)": str(tow),
            "ECEF X (m)": "1",
            "ECEF Y (m)": "2",
            "ECEF Z (m)": "3",
        }
        for tow in (1.0, 1.2, 1.4)
    ]
    manifest = {
        "fault": "cycle_slip",
        "events": [{"start_tow": 1.2, "end_tow": 1.2}],
    }

    result = analyze(debug, positions, reference, manifest)

    assert result["fixed_epochs_during_fault"] == 1
    assert result["pass_no_fix_during_fault"] is False
    assert result["false_fixed_epochs_during_fault"] == 0
    assert result["pass_fault_window_false_fix_zero"] is True
    assert result["pass_lost_zero"] is True


def test_postfilter_output_omission_counts_as_lost() -> None:
    debug = [{"tow": "1.0", "status": "3"}]
    manifest = {"fault": "outage", "events": []}

    result = analyze(debug, {}, [], manifest)

    assert result["solver_status_lost_epochs"] == 0
    assert result["lost_epochs"] == 1
    assert result["pass_lost_zero"] is False


def test_missing_fixed_truth_fails_closed() -> None:
    debug = [{"tow": "1.0", "status": "4"}]
    positions = {
        1.0: {
            "ecef_x": 1.0,
            "ecef_y": 2.0,
            "ecef_z": 3.0,
            "status": 4,
        }
    }
    manifest = {"fault": "nlos", "events": []}

    result = analyze(debug, positions, [], manifest)

    assert result["unlabeled_fixed_epochs"] == 1
    assert result["pass_fixed_truth_coverage"] is False
    assert result["pass_false_fix_zero"] is False


def test_safe_fix_shadow_uses_candidate_ecef_not_solver_status() -> None:
    debug = [
        {
            "tow": "1.0",
            "status": "3",
            "safe_fix_shadow_declared_fixed": "1",
            "lambda_shadow_best_ecef_x": "1",
            "lambda_shadow_best_ecef_y": "2",
            "lambda_shadow_best_ecef_z": "3",
        }
    ]
    reference = [
        {
            "GPS TOW (s)": "1.0",
            "ECEF X (m)": "1",
            "ECEF Y (m)": "2",
            "ECEF Z (m)": "3",
        }
    ]
    manifest = {"fault": "cycle_slip", "events": []}

    result = analyze(
        debug,
        {},
        reference,
        manifest,
        fix_source="safe_fix_shadow",
    )

    assert result["fix_source"] == "safe_fix_shadow"
    assert result["truth_labeled_fixed_epochs"] == 1
    assert result["pass_false_fix_zero"] is True
