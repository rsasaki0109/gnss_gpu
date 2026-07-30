from experiments.analyze_wp175_library_fix_quality_gate import (
    QualityGatePolicy,
    analyze,
    gate_branches,
)


def test_gate_is_truth_free_and_fail_closed_on_missing_evidence() -> None:
    policy = QualityGatePolicy()
    assert gate_branches({}, policy) == (False, False, False)
    assert gate_branches(
        {"safe_fix_shadow_declared_fixed": "1"}, policy
    ) == (True, True, False)


def test_gate_accepts_each_structural_branch() -> None:
    policy = QualityGatePolicy()
    covariance_row = {
        "float_position_covariance_trace_m2": "0.00025",
        "float_update_nis_per_observation": "10",
        "float_update_observation_count": "8",
    }
    strong_innovation_row = {
        "float_position_covariance_trace_m2": "1",
        "float_update_nis_per_observation": "1",
        "float_update_observation_count": "28",
        "float_update_suppressed_outliers": "14",
    }
    assert gate_branches(covariance_row, policy) == (True, False, True)
    assert gate_branches(strong_innovation_row, policy) == (
        True,
        False,
        True,
    )


def test_strong_branch_rejects_majority_suppression() -> None:
    policy = QualityGatePolicy()
    row = {
        "float_position_covariance_trace_m2": "1",
        "float_update_nis_per_observation": "0.1",
        "float_update_observation_count": "60",
        "float_update_suppressed_outliers": "31",
    }
    assert gate_branches(row, policy) == (False, False, False)


def test_analyze_uses_library_status_as_fix_authority() -> None:
    rows = [
        {
            "tow": "1",
            "safe_fix_shadow_declared_fixed": "0",
            "float_position_covariance_trace_m2": "0.0001",
            "float_update_nis_per_observation": "2",
            "float_update_observation_count": "12",
        },
        {
            "tow": "2",
            "safe_fix_shadow_declared_fixed": "1",
        },
    ]
    truth = {1.0: (0.0, 0.0, 0.0), 2.0: (0.0, 0.0, 0.0)}
    positions = {
        1.0: {
            "status": 4,
            "ecef_x": 0.1,
            "ecef_y": 0.0,
            "ecef_z": 0.0,
        },
        2.0: {
            "status": 3,
            "ecef_x": 0.0,
            "ecef_y": 0.0,
            "ecef_z": 0.0,
        },
    }
    result = analyze(rows, truth, positions, "tokyo", block_count=1)
    assert result["original_library_fixed_epochs"] == 1
    assert result["retained_library_fixed_epochs"] == 1
    assert result["retained_false_fixed_epochs"] == 0
