import pytest

from experiments.analyze_wp175_library_fix_integrity import analyze


def test_library_fix_integrity_requires_budget_for_every_status4() -> None:
    integrity = [
        {
            "tow": "1.0",
            "failure_budget_passed": "1",
            "independent_families": "2",
            "quality_gate_passed": "1",
            "inertial_available": "1",
            "inertial_healthy_anchor": "1",
            "inertial_passed": "1",
            "processing_runtime_ms": "8",
        },
        {
            "tow": "2.0",
            "failure_budget_passed": "0",
            "independent_families": "1",
            "quality_gate_passed": "0",
            "inertial_available": "1",
            "inertial_healthy_anchor": "0",
            "inertial_passed": "0",
            "processing_runtime_ms": "12",
        },
    ]
    positions = {
        1.0: {"status": 4, "ecef_x": 0.0, "ecef_y": 0.0, "ecef_z": 0.0},
        2.0: {"status": 3, "ecef_x": 1.0, "ecef_y": 0.0, "ecef_z": 0.0},
    }
    reference = [
        {
            "GPS TOW (s)": "1.0",
            "ECEF X (m)": "0",
            "ECEF Y (m)": "0",
            "ECEF Z (m)": "0",
        },
        {
            "GPS TOW (s)": "2.0",
            "ECEF X (m)": "1",
            "ECEF Y (m)": "0",
            "ECEF Z (m)": "0",
        },
    ]
    result = analyze(integrity, positions, reference, "tokyo")
    assert result["every_fixed_has_two_family_budget"] is True
    assert result["every_fixed_passed_quality_gate"] is True
    assert result["every_fixed_passed_all_integrity_gates"] is True
    assert result["library_fixed_epochs"] == 1
    assert result["observed_false_fixed_epochs"] == 0
    assert result["processing_runtime_p95_ms"] == pytest.approx(11.8)
    assert result["runtime_p95_100ms_pass"] is True


def test_library_fix_integrity_rejects_status4_without_quality_gate() -> None:
    integrity = [
        {
            "tow": "1.0",
            "failure_budget_passed": "1",
            "independent_families": "2",
            "quality_gate_passed": "0",
            "processing_runtime_ms": "8",
        }
    ]
    positions = {
        1.0: {"status": 4, "ecef_x": 0.0, "ecef_y": 0.0, "ecef_z": 0.0}
    }
    reference = [
        {
            "GPS TOW (s)": "1.0",
            "ECEF X (m)": "0",
            "ECEF Y (m)": "0",
            "ECEF Z (m)": "0",
        }
    ]
    result = analyze(integrity, positions, reference, "tokyo")
    assert result["every_fixed_has_two_family_budget"] is True
    assert result["every_fixed_passed_quality_gate"] is False
    assert result["every_fixed_passed_all_integrity_gates"] is False
    assert result["fixed_without_quality_gate_epochs"] == [1.0]
