import pytest

from experiments.analyze_wp174_fault_injection import inject_audit_row


@pytest.mark.parametrize(
    ("fault", "field", "expected"),
    [
        ("outage", "pair_count", "0"),
        ("cycle_slip", "lambda_shadow_best_correction_x", ""),
        ("satellite_loss", "pair_count", "0"),
        ("nlos", "float_update_nis_per_observation", "inf"),
    ],
)
def test_fault_mutations_fail_closed(
    fault: str,
    field: str,
    expected: str,
) -> None:
    row = {
        "pair_count": "12",
        "float_update_nis_per_observation": "1",
        "lambda_shadow_best_correction_x": "0.1",
        "lambda_shadow_best_correction_y": "0.2",
        "lambda_shadow_best_correction_z": "0.3",
    }
    assert inject_audit_row(row, fault)[field] == expected
