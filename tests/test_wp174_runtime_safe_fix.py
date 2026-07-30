from __future__ import annotations

from experiments.analyze_wp174_ffrt_calibration import Policy
from experiments.analyze_wp174_runtime_safe_fix import analyze
from experiments.analyze_wp174_safe_union import StateMachineConfig


def _row(tow: float, good: bool) -> dict[str, str]:
    return {
        "tow": str(tow),
        "block": "0",
        "shadow_best_sub50cm": "1" if good else "0",
        "pair_count": "16",
        "lambda_shadow_ratio": "20",
        "lambda_shadow_bsr_qscale16": "0.99999",
        "lambda_shadow_second_position_delta_m": "0.01",
        "float_update_nis_per_observation": "0.5",
        "lambda_shadow_best_correction_x": str((tow - 1.0) * 0.01),
        "lambda_shadow_best_correction_y": "0",
        "lambda_shadow_best_correction_z": "0",
    }


def test_runtime_policy_audit_reports_truth_only_after_declaration() -> None:
    result = analyze(
        {"tokyo": [_row(1.0, True), _row(1.2, False)]},
        Policy(16, 16, 0.05, 3.0),
        StateMachineConfig(
            acquisition_streak=2,
            maximum_correction_jump_m=0.03,
        ),
    )

    assert result["selection_status"].startswith("posthoc")
    assert result["domains"]["tokyo"]["declared_fix_epochs"] == 1
    assert result["domains"]["tokyo"]["accepted_bad_epochs"] == 1
