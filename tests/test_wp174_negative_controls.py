from __future__ import annotations

from experiments.analyze_wp174_negative_controls import analyze


def test_all_negative_controls_fail_closed() -> None:
    rows = [
        {
            "tow": str(1.0 + 0.2 * index),
            "block": "0",
            "pair_count": "16",
            "lambda_shadow_ratio": "20",
            "lambda_shadow_bsr": "0.99999",
            "lambda_shadow_bsr_qscale2": "0.99999",
            "lambda_shadow_bsr_qscale4": "0.99999",
            "lambda_shadow_bsr_qscale8": "0.99999",
            "lambda_shadow_bsr_qscale16": "0.99999",
            "lambda_shadow_second_position_delta_m": "0.01",
            "float_update_nis_per_observation": "0.5",
            "lambda_shadow_best_correction_x": "0",
            "lambda_shadow_best_correction_y": "0",
            "lambda_shadow_best_correction_z": "0",
        }
        for index in range(20)
    ]
    policy = {
        "covariance_scale": 16,
        "minimum_pairs": 16,
        "maximum_second_position_delta_m": 0.05,
        "maximum_nis_per_observation": 1.0,
    }
    fold = {
        "test_domain": "tokyo",
        "test_block": 0,
        "selected_policy": policy,
    }
    cv = {
        "folds": [fold],
        "confirmed_policy_folds": [fold],
        "temporal_policy_diagnostic_only": {"folds": [fold]},
    }
    result = analyze(rows, cv, "tokyo")
    assert result["all_pass"] is True
    assert result["total_negative_fix_epochs"] == 0
