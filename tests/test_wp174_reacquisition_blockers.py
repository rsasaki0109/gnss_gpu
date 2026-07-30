from __future__ import annotations

from experiments.analyze_wp174_reacquisition_blockers import diagnose


def _row(tow: float, solved: bool, pairs: int) -> dict[str, str]:
    return {
        "tow": str(tow),
        "block": "0",
        "lambda_shadow_solved": "1" if solved else "0",
        "pair_count": str(pairs),
        "lambda_shadow_ratio": "20",
        "lambda_shadow_bsr_qscale16": "0.99999",
        "lambda_shadow_second_position_delta_m": "0.01",
        "float_update_nis_per_observation": "0.2",
        "float_update_prefit_residual_rms_m": "1",
        "lambda_shadow_best_correction_x": "0",
        "lambda_shadow_best_correction_y": "0",
        "lambda_shadow_best_correction_z": "0",
    }


def test_diagnose_counts_truth_free_long_gap_blockers() -> None:
    rows = [_row(index * 0.2, False, 0) for index in range(60)]
    rows += [_row(12.0 + index * 0.2, True, 16) for index in range(12)]
    result = diagnose(rows)

    assert result["long_gap_epochs"] == 71
    assert result["blocker_counts"]["lambda_not_solved"] == 60
    assert result["blocker_counts"]["eligible_but_temporally_unconfirmed"] == 11
