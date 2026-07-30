from __future__ import annotations

from experiments.analyze_wp174_strong_instant_policy import _decisions


def _row(
    tow: float,
    correction: float,
    *,
    ratio: float = 1.2,
    good: str = "1",
) -> dict[str, str]:
    return {
        "tow": str(tow),
        "shadow_best_sub50cm": good,
        "pair_count": "20",
        "lambda_shadow_ratio": str(ratio),
        "lambda_shadow_bsr_qscale16": "0.9999",
        "lambda_shadow_second_position_delta_m": "0.05",
        "float_update_nis_per_observation": "1",
        "float_update_prefit_residual_rms_m": "1",
        "lambda_shadow_best_correction_x": str(correction),
        "lambda_shadow_best_correction_y": "0",
        "lambda_shadow_best_correction_z": "0",
        "lambda_shadow_best_ecef_x": str(correction),
        "lambda_shadow_best_ecef_y": "0",
        "lambda_shadow_best_ecef_z": "0",
        "lambda_satellite_par_shadow_solved": "1",
        "lambda_satellite_par_shadow_best_correction_x": str(correction),
        "lambda_satellite_par_shadow_best_correction_y": "0",
        "lambda_satellite_par_shadow_best_correction_z": "0",
        "lambda_satellite_par_shadow_best_ecef_x": str(correction),
        "lambda_satellite_par_shadow_best_ecef_y": "0",
        "lambda_satellite_par_shadow_best_ecef_z": "0",
    }


def test_change_point_requires_jump_and_three_contiguous_stable_rows() -> None:
    rows = [
        _row(1.0, 0.0),
        _row(1.2, 0.5),
        _row(1.4, 0.505),
        _row(1.6, 0.510),
    ]
    declared, strong, change_point = _decisions(rows)

    assert declared == [False, False, False, True]
    assert strong == [False] * 4
    assert change_point == [False, False, False, True]


def test_change_point_streak_fails_closed_across_epoch_gap() -> None:
    rows = [
        _row(1.0, 0.0),
        _row(1.2, 0.5),
        _row(1.4, 0.505),
        _row(2.0, 0.510),
        _row(2.2, 0.515),
        _row(2.4, 0.520),
    ]
    declared, _, change_point = _decisions(rows)

    assert not any(declared)
    assert not any(change_point)
