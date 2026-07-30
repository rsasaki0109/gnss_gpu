from __future__ import annotations

from experiments.analyze_wp174_temporal_consensus_calibration import (
    TemporalConsensusPolicy,
    _declared,
)


def _row(tow: float, correction_x: float) -> dict[str, str]:
    return {
        "tow": str(tow),
        "shadow_best_sub50cm": "1",
        "pair_count": "16",
        "lambda_shadow_bsr_qscale16": "0.999999",
        "lambda_shadow_ratio": "100",
        "lambda_shadow_second_position_delta_m": "0.01",
        "float_update_nis_per_observation": "1",
        "lambda_shadow_best_ecef_x": "1",
        "lambda_shadow_best_ecef_y": "2",
        "lambda_shadow_best_ecef_z": "3",
        "lambda_satellite_par_shadow_best_ecef_x": "1",
        "lambda_satellite_par_shadow_best_ecef_y": "2",
        "lambda_satellite_par_shadow_best_ecef_z": "3",
        "lambda_shadow_best_correction_x": str(correction_x),
        "lambda_shadow_best_correction_y": "0",
        "lambda_shadow_best_correction_z": "0",
    }


def test_temporal_consensus_requires_contiguous_stable_streak() -> None:
    policy = TemporalConsensusPolicy(12, 0.25, 0.25, 0.02, 3)
    rows = [
        _row(1.0, 0.000),
        _row(1.2, 0.005),
        _row(1.4, 0.010),
        _row(2.0, 0.011),
        _row(2.2, 0.050),
    ]

    assert _declared(rows, policy) == [False, False, True, False, False]
