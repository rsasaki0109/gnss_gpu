import pytest

from experiments.eval_wp27_online_calibration import _evaluate, _wilson_upper


def _row(*, error, gamma=0.2, dwell=5, float_separation=0.1, ddpr_separation=0.2):
    return {
        "integrity_position_ball_gamma": str(gamma),
        "integrity_dwell_epochs": str(dwell),
        "integrity_map_float_separation_m": str(float_separation),
        "integrity_map_ddpr_separation_m": str(ddpr_separation),
        "last_ddpr_pairs": "10",
        "ddpr_age_epochs": "2",
        "integrity_map_error_m": str(error),
    }


def test_calibration_grid_counts_false_acceptance_and_wilson_bound():
    result = _evaluate(
        [_row(error=0.2), _row(error=1.2)],
        (0.1, 3, 0.5, 1.0),
    )
    assert result["accepted"] == 2
    assert result["correct"] == 1
    assert result["false"] == 1
    assert result["false_pct"] == 50.0
    assert result["false_wilson95_upper_pct"] > 50.0


def test_empty_acceptance_does_not_claim_statistical_safety():
    assert _wilson_upper(0, 0) == 1.0
    result = _evaluate([_row(error=0.2, gamma=0.01)], (0.1, 3, 0.5, 1.0))
    assert result["accepted"] == 0
    assert result["false_pct"] == 0.0
    assert result["false_wilson95_upper_pct"] == pytest.approx(100.0)
