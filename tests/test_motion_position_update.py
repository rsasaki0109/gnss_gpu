import numpy as np

from gnss_gpu.motion_position_update import evaluate_motion_position_update


def test_motion_position_update_applies_velocity_delta():
    decision = evaluate_motion_position_update(
        np.array([1.0, 2.0, 3.0]),
        np.array([4.0, 5.0, 6.0]),
        0.5,
    )

    assert decision.apply_update is True
    assert decision.reason == "ok"
    np.testing.assert_allclose(decision.predicted_position, [3.0, 4.5, 6.0])


def test_motion_position_update_skips_missing_previous_estimate():
    decision = evaluate_motion_position_update(None, np.ones(3), 1.0)

    assert decision.apply_update is False
    assert decision.predicted_position is None
    assert decision.reason == "no_previous_estimate"


def test_motion_position_update_skips_invalid_velocity():
    decision = evaluate_motion_position_update(
        np.zeros(3),
        np.array([np.nan, 1.0, 2.0]),
        1.0,
    )

    assert decision.apply_update is False
    assert decision.predicted_position is None
    assert decision.reason == "invalid_velocity"


def test_motion_position_update_skips_nonpositive_dt():
    decision = evaluate_motion_position_update(np.zeros(3), np.ones(3), 0.0)

    assert decision.apply_update is False
    assert decision.reason == "invalid_dt"
