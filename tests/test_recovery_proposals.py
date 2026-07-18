import numpy as np
import pytest

from gnss_gpu.recovery_proposals import (
    RecoveryPositionBank,
    covariance_axis_position_seeds,
)


def test_covariance_axis_seeds_are_deterministic_and_symmetric():
    center = np.array([1.0, 2.0, 3.0])
    covariance = np.diag([9.0, 4.0, 1.0])
    seeds = covariance_axis_position_seeds(center, covariance, (5.0, 10.0))
    assert len(seeds) == 13
    np.testing.assert_allclose(seeds[0], center)
    offsets = np.asarray(seeds[1:]) - center
    np.testing.assert_allclose(np.linalg.norm(offsets, axis=1), [5.0] * 6 + [10.0] * 6)
    for index in range(0, len(offsets), 2):
        np.testing.assert_allclose(offsets[index], -offsets[index + 1])


def test_covariance_axis_seeds_reject_invalid_radius():
    with pytest.raises(ValueError, match="radii"):
        covariance_axis_position_seeds(np.zeros(3), np.eye(3), (0.0,))


def test_cube26_seed_mode_covers_dense_shell():
    seeds = covariance_axis_position_seeds(
        np.zeros(3), np.eye(3), (5.0,), direction_mode="cube26"
    )
    assert len(seeds) == 27
    np.testing.assert_allclose(
        np.linalg.norm(np.asarray(seeds[1:]), axis=1), np.full(26, 5.0)
    )


def test_recovery_position_bank_keeps_distinct_history_until_expiry():
    bank = RecoveryPositionBank(max_seeds=3, separation_m=1.0, max_age_epochs=2)
    bank.update(
        0,
        np.asarray([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [5.0, 0.0, 0.0]]),
        np.asarray([0.0, -0.1, -2.0]),
    )
    assert len(bank.positions) == 2
    bank.update(1, np.asarray([[0.1, 0.0, 0.0]]), np.asarray([0.0]))
    assert any(position[0] == 5.0 for position in bank.positions)
    bank.update(3, np.asarray([[0.1, 0.0, 0.0]]), np.asarray([0.0]))
    assert all(position[0] != 5.0 for position in bank.positions)
