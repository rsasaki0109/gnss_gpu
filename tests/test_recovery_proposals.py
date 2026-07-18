import numpy as np
import pytest

from gnss_gpu.recovery_proposals import (
    RecoveryAssignmentBank,
    RecoveryPositionBank,
    complete_versioned_assignment,
    covariance_axis_position_seeds,
)


def _assignment(value: int, generation: int = 0):
    return {
        (("G01", f"G{sat:02d}", 190293673), generation): value + sat
        for sat in range(2, 10)
    }


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


def test_recovery_position_bank_propagates_retained_velocity():
    bank = RecoveryPositionBank(max_seeds=2, separation_m=1.0, max_age_epochs=5)
    bank.update(
        0,
        np.asarray([[0.0, 0.0, 0.0]]),
        np.asarray([0.0]),
        velocities_ecef=np.asarray([[2.0, 0.0, 0.0]]),
    )
    bank.update(
        1,
        np.asarray([[10.0, 0.0, 0.0]]),
        np.asarray([1.0]),
        velocities_ecef=np.asarray([[0.0, 0.0, 0.0]]),
        dt_seconds=0.5,
    )
    assert any(np.isclose(position[0], 1.0) for position in bank.positions)


def test_recovery_position_bank_propagates_retained_tdcp_displacement():
    bank = RecoveryPositionBank(max_seeds=2, separation_m=1.0, max_age_epochs=5)
    bank.update(0, np.asarray([[0.0, 0.0, 0.0]]), np.asarray([0.0]))
    bank.update(
        1,
        np.asarray([[10.0, 0.0, 0.0]]),
        np.asarray([1.0]),
        displacement_ecef_m=np.asarray([0.5, 0.25, 0.0]),
    )
    assert any(np.allclose(position, [0.5, 0.25, 0.0]) for position in bank.positions)


def test_recovery_position_bank_farthest_mode_preserves_low_weight_spatial_mode():
    bank = RecoveryPositionBank(
        max_seeds=2,
        separation_m=0.1,
        max_age_epochs=5,
        selection_mode="farthest",
    )
    bank.update(
        0,
        np.asarray([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [5.0, 0.0, 0.0]]),
        np.asarray([0.0, -0.1, -10.0]),
    )
    assert any(np.isclose(position[0], 5.0) for position in bank.positions)


def test_recovery_position_bank_limits_reference_distance():
    bank = RecoveryPositionBank(max_seeds=3, separation_m=0.1, max_age_epochs=5)
    bank.update(
        0,
        np.asarray([[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]]),
        np.asarray([0.0, 1.0]),
        reference_position_ecef=np.zeros(3),
        max_reference_distance_m=10.0,
    )
    assert len(bank.positions) == 1
    np.testing.assert_allclose(bank.positions[0], np.zeros(3))


def test_assignment_bank_projects_only_exact_active_generations():
    bank = RecoveryAssignmentBank(
        max_assignments=4, max_age_epochs=10, min_assignment_size=8
    )
    bank.update(0, [_assignment(3), _assignment(4)], [0.0, -1.0])
    active = set(_assignment(0))
    observed = {key[0] for key in active}
    compatible = bank.compatible_assignments(active, observed)
    assert len(compatible) == 2
    assert all(len(assignment) == 8 for assignment in compatible)

    slipped = {((raw_key), 1) for raw_key, _generation in active}
    assert bank.compatible_assignments(slipped, observed) == ()


def test_assignment_bank_expires_and_deduplicates():
    bank = RecoveryAssignmentBank(
        max_assignments=2, max_age_epochs=2, min_assignment_size=8
    )
    bank.update(0, [_assignment(1), _assignment(1), _assignment(2)], [0.0, 1.0, -1.0])
    active = set(_assignment(0))
    observed = {key[0] for key in active}
    assert len(bank.compatible_assignments(active, observed)) == 2
    bank.update(3, [_assignment(3)], [0.0])
    compatible = bank.compatible_assignments(active, observed)
    assert len(compatible) == 1
    assert next(iter(compatible[0].values())) >= 5


def test_assignment_bank_rebases_integer_differences_to_new_pivot():
    wavelength = 190293673
    old = {
        (("G01", "G02", wavelength), 0): 5,
        (("G01", "G03", wavelength), 0): 8,
    }
    current = {
        (("G02", "G01", wavelength), 4),
        (("G02", "G03", wavelength), 7),
    }
    bank = RecoveryAssignmentBank(
        max_assignments=4, max_age_epochs=10, min_assignment_size=2
    )
    bank.update(0, [old], [0.0])
    rebased = bank.rebased_assignments(current, [key[0] for key in current])
    assert rebased == (
        {
            (("G02", "G01", wavelength), 4): -5,
            (("G02", "G03", wavelength), 7): 3,
        },
    )
    bank.clear()
    assert bank.rebased_assignments(current, [key[0] for key in current]) == ()


def test_assignment_completion_preserves_stable_and_fills_current_generation():
    raw_keys = tuple(("G01", f"G{sat:02d}", 190293673) for sat in range(2, 12))
    generations = {key: (1 if index >= 4 else 0) for index, key in enumerate(raw_keys)}
    stable = {(raw_keys[index], 0): index + 10 for index in range(4)}
    proposals = complete_versioned_assignment(
        raw_keys,
        generations,
        np.arange(10, dtype=np.float64) + 10.2,
        np.eye(10),
        stable,
        target_size=8,
        n_candidates=2,
    )
    assert len(proposals) == 2
    for assignment, distance in proposals:
        assert len(assignment) == 8
        assert all(assignment[key] == value for key, value in stable.items())
        assert all(key[1] == generations[key[0]] for key in assignment)
        assert np.isfinite(distance)
