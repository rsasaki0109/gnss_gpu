import numpy as np
import pytest

from gnss_gpu.candidate_3dma import (
    cn0_to_los_probability,
    horizontal_candidates_ecef,
    multipivot_consensus_scores,
    recurrence_vector_scores,
    road_mode_trigger,
    robust_subset_consensus_scores,
    score_candidate_positions,
    solve_four_satellite_position,
    temporal_bias_consistency_scores,
    visibility_mode_cluster_scores,
)


def test_four_satellite_position_reconstructs_position_and_clock():
    truth = np.array([4_000_000.0, 3_000_000.0, 2_000_000.0])
    satellites = _synthetic_satellites(truth)[:4]
    clock = 82_000.0
    pseudoranges = np.linalg.norm(satellites - truth, axis=1) + clock
    solved, solved_clock = solve_four_satellite_position(
        satellites, pseudoranges, truth + np.array([100.0, -50.0, 20.0])
    )
    assert np.allclose(solved, truth, atol=1e-4)
    assert abs(solved_clock - clock) < 1e-4


def test_recurrence_vector_uses_actual_subset_solutions_and_los_projection():
    truth = np.array([4_000_000.0, 3_000_000.0, 2_000_000.0])
    satellites = _synthetic_satellites(truth)[:5]
    clock = 50_000.0
    pseudoranges = np.linalg.norm(satellites - truth, axis=1) + clock
    candidates = np.vstack([truth, truth + np.array([20.0, 0.0, 0.0])])
    predicted_los = np.ones((2, 5), dtype=bool)
    out = recurrence_vector_scores(
        candidates,
        satellites,
        pseudoranges,
        predicted_los,
        truth + np.array([5.0, 2.0, 0.0]),
        sigma_los_m=2.0,
    )
    assert out.subset_indices.shape == (5, 4)
    assert np.max(np.linalg.norm(out.subset_positions_ecef - truth, axis=1)) < 1e-3
    assert np.max(np.abs(out.ranging_errors_m[0])) < 1e-3
    assert out.best_index == 0


def test_recurrence_visibility_probability_rewards_matching_nlos_map():
    truth = np.array([4_000_000.0, 3_000_000.0, 2_000_000.0])
    satellites = _synthetic_satellites(truth)[:6]
    pseudoranges = np.linalg.norm(satellites - truth, axis=1) + 60_000.0
    pseudoranges[[0, 1]] += 25.0
    candidates = np.vstack([truth, truth + np.array([15.0, 0.0, 0.0])])
    predicted_los = np.ones((2, 6), dtype=bool)
    predicted_los[0, [0, 1]] = False
    out = recurrence_vector_scores(
        candidates,
        satellites,
        pseudoranges,
        predicted_los,
        truth,
        sigma_los_m=3.0,
        nlos_bias_m=20.0,
        sigma_nlos_m=10.0,
    )
    assert out.scores[0] > out.scores[1]


def _synthetic_satellites(center):
    directions = np.asarray(
        [
            [0.8, 0.1, 0.6],
            [-0.7, 0.2, 0.7],
            [0.1, 0.9, 0.6],
            [0.0, -0.8, 0.7],
            [0.6, 0.6, 0.5],
            [-0.5, -0.6, 0.7],
        ],
        dtype=np.float64,
    )
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    return np.asarray(center)[None, :] + 21_000_000.0 * directions


def test_horizontal_candidates_grid_and_paired_offsets():
    center = np.array([3_900_000.0, 3_400_000.0, 3_700_000.0])
    grid = horizontal_candidates_ecef(center, [-1.0, 1.0], [-2.0, 0.0, 2.0])
    paired = horizontal_candidates_ecef(
        center, [-1.0, 1.0], [-2.0, 2.0], grid=False
    )

    assert grid.shape == (6, 3)
    assert paired.shape == (2, 3)
    assert np.allclose(np.linalg.norm(paired - center, axis=1), np.sqrt(5.0))


def test_clock_free_likelihood_selects_true_horizontal_candidate():
    center = np.array([3_900_000.0, 3_400_000.0, 3_700_000.0])
    candidates = horizontal_candidates_ecef(
        center,
        np.zeros(7),
        np.arange(-3.0, 4.0),
        grid=False,
    )
    true_index = 5
    satellites = _synthetic_satellites(center)
    pseudoranges = np.linalg.norm(satellites - candidates[true_index], axis=1) + 82_000.0
    predicted_los = np.ones((len(candidates), len(satellites)), dtype=bool)

    result = score_candidate_positions(
        candidates,
        satellites,
        pseudoranges,
        predicted_los,
        sigma_los_m=0.25,
        apply_sagnac=False,
    )

    assert result.best_index == true_index
    assert result.probabilities.sum() == pytest.approx(1.0)
    assert result.clock_bias_m[true_index] == pytest.approx(82_000.0, abs=1e-6)
    assert np.max(np.abs(result.innovations_m[true_index])) < 1e-6


def test_nlos_asymmetric_model_rewards_consistent_visibility():
    center = np.array([3_900_000.0, 3_400_000.0, 3_700_000.0])
    candidates = np.repeat(center[None, :], 2, axis=0)
    satellites = _synthetic_satellites(center)
    pseudoranges = np.linalg.norm(satellites - center, axis=1) + 10_000.0
    pseudoranges[-1] += 18.0
    predicted_los = np.ones((2, len(satellites)), dtype=bool)
    predicted_los[1, -1] = False

    result = score_candidate_positions(
        candidates,
        satellites,
        pseudoranges,
        predicted_los,
        sigma_los_m=1.0,
        nlos_bias_m=18.0,
        sigma_nlos_negative_m=4.0,
        sigma_nlos_positive_m=12.0,
        apply_sagnac=False,
    )

    assert result.best_index == 1
    assert result.pseudorange_scores[1] > result.pseudorange_scores[0]


def test_visibility_and_road_terms_break_equal_geometry_tie():
    center = np.array([3_900_000.0, 3_400_000.0, 3_700_000.0])
    candidates = np.repeat(center[None, :], 2, axis=0)
    satellites = _synthetic_satellites(center)
    pseudoranges = np.linalg.norm(satellites - center, axis=1)
    predicted_los = np.ones((2, len(satellites)), dtype=bool)
    predicted_los[0, 0] = False

    result = score_candidate_positions(
        candidates,
        satellites,
        pseudoranges,
        predicted_los,
        observed_los_probability=np.full(len(satellites), 0.95),
        road_outside_distance_m=[4.0, 0.0],
        apply_sagnac=False,
    )

    assert result.best_index == 1
    assert result.visibility_scores[1] > result.visibility_scores[0]
    assert result.road_scores[1] > result.road_scores[0]


def test_cn0_probability_is_bounded_and_monotonic():
    probability = cn0_to_los_probability([15.0, 32.0, 50.0])

    assert np.all((probability > 0.0) & (probability < 1.0))
    assert np.all(np.diff(probability) > 0.0)
    assert probability[1] == pytest.approx(0.5)


def test_candidate_scorer_rejects_bad_shapes_and_negative_weights():
    candidates = np.ones((2, 3)) * 4_000_000.0
    satellites = np.ones((4, 3)) * 20_000_000.0
    pseudoranges = np.ones(4) * 20_000_000.0
    los = np.ones((2, 4), dtype=bool)

    with pytest.raises(ValueError, match="predicted_los"):
        score_candidate_positions(candidates, satellites, pseudoranges, los[:, :3])
    with pytest.raises(ValueError, match="satellite_weights"):
        score_candidate_positions(
            candidates,
            satellites,
            pseudoranges,
            los,
            satellite_weights=[1.0, 1.0, -1.0, 1.0],
        )
    with pytest.raises(ValueError, match="clock_group_ids"):
        score_candidate_positions(
            candidates,
            satellites,
            pseudoranges,
            los,
            clock_group_ids=[0, 1, 2],
        )


def test_constellation_specific_clocks_are_removed():
    center = np.array([3_900_000.0, 3_400_000.0, 3_700_000.0])
    candidates = horizontal_candidates_ecef(
        center, np.zeros(5), np.arange(-2.0, 3.0), grid=False
    )
    satellites = _synthetic_satellites(center)
    groups = np.array([0, 0, 0, 1, 1, 1])
    clock = np.where(groups == 0, 20_000.0, -35_000.0)
    true_index = 3
    pseudoranges = np.linalg.norm(satellites - candidates[true_index], axis=1) + clock

    result = score_candidate_positions(
        candidates,
        satellites,
        pseudoranges,
        np.ones((len(candidates), len(satellites)), dtype=bool),
        clock_group_ids=groups,
        sigma_los_m=0.25,
        apply_sagnac=False,
    )

    assert result.best_index == true_index
    assert np.max(np.abs(result.innovations_m[true_index])) < 1e-6


def test_multipivot_consensus_tolerates_one_low_quality_outlier():
    innovations = np.array(
        [
            [0.0, 0.1, -0.1, 0.2, 0.0, 18.0],
            [2.5, -2.0, 2.0, -2.5, 2.2, -2.2],
        ]
    )
    observed_los = np.array([0.95, 0.95, 0.9, 0.9, 0.85, 0.1])

    scores = multipivot_consensus_scores(
        innovations,
        np.ones_like(innovations, dtype=bool),
        observed_los_probability=observed_los,
        scale_m=1.0,
        max_pivots=5,
    )

    assert int(np.argmax(scores)) == 0


def test_multipivot_penalizes_only_candidates_without_a_los_pair():
    innovations = np.zeros((2, 4), dtype=np.float64)
    predicted_los = np.array(
        [[True, True, True, True], [True, False, False, False]], dtype=bool
    )

    scores = multipivot_consensus_scores(
        innovations, predicted_los, max_pivots=4
    )

    assert np.all(np.isfinite(scores))
    assert scores[0] > scores[1]


def test_temporal_bias_consistency_removes_satellite_intercepts():
    sat_ids = [["G01", "G02", "G03", "G04"] for _ in range(12)]
    fixed_bias = np.array([12.0, -7.0, 4.0, 20.0])
    innovations = []
    for epoch in range(12):
        correct = fixed_bias
        wrong = fixed_bias + epoch * np.array([0.5, -0.3, 0.2, -0.4])
        innovations.append(np.stack([correct, wrong], axis=0))

    scores = temporal_bias_consistency_scores(
        innovations,
        sat_ids,
        scale_m=1.0,
        min_epochs_per_satellite=8,
    )

    assert int(np.argmax(scores)) == 0
    assert scores[0] == pytest.approx(0.0)


def test_robust_subset_finds_clean_consensus_among_outliers():
    innovations = np.array(
        [
            [0.0, 0.1, -0.1, 0.2, 16.0, -20.0, 25.0],
            [2.5, -2.5, 2.0, -2.0, 2.2, -2.2, 2.4],
        ]
    )
    observed_los = np.array([0.95, 0.94, 0.93, 0.92, 0.5, 0.4, 0.3])

    scores = robust_subset_consensus_scores(
        innovations,
        np.ones_like(innovations, dtype=bool),
        observed_los_probability=observed_los,
        scale_m=1.0,
        subset_size=4,
        max_satellites=7,
        subset_quantile=0.0,
    )

    assert int(np.argmax(scores)) == 0


def test_visibility_cluster_prefers_supported_mode_over_singleton_peak():
    scores = np.array(
        [
            10.0, 0.0, 0.0,
            0.0, 9.5, 9.5,
            0.0, 9.5, 9.5,
        ]
    )
    los = np.ones((9, 5), dtype=bool)
    los[0, 0] = False

    adjusted = visibility_mode_cluster_scores(
        scores,
        los,
        (3, 3),
        score_margin=1.0,
        max_hamming=0,
        outside_penalty=5.0,
    )

    assert int(np.argmax(adjusted)) in {4, 5, 7, 8}
    assert adjusted[0] == pytest.approx(5.0)


def test_road_mode_trigger_requires_a_contiguous_source_mismatch():
    distances = [2.6, 2.8, 1.0, 2.7, 2.9, 3.0]

    assert road_mode_trigger(
        distances, min_distance_m=2.5, min_contiguous_epochs=3
    )
    assert not road_mode_trigger(
        distances, min_distance_m=2.5, min_contiguous_epochs=4
    )


def test_road_mode_trigger_treats_missing_distance_as_a_break():
    assert not road_mode_trigger(
        [3.0, np.nan, 3.0], min_distance_m=2.5, min_contiguous_epochs=2
    )


def test_road_mode_trigger_requires_the_grid_to_reach_the_road():
    assert road_mode_trigger(
        [3.0, 3.1],
        closest_candidate_road_distances_m=[0.1, 0.2],
        min_contiguous_epochs=2,
    )
    assert not road_mode_trigger(
        [8.0, 8.1],
        closest_candidate_road_distances_m=[5.0, 5.1],
        min_contiguous_epochs=2,
    )
