from __future__ import annotations

import numpy as np
import pytest

from experiments.run_wp29_tdcp_anchor_smoother import (
    _close_static_anchor_gaps,
    _has_recent_external_position_seed,
    _interval_dt_s,
    _tdcp_doppler_gate_reason,
)

from gnss_gpu.tdcp_anchor_smoother import (
    AnchorCandidateEpoch,
    anchored_viterbi_path,
    constrained_assignment_greedy_path,
    constrained_assignment_viterbi_path,
    constrained_assignment_viterbi_audit,
    constrained_greedy_path,
    constrained_viterbi_audit,
    constrained_viterbi_path,
    interpolate_path_position,
)
from gnss_gpu.tdcp_anchor_smoother import _transition_scores
from gnss_gpu.tdcp_anchor_smoother import _assignment_continuity_score


def test_recent_external_position_seed_provenance_is_age_gated() -> None:
    row = {
        "proposal_sources": (
            "95:external_position:0:1|100:trusted_float_line:2:3"
        )
    }

    assert _has_recent_external_position_seed(row, 100, 5)
    assert not _has_recent_external_position_seed(row, 101, 5)
    assert not _has_recent_external_position_seed(row, 94, 5)


def test_tdcp_doppler_gate_rejects_only_finite_gross_vector_conflict() -> None:
    tdcp = np.array([9.0, 0.0, 0.0])
    doppler = np.array([0.2, 0.0, 0.0])

    assert _tdcp_doppler_gate_reason(
        tdcp, doppler, max_vector_difference_m=0.75
    ) == "tdcp_doppler_vector_conflict"
    assert _tdcp_doppler_gate_reason(
        np.array([0.4, 0.0, 0.0]), doppler, max_vector_difference_m=0.75
    ) is None
    assert _tdcp_doppler_gate_reason(
        tdcp, None, max_vector_difference_m=0.75
    ) is None
    assert _tdcp_doppler_gate_reason(
        tdcp, np.full(3, np.nan), max_vector_difference_m=0.75
    ) is None
    assert _tdcp_doppler_gate_reason(
        tdcp, doppler, max_vector_difference_m=float("inf")
    ) is None


def test_interval_dt_uses_recorded_time_across_dropout_and_nominal_fallback() -> None:
    times = np.array([10.0, 10.2, 24.0, np.nan, 24.4])

    assert _interval_dt_s(times, 1, 0.2) == pytest.approx(0.2)
    assert _interval_dt_s(times, 2, 0.2) == pytest.approx(13.8)
    assert _interval_dt_s(times, 3, 0.2) == pytest.approx(0.2)
    assert _interval_dt_s(times, 4, 0.2) == pytest.approx(0.2)


def test_static_anchor_endpoint_residual_is_applied_only_to_largest_time_gap() -> None:
    displacements = [np.zeros(3) for _ in range(7)]
    displacements[2:] = [np.array([1.0, 0.0, 0.0]) for _ in range(5)]
    times = np.array([0.0, 0.2, 0.4, 2.4, 2.6, 2.8, 3.0])
    spans = [
        (0, 2, np.array([0.0, 0.0, 0.0]), 1, "clear_widelane"),
        (6, 7, np.array([10.0, 0.0, 0.0]), 2, "clear_widelane"),
    ]

    reports = _close_static_anchor_gaps(
        displacements, times, spans, nominal_dt_s=0.2
    )

    assert len(reports) == 1
    assert reports[0]["bridge_epoch"] == 3
    assert reports[0]["raw_endpoint_residual_m"] == pytest.approx(5.0)
    np.testing.assert_allclose(displacements[3], [6.0, 0.0, 0.0])
    np.testing.assert_allclose(sum(displacements[2:7]), [10.0, 0.0, 0.0])


def test_trusted_anchor_and_motion_select_consistent_path_both_directions() -> None:
    epochs = [
        AnchorCandidateEpoch(0, np.array([[0.0, 0.0, 0.0], [0.0, 5.0, 0.0]]), np.zeros(2)),
        AnchorCandidateEpoch(5, np.array([[1.0, 0.0, 0.0], [1.0, 5.0, 0.0]]), np.zeros(2)),
        AnchorCandidateEpoch(10, np.array([[2.0, 0.0, 0.0], [2.0, 5.0, 0.0]]), np.zeros(2)),
    ]

    path = anchored_viterbi_path(
        epochs,
        {(0, 5): np.array([1.0, 0.0, 0.0]), (5, 10): np.array([1.0, 0.0, 0.0])},
        anchor_epoch=5,
        anchor_index=1,
        transition_sigma_m=0.5,
        emission_weight=0.0,
    )

    assert path == {0: 1, 5: 1, 10: 1}


def test_interpolate_path_position_requires_two_sided_support() -> None:
    anchors = {0: np.array([0.0, 0.0, 0.0]), 5: np.array([5.0, 0.0, 0.0])}

    np.testing.assert_allclose(interpolate_path_position(2, anchors), [2.0, 0.0, 0.0])
    np.testing.assert_allclose(interpolate_path_position(5, anchors), [5.0, 0.0, 0.0])
    assert interpolate_path_position(-1, anchors) is None
    assert interpolate_path_position(6, anchors) is None


def test_smoother_rejects_missing_anchor() -> None:
    with pytest.raises(ValueError, match="anchor_epoch"):
        anchored_viterbi_path(
            [AnchorCandidateEpoch(0, np.zeros((1, 3)), np.zeros(1))],
            {},
            anchor_epoch=5,
            anchor_index=0,
            transition_sigma_m=1.0,
            emission_weight=0.0,
        )


def test_multiple_trusted_constraints_are_hard_path_anchors() -> None:
    epochs = [
        AnchorCandidateEpoch(0, np.array([[0.0, 0.0, 0.0], [0.0, 5.0, 0.0]]), np.zeros(2)),
        AnchorCandidateEpoch(5, np.array([[1.0, 0.0, 0.0], [1.0, 5.0, 0.0]]), np.zeros(2)),
        AnchorCandidateEpoch(10, np.array([[2.0, 0.0, 0.0], [2.0, 5.0, 0.0]]), np.zeros(2)),
    ]

    path = constrained_viterbi_path(
        epochs,
        {(0, 5): np.array([1.0, 0.0, 0.0]), (5, 10): np.array([1.0, 0.0, 0.0])},
        constrained_indices={0: 0, 10: 1},
        transition_sigma_m=1.0,
        emission_weight=0.0,
    )

    assert path[0] == 0
    assert path[10] == 1


def test_constrained_viterbi_audit_max_marginal_rewards_motion_path() -> None:
    epochs = [
        AnchorCandidateEpoch(0, np.array([[0.0, 0.0, 0.0], [0.0, 5.0, 0.0]]), np.zeros(2)),
        AnchorCandidateEpoch(5, np.array([[1.0, 0.0, 0.0], [1.0, 5.0, 0.0]]), np.zeros(2)),
        AnchorCandidateEpoch(10, np.array([[2.0, 0.0, 0.0], [2.0, 5.0, 0.0]]), np.zeros(2)),
    ]

    audit = constrained_viterbi_audit(
        epochs,
        {(0, 5): np.array([1.0, 0.0, 0.0]), (5, 10): np.array([1.0, 0.0, 0.0])},
        constrained_indices={0: 0, 10: 0},
        transition_sigma_m=0.5,
        emission_weight=0.0,
    )

    assert audit.max_marginal_relative[5][0] == pytest.approx(0.0)
    assert audit.max_marginal_relative[5][1] < -10.0
    assert np.isneginf(audit.max_marginal_relative[0][1])


def test_greedy_path_tracks_motion_forward_and_preserves_later_constraint() -> None:
    epochs = [
        AnchorCandidateEpoch(0, np.array([[0.0, 0.0, 0.0], [0.0, 5.0, 0.0]]), np.zeros(2)),
        AnchorCandidateEpoch(5, np.array([[1.0, 0.0, 0.0], [1.0, 5.0, 0.0]]), np.zeros(2)),
        AnchorCandidateEpoch(10, np.array([[2.0, 0.0, 0.0], [2.0, 5.0, 0.0]]), np.zeros(2)),
    ]

    path = constrained_greedy_path(
        epochs,
        {(0, 5): np.array([1.0, 0.0, 0.0]), (5, 10): np.array([1.0, 0.0, 0.0])},
        constrained_indices={0: 0, 10: 1},
        transition_sigma_m=0.5,
        emission_weight=0.0,
    )

    assert path == {0: 0, 5: 0, 10: 1}


def test_assignment_greedy_prefers_persistent_integer_branch_over_closer_motion() -> None:
    epochs = [
        AnchorCandidateEpoch(0, np.array([[0.0, 0.0, 0.0]]), np.zeros(1)),
        AnchorCandidateEpoch(
            5,
            np.array([[1.0, 0.3, 0.0], [1.0, 0.0, 0.0]]),
            np.zeros(2),
        ),
        AnchorCandidateEpoch(
            10,
            np.array([[2.0, 0.3, 0.0], [2.0, 0.0, 0.0]]),
            np.zeros(2),
        ),
    ]
    branch = {("G01", "G02", 190): 10, ("G01", "G03", 244): 20}
    conflict = {("G01", "G02", 190): 11, ("G01", "G03", 244): 21}

    path = constrained_assignment_greedy_path(
        epochs,
        {(0, 5): np.array([1.0, 0.0, 0.0]), (5, 10): np.array([1.0, 0.0, 0.0])},
        {0: [branch], 5: [branch, conflict], 10: [branch, conflict]},
        constrained_indices={0: 0},
        transition_sigma_m=0.5,
        emission_weight=0.0,
        assignment_match_bonus=0.5,
        assignment_conflict_penalty=1.0,
    )

    assert path == {0: 0, 5: 0, 10: 0}


def test_assignment_viterbi_uses_future_continuity_to_select_branch() -> None:
    branch_start = {("E01", "E02", 254): 3}
    branch_a = {("G01", "G02", 190): 10}
    branch_b = {("G01", "G02", 190): 11}
    epochs = [
        AnchorCandidateEpoch(0, np.array([[0.0, 0.0, 0.0]]), np.zeros(1)),
        AnchorCandidateEpoch(
            5, np.array([[1.0, 0.0, 0.0], [1.0, 0.1, 0.0]]), np.zeros(2)
        ),
        AnchorCandidateEpoch(10, np.array([[2.0, 0.1, 0.0]]), np.zeros(1)),
    ]

    path = constrained_assignment_viterbi_path(
        epochs,
        {(0, 5): np.array([1.0, 0.0, 0.0]), (5, 10): np.array([1.0, 0.0, 0.0])},
        {0: [branch_start], 5: [branch_a, branch_b], 10: [branch_b]},
        constrained_indices={0: 0, 10: 0},
        transition_sigma_m=1.0,
        emission_weight=0.0,
        assignment_match_bonus=2.0,
        assignment_conflict_penalty=4.0,
    )

    assert path == {0: 0, 5: 1, 10: 0}

    audit = constrained_assignment_viterbi_audit(
        epochs,
        {(0, 5): np.array([1.0, 0.0, 0.0]), (5, 10): np.array([1.0, 0.0, 0.0])},
        {0: [branch_start], 5: [branch_a, branch_b], 10: [branch_b]},
        constrained_indices={0: 0, 10: 0},
        transition_sigma_m=1.0,
        emission_weight=0.0,
        assignment_match_bonus=2.0,
        assignment_conflict_penalty=4.0,
    )
    assert int(np.argmax(audit.max_marginal_relative[5])) == 1


def test_assignment_greedy_validates_constraint_index() -> None:
    epoch = AnchorCandidateEpoch(0, np.zeros((1, 3)), np.zeros(1))

    with pytest.raises(ValueError, match="out of range"):
        constrained_assignment_greedy_path(
            [epoch],
            {},
            {0: [{}]},
            constrained_indices={0: 1},
            transition_sigma_m=1.0,
            emission_weight=0.0,
        )


def test_assignment_continuity_counts_evidence_and_empty_overlap_is_neutral() -> None:
    left = {("G01", "G02", 190): 10, ("G01", "G03", 244): 20}
    one_match = {("G01", "G02", 190): 10, ("G01", "G03", 244): 21}

    assert _assignment_continuity_score(
        left, one_match, match_bonus=2.0, conflict_penalty=4.0
    ) == pytest.approx(-2.0)
    assert _assignment_continuity_score(
        left, {("E01", "E02", 254): 5}, match_bonus=2.0, conflict_penalty=4.0
    ) == 0.0


def test_assignment_reacquisition_uses_flag_only_after_exact_branch_break() -> None:
    old = {
        ("G01", "G02", 190): 10,
        ("G01", "G03", 190): 20,
        ("G01", "G04", 190): 30,
        ("G01", "G05", 190): 40,
    }
    wrong = {key: value + 1 for key, value in old.items()}
    snapshot = {key: value - 1 for key, value in old.items()}
    epochs = [
        AnchorCandidateEpoch(0, np.array([[0.0, 0.0, 0.0]]), np.zeros(1)),
        AnchorCandidateEpoch(
            5, np.array([[1.0, 0.0, 0.0], [1.0, 0.2, 0.0]]), np.zeros(2)
        ),
        AnchorCandidateEpoch(
            10, np.array([[2.0, 0.0, 0.0], [2.0, 0.2, 0.0]]), np.zeros(2)
        ),
    ]

    path = constrained_assignment_greedy_path(
        epochs,
        {(0, 5): np.array([1.0, 0.0, 0.0]), (5, 10): np.array([1.0, 0.0, 0.0])},
        {0: [old], 5: [wrong, snapshot], 10: [wrong, snapshot]},
        constrained_indices={0: 0},
        transition_sigma_m=0.5,
        emission_weight=0.0,
        candidate_reacquisition_flags={0: [False], 5: [False, True], 10: [False, True]},
        reacquisition_min_stable_anchors=1,
    )

    assert path == {0: 0, 5: 1, 10: 1}

    guarded = constrained_assignment_greedy_path(
        epochs,
        {(0, 5): np.array([1.0, 0.0, 0.0]), (5, 10): np.array([1.0, 0.0, 0.0])},
        {0: [old], 5: [wrong, snapshot], 10: [wrong, snapshot]},
        constrained_indices={0: 0},
        transition_sigma_m=0.5,
        emission_weight=0.0,
        candidate_reacquisition_flags={0: [False], 5: [False, True], 10: [False, True]},
        reacquisition_min_stable_anchors=10,
    )

    assert guarded[5] == 0


def test_reacquisition_can_drop_stale_assignment_score_inside_source_pool() -> None:
    old = {("G01", f"G0{sat}", 190): 10 * sat for sat in range(2, 6)}
    stale = dict(old)
    stale[("G01", "G05", 190)] += 1
    fresh = {key: value + 1 for key, value in old.items()}
    epochs = [
        AnchorCandidateEpoch(0, np.array([[0.0, 0.0, 0.0]]), np.zeros(1)),
        AnchorCandidateEpoch(
            5, np.array([[1.0, 0.3, 0.0], [1.0, 0.0, 0.0]]), np.zeros(2)
        ),
    ]

    path = constrained_assignment_greedy_path(
        epochs,
        {(0, 5): np.array([1.0, 0.0, 0.0])},
        {0: [old], 5: [stale, fresh]},
        constrained_indices={0: 0},
        transition_sigma_m=0.5,
        emission_weight=0.0,
        candidate_reacquisition_flags={0: [False], 5: [True, True]},
        reacquisition_min_stable_anchors=1,
        reacquisition_ignore_assignment=True,
    )

    assert path == {0: 0, 5: 1}


def test_reacquisition_dead_reckoning_does_not_accumulate_candidate_snap() -> None:
    old = {("G01", f"G0{sat}", 190): 10 * sat for sat in range(2, 6)}
    shifted = {key: value + 1 for key, value in old.items()}
    shifted_again = {key: value + 2 for key, value in old.items()}
    epochs = [
        AnchorCandidateEpoch(0, np.array([[0.0, 0.0, 0.0]]), np.zeros(1)),
        AnchorCandidateEpoch(5, np.array([[1.0, 0.5, 0.0]]), np.zeros(1)),
        AnchorCandidateEpoch(
            10, np.array([[2.0, 0.0, 0.0], [2.0, 0.5, 0.0]]), np.zeros(2)
        ),
    ]

    path = constrained_assignment_greedy_path(
        epochs,
        {(0, 5): np.array([1.0, 0.0, 0.0]), (5, 10): np.array([1.0, 0.0, 0.0])},
        {0: [old], 5: [shifted], 10: [shifted_again, shifted]},
        constrained_indices={0: 0},
        transition_sigma_m=0.5,
        emission_weight=0.0,
        candidate_reacquisition_flags={0: [False], 5: [True], 10: [True, True]},
        reacquisition_min_stable_anchors=1,
        reacquisition_window_anchors=2,
        reacquisition_ignore_assignment=True,
        reacquisition_dead_reckon=True,
    )

    assert path == {0: 0, 5: 0, 10: 0}


def test_cauchy_transition_penalty_is_robust_to_large_jumps() -> None:
    distances = np.array([0.5, 10.0])
    gaussian = _transition_scores(distances, 0.5, "gaussian")
    cauchy = _transition_scores(distances, 0.5, "cauchy")

    assert cauchy[1] > gaussian[1]
    assert cauchy[0] < 0.0


def test_huber_transition_penalty_lies_between_gaussian_and_cauchy_for_large_jump() -> None:
    distances = np.array([0.25, 10.0])
    gaussian = _transition_scores(distances, 0.5, "gaussian")
    huber = _transition_scores(distances, 0.5, "huber")
    cauchy = _transition_scores(distances, 0.5, "cauchy")

    assert huber[0] == pytest.approx(gaussian[0])
    assert gaussian[1] < huber[1] < cauchy[1]
