import numpy as np

from gnss_gpu.temporal_ambiguity import (
    TemporalAmbiguityCandidate,
    TemporalAmbiguityConfig,
    TemporalAmbiguityFilter,
)


def _assignment(satellite: str, integer: int = 1, generation: int = 0):
    return (((("G01", satellite, 190000000), generation), integer),)


def _candidate(name, assignment, likelihood, position=0.0, velocity=0.0):
    return TemporalAmbiguityCandidate(
        candidate_id=name,
        assignment=assignment,
        epoch_log_likelihood=likelihood,
        position_ecef=np.array([position, 0.0, 0.0]),
        velocity_ecef=np.array([velocity, 0.0, 0.0]),
    )


def test_persistent_lineage_beats_alternating_single_epoch_distractors():
    tracker = TemporalAmbiguityFilter(
        TemporalAmbiguityConfig(birth_mass=0.02, incompatible_cost=15.0)
    )
    posterior = None
    for epoch in range(6):
        distractor = f"wrong-{epoch}"
        posterior = tracker.step(
            epoch,
            1.0,
            [
                _candidate("correct", _assignment("G02"), -0.5, position=float(epoch), velocity=1.0),
                _candidate(distractor, _assignment(f"G{epoch + 10:02d}"), 0.0, position=20.0),
            ],
        )
    assert posterior is not None
    assert posterior.map_candidate_id == "correct"
    assert posterior.dwell_epochs >= 5
    assert tracker.viterbi_path(4) == ("correct",) * 4


def test_partial_adopt_is_preferred_over_conflicting_assignment():
    tracker = TemporalAmbiguityFilter(
        TemporalAmbiguityConfig(birth_mass=0.01, assignment_change_cost=1.0)
    )
    shared = _assignment("G02")
    tracker.step(0, 0.0, [_candidate("a", shared, 0.0)])
    compatible = shared + _assignment("G03")
    conflicting = _assignment("G02", integer=2)
    posterior = tracker.step(
        1,
        1.0,
        [
            _candidate("compatible", compatible, 0.0),
            _candidate("conflict", conflicting, 0.0),
        ],
    )
    assert posterior.map_candidate_id == "compatible"


def test_generation_change_is_not_exact_stay_and_probabilities_normalize():
    tracker = TemporalAmbiguityFilter()
    tracker.step(0, 0.0, [_candidate("old", _assignment("G02", generation=0), 0.0)])
    posterior = tracker.step(
        1,
        1.0,
        [
            _candidate("new", _assignment("G02", generation=1), 0.0),
            _candidate("other", _assignment("G03"), -1.0),
        ],
    )
    probabilities = np.exp([item.log_probability for item in tracker.hypotheses])
    np.testing.assert_allclose(probabilities.sum(), 1.0)
    assert posterior.dwell_epochs == 1


def test_empty_epoch_resets_active_lineage():
    tracker = TemporalAmbiguityFilter()
    tracker.step(0, 0.0, [_candidate("a", _assignment("G02"), 0.0)])
    empty = tracker.step(1, 1.0, [])
    assert empty.n_candidates == 0
    restarted = tracker.step(2, 1.0, [_candidate("a", _assignment("G02"), 0.0)])
    assert restarted.dwell_epochs == 1


def test_external_displacement_selects_motion_consistent_successor():
    tracker = TemporalAmbiguityFilter(
        TemporalAmbiguityConfig(birth_mass=0.01, incompatible_cost=1.0)
    )
    shared = _assignment("G02")
    tracker.step(0, 0.0, [_candidate("start", shared, 0.0)])
    posterior = tracker.step(
        1,
        1.0,
        [
            _candidate("near", shared + _assignment("G03"), 0.0, position=1.0),
            _candidate("far", shared + _assignment("G04"), 0.2, position=5.0),
        ],
        motion_mode="external",
        external_displacement_ecef_m=np.array([1.0, 0.0, 0.0]),
        external_covariance_m2=np.eye(3) * 0.1**2,
    )
    assert posterior.map_candidate_id == "near"


def test_map_position_ball_is_non_chaining_and_normalized():
    tracker = TemporalAmbiguityFilter()
    tracker.step(
        0,
        0.0,
        [
            _candidate("map", _assignment("G02"), 0.0, position=0.0),
            _candidate("near", _assignment("G03"), -0.2, position=0.4),
            _candidate("chain", _assignment("G04"), -0.4, position=0.8),
        ],
    )
    ball = tracker.map_position_ball(0.5)
    assert ball.map_candidate_id == "map"
    assert ball.n_members == 2
    assert 0.0 < ball.probability < 1.0
    assert 0.0 < ball.mean_position_ecef[0] < 0.4
    assert ball.rms_spread_m > 0.0


def test_map_position_ball_handles_empty_filter():
    ball = TemporalAmbiguityFilter().map_position_ball(0.5)
    assert ball.probability == 0.0
    assert ball.n_members == 0
    assert np.isnan(ball.mean_position_ecef).all()
