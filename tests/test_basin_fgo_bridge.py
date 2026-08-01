from __future__ import annotations

import json

import numpy as np
import pytest

from gnss_gpu.ambiguity_basin_pf import (
    AmbiguityBasinParticleFilter,
    BasinKalmanState,
)
from gnss_gpu.basin_fgo_bridge import (
    ConditionedFGOHypothesis,
    apply_conditioned_batch,
    evaluate_conditioned_fgo_batch,
    parse_native_fgo_hypotheses,
    parse_native_fgo_jsonl,
    spawn_native_fgo_candidates,
    transition_native_fgo_candidates,
)


def _hypothesis(basin_id: str, target: float, *, column: int = 0):
    design = np.zeros((1, 6), dtype=np.float64)
    design[0, column] = 1.0
    return ConditionedFGOHypothesis(basin_id, design, np.array([target]), 1.0)


def _two_basin_filter() -> AmbiguityBasinParticleFilter:
    pf = AmbiguityBasinParticleFilter(
        min_fixed_ambiguities=0,
        fix_min_streak=1,
        dedup_position_radius_m=0.1,
    )
    states = [
        BasinKalmanState.from_position(np.array([0.0, 0.0, 0.0]), np.eye(3)),
        BasinKalmanState.from_position(np.array([5.0, 0.0, 0.0]), np.eye(3)),
    ]
    pf.spawn([{}, {}], states)
    return pf


def test_common_pattern_uses_one_factorization_and_multi_rhs() -> None:
    batch = evaluate_conditioned_fgo_batch(
        np.eye(6),
        np.zeros(6),
        np.zeros(6),
        [_hypothesis("correct", 0.0), _hypothesis("wrong", 4.0)],
    )
    assert batch.all_succeeded
    assert batch.factorization_count == 1
    assert batch.rhs_columns == 2
    correct, wrong = batch.hypotheses
    np.testing.assert_allclose(correct.delta, np.zeros(6))
    assert wrong.delta is not None
    assert wrong.delta[0] == pytest.approx(2.0)
    assert correct.relative_log_evidence - wrong.relative_log_evidence == pytest.approx(4.0)
    np.testing.assert_allclose(correct.covariance, np.diag([0.5, 1, 1, 1, 1, 1]))


def test_different_constraint_patterns_are_factored_separately() -> None:
    batch = evaluate_conditioned_fgo_batch(
        np.eye(6),
        np.zeros(6),
        np.zeros(6),
        [_hypothesis("x", 1.0, column=0), _hypothesis("y", 1.0, column=1)],
    )
    assert batch.all_succeeded
    assert batch.factorization_count == 2
    assert batch.rhs_columns == 2


def test_non_positive_definite_group_fails_explicitly() -> None:
    normal = np.eye(6)
    normal[1, 1] = -1.0
    batch = evaluate_conditioned_fgo_batch(
        normal,
        np.zeros(6),
        np.zeros(6),
        [_hypothesis("bad", 0.0)],
    )
    assert not batch.all_succeeded
    assert batch.rhs_columns == 0
    assert batch.hypotheses[0].failure_reason == "non_positive_definite_conditioned_normal"


def test_apply_is_atomic_and_updates_conditional_states_and_weights() -> None:
    pf = _two_basin_filter()
    basin_ids = [basin.basin_id for basin in pf.basins]
    batch = evaluate_conditioned_fgo_batch(
        np.eye(6),
        np.zeros(6),
        np.zeros(6),
        [_hypothesis(basin_ids[0], 0.0), _hypothesis(basin_ids[1], 4.0)],
    )
    assert apply_conditioned_batch(
        pf, batch, navigation_indices=(0, 1, 2, 3, 4, 5)
    )
    posterior = pf.posterior_snapshot()
    assert posterior.gamma > 0.98
    assert pf.basins[0].conditional.covariance[0, 0] == pytest.approx(0.5)


def test_incomplete_batch_does_not_mutate_filter() -> None:
    pf = _two_basin_filter()
    before = [
        (basin.conditional.mean.copy(), basin.log_weight) for basin in pf.basins
    ]
    batch = evaluate_conditioned_fgo_batch(
        np.eye(6),
        np.zeros(6),
        np.zeros(6),
        [_hypothesis(pf.basins[0].basin_id, 2.0)],
    )
    assert not apply_conditioned_batch(
        pf, batch, navigation_indices=(0, 1, 2, 3, 4, 5)
    )
    for basin, (mean, log_weight) in zip(pf.basins, before):
        np.testing.assert_array_equal(basin.conditional.mean, mean)
        assert basin.log_weight == log_weight


@pytest.mark.parametrize("temperature", [0.0, -0.1, 1.1, np.nan])
def test_invalid_likelihood_temperature_is_rejected(temperature: float) -> None:
    pf = _two_basin_filter()
    ids = [basin.basin_id for basin in pf.basins]
    batch = evaluate_conditioned_fgo_batch(
        np.eye(6),
        np.zeros(6),
        np.zeros(6),
        [_hypothesis(ids[0], 0.0), _hypothesis(ids[1], 1.0)],
    )
    with pytest.raises(ValueError, match="likelihood_temperature"):
        apply_conditioned_batch(
            pf,
            batch,
            navigation_indices=(0, 1, 2, 3, 4, 5),
            likelihood_temperature=temperature,
        )


def _native_row(rank: int, fixed_cycles: int, evidence: float) -> dict:
    return {
        "rank": rank,
        "group_index": 0,
        "group_rank": rank,
        "evaluated": True,
        "position_ecef": [float(rank), 2.0, 3.0],
        "velocity_valid": False,
        "velocity_ecef_mps": [0.0, 0.0, 0.0],
        "position_covariance_valid": True,
        "position_covariance_m2": [
            0.25,
            0.0,
            0.0,
            0.0,
            0.36,
            0.0,
            0.0,
            0.0,
            0.49,
        ],
        "relative_log_evidence": evidence,
        "incremental_log_likelihood": evidence,
        "incremental_likelihood_rows": 8,
        "fixed_integers": [
            {
                "satellite": "G02",
                "reference_satellite": "G01",
                "signal": 0,
                "segment_index": 3,
                "reference_segment_index": 2,
                "wavelength_m": 0.1902936728,
                "fixed_cycles": fixed_cycles,
            }
        ],
    }


def test_native_fgo_details_create_versioned_integer_basin_births() -> None:
    payload = {
        "multisd_validation_hypothesis_details": [
            _native_row(0, 8, -1.0),
            _native_row(1, 9, -5.0),
        ]
    }
    candidates = parse_native_fgo_hypotheses(payload, group_index=0)
    assert len(candidates) == 2
    key = next(iter(candidates[0].assignment))
    assert key[0] == ("G01", "G02", 190293673)
    assert key[1] == (3 << 32) | 2

    pf = AmbiguityBasinParticleFilter(
        min_fixed_ambiguities=1,
        dedup_position_radius_m=0.1,
    )
    assert (
        spawn_native_fgo_candidates(
            pf,
            candidates,
            prior_mass=1.0,
            fallback_velocity_ecef_mps=np.array([1.0, 0.0, 0.0]),
        )
        == 2
    )
    assert len(pf.basins) == 2
    assert pf.posterior_snapshot().gamma > 0.98
    assert all(
        basin.proposal_sources[0].startswith("native_fgo:g0:r")
        for basin in pf.basins
    )


def test_native_fgo_parser_rejects_cross_group_evidence_mix() -> None:
    payload = {
        "multisd_validation_hypothesis_details": [_native_row(0, 8, -1.0)]
    }
    first = parse_native_fgo_hypotheses(payload, group_index=0)[0]
    second = type(first)(
        group_index=1,
        rank=first.rank,
        assignment=first.assignment,
        position_ecef_m=first.position_ecef_m,
        position_covariance_m2=first.position_covariance_m2,
        velocity_ecef_mps=first.velocity_ecef_mps,
        relative_log_evidence=first.relative_log_evidence,
        incremental_likelihood_rows=first.incremental_likelihood_rows,
        source_id="native_fgo:g1:r0",
        validation_pass=first.validation_pass,
    )
    with pytest.raises(ValueError, match="across PAR groups"):
        spawn_native_fgo_candidates(
            AmbiguityBasinParticleFilter(),
            [first, second],
            prior_mass=1.0,
            fallback_velocity_ecef_mps=np.zeros(3),
        )


def test_native_basin_jsonl_selects_epoch_without_truth(tmp_path) -> None:
    first = _native_row(0, 8, -1.0)
    first.update({"schema": "gnsspp_multisd_basin_v1", "epoch_index": 10})
    second = _native_row(0, 9, -2.0)
    second.update({"schema": "gnsspp_multisd_basin_v1", "epoch_index": 11})
    path = tmp_path / "basins.jsonl"
    path.write_text(
        "\n".join([json.dumps(first), json.dumps(second)]) + "\n",
        encoding="utf-8",
    )
    selected = parse_native_fgo_jsonl(path, epoch_index=11, group_index=0)
    assert len(selected) == 1
    assert next(iter(selected[0].assignment.values())) == 9


def test_native_fgo_transition_accumulates_lineage_and_penalizes_conflict() -> None:
    pf = AmbiguityBasinParticleFilter(
        min_fixed_ambiguities=1,
        dedup_position_radius_m=0.1,
    )
    epoch0 = parse_native_fgo_hypotheses(
        {
            "multisd_validation_hypothesis_details": [
                _native_row(0, 8, -1.0),
                _native_row(1, 9, -2.0),
            ]
        },
        group_index=0,
    )
    first = transition_native_fgo_candidates(
        pf,
        epoch0,
        fallback_velocity_ecef_mps=np.zeros(3),
    )
    assert first.resulting_basins == 2
    parent_ids = {basin.basin_id for basin in pf.basins}

    epoch1 = parse_native_fgo_hypotheses(
        {
            "multisd_validation_hypothesis_details": [
                _native_row(0, 8, -0.5),
                _native_row(1, 10, -1.0),
            ]
        },
        group_index=0,
    )
    second = transition_native_fgo_candidates(
        pf,
        epoch1,
        fallback_velocity_ecef_mps=np.zeros(3),
        parents_per_candidate=2,
        integer_conflict_log_penalty=8.0,
    )
    assert second.parent_child_branches == 4
    assert second.minimum_conflicts == 0
    assert second.maximum_conflicts == 1
    assert all(basin.parent_basin_id in parent_ids for basin in pf.basins)
    assert next(iter(pf.map_basin().assignment_dict.values())) == 8


def test_native_fgo_transition_prefers_kinematically_consistent_parent() -> None:
    assignment = {(('G01', 'G02', 190293673), 0): 8}
    pf = AmbiguityBasinParticleFilter(
        min_fixed_ambiguities=1,
        dedup_position_radius_m=0.1,
    )
    pf.spawn(
        [assignment, assignment],
        [
            BasinKalmanState.from_position(np.array([0.0, 0.0, 0.0]), np.eye(3)),
            BasinKalmanState.from_position(np.array([10.0, 0.0, 0.0]), np.eye(3)),
        ],
    )
    expected_parent = max(pf.basins, key=lambda basin: basin.conditional.mean[0])
    row = _native_row(0, 8, 0.0)
    row["position_ecef"] = [9.9, 0.0, 0.0]
    candidate = parse_native_fgo_hypotheses(
        {"multisd_validation_hypothesis_details": [row]}, group_index=0
    )
    transition_native_fgo_candidates(
        pf,
        candidate,
        fallback_velocity_ecef_mps=np.zeros(3),
        parents_per_candidate=1,
    )
    assert len(pf.basins) == 1
    assert pf.basins[0].parent_basin_id == expected_parent.basin_id
