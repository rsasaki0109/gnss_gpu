from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from analyze_wp29_static_reanchor_shadow import (
    _assignment_integers,
    offset_seed_candidates,
    parse_ecef,
    recurring_position_candidates,
)
from analyze_wp29_static_grid_integrity_shadow import (
    fixed_satellite_mean,
    temporal_arc_centered_score,
    temporal_window_scores,
    trimmed_satellite_mean,
)
from analyze_wp29_static_grid_osm_shadow import rank_road_distance
from analyze_wp29_static_grid_widelane_shadow import widelane_residual_scores
from select_wp29_static_grid_fusion_shadow import select_static_grid_fusion
from build_wp29_reverse_static_seed_trace import reverse_integrate
from build_wp29_static_grid_basin_trace import select_candidate_ids
from analyze_wp29_zupt_motion_shadow import active_completed_bias, parse_segments
from run_wp29_tdcp_anchor_smoother import (
    _current_epoch_seed_support,
    _load_fusion_static_override,
    _robust_static_velocity_bias,
    _select_static_anchor_candidate,
)


def test_current_epoch_seed_support_ignores_old_and_non_position_sources() -> None:
    row = {"proposal_sources": "1050:1|1050:7|1045:1|1050:assignment:0"}

    assert _current_epoch_seed_support(row, 1050) == 2


def test_recurring_candidates_rank_epoch_coverage_before_mass() -> None:
    positions = {
        epoch: np.asarray([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        for epoch in range(5)
    }
    positions[4] = np.asarray([[0.0, 0.0, 0.0]])

    candidates = recurring_position_candidates(
        positions,
        0,
        5,
        radius_m=0.1,
        sample_stride_epochs=1,
        dedup_radius_m=0.1,
    )

    assert candidates[0]["coverage_epochs"] == 5
    assert np.allclose(candidates[0]["position_ecef"], [0.0, 0.0, 0.0])
    assert candidates[1]["coverage_epochs"] == 4


def test_recurring_candidates_can_be_disabled_for_external_seed_only() -> None:
    positions = {0: np.asarray([[0.0, 0.0, 0.0]])}

    assert recurring_position_candidates(positions, 0, 1, max_candidates=0) == []


def test_assignment_integer_parser_preserves_frequency_family_keys() -> None:
    row = {
        "assignment_json": (
            '[["G01@L1_E1_B1","G02@L1_E1_B1",190293673,0,12],'
            '["G01@L2_E5B_B2","G02@L2_E5B_B2",244210213,0,7]]'
        )
    }

    assert _assignment_integers(row) == {
        ("G01@L1_E1_B1", "G02@L1_E1_B1", 190293673): 12,
        ("G01@L2_E5B_B2", "G02@L2_E5B_B2", 244210213): 7,
    }


def test_cube26_offset_seeds_have_requested_radius() -> None:
    center = np.array([10.0, 20.0, 30.0])
    seeds = offset_seed_candidates(center, (2.0,), directions="cube26")

    assert len(seeds) == 26
    distances = [np.linalg.norm(seed["position_ecef"] - center) for seed in seeds]
    np.testing.assert_allclose(distances, 2.0)


def test_explicit_truth_free_ecef_seed_requires_three_values() -> None:
    np.testing.assert_allclose(parse_ecef("1.0,2.0,3.0"), [1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="exactly three"):
        parse_ecef("1.0,2.0")


def test_satellite_trim_drops_all_rows_incident_to_worst_satellite() -> None:
    rows = [
        (9.0, "G01@L1", "G02@L1", 0),
        (8.0, "G01@L1", "G03@L1", 0),
        (1.0, "E01@L1", "E02@L1", 0),
        (1.0, "E01@L1", "E03@L1", 0),
    ]

    mean, excluded, retained = trimmed_satellite_mean(rows, 1)

    assert excluded == ("G02",)
    assert mean == pytest.approx((8.0 + 1.0 + 1.0) / 3.0)
    assert retained == 3


def test_fixed_satellite_exclusion_uses_physical_id_across_families() -> None:
    rows = [
        (9.0, "G01@L1", "G02@L1", 0),
        (8.0, "G01@L5", "G03@L5", 0),
        (1.0, "E01@L1", "E02@L1", 0),
    ]

    mean, retained = fixed_satellite_mean(rows, ("G01",))

    assert mean == pytest.approx(1.0)
    assert retained == 1


def test_temporal_arc_score_removes_constant_circular_phase_offset() -> None:
    stable = [
        (0.49, "G01@L1", "G02@L1", 0),
        (-0.49, "G01@L1", "G02@L1", 1),
        (0.48, "G01@L1", "G02@L1", 2),
        (-0.48, "G01@L1", "G02@L1", 3),
        (0.50, "G01@L1", "G02@L1", 4),
    ]
    varying = [
        (-0.4, "G01@L1", "G02@L1", 0),
        (-0.2, "G01@L1", "G02@L1", 1),
        (0.0, "G01@L1", "G02@L1", 2),
        (0.2, "G01@L1", "G02@L1", 3),
        (0.4, "G01@L1", "G02@L1", 4),
    ]

    stable_cost, _median, stable_arcs = temporal_arc_centered_score(
        stable, min_samples=5, sigma_cycles=0.1
    )
    varying_cost, _median, varying_arcs = temporal_arc_centered_score(
        varying, min_samples=5, sigma_cycles=0.1
    )

    assert stable_arcs == varying_arcs == 1
    assert stable_cost < varying_cost


def test_temporal_window_scores_do_not_share_phase_centers_across_windows() -> None:
    rows = [
        (0.10, "G01@L1", "G02@L1", 0),
        (0.11, "G01@L1", "G02@L1", 1),
        (0.09, "G01@L1", "G02@L1", 2),
        (0.40, "G01@L1", "G02@L1", 3),
        (0.41, "G01@L1", "G02@L1", 4),
        (0.39, "G01@L1", "G02@L1", 5),
    ]

    scores = temporal_window_scores(
        rows, n_epochs=6, n_windows=2, min_samples=3, sigma_cycles=0.1
    )

    assert len(scores) == 2
    assert max(scores) < 0.01


def test_static_grid_trace_selector_uses_only_truth_free_score() -> None:
    integrity = {
        "candidates": [
            {"candidate_id": 1, "score": 0.3, "final_error_m": 0.1},
            {"candidate_id": 2, "score": 0.1, "final_error_m": 9.0},
            {"candidate_id": 3, "score": 0.2, "final_error_m": 4.0},
        ]
    }

    assert select_candidate_ids(integrity, score_name="score", top_k=2) == [2, 3]


def test_static_grid_osm_rank_uses_only_road_distance() -> None:
    rows = [
        {"candidate_id": 1, "road_distance_m": 3.0, "final_error_m": 0.1},
        {"candidate_id": 2, "road_distance_m": 1.0, "final_error_m": 9.0},
    ]

    assert [row["candidate_id"] for row in rank_road_distance(rows)] == [2, 1]


def test_widelane_residual_scores_reward_absolute_range_consistency() -> None:
    clean = widelane_residual_scores([0.1, -0.2, 0.0], sigma_m=1.0)
    biased = widelane_residual_scores([1.1, 0.8, 1.0], sigma_m=1.0)

    assert clean["widelane_rms_m"] < biased["widelane_rms_m"]
    assert clean["widelane_cauchy_mean"] < biased["widelane_cauchy_mean"]


def _fusion_rows() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    temporal = [
        {"candidate_id": 1, "carrier_temporal_window_mean": 0.1, "carrier_temporal_window_mean_rank": 1},
        {"candidate_id": 2, "carrier_temporal_window_mean": 0.2, "carrier_temporal_window_mean_rank": 2},
    ]
    widelane = [
        {"candidate_id": 1, "widelane_median_abs_m": 1.0, "widelane_median_abs_m_rank": 2},
        {"candidate_id": 2, "widelane_median_abs_m": 0.8, "widelane_median_abs_m_rank": 1},
    ]
    return temporal, widelane


def test_static_grid_fusion_accepts_clear_widelane_winner() -> None:
    temporal, widelane = _fusion_rows()
    widelane[1]["widelane_median_abs_m"] = 0.2

    result = select_static_grid_fusion(
        temporal, widelane, evidence_epochs=7, candidate_pairs=20, fixed_pairs=15
    )

    assert result["selected_candidate_id"] == 2
    assert result["reason"] == "clear_widelane"


def test_static_grid_fusion_accepts_rank_consensus() -> None:
    temporal, widelane = _fusion_rows()
    temporal[0]["carrier_temporal_window_mean_rank"] = 4
    temporal[1]["carrier_temporal_window_mean_rank"] = 8

    result = select_static_grid_fusion(
        temporal, widelane, evidence_epochs=20, candidate_pairs=20, fixed_pairs=15
    )

    assert result["selected_candidate_id"] == 1
    assert result["reason"] == "temporal_widelane_consensus"


def test_static_grid_fusion_rejects_nonfinite_temporal_consensus() -> None:
    temporal, widelane = _fusion_rows()
    for row in temporal:
        row["carrier_temporal_window_mean"] = float("inf")

    result = select_static_grid_fusion(
        temporal, widelane, evidence_epochs=20, candidate_pairs=20, fixed_pairs=15
    )

    assert result["selected_candidate_id"] is None
    assert result["reason"] == "insufficient_consensus_candidates"


def test_static_grid_fusion_accepts_strict_high_evidence_low_fix_consensus() -> None:
    temporal = [
        {"candidate_id": 1, "carrier_temporal_window_mean": 0.1, "carrier_temporal_window_mean_rank": 1},
        {"candidate_id": 2, "carrier_temporal_window_mean": 0.2, "carrier_temporal_window_mean_rank": 5},
    ]
    widelane = [
        {"candidate_id": 1, "widelane_median_abs_m": 0.7, "widelane_median_abs_m_rank": 3},
        {"candidate_id": 2, "widelane_median_abs_m": 0.5, "widelane_median_abs_m_rank": 1},
    ]

    accepted = select_static_grid_fusion(
        temporal, widelane, evidence_epochs=32, candidate_pairs=394, fixed_pairs=188
    )
    rejected_short = select_static_grid_fusion(
        temporal, widelane, evidence_epochs=20, candidate_pairs=394, fixed_pairs=188
    )

    assert accepted["selected_candidate_id"] == 1
    assert accepted["reason"] == "high_evidence_temporal_widelane_consensus"
    assert rejected_short["selected_candidate_id"] is None
    assert rejected_short["reason"] == "insufficient_widelane_fix_rate"


def test_reverse_integrate_subtracts_forward_displacements() -> None:
    positions = reverse_integrate(
        np.array([3.0, 2.0, 0.0]),
        [np.array([1.0, 0.0, 0.0]), np.array([0.0, 2.0, 0.0])],
    )

    np.testing.assert_allclose(positions, [[2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [3.0, 2.0, 0.0]])


def test_zupt_bias_changes_only_after_completed_stop() -> None:
    segments = parse_segments("1:5,10:15")
    biases = (np.array([1.0, 0.0, 0.0]), np.array([2.0, 0.0, 0.0]))

    assert active_completed_bias(4, segments, biases) is None
    np.testing.assert_allclose(active_completed_bias(5, segments, biases), biases[0])
    np.testing.assert_allclose(active_completed_bias(14, segments, biases), biases[0])
    np.testing.assert_allclose(active_completed_bias(15, segments, biases), biases[1])


def test_static_anchor_gate_accepts_clear_low_rms_winner() -> None:
    result = {
        "candidates": [
            {"applied": True, "final_norm_rms": 0.0100},
            {"applied": True, "final_norm_rms": 0.0110},
        ]
    }

    assert (
        _select_static_anchor_candidate(
            result, max_norm_rms=0.015, max_runner_up_ratio=0.97
        )["final_norm_rms"]
        == 0.0100
    )


def test_static_anchor_gate_uses_explicit_fusion_candidate() -> None:
    result = {
        "candidates": [
            {"candidate_id": 1, "applied": True, "final_norm_rms": 0.01},
            {"candidate_id": 7, "applied": True, "final_norm_rms": 1.0},
        ]
    }

    selected = _select_static_anchor_candidate(
        result,
        max_norm_rms=0.015,
        max_runner_up_ratio=0.95,
        selected_candidate_id=7,
    )

    assert selected["candidate_id"] == 7


def test_fusion_static_override_loads_only_accepted_candidate(tmp_path: Path) -> None:
    static_path = tmp_path / "static.json"
    fusion_path = tmp_path / "fusion.json"
    static_path.write_text(
        '{"segment":[10,20],"candidates":['
        '{"candidate_id":7,"applied":true,"position_ecef":[1,2,3]}]}',
        encoding="utf-8",
    )
    fusion_path.write_text(
        '{"selected_candidate_id":7,"reason":"clear_widelane"}',
        encoding="utf-8",
    )

    start, end, position, candidate_id, reason = _load_fusion_static_override(
        static_path, fusion_path
    )

    assert (start, end, candidate_id, reason) == (10, 20, 7, "clear_widelane")
    np.testing.assert_allclose(position, [1, 2, 3])


def test_static_anchor_gate_accepts_all_block_bootstrap_winner() -> None:
    result = {
        "candidates": [
            {
                "applied": True,
                "final_norm_rms": 0.28,
                "bootstrap_wins": 4,
                "bootstrap_norm_rms": [0.27, 0.28, 0.28, 0.29],
            },
            {
                "applied": True,
                "final_norm_rms": 0.29,
                "bootstrap_wins": 0,
                "bootstrap_norm_rms": [0.30, 0.31, 0.31, 0.32],
            },
        ]
    }

    selected = _select_static_anchor_candidate(
        result, max_norm_rms=0.015, max_runner_up_ratio=0.95
    )

    assert selected["bootstrap_wins"] == 4


def test_static_velocity_bias_rejects_large_outlier() -> None:
    samples = [np.array([1.0, -2.0, 0.5]) for _ in range(8)]
    samples.append(np.array([40.0, 30.0, -20.0]))

    bias = _robust_static_velocity_bias(samples)

    np.testing.assert_allclose(bias, [1.0, -2.0, 0.5])


@pytest.mark.parametrize(
    "first,second,message",
    [
        (0.0190, 0.0200, "normalized-RMS"),
        (0.0100, 0.0102, "runner-up"),
    ],
)
def test_static_anchor_gate_rejects_weak_or_ambiguous_winner(
    first: float, second: float, message: str
) -> None:
    result = {
        "candidates": [
            {"applied": True, "final_norm_rms": first},
            {"applied": True, "final_norm_rms": second},
        ]
    }

    with pytest.raises(RuntimeError, match=message):
        _select_static_anchor_candidate(
            result, max_norm_rms=0.015, max_runner_up_ratio=0.97
        )
