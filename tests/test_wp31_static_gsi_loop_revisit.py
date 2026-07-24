from __future__ import annotations

from experiments.select_wp31_static_gsi_loop_revisit import select_loop_revisit
from experiments.run_wp29_tdcp_anchor_smoother import _load_static_position_override


def _road(candidate_id: int, position: list[float], east: float, nearest: float) -> dict:
    return {
        "candidate_id": candidate_id,
        "position_ecef": position,
        "road_offset_east_m": east,
        "road_offset_north_m": 0.0,
        "nearest_road_east_m": nearest,
        "nearest_road_north_m": 0.0,
    }


def test_selects_matching_height_and_road_revisit_parent() -> None:
    reference = [{"candidate_id": 0, "position_ecef": [6378177.0, 0.0, 0.0]}]
    corrected = [
        {"candidate_id": 7, "position_ecef": [6378177.0, 1.0, 0.0]},
        {"candidate_id": 8, "position_ecef": [6378177.0, 8.0, 0.0]},
    ]
    height = {
        "reason": "weak_absolute_height_match",
        "best_candidate_id": 7,
        "runner_gap_m": 1.2,
        "best_height_residual_m": 1.4,
        "predicted_antenna_ellipsoid_height_m": 40.0,
    }
    refined = {
        "seed_center_ecef": corrected[0]["position_ecef"],
        "candidates": [{
            "candidate_id": 24,
            "position_ecef": [6378177.0, 1.05, 0.0],
            "applied": True,
            "reason": "converged",
            "update_norm_m": 0.05,
            "final_norm_rms": 0.2,
            "n_observations": 500,
        }],
    }
    result = select_loop_revisit(
        reference, 0, [_road(0, reference[0]["position_ecef"], 100.0, 5.0)],
        corrected,
        [_road(7, corrected[0]["position_ecef"], 101.0, 5.5), _road(8, corrected[1]["position_ecef"], 108.0, 10.0)],
        height, refined,
    )
    assert result["reason"] == "gsi_height_osm_loop_revisit_unique"
    assert result["selected_candidate_id"] == 7


def test_rejects_height_road_disagreement() -> None:
    reference = [{"candidate_id": 0, "position_ecef": [6378177.0, 0.0, 0.0]}]
    corrected = [
        {"candidate_id": 7, "position_ecef": [6378177.0, 1.0, 0.0]},
        {"candidate_id": 8, "position_ecef": [6378177.0, 2.0, 0.0]},
    ]
    height = {"reason": "weak_absolute_height_match", "best_candidate_id": 7, "runner_gap_m": 1.2, "best_height_residual_m": 1.4}
    result = select_loop_revisit(
        reference, 0, [_road(0, reference[0]["position_ecef"], 100.0, 5.0)], corrected,
        [_road(7, corrected[0]["position_ecef"], 108.0, 6.0), _road(8, corrected[1]["position_ecef"], 101.0, 5.5)],
        height, {},
    )
    assert result["reason"] == "height_road_parent_disagree"


def test_position_override_accepts_loop_revisit_reason(tmp_path) -> None:
    import json

    path = tmp_path / "loop.json"
    path.write_text(json.dumps({
        "selected_candidate_id": 7,
        "reason": "gsi_height_osm_loop_revisit_unique",
        "segment": [11844, 11924],
        "position_ecef": [1.0, 2.0, 3.0],
    }), encoding="utf-8")
    span = _load_static_position_override(path)
    assert span[0:2] == (11844, 11924)
    assert span[3:] == (7, "gsi_height_osm_loop_revisit_unique")
