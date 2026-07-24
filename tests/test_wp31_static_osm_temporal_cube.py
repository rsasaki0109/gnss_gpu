from __future__ import annotations

from experiments.select_wp31_static_osm_temporal_cube import select_cube_consensus
from experiments.run_wp29_tdcp_anchor_smoother import _load_static_position_override


def test_selects_stable_cube_candidate() -> None:
    candidates = [{"candidate_id": 1, "position_ecef": [1, 0, 0], "applied": True, "reason": "converged"}, {"candidate_id": 2, "position_ecef": [2, 0, 0], "applied": True, "reason": "converged"}]
    road = [{"candidate_id": row["candidate_id"], "position_ecef": row["position_ecef"], "road_distance_m": 0.5} for row in candidates]
    integrity = [{"candidate_id": 1, "position_ecef": [1, 0, 0], "carrier_cauchy_mean": 0.1, "carrier_temporal_arc_cauchy_mean": 0.1, "carrier_temporal_m5_s0010": 0.1, "carrier_temporal_m10_s0030": 0.2, "carrier_temporal_m20_s0050": 1.0}, {"candidate_id": 2, "position_ecef": [2, 0, 0], "carrier_cauchy_mean": 1.0, "carrier_temporal_arc_cauchy_mean": 1.0, "carrier_temporal_m5_s0010": 1.0, "carrier_temporal_m10_s0030": 2.0, "carrier_temporal_m20_s0050": 0.1}]
    result = select_cube_consensus(candidates, road, integrity, [0.3, 0.5, 0.7, 0.9, 1.1], min_grid_scores=3, min_win_gap=1)
    assert result["reason"] == "gsi_osm_carrier_temporal_cube_consensus"
    assert result["selected_candidate_id"] == 1


def test_rejects_metric_disagreement() -> None:
    candidates = [{"candidate_id": 1, "position_ecef": [1, 0, 0], "applied": True, "reason": "converged"}, {"candidate_id": 2, "position_ecef": [2, 0, 0], "applied": True, "reason": "converged"}]
    road = [{"candidate_id": row["candidate_id"], "position_ecef": row["position_ecef"], "road_distance_m": 0.5} for row in candidates]
    integrity = [{"candidate_id": 1, "position_ecef": [1, 0, 0], "carrier_cauchy_mean": 1.0, "carrier_temporal_arc_cauchy_mean": 0.1, "carrier_temporal_m5_s0010": 0.1, "carrier_temporal_m10_s0030": 0.2, "carrier_temporal_m20_s0050": 1.0}, {"candidate_id": 2, "position_ecef": [2, 0, 0], "carrier_cauchy_mean": 0.1, "carrier_temporal_arc_cauchy_mean": 1.0, "carrier_temporal_m5_s0010": 1.0, "carrier_temporal_m10_s0030": 2.0, "carrier_temporal_m20_s0050": 0.1}]
    result = select_cube_consensus(candidates, road, integrity, [0.3, 0.5, 0.7, 0.9, 1.1], min_grid_scores=3, min_win_gap=1)
    assert result["reason"] == "carrier_metrics_disagree"


def test_position_override_accepts_cube_consensus(tmp_path) -> None:
    import json

    path = tmp_path / "cube.json"
    path.write_text(json.dumps({"selected_candidate_id": 25, "reason": "gsi_osm_carrier_temporal_cube_consensus", "segment": [9218, 9540], "position_ecef": [1.0, 2.0, 3.0]}), encoding="utf-8")
    span = _load_static_position_override(path)
    assert span[0:2] == (9218, 9540)
    assert span[3:] == (25, "gsi_osm_carrier_temporal_cube_consensus")
