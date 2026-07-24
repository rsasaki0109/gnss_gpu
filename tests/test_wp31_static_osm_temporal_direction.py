from __future__ import annotations

from experiments.select_wp31_static_osm_temporal_direction import select_direction_consensus
from experiments.run_wp29_tdcp_anchor_smoother import _load_static_position_override


def _candidate(candidate_id: int, direction: int, radius: float, x: float) -> dict:
    return {"candidate_id": candidate_id, "proposal_kind": "horizontal_ring", "direction_index": direction, "radius_m": radius, "position_ecef": [x, 0.0, 0.0]}


def test_selects_stable_complete_direction_cluster() -> None:
    shell = [_candidate(1, 0, 1.0, 1.0), _candidate(2, 0, 1.2, 1.2), _candidate(3, 1, 1.0, -1.0), _candidate(4, 1, 1.2, -1.2)]
    road = [{"candidate_id": row["candidate_id"], "position_ecef": row["position_ecef"], "road_distance_m": 0.5} for row in shell]
    integrity = []
    for row in shell:
        preferred = row["direction_index"] == 0
        integrity.append({"candidate_id": row["candidate_id"], "position_ecef": row["position_ecef"], "carrier_temporal_m5_s0010": 0.1 if preferred else 1.0, "carrier_temporal_m10_s0030": 0.2 if preferred else 1.1, "carrier_temporal_m20_s0050": 0.3 if preferred else 0.05})
    result = select_direction_consensus(shell, road, integrity, [0.4, 0.6, 0.8], min_grid_scores=3, min_win_gap=1)
    assert result["reason"] == "gsi_osm_carrier_temporal_direction_consensus"
    assert result["selected_cluster_candidate_ids"] == [1, 2]


def test_rejects_unstable_direction() -> None:
    shell = [_candidate(1, 0, 1.0, 1.0), _candidate(2, 1, 1.0, -1.0)]
    road = [{"candidate_id": row["candidate_id"], "position_ecef": row["position_ecef"], "road_distance_m": 0.5} for row in shell]
    integrity = [{"candidate_id": 1, "position_ecef": [1.0, 0.0, 0.0], "carrier_temporal_m5_s0010": 0.1, "carrier_temporal_m10_s0030": 1.0}, {"candidate_id": 2, "position_ecef": [-1.0, 0.0, 0.0], "carrier_temporal_m5_s0010": 1.0, "carrier_temporal_m10_s0030": 0.1}]
    result = select_direction_consensus(shell, road, integrity, [0.4, 0.6, 0.8], min_grid_scores=2, min_win_gap=1)
    assert result["reason"] == "temporal_direction_not_stable"


def test_ignores_shell_center_without_direction() -> None:
    shell = [{"candidate_id": 0, "proposal_kind": "horizontal_shell_center", "position_ecef": [0.0, 0.0, 0.0]}, _candidate(1, 0, 1.0, 1.0), _candidate(2, 0, 1.2, 1.2), _candidate(3, 1, 1.0, -1.0), _candidate(4, 1, 1.2, -1.2)]
    road = [{"candidate_id": row["candidate_id"], "position_ecef": row["position_ecef"], "road_distance_m": 0.5} for row in shell]
    integrity = [{"candidate_id": row["candidate_id"], "position_ecef": row["position_ecef"], "carrier_temporal_m5_s0010": 0.0 if row["candidate_id"] == 0 else (0.1 if row["candidate_id"] in (1, 2) else 1.0), "carrier_temporal_m10_s0030": 0.0 if row["candidate_id"] == 0 else (0.2 if row["candidate_id"] in (1, 2) else 1.1), "carrier_temporal_m20_s0050": 0.0 if row["candidate_id"] == 0 else (1.0 if row["candidate_id"] in (1, 2) else 0.1)} for row in shell]
    result = select_direction_consensus(shell, road, integrity, [0.4, 0.6, 0.8], min_grid_scores=3, min_win_gap=1)
    assert result["selected_cluster_candidate_ids"] == [1, 2]


def test_position_override_accepts_direction_consensus(tmp_path) -> None:
    import json

    path = tmp_path / "direction.json"
    path.write_text(json.dumps({"selected_candidate_id": 68, "reason": "gsi_osm_carrier_temporal_direction_consensus", "segment": [9883, 10248], "position_ecef": [1.0, 2.0, 3.0]}), encoding="utf-8")
    span = _load_static_position_override(path)
    assert span[0:2] == (9883, 10248)
    assert span[3:] == (68, "gsi_osm_carrier_temporal_direction_consensus")
