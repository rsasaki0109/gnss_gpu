from experiments.select_wp31_static_multimode_cluster import select_multimode_cluster


def test_cluster_beats_isolated_rank_winner_only_with_member_gate():
    candidates = [
        {"candidate_id": 0, "position_ecef": [0.0, 0.0, 0.0]},
        {"candidate_id": 1, "position_ecef": [10.0, 0.0, 0.0]},
        {"candidate_id": 2, "position_ecef": [10.2, 0.0, 0.0]},
        {"candidate_id": 3, "position_ecef": [10.1, 0.2, 0.0]},
    ]
    wide = [
        {"candidate_id": 0, "widelane_median_abs_m_rank": 1},
        {"candidate_id": 1, "widelane_median_abs_m_rank": 3},
        {"candidate_id": 2, "widelane_median_abs_m_rank": 4},
        {"candidate_id": 3, "widelane_median_abs_m_rank": 5},
    ]
    temporal = [
        {"candidate_id": 0, "carrier_temporal_arc_cauchy_mean_rank": 1},
        {"candidate_id": 1, "carrier_temporal_arc_cauchy_mean_rank": 4},
        {"candidate_id": 2, "carrier_temporal_arc_cauchy_mean_rank": 5},
        {"candidate_id": 3, "carrier_temporal_arc_cauchy_mean_rank": 3},
    ]
    result = select_multimode_cluster(candidates, wide, temporal, min_cluster_score=0.5)
    assert result["selected_candidate_ids"] == [1, 2, 3]
    assert result["reason"] == "compact_multimode_rank_cluster_development"


def test_cluster_fails_closed_without_three_members():
    candidates = [{"candidate_id": 0, "position_ecef": [0.0, 0.0, 0.0]}]
    wide = [{"candidate_id": 0, "widelane_median_abs_m_rank": 1}]
    temporal = [{"candidate_id": 0, "carrier_temporal_arc_cauchy_mean_rank": 1}]
    result = select_multimode_cluster(candidates, wide, temporal)
    assert result["selected_candidate_ids"] == []
