from experiments.select_wp32_static_compact_posterior import (
    select_compact_posterior,
)


def test_compact_posterior_avoids_single_link_chain():
    candidates = [
        {"candidate_id": index, "position_ecef": [0.2 * index, 0.0, 0.0]}
        for index in range(7)
    ]
    wide_lane = [
        {"candidate_id": index, "widelane_median_abs_m_rank": index + 1}
        for index in range(7)
    ]
    temporal = [
        {"candidate_id": index, "carrier_temporal_arc_cauchy_mean_rank": index + 1}
        for index in range(7)
    ]
    result = select_compact_posterior(
        candidates,
        wide_lane,
        temporal,
        ball_radius_m=0.5,
        max_spread_m=0.5,
        min_members=3,
        min_score=0.5,
    )
    assert result["reason"] == "compact_rank_posterior_development"
    assert len(result["selected_candidate_ids"]) < 7
    assert result["selected_spread_m"] <= 0.5


def test_compact_posterior_fails_closed_on_weak_rank_mass():
    candidates = [
        {"candidate_id": index, "position_ecef": [0.1 * index, 0.0, 0.0]}
        for index in range(3)
    ]
    evidence = [
        {
            "candidate_id": index,
            "widelane_median_abs_m_rank": 100,
            "carrier_temporal_arc_cauchy_mean_rank": 100,
        }
        for index in range(3)
    ]
    result = select_compact_posterior(candidates, evidence, evidence)
    assert result["reason"] == "no_eligible_compact_posterior"
