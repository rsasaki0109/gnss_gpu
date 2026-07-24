from experiments.build_wp32_cluster_resample_parents import proposal_parents


def _cluster(score: float, members: int, spread: float, x: float):
    return {
        "score": score,
        "members": members,
        "spread_m": spread,
        "position_ecef": [x, 2.0, 3.0],
        "member_ids": list(range(members)),
    }


def test_proposal_parents_keeps_all_top_weak_components_without_truth():
    rows = [
        _cluster(0.41, 3, 0.8, 1.0),
        _cluster(0.7, 4, 0.6, 2.0),
        _cluster(0.39, 5, 0.2, 3.0),
        _cluster(0.5, 2, 0.1, 4.0),
    ]
    result = proposal_parents(rows, min_members=3, min_score=0.4, max_parents=3)
    assert [row["position_ecef"][0] for row in result] == [2.0, 1.0]
    assert [row["candidate_id"] for row in result] == [0, 1]
