from experiments.select_wp31_static_multimode_ddpr import select_multimode_ddpr


def _cluster(reason="compact_multimode_rank_cluster_development"):
    return {"reason": reason, "selected_candidate_ids": [1, 2, 3]}


def _ddpr():
    return {
        "evidence_epochs": 10,
        "candidates": [
            {"candidate_id": 1, "position_ecef": [0.0, 0.0, 0.0], "ddpr_median_abs_m": 0.4},
            {"candidate_id": 2, "position_ecef": [0.2, 0.0, 0.0], "ddpr_median_abs_m": 0.45},
            {"candidate_id": 3, "position_ecef": [1.0, 0.0, 0.0], "ddpr_median_abs_m": 0.8},
        ],
    }


def test_selects_two_compact_ddpr_supported_cluster_members():
    result = select_multimode_ddpr(_cluster(), _ddpr())
    assert result["production_promoted"] is True
    assert result["selected_candidate_ids"] == [1, 2]
    assert result["reason"] == "multimode_ddpr_consensus"


def test_rejects_without_preexisting_multimode_cluster():
    result = select_multimode_ddpr(_cluster("no_eligible_multimode_cluster"), _ddpr())
    assert result["production_promoted"] is False
    assert result["reason"] == "no_multimode_cluster"
