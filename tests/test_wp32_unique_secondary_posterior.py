from experiments.select_wp32_unique_secondary_posterior import (
    select_unique_secondary_posterior,
)


def _posterior(reason="compact_rank_posterior_development"):
    return {
        "reason": reason,
        "selected_candidate_ids": [1, 2, 3],
        "selected_score": 0.8,
    }


def _secondary(offset=0.0):
    return {
        "pseudorange_family": "secondary",
        "evidence_epochs": 10,
        "candidates": [
            {
                "candidate_id": index,
                "position_ecef": [offset + 0.1 * index, 0.0, 0.0],
                "ddpr_median_abs_m": 0.35 + 0.02 * index,
            }
            for index in (1, 2, 3)
        ],
    }


def test_selects_only_eligible_parent_top3():
    result = select_unique_secondary_posterior(
        [_posterior("none"), _posterior()], [_secondary(), _secondary(1.0)]
    )
    assert result["selected_parent_id"] == 1
    assert result["selected_candidate_ids"] == [1, 2, 3]


def test_rejects_two_eligible_parents_as_ambiguous():
    result = select_unique_secondary_posterior(
        [_posterior(), _posterior()], [_secondary(), _secondary(1.0)]
    )
    assert result["reason"] == "ambiguous_secondary_posteriors"
    assert result["production_promoted"] is False
