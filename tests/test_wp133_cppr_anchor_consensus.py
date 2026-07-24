from __future__ import annotations

from experiments.select_wp133_cppr_anchor_consensus import anchor


def _fusion(cppr_rank: int, *, production_truth: bool = False) -> dict:
    winner = {
        "candidate_id": 10,
        "family_ranks": {
            "cross_consensus": 3,
            "carrier_rms": 3,
            "cppr": cppr_rank,
        },
    }
    return {
        "production_input_truth": production_truth,
        "segment": [10, 20],
        "reason": "cross_basis_cppr_family_or_margin_gate_failed",
        "mode_count": 8,
        "winner": winner,
        "runner": {"candidate_id": 1},
        "runner_margin": 3 / 7,
        "modes": [winner, {"candidate_id": 1}],
        "gate": {"max_cross_refit_disagreement_m": 0.1},
    }


def test_cppr_anchor_accepts_top_cppr_with_bounded_other_families() -> None:
    result = anchor(_fusion(1))

    assert result["accepted"] is True
    assert result["selected_candidate_id"] == 10
    assert result["other_family_rank_limit"] == 4
    assert result["reason"] == "unique_cppr_anchor_cross_basis_mode"
    assert "audit" not in str(result)


def test_cppr_anchor_rejects_cross_carrier_winner_without_top_cppr() -> None:
    result = anchor(_fusion(7))

    assert result["accepted"] is False
    assert result["cppr_anchor_pass"] is False
    assert result["reason"] == "cppr_anchor_family_or_margin_gate_failed"


def test_cppr_anchor_fails_closed_without_cppr_evidence() -> None:
    result = anchor(
        {
            "production_input_truth": False,
            "segment": [10, 20],
            "reason": "cppr_evidence_unavailable",
            "mode_count": 0,
        }
    )

    assert result["accepted"] is False
    assert result["reason"] == "cppr_anchor_evidence_unavailable"
