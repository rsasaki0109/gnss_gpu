from __future__ import annotations

from experiments.select_wp138_stability_cppr_consensus import rerank


def _mode(candidate_id: int, values: tuple[float, float, float, float]) -> dict:
    return {
        "candidate_id": candidate_id,
        "cross_refit_disagreement_m": values[0],
        "block_spread_m": values[1],
        "max_cross_basis_carrier_rms_cycles": values[2],
        "cppr_rank_sum": values[3],
    }


def test_stability_cppr_rerank_selects_balanced_unique_mode() -> None:
    fused = {
        "production_input_truth": False,
        "segment": [10, 20],
        "reason": "cross_basis_cppr_family_or_margin_gate_failed",
        "modes": [
            _mode(1, (0.03, 0.03, 0.22, 3.0)),
            _mode(2, (0.02, 0.08, 0.30, 8.0)),
            _mode(3, (0.08, 0.07, 0.32, 9.0)),
        ],
        "gate": {"max_cross_refit_disagreement_m": 0.1},
    }

    result = rerank(fused, max_family_rank_fraction=1.0)

    assert result["accepted"] is True
    assert result["selected_candidate_id"] == 1
    assert result["winner"]["stability_rank_sum"] == 5
    assert "audit" not in str(result)


def test_stability_cppr_rerank_rejects_tied_winners() -> None:
    fused = {
        "production_input_truth": False,
        "segment": [10, 20],
        "reason": "cross_basis_cppr_family_or_margin_gate_failed",
        "modes": [
            _mode(1, (0.01, 0.04, 0.20, 8.0)),
            _mode(2, (0.04, 0.01, 0.80, 1.0)),
        ],
        "gate": {},
    }

    result = rerank(fused, max_family_rank_fraction=1.0)

    assert result["accepted"] is False
    assert result["runner_margin_pass"] is False


def test_stability_cppr_rerank_fails_closed_without_modes() -> None:
    result = rerank(
        {
            "production_input_truth": False,
            "segment": [10, 20],
            "reason": "cppr_evidence_unavailable",
            "modes": [],
        }
    )

    assert result["accepted"] is False
    assert result["reason"] == "stability_cppr_evidence_unavailable"
