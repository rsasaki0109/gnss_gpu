from __future__ import annotations

from experiments.select_wp131_cross_basis_cppr_consensus import select


def _hypothesis(seed_id: int, offset: float, cppr: float) -> dict:
    return {
        "seed_id": seed_id,
        "offset_ecef_m": [offset, 0.0, 0.0],
        "block_offsets_ecef_m": [[offset, 0.0, 0.0]],
        "block_spread_m": 0.1,
        "carrier_rms_cycles": 0.2,
        "cp_pr_consistency": {
            "checked_pairs": 100,
            "bad_pairs": 0,
            "median_abs_innovation_m": cppr,
            "p95_abs_innovation_m": cppr,
        },
    }


def test_cross_basis_cppr_fusion_rejects_consensus_only_distractor() -> None:
    source = {
        "production_input_truth": False,
        "segment": [10, 20],
        "hypotheses": [
            _hypothesis(1, 1.0, 0.1),
            _hypothesis(2, 2.0, 2.0),
            _hypothesis(3, 3.0, 3.0),
        ],
    }
    cross = {
        "production_input_truth": False,
        "segment": [10, 20],
        "candidates": [
            {"source_candidate_id": 1, "consensus_score_m": 0.02,
             "rank0_to_rank2_m": 0.02, "max_carrier_rms_cycles": 0.20,
             "max_block_spread_m": 0.1},
            {"source_candidate_id": 2, "consensus_score_m": 0.01,
             "rank0_to_rank2_m": 0.01, "max_carrier_rms_cycles": 0.25,
             "max_block_spread_m": 0.1},
            {"source_candidate_id": 3, "consensus_score_m": 0.08,
             "rank0_to_rank2_m": 0.08, "max_carrier_rms_cycles": 0.30,
             "max_block_spread_m": 0.1},
        ],
    }

    result = select(source, cross, max_family_rank_fraction=0.67)

    assert result["accepted"] is True
    assert result["selected_candidate_id"] == 1
    assert result["winner"]["family_ranks"] == {
        "cross_consensus": 2,
        "carrier_rms": 1,
        "cppr": 1,
    }
    assert "audit" not in str(result)


def test_cross_basis_cppr_fusion_keeps_absolute_cross_gate() -> None:
    source = {
        "production_input_truth": False,
        "segment": [10, 20],
        "hypotheses": [_hypothesis(1, 1.0, 0.1), _hypothesis(2, 2.0, 0.2)],
    }
    cross = {
        "production_input_truth": False,
        "segment": [10, 20],
        "candidates": [
            {"source_candidate_id": candidate_id, "consensus_score_m": 0.2,
             "rank0_to_rank2_m": 0.11, "max_carrier_rms_cycles": 0.2,
             "max_block_spread_m": 0.1}
            for candidate_id in (1, 2)
        ],
    }

    result = select(source, cross)

    assert result["accepted"] is False
    assert result["reason"] == "fewer_than_two_cross_basis_cppr_modes"


def test_cross_basis_cppr_fusion_abstains_when_cppr_is_unavailable() -> None:
    source = {
        "production_input_truth": False,
        "segment": [10, 20],
        "hypotheses": [
            {"seed_id": 1, "offset_ecef_m": [1.0, 0.0, 0.0]},
            {"seed_id": 2, "offset_ecef_m": [2.0, 0.0, 0.0]},
        ],
    }
    cross = {
        "production_input_truth": False,
        "segment": [10, 20],
        "candidates": [],
    }

    result = select(source, cross)

    assert result["accepted"] is False
    assert result["reason"] == "cppr_evidence_unavailable"
