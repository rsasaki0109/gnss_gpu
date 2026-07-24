from __future__ import annotations

from experiments.select_wp159_screened_stability_consensus import rerank

_RANKING_FAMILIES = (
    "block_spread_m",
    "max_cross_basis_carrier_rms_cycles",
    "cppr_rank_sum",
)


def _mode(
    candidate_id: int,
    *,
    cross_refit_disagreement_m: float,
    block_spread_m: float,
    max_cross_basis_carrier_rms_cycles: float,
    cppr_rank_sum: float,
) -> dict:
    return {
        "candidate_id": candidate_id,
        "cross_refit_disagreement_m": cross_refit_disagreement_m,
        "block_spread_m": block_spread_m,
        "max_cross_basis_carrier_rms_cycles": max_cross_basis_carrier_rms_cycles,
        "cppr_rank_sum": cppr_rank_sum,
    }


def test_screened_stability_rerank_selects_balanced_unique_mode() -> None:
    fused = {
        "production_input_truth": False,
        "segment": [10, 20],
        "reason": "cross_basis_cppr_family_or_margin_gate_failed",
        "ddpr_excluded_satellites": ["G01"],
        "modes": [
            _mode(
                1,
                cross_refit_disagreement_m=0.03,
                block_spread_m=0.03,
                max_cross_basis_carrier_rms_cycles=0.22,
                cppr_rank_sum=3.0,
            ),
            _mode(
                2,
                cross_refit_disagreement_m=0.02,
                block_spread_m=0.08,
                max_cross_basis_carrier_rms_cycles=0.30,
                cppr_rank_sum=8.0,
            ),
            _mode(
                3,
                cross_refit_disagreement_m=0.08,
                block_spread_m=0.07,
                max_cross_basis_carrier_rms_cycles=0.32,
                cppr_rank_sum=9.0,
            ),
        ],
        "gate": {"max_cross_refit_disagreement_m": 0.1},
    }

    result = rerank(fused, max_family_rank_fraction=1.0)

    assert result["accepted"] is True
    assert result["selected_candidate_id"] == 1
    # Candidate 1 ranks first in all three retained families -> sum of 3.
    assert result["winner"]["stability_rank_sum"] == 3
    assert result["ranking_families"] == list(_RANKING_FAMILIES)
    assert "cross_refit_disagreement_m" not in result["winner"]["stability_family_ranks"]
    assert result["runner_margin"] >= 0.2
    assert "audit" not in str(result)


def test_screened_stability_rerank_rejects_on_runner_margin_only() -> None:
    fused = {
        "production_input_truth": False,
        "segment": [10, 20],
        "reason": "cross_basis_cppr_family_or_margin_gate_failed",
        "ddpr_excluded_satellites": ["G01"],
        "modes": [
            _mode(
                1,
                cross_refit_disagreement_m=0.03,
                block_spread_m=0.03,
                max_cross_basis_carrier_rms_cycles=0.20,
                cppr_rank_sum=5.0,
            ),
            _mode(
                2,
                cross_refit_disagreement_m=0.02,
                block_spread_m=0.03,
                max_cross_basis_carrier_rms_cycles=0.30,
                cppr_rank_sum=2.0,
            ),
            _mode(
                3,
                cross_refit_disagreement_m=0.04,
                block_spread_m=0.05,
                max_cross_basis_carrier_rms_cycles=0.30,
                cppr_rank_sum=5.0,
            ),
        ],
        "gate": {"max_cross_refit_disagreement_m": 0.1},
    }

    result = rerank(fused, max_family_rank_fraction=1.0)

    # Candidate 1 and 2 tie on rank sum (4 each); the family-rank gate is
    # wide open (fraction=1.0) but the margin between winner and runner is
    # zero, so only the margin gate fails.
    assert result["family_rank_pass"] is True
    assert result["runner_margin_pass"] is False
    assert result["accepted"] is False
    assert result["reason"] == "stability_cppr_family_or_margin_gate_failed"


def test_screened_stability_rerank_rejects_on_family_rank_limit_only() -> None:
    fused = {
        "production_input_truth": False,
        "segment": [10, 20],
        "reason": "cross_basis_cppr_family_or_margin_gate_failed",
        "ddpr_excluded_satellites": ["G01"],
        "modes": [
            _mode(
                1,
                cross_refit_disagreement_m=0.02,
                block_spread_m=0.02,
                max_cross_basis_carrier_rms_cycles=0.20,
                cppr_rank_sum=9.0,
            ),
            _mode(
                2,
                cross_refit_disagreement_m=0.02,
                block_spread_m=0.05,
                max_cross_basis_carrier_rms_cycles=0.35,
                cppr_rank_sum=1.0,
            ),
            _mode(
                3,
                cross_refit_disagreement_m=0.02,
                block_spread_m=0.08,
                max_cross_basis_carrier_rms_cycles=0.28,
                cppr_rank_sum=5.0,
            ),
        ],
        "gate": {"max_cross_refit_disagreement_m": 0.1},
    }

    # Default max_family_rank_fraction=0.4 -> family_rank_limit = ceil(3*0.4) = 2.
    result = rerank(fused)

    assert result["selected_candidate_id"] is None
    assert result["winner"]["candidate_id"] == 1
    assert result["runner"]["candidate_id"] == 2
    assert result["runner_margin"] == 0.2
    assert result["runner_margin_pass"] is True
    assert result["family_rank_pass"] is False
    assert result["accepted"] is False
    assert result["reason"] == "stability_cppr_family_or_margin_gate_failed"


def test_screened_stability_rerank_ignores_cross_refit_disagreement_in_ranking() -> None:
    def _fused(winner_disagreement: float) -> dict:
        return {
            "production_input_truth": False,
            "segment": [10, 20],
            "reason": "cross_basis_cppr_family_or_margin_gate_failed",
            "ddpr_excluded_satellites": ["G01"],
            "modes": [
                _mode(
                    1,
                    cross_refit_disagreement_m=winner_disagreement,
                    block_spread_m=0.03,
                    max_cross_basis_carrier_rms_cycles=0.22,
                    cppr_rank_sum=3.0,
                ),
                _mode(
                    2,
                    cross_refit_disagreement_m=0.01,
                    block_spread_m=0.08,
                    max_cross_basis_carrier_rms_cycles=0.30,
                    cppr_rank_sum=8.0,
                ),
                _mode(
                    3,
                    cross_refit_disagreement_m=0.01,
                    block_spread_m=0.07,
                    max_cross_basis_carrier_rms_cycles=0.32,
                    cppr_rank_sum=9.0,
                ),
            ],
            "gate": {"max_cross_refit_disagreement_m": 0.1},
        }

    # Winner's cross_refit_disagreement_m is bad-but-still-under-gate (0.09,
    # vs the 0.10 absolute eligibility gate enforced upstream in wp131) while
    # the other candidates have good values. Since the family is dropped from
    # ranking, this must not move the outcome at all relative to a run where
    # the winner's disagreement is excellent (0.001).
    bad_result = rerank(_fused(0.09), max_family_rank_fraction=1.0)
    good_result = rerank(_fused(0.001), max_family_rank_fraction=1.0)

    assert bad_result["accepted"] is True
    assert bad_result["selected_candidate_id"] == 1
    assert bad_result["winner"]["stability_rank_sum"] == 3
    assert "cross_refit_disagreement_m" not in bad_result["winner"]["stability_family_ranks"]

    assert bad_result["accepted"] == good_result["accepted"]
    assert bad_result["selected_candidate_id"] == good_result["selected_candidate_id"]
    assert bad_result["winner"]["stability_rank_sum"] == good_result["winner"]["stability_rank_sum"]
    assert bad_result["runner_margin"] == good_result["runner_margin"]


def test_screened_stability_rerank_fails_closed_without_modes() -> None:
    result = rerank(
        {
            "production_input_truth": False,
            "segment": [10, 20],
            "reason": "cppr_evidence_unavailable",
            "ddpr_excluded_satellites": ["G01"],
            "modes": [],
        }
    )

    assert result["accepted"] is False
    assert result["reason"] == "stability_cppr_evidence_unavailable"
    assert result["base_reason"] == "cppr_evidence_unavailable"


def test_screened_stability_rerank_fails_closed_without_ddpr_screen_evidence() -> None:
    missing = rerank(
        {
            "production_input_truth": False,
            "segment": [10, 20],
            "reason": "cross_basis_cppr_family_or_margin_gate_failed",
            "modes": [
                _mode(
                    1,
                    cross_refit_disagreement_m=0.03,
                    block_spread_m=0.03,
                    max_cross_basis_carrier_rms_cycles=0.22,
                    cppr_rank_sum=3.0,
                ),
                _mode(
                    2,
                    cross_refit_disagreement_m=0.02,
                    block_spread_m=0.08,
                    max_cross_basis_carrier_rms_cycles=0.30,
                    cppr_rank_sum=8.0,
                ),
            ],
        }
    )

    assert missing["accepted"] is False
    assert missing["reason"] == "screen_evidence_required"
    assert missing["selected_candidate_id"] is None
    assert missing["mode_count"] == 0
    assert missing["ddpr_excluded_satellites"] == []
    assert "winner" not in missing

    empty = rerank(
        {
            "production_input_truth": False,
            "segment": [10, 20],
            "reason": "cross_basis_cppr_family_or_margin_gate_failed",
            "ddpr_excluded_satellites": [],
            "modes": [],
        }
    )

    assert empty["accepted"] is False
    assert empty["reason"] == "screen_evidence_required"
    assert empty["selected_candidate_id"] is None
    assert empty["mode_count"] == 0
    assert empty["ddpr_excluded_satellites"] == []
