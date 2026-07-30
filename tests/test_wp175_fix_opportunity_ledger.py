from __future__ import annotations

from experiments.analyze_wp175_fix_opportunity_ledger import analyze


def _row(tow: float, *, fixed: bool, solved: bool, pairs: int) -> dict[str, str]:
    return {
        "tow": str(tow),
        "block": "0",
        "safe_fix_shadow_declared_fixed": "1" if fixed else "0",
        "lambda_shadow_attempted": "1",
        "lambda_shadow_solved": "1" if solved else "0",
        "pair_count": str(pairs),
        "full_ratio": "20",
        "lambda_shadow_bsr_qscale16": "0.999999",
        "lambda_shadow_second_position_delta_m": "0.01",
        "float_update_nis_per_observation": "0.1",
        "float_update_prefit_residual_rms_m": "1",
        "safe_fix_shadow_independent_consensus_delta_m": "0.01",
        "lambda_shadow_best_ecef_x": "1",
        "lambda_shadow_best_ecef_y": "2",
        "lambda_shadow_best_ecef_z": "3",
        "lambda_shadow_candidate_1_ecef_x": "1",
        "lambda_shadow_candidate_1_ecef_y": "2",
        "lambda_shadow_candidate_1_ecef_z": "3",
    }


def test_ledger_separates_truth_free_blockers_from_candidate_oracle() -> None:
    rows = [
        _row(0.0, fixed=True, solved=True, pairs=12),
        _row(0.2, fixed=False, solved=False, pairs=0),
        _row(0.4, fixed=False, solved=True, pairs=12),
    ]
    truth = {tow: (1.0, 2.0, 3.0) for tow in (0.0, 0.2, 0.4)}

    positions = {
        0.0: {"status": 4, "ecef_x": 1.0, "ecef_y": 2.0, "ecef_z": 3.0},
        0.2: {"status": 3, "ecef_x": 1.0, "ecef_y": 2.0, "ecef_z": 3.0},
        0.4: {"status": 3, "ecef_x": 1.0, "ecef_y": 2.0, "ecef_z": 3.0},
    }
    result = analyze(rows, truth, positions, "tokyo", block_count=2)

    assert result["truth_free_primary_blockers"] == {
        "eligible_but_temporally_unconfirmed": 1,
        "lambda_not_solved": 1,
    }
    assert (
        result["post_selection_candidate_oracles"]["lambda_topk"][
            "sub50cm_oracle_epochs"
        ]
        == 1
    )
    assert (
        result[
            "additional_correct_fix_epochs_required_after_false_demotion"
        ]
        == 1
    )
    assert result["post_selection_union_covers_required_increment"]
    assert len(result["contiguous_nested_cv_blocks"]) == 2
