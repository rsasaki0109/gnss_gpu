from __future__ import annotations

from experiments.analyze_wp174_consensus_calibration import analyze


def _row(domain: str, block: int, good: bool) -> dict[str, str]:
    return {
        "_domain": domain,
        "block": str(block),
        "pair_count": "16",
        "shadow_best_sub50cm": "1" if good else "0",
        "lambda_shadow_ratio": "20",
        "lambda_shadow_bsr": "0.99999",
        "lambda_shadow_bsr_qscale2": "0.99999",
        "lambda_shadow_bsr_qscale4": "0.99999",
        "lambda_shadow_bsr_qscale8": "0.99999",
        "lambda_shadow_bsr_qscale16": "0.99999",
        "lambda_shadow_second_position_delta_m": "0.01",
        "float_update_nis_per_observation": "0.5",
        "lambda_shadow_best_ecef_x": "1.0",
        "lambda_shadow_best_ecef_y": "2.0",
        "lambda_shadow_best_ecef_z": "3.0",
        "lambda_satellite_par_shadow_best_ecef_x": "1.001",
        "lambda_satellite_par_shadow_best_ecef_y": "2.0",
        "lambda_satellite_par_shadow_best_ecef_z": "3.0",
    }


def test_blocked_cv_keeps_held_out_truth_out_of_selection() -> None:
    rows = [
        _row(domain, block, good=not (domain == "tokyo" and block == 1))
        for domain in ("tokyo", "nagoya")
        for block in range(3)
    ]

    result = analyze(rows, purge_blocks=0)

    assert result["truth_labeled_epochs"] == 6
    assert result["out_of_fold"]["accepted_bad_epochs"] == 1
    assert result["out_of_fold"]["by_domain"]["tokyo"][
        "accepted_bad_epochs"
    ] == 1
    bad_fold = next(
        fold
        for fold in result["folds"]
        if fold["test_domain"] == "tokyo" and fold["test_block"] == 1
    )
    assert bad_fold["selected_policy"] is not None
    assert bad_fold["test_bad_epochs"] == 1


def test_nonfinite_satellite_solution_fails_closed() -> None:
    rows = [_row("tokyo", block, good=True) for block in range(3)]
    rows[1]["lambda_satellite_par_shadow_best_ecef_x"] = ""

    result = analyze(rows, purge_blocks=0)

    missing_solution_fold = next(
        fold for fold in result["folds"] if fold["test_block"] == 1
    )
    assert missing_solution_fold["test_good_epochs"] == 0
