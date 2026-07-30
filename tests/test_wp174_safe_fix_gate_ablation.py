from __future__ import annotations

from experiments.analyze_wp174_safe_fix_gate_ablation import compare


def _row(tow: float, pair_count: int, good: bool) -> dict[str, str]:
    return {
        "tow": str(tow),
        "block": "0",
        "shadow_best_sub50cm": "1" if good else "0",
        "pair_count": str(pair_count),
        "lambda_shadow_ratio": "20",
        "lambda_shadow_bsr_qscale16": "0.99999",
        "lambda_shadow_second_position_delta_m": "0.01",
        "float_update_nis_per_observation": "0.2",
        "float_update_prefit_residual_rms_m": "22",
        "lambda_shadow_best_correction_x": "0",
        "lambda_shadow_best_correction_y": "0",
        "lambda_shadow_best_correction_z": "0",
    }


def test_selected_gate_records_nlos_pair_geometry_defense() -> None:
    result = compare(
        {"nagoya_nlos": [_row(1.0 + 0.2 * index, 6, False) for index in range(12)]}
    )

    assert result["selected"] == "prefit50_pairs16_selected"
    assert result["nlos_counterexample"]["pair_count"] == 6
    selected = result["candidates"][result["selected"]]
    assert selected["total_accepted_bad_epochs"] == 0
