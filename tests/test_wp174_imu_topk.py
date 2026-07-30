from __future__ import annotations

import pandas as pd

from experiments.analyze_wp174_imu_topk import analyze


def _debug(candidate_1: float, candidate_2: float) -> pd.DataFrame:
    row: dict[str, float] = {"tow": 10.0}
    for ordinal, position in enumerate((candidate_1, candidate_2), start=1):
        row[f"lambda_shadow_candidate_{ordinal}_cost"] = (
            ordinal - 1.0
        ) * 0.1
        row[f"lambda_shadow_candidate_{ordinal}_ecef_x"] = position
        row[f"lambda_shadow_candidate_{ordinal}_ecef_y"] = 0.0
        row[f"lambda_shadow_candidate_{ordinal}_ecef_z"] = 0.0
    return pd.DataFrame([row])


def _bridge(position: float, age: float = 0.2) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "tow": 10.0,
                "anchor": 0,
                "anchor_age_s": age,
                "bridge_ecef_x": position,
                "bridge_ecef_y": 0.0,
                "bridge_ecef_z": 0.0,
                "initialized": 1,
                "heading_converged": 1,
                "position_sigma_max_m": 0.1,
            }
        ]
    )


def _reference() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "tow": 10.0,
                "truth_x": 1.0,
                "truth_y": 0.0,
                "truth_z": 0.0,
            }
        ]
    )


def test_imu_score_can_select_accurate_alternate_without_truth() -> None:
    result = analyze(
        _debug(0.0, 1.0), _bridge(1.0), _reference(), "synthetic"
    )

    assert result["baseline_best_sub50cm_epochs"] == 0
    assert result["oracle_topk_sub50cm_epochs"] == 1
    assert result["imu_ranked_sub50cm_epochs"] == 1
    assert result["imu_ranking_gains"] == 1
    assert result["imu_ranking_harms"] == 0


def test_stale_imu_cannot_rerank_candidates() -> None:
    result = analyze(
        _debug(1.0, 0.0),
        _bridge(0.0, age=0.4),
        _reference(),
        "synthetic",
    )

    assert result["imu_eligible_epochs"] == 0
    assert result["imu_alternate_selections"] == 0
    assert result["imu_ranked_sub50cm_epochs"] == 1
