from __future__ import annotations

import math
import json
from pathlib import Path

import pytest

from experiments.analyze_wp174_topk_shadow import (
    _feature_profile,
    _quantile,
    _shadow_candidate_error,
    _shadow_ratio,
    _vector_norm,
)
from experiments.analyze_wp174_ffrt_calibration import (
    Policy,
    _accepts,
    _confirmed_metrics,
    analyze as analyze_calibration,
)
from experiments.wp174_ffrt import (
    minimum_second_to_best_ratio,
    passes_ffrt,
)


def test_quantile_interpolates_and_empty_is_none() -> None:
    assert _quantile([], 0.5) is None
    assert _quantile([1.0, 3.0], 0.5) == 2.0
    assert _quantile([1.0, 2.0, 3.0], 0.9) == pytest.approx(2.8)


def test_feature_profile_ignores_nonfinite_and_missing_values() -> None:
    rows = [
        {"ratio": "2.0", "lambda_shadow_bsr": "nan"},
        {"ratio": "4.0", "lambda_shadow_bsr": "0.9"},
        {"ratio": "", "lambda_shadow_bsr": "1.0"},
    ]
    profile = _feature_profile(rows)

    assert profile["ratio"]["count"] == 2
    assert profile["ratio"]["p50"] == 3.0
    assert profile["lambda_shadow_bsr"]["count"] == 2
    assert profile["lambda_shadow_bsr"]["min"] == 0.9


def test_shadow_ratio_uses_topk_costs_and_fails_closed() -> None:
    assert _shadow_ratio(
        {
            "lambda_shadow_best_cost": "2",
            "lambda_shadow_second_cost": "6",
        }
    ) == 3.0
    assert _shadow_ratio(
        {
            "lambda_shadow_best_cost": "0",
            "lambda_shadow_second_cost": "6",
        }
    ) is None
    assert _shadow_ratio(
        {
            "lambda_shadow_best_cost": "6",
            "lambda_shadow_second_cost": "2",
        }
    ) is None


def test_shadow_candidate_error_requires_finite_ecef() -> None:
    telemetry = {
        "lambda_shadow_best_ecef_x": "1",
        "lambda_shadow_best_ecef_y": "2",
        "lambda_shadow_best_ecef_z": "3",
    }
    assert _shadow_candidate_error(telemetry, (1.0, 2.0, 5.0)) == 2.0
    telemetry["lambda_shadow_best_ecef_z"] = "nan"
    assert _shadow_candidate_error(telemetry, (1.0, 2.0, 5.0)) is None


def test_correction_vector_norm_requires_all_finite_components() -> None:
    telemetry = {
        "lambda_shadow_best_correction_x": "3",
        "lambda_shadow_best_correction_y": "4",
        "lambda_shadow_best_correction_z": "0",
    }
    assert _vector_norm(
        telemetry,
        prefix="lambda_shadow_best_correction",
    ) == 5.0
    telemetry["lambda_shadow_best_correction_z"] = "nan"
    assert (
        _vector_norm(
            telemetry,
            prefix="lambda_shadow_best_correction",
        )
        is None
    )


def test_paper_locked_ffrt_matches_cpp_reference_case() -> None:
    expected_mu = 0.1751 * 0.05**-0.2605 - 0.0404
    threshold = minimum_second_to_best_ratio(8, 0.95)
    assert threshold == pytest.approx(1.0 / expected_mu)
    assert minimum_second_to_best_ratio(8, 0.9995) == 1.0
    assert math.isinf(minimum_second_to_best_ratio(8, 0.8))
    assert minimum_second_to_best_ratio(67, 0.99) is None
    assert passes_ffrt(8, 0.95, threshold)


def test_calibration_policy_uses_only_decision_time_fields() -> None:
    row = {
        "pair_count": "12",
        "lambda_shadow_ratio": "3",
        "lambda_shadow_bsr": "0.9995",
        "lambda_shadow_second_position_delta_m": "0.02",
        "float_update_nis_per_observation": "1.5",
    }
    policy = Policy(1, 10, 0.03, 2.0)
    assert _accepts(row, policy)
    row["lambda_shadow_second_position_delta_m"] = "0.04"
    assert not _accepts(row, policy)


def test_causal_confirmation_rejects_isolated_candidate() -> None:
    policy = Policy(1, 10, 0.03, 2.0)
    rows = []
    for index, good in enumerate((1, 1, 0)):
        rows.append(
            {
                "tow": str(100.0 + 0.2 * index),
                "shadow_best_sub50cm": str(good),
                "pair_count": "12",
                "lambda_shadow_ratio": "3",
                "lambda_shadow_bsr": "0.9995",
                "lambda_shadow_second_position_delta_m": (
                    "0.02" if good else "0.04"
                ),
                "float_update_nis_per_observation": "1.5",
                "lambda_shadow_best_ecef_x": str(index * 0.01),
                "lambda_shadow_best_ecef_y": "0",
                "lambda_shadow_best_ecef_z": "0",
                "lambda_shadow_best_correction_x": str(index * 0.01),
                "lambda_shadow_best_correction_y": "0",
                "lambda_shadow_best_correction_z": "0",
            }
        )
    metrics = _confirmed_metrics(
        rows,
        policy,
        minimum_contiguous_epochs=2,
        maximum_correction_jump_m=0.25,
    )
    assert metrics["accepted_good_epochs"] == 1
    assert metrics["accepted_bad_epochs"] == 0


def test_calibration_reports_exploratory_holdout_blocker() -> None:
    rows = []
    for domain in ("tokyo", "nagoya"):
        for block in range(3):
            rows.append(
                    {
                        "_domain": domain,
                        "block": str(block),
                        "tow": str(100.0 + block),
                        "shadow_best_sub50cm": "1",
                    "pair_count": "12",
                    "lambda_shadow_ratio": "3",
                    "lambda_shadow_bsr": "0.9995",
                    "lambda_shadow_bsr_qscale2": "0.9995",
                    "lambda_shadow_bsr_qscale4": "0.9995",
                    "lambda_shadow_bsr_qscale8": "0.9995",
                    "lambda_shadow_bsr_qscale16": "0.9995",
                        "lambda_shadow_second_position_delta_m": "0.02",
                        "float_update_nis_per_observation": "1.5",
                        "lambda_shadow_best_ecef_x": str(block * 0.01),
                        "lambda_shadow_best_ecef_y": "0",
                        "lambda_shadow_best_ecef_z": "0",
                        "lambda_shadow_best_correction_x": str(block * 0.01),
                        "lambda_shadow_best_correction_y": "0",
                        "lambda_shadow_best_correction_z": "0",
                    }
            )
    summary = analyze_calibration(rows, purge_blocks=1)
    assert summary["promotion_ready"] is False
    assert summary["out_of_fold"]["accepted_bad_epochs"] == 0
    assert len(summary["folds"]) == 6


def test_ffrt_contract_is_shadow_only_and_fail_closed() -> None:
    root = Path(__file__).resolve().parents[1]
    contract = json.loads(
        (root / "configs/evaluation/wp174_ffrt.json").read_text(
            encoding="utf-8"
        )
    )
    assert contract["mode"] == "shadow_only"
    assert contract["runtime_fgo"] is False
    assert contract["truth_used_for_selection"] is False
    assert contract["implemented_table"]["tolerable_failure_rate"] == 0.001
    assert contract["implemented_table"]["unsupported_request"] == "fail_closed"
