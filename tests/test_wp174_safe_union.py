from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.analyze_wp174_ffrt_calibration import Policy
from experiments.analyze_wp174_safe_union import (
    StateMachineConfig,
    _policy_by_fold,
    _union_reacquisition_metrics,
    causal_candidate_declarations,
)


def _row(tow: float, *, correction_x: str = "0") -> dict[str, str]:
    return {
        "tow": str(tow),
        "block": "0",
        "pair_count": "12",
        "lambda_shadow_ratio": "3",
        "lambda_shadow_bsr": "0.9995",
        "lambda_shadow_bsr_qscale2": "0.9995",
        "lambda_shadow_bsr_qscale4": "0.9995",
        "lambda_shadow_bsr_qscale8": "0.9995",
        "lambda_shadow_bsr_qscale16": "0.9995",
        "lambda_shadow_second_position_delta_m": "0.02",
        "float_update_nis_per_observation": "1.5",
        "lambda_shadow_best_correction_x": correction_x,
        "lambda_shadow_best_correction_y": "0",
        "lambda_shadow_best_correction_z": "0",
    }


def test_causal_state_machine_acquires_and_immediately_revokes() -> None:
    rows = [_row(1.0), _row(1.2, correction_x="0.01"), _row(1.4)]
    rows[-1]["lambda_shadow_second_position_delta_m"] = "0.5"
    declared, metrics = causal_candidate_declarations(
        rows,
        {0: Policy(1, 10, 0.03, 2.0)},
        StateMachineConfig(),
    )
    assert declared == {1.2}
    assert metrics["transitions"] == {"acquire": 1, "hold": 0, "revoke": 1}


def test_causal_state_machine_fails_closed_on_missing_correction() -> None:
    rows = [_row(1.0), _row(1.2)]
    rows[1]["lambda_shadow_best_correction_z"] = ""
    declared, _ = causal_candidate_declarations(
        rows,
        {0: Policy(1, 10, 0.03, 2.0)},
        StateMachineConfig(),
    )
    assert declared == set()


def test_guarded_hold_requires_correction_continuity() -> None:
    rows = [_row(1.0), _row(1.2, correction_x="0.01")]
    rows.extend(
        [
            _row(1.4, correction_x="0.02"),
            _row(1.6, correction_x="0.20"),
        ]
    )
    for row in rows[2:]:
        row["lambda_shadow_second_position_delta_m"] = "0.5"
    declared, metrics = causal_candidate_declarations(
        rows,
        {0: Policy(1, 10, 0.03, 2.0)},
        StateMachineConfig(
            maximum_hold_epochs=2,
            maximum_hold_correction_jump_m=0.03,
        ),
    )
    assert declared == {1.2, 1.4}
    assert metrics["transitions"]["hold"] == 1
    assert metrics["transitions"]["revoke"] == 1


def test_union_reacquisition_uses_final_fix_sequence() -> None:
    rows = [
        {"tow": 1.0, "union_fix": 1},
        {"tow": 1.2, "union_fix": 0},
        {"tow": 1.4, "union_fix": 0},
        {"tow": 1.6, "union_fix": 1},
    ]
    metrics = _union_reacquisition_metrics(rows)
    assert metrics["union_reacquisition_events"] == 1
    assert metrics["union_reacquisition_p95_s"] == pytest.approx(0.4)


def test_temporal_policy_reads_diagnostic_fold_family() -> None:
    payload = {
        "folds": [],
        "temporal_policy_diagnostic_only": {
            "folds": [
                {
                    "test_domain": "tokyo",
                    "test_block": 3,
                    "selected_policy": {
                        "covariance_scale": 16,
                        "minimum_pairs": 8,
                        "maximum_second_position_delta_m": 0.25,
                        "maximum_nis_per_observation": 1.0,
                    },
                }
            ]
        },
    }
    policies = _policy_by_fold(payload, "tokyo", "temporal")
    assert policies[3] == Policy(16, 8, 0.25, 1.0)


def test_safe_union_contract_remains_shadow_only_and_blocked() -> None:
    root = Path(__file__).resolve().parents[1]
    contract = json.loads(
        (root / "configs/evaluation/wp174_safe_union.json").read_text(
            encoding="utf-8"
        )
    )
    assert contract["mode"] == "shadow_only"
    assert contract["promotion_allowed"] is False
    assert contract["runtime_fgo"] is False
    assert contract["exploratory_observed"]["tokyo"][
        "reacquisition_p95_s"
    ] > contract["formal_targets"]["reacquisition_p95_s"]
