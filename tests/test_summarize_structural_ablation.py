from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


MODULE_PATH = Path(__file__).parents[1] / "experiments" / "summarize_structural_ablation.py"
SPEC = importlib.util.spec_from_file_location("summarize_structural_ablation", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_summarize_emits_pooled_run_and_predeclared_span_metrics() -> None:
    epochs = pd.DataFrame(
        {
            "city": ["tokyo"] * 4,
            "run": ["run1"] * 4,
            "epoch": [9, 10, 11, 12],
            "emit_to_ref_m": [0.25, 0.75, 2.0, 4.0],
            "pf_mode_count": [1, 2, 3, 2],
            "pf_mode_selection_accepted": [False, True, False, True],
            "pf_mode_weighted_mean_distance_m": [0.1, 0.2, 0.4, 0.8],
            "pf_mode_selected_x": [0.0, 0.5, 3.0, 2.0],
            "pf_mode_selected_y": [0.0, 0.0, 0.0, 0.0],
            "pf_mode_selected_z": [0.0, 0.0, 0.0, 0.0],
            "ref_x": [0.0, 0.0, 0.0, 0.0],
            "ref_y": [0.0, 0.0, 0.0, 0.0],
            "ref_z": [0.0, 0.0, 0.0, 0.0],
            "pf_mode_emit_epoch_eligible": [False, True, True, True],
            "emitted_source": ["pf", "pf", "hybrid", "pf"],
            "doppler_signal_wavelength_known_count": [10, 12, 14, 16],
            "doppler_signal_wavelength_unknown_count": [2, 2, 4, 4],
            "doppler_update_applied": [True, True, False, True],
            "doppler_clock_group_count": [3, 3, 4, 4],
            "pf_ffbsi_available": [False, True, True, True],
            "pf_ffbsi_applied": [False, True, False, True],
            "pf_ffbsi_correction_m": [None, 0.2, None, 0.4],
            "pf_ffbsi_to_ref_m": [None, 0.5, None, 3.0],
        }
    )
    spans = pd.DataFrame(
        {
            "span_id": ["fixed"],
            "city": ["tokyo"],
            "run": ["run1"],
            "start_epoch": [10],
            "end_epoch_exclusive": [12],
            "evaluation_role": ["holdout"],
        }
    )
    runs = pd.DataFrame({"city": ["tokyo"], "run": ["run1"], "ms_per_epoch": [2.5]})

    result = MODULE.summarize(epochs, spans, runs, holdout_start_epoch=10)
    assert list(result["scope"]) == [
        "pooled",
        "pooled_holdout",
        "run",
        "run_holdout",
        "blocked_span",
    ]
    pooled = result.iloc[0]
    assert pooled["reference_coverage"] == 1.0
    assert pooled["pass_0p5"] == 0.25
    assert pooled["error_p95_m"] == pytest.approx(3.7)
    assert pooled["mode_abstention_rate"] == 0.5
    assert pooled["mode_counterfactual_emissions"] == 2
    assert pooled["mode_counterfactual_error_delta_mean_m"] == pytest.approx(-1.125)
    assert pooled["doppler_update_rate"] == 0.75
    assert pooled["ffbsi_abstention_rate"] == pytest.approx(1 / 3)
    assert pooled["ffbsi_counterfactual_epochs"] == 2
    assert pooled["ffbsi_improved_rate"] == 1.0
    assert pooled["ffbsi_error_delta_mean_m"] == pytest.approx(-0.625)
    assert pooled["ms_per_epoch"] == 2.5
    assert pooled["runtime_scope"] == "mean_of_measured_full_runs"
    assert result.iloc[1]["epochs"] == 3
    assert result.iloc[1]["evaluation_role"] == "holdout"
    blocked = result.iloc[4]
    assert blocked["epochs"] == 2
    assert blocked["pass_1m"] == 0.5
    assert blocked["ms_per_epoch"] == 2.5
    assert blocked["runtime_scope"] == "full_run_average_proxy"


def test_summarize_uses_pre_ffbsi_error_after_emit_metric_is_replaced() -> None:
    epochs = pd.DataFrame(
        {
            "city": ["tokyo", "tokyo"],
            "run": ["run1", "run1"],
            "epoch": [200, 201],
            "emit_to_ref_m": [0.8, 1.2],
            "pf_before_emit_to_ref_m": [1.0, 1.0],
            "pf_ffbsi_available": [True, True],
            "pf_ffbsi_applied": [True, True],
            "pf_ffbsi_to_ref_m": [0.8, 1.2],
        }
    )

    result = MODULE.summarize(epochs, pd.DataFrame(), holdout_start_epoch=200)
    pooled = result.iloc[0]
    assert pooled["ffbsi_counterfactual_epochs"] == 2
    assert pooled["ffbsi_improved_rate"] == 0.5
    assert pooled["ffbsi_error_delta_mean_m"] == pytest.approx(0.0)
    assert pooled["ffbsi_error_delta_p95_m"] == pytest.approx(0.18)


def test_summarize_rejects_internal_csv_without_identity_columns() -> None:
    with pytest.raises(ValueError, match="lacks columns"):
        MODULE.summarize(pd.DataFrame({"epoch": [0]}), pd.DataFrame())
