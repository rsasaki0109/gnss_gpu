"""Unit tests for experiments/diag_canyon_forensics_wp8.py's pure helpers."""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in (_PROJECT_ROOT / "experiments",):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from diag_canyon_forensics_wp8 import (  # noqa: E402
    summarize_debug_log,
    summarize_lli,
    summarize_residuals_by_los,
)


def _row(**kwargs) -> dict:
    base = {
        "tow": "189000.0", "gf_slip_count": "0", "doppler_slip_l1_count": "0",
        "doppler_slip_l2_count": "0", "code_slip_l1_count": "0", "code_slip_l2_count": "0",
        "lli_slip_l1_count": "0", "lli_slip_l2_count": "0",
        "ambiguity_reset_l1_count": "0", "ambiguity_reset_l2_count": "0",
        "float_update_post_suppression_residual_rms_m": "0.05",
        "float_update_prefit_residual_rms_m": "0.10",
        "float_update_nis_per_observation": "1.2",
        "float_position_covariance_trace_m2": "0.02",
        "num_sats": "8",
        "reject_reason": "",
    }
    base.update(kwargs)
    return base


def test_summarize_debug_log_empty() -> None:
    assert summarize_debug_log([]) == {"n_epochs": 0}


def test_summarize_debug_log_totals_slip_counters() -> None:
    rows = [_row(gf_slip_count="1"), _row(gf_slip_count="2"), _row(doppler_slip_l1_count="3")]
    summary = summarize_debug_log(rows)
    assert summary["n_epochs"] == 3
    assert summary["slip_totals"]["gf_slip_count"] == 3
    assert summary["slip_totals"]["doppler_slip_l1_count"] == 3


def test_summarize_debug_log_covariance_trace_stats() -> None:
    rows = [
        _row(float_position_covariance_trace_m2="0.01"),
        _row(float_position_covariance_trace_m2="0.03"),
        _row(float_position_covariance_trace_m2="0.02"),
    ]
    summary = summarize_debug_log(rows)
    stats = summary["float_position_covariance_trace_m2"]
    assert stats["n"] == 3
    assert stats["median"] == 0.02
    assert stats["min"] == 0.01
    assert stats["max"] == 0.03


def test_summarize_debug_log_covariance_trace_regime_buckets() -> None:
    rows = [
        _row(float_position_covariance_trace_m2="2700.0"),  # wide/untrusted reset
        _row(float_position_covariance_trace_m2="900.0"),   # wide/untrusted reset
        _row(float_position_covariance_trace_m2="200.0"),   # partially shrunk
        _row(float_position_covariance_trace_m2="0.5"),     # fully converged
        _row(float_position_covariance_trace_m2="0.0005"),  # fully converged
    ]
    summary = summarize_debug_log(rows)
    regime = summary["float_position_covariance_trace_regime"]
    assert regime["n"] == 5
    assert regime["frac_wide_untrusted_reset_gt500"] == 0.4
    assert regime["frac_partially_shrunk_50_500"] == 0.2
    assert regime["frac_converged_lt50"] == 0.4
    assert regime["frac_fully_converged_lt1"] == 0.4


def test_summarize_debug_log_handles_missing_numeric_fields() -> None:
    rows = [_row(float_position_covariance_trace_m2="")]
    summary = summarize_debug_log(rows)
    assert summary["float_position_covariance_trace_m2"]["n"] == 0


def test_summarize_debug_log_counts_reject_reasons() -> None:
    rows = [
        _row(reject_reason="max_position_jump"),
        _row(reject_reason="max_position_jump"),
        _row(reject_reason="postfix_rms"),
        _row(reject_reason=""),
    ]
    summary = summarize_debug_log(rows)
    assert summary["reject_reasons"] == {"max_position_jump": 2, "postfix_rms": 1}


def test_summarize_lli_flags_odd_lli_values_as_slip() -> None:
    series = {
        "G01": [(100.0, 0), (100.2, 1), (100.4, 0)],  # 1 slip flag (odd LLI)
        "G02": [(100.0, 4), (100.2, 6)],  # even LLI values -> no slip flag
    }
    summary = summarize_lli(series)
    assert summary["G01"]["n_slip_flagged"] == 1
    assert summary["G01"]["slip_tows"] == [100.2]
    assert summary["G02"]["n_slip_flagged"] == 0


def test_summarize_residuals_by_los_splits_correctly() -> None:
    residual_rows = [
        {"tow": 100.0, "sat_id": "G01", "common_mode_removed_residual_m": 1.0},
        {"tow": 100.0, "sat_id": "G02", "common_mode_removed_residual_m": -50.0},
        {"tow": 100.0, "sat_id": "G03", "common_mode_removed_residual_m": 2.0},
    ]
    nlos_mask = {
        (100.0, "G01"): True,   # LOS
        (100.0, "G02"): False,  # NLOS -- the big residual
        # G03 absent from mask -> "unknown"
    }
    result = summarize_residuals_by_los(residual_rows, nlos_mask)
    assert result["los"]["n"] == 1
    assert result["los"]["median"] == 1.0
    assert result["nlos"]["n"] == 1
    assert result["nlos"]["median"] == 50.0
    assert result["unknown"]["n"] == 1


def test_summarize_residuals_by_los_respects_tow_tolerance() -> None:
    residual_rows = [{"tow": 100.5, "sat_id": "G01", "common_mode_removed_residual_m": 3.0}]
    nlos_mask = {(100.0, "G01"): False}
    # Default tolerance is 0.3s; 0.5s away should not match -> unknown.
    result = summarize_residuals_by_los(residual_rows, nlos_mask, tow_tolerance=0.3)
    assert result["nlos"]["n"] == 0
    assert result["unknown"]["n"] == 1
    # Widening tolerance should pick it up.
    result_wide = summarize_residuals_by_los(residual_rows, nlos_mask, tow_tolerance=1.0)
    assert result_wide["nlos"]["n"] == 1
