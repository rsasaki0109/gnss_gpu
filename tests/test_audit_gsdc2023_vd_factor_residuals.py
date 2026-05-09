from __future__ import annotations

from pathlib import Path

import pandas as pd

from experiments.audit_gsdc2023_vd_factor_residuals import (
    DEFAULT_VD_FACTOR_RESIDUAL_TRIPS,
    _guard_reason,
    guard_segment_summary_payload,
    vd_factor_residual_audit,
)


def test_vd_factor_residual_audit_summarizes_nested_metrics(tmp_path: Path) -> None:
    def fake_diagnose(args):
        return {
            "trip": args.trip,
            "n_epochs": 10,
            "n_sat_slots": 3,
            "tdcp_use_drift": args.trip.endswith("pixel6pro"),
            "doppler_seed_residual": {"count": 5, "weighted_rms": 4.0, "abs_p95": 6.0, "abs_max": 7.0},
            "tdcp_seed_residual": {"count": 4, "weighted_rms": 20.0, "abs_p95": 30.0, "abs_max": 40.0},
            "pseudorange_mse": {"baseline": 8.0, "raw_wls": 3.0},
        }

    summary, payload = vd_factor_residual_audit(
        tmp_path,
        ["test/course/pixel6pro", "train/course/pixel5"],
        start_epoch=0,
        max_epochs=50,
        output_dir=tmp_path / "out",
        multi_gnss=True,
        dual_frequency=False,
        observation_mask=True,
        tdcp_use_drift="auto",
        top=3,
        diagnose_fn=fake_diagnose,
        verbose=True,
    )

    assert summary["trip"].tolist() == ["test/course/pixel6pro", "train/course/pixel5"]
    assert summary["doppler_weighted_rms_mps"].tolist() == [4.0, 4.0]
    assert summary["tdcp_weighted_rms_m"].tolist() == [20.0, 20.0]
    assert bool(summary.loc[summary["trip"].eq("test/course/pixel6pro"), "tdcp_use_drift"].iloc[0]) is True
    assert payload["passed"] is True
    assert payload["completed_trip_count"] == 2
    assert payload["max_tdcp_weighted_rms_m"] == 20.0
    assert payload["residual_threshold_failure_count"] == 0
    assert summary["guard_threshold_would_reject"].tolist() == [False, False]


def test_vd_factor_residual_audit_can_require_clean_guard_enabled_phones(tmp_path: Path) -> None:
    def fake_diagnose(args):
        return {
            "trip": args.trip,
            "n_epochs": 10,
            "n_sat_slots": 3,
            "tdcp_use_drift": args.trip.endswith("pixel6pro"),
            "doppler_seed_residual": {"count": 20, "weighted_rms": 8.5, "abs_p95": 9.0, "abs_max": 10.0},
            "tdcp_seed_residual": {"count": 20, "weighted_rms": 4.0, "abs_p95": 5.0, "abs_max": 6.0},
            "pseudorange_mse": {"baseline": 8.0, "raw_wls": 3.0},
        }

    summary, payload = vd_factor_residual_audit(
        tmp_path,
        ["test/course/pixel6pro", "train/course/pixel5"],
        start_epoch=0,
        max_epochs=50,
        output_dir=tmp_path / "out",
        multi_gnss=True,
        dual_frequency=False,
        observation_mask=True,
        tdcp_use_drift="auto",
        top=3,
        require_guard_clean=True,
        diagnose_fn=fake_diagnose,
    )

    assert payload["passed"] is False
    assert payload["residual_threshold_failure_count"] == 2
    assert payload["guard_enabled_residual_threshold_failure_count"] == 1
    assert payload["guard_disabled_residual_threshold_failure_count"] == 1
    assert summary["guard_threshold_reject_reason"].tolist() == ["doppler", "doppler"]
    assert summary["guard_effective_reject"].tolist() == [True, False]


def test_guard_segment_summary_payload_counts_effective_and_disabled_rejects() -> None:
    frame = pd.DataFrame(
        [
            {"would_reject": True, "effective_reject": True, "segment_epochs": 200, "reject_reason": "doppler"},
            {"would_reject": True, "effective_reject": False, "segment_epochs": 200, "reject_reason": "tdcp"},
            {"would_reject": False, "effective_reject": False, "segment_epochs": 100, "reject_reason": ""},
        ],
    )

    payload = guard_segment_summary_payload(frame)

    assert payload["guard_segment_count"] == 3
    assert payload["guard_threshold_rejected_segment_count"] == 2
    assert payload["guard_threshold_rejected_epoch_count"] == 400
    assert payload["guard_rejected_segment_count"] == 1
    assert payload["guard_rejected_epoch_count"] == 200
    assert payload["guard_disabled_threshold_rejected_segment_count"] == 1
    assert payload["guard_disabled_threshold_rejected_epoch_count"] == 200
    assert payload["guard_threshold_reject_reason_counts"] == {"doppler": 1, "tdcp": 1}
    assert payload["guard_effective_reject_reason_counts"] == {"doppler": 1}


def test_default_vd_factor_residual_trip_set_includes_pixel6pro_cases() -> None:
    assert any(trip.endswith("/pixel6pro") for trip in DEFAULT_VD_FACTOR_RESIDUAL_TRIPS)


def test_guard_reason_prefers_doppler_then_tdcp_thresholds() -> None:
    assert (
        _guard_reason(doppler_count=20, doppler_rms=8.1, tdcp_count=20, tdcp_rms=100.0)
        == "doppler"
    )
    assert _guard_reason(doppler_count=19, doppler_rms=100.0, tdcp_count=20, tdcp_rms=50.1) == "tdcp"
    assert _guard_reason(doppler_count=20, doppler_rms=8.0, tdcp_count=20, tdcp_rms=50.0) == ""
