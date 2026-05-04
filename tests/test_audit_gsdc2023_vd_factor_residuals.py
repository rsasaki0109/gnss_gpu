from __future__ import annotations

from pathlib import Path

from experiments.audit_gsdc2023_vd_factor_residuals import (
    DEFAULT_VD_FACTOR_RESIDUAL_TRIPS,
    _guard_reason,
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


def test_default_vd_factor_residual_trip_set_includes_pixel6pro_cases() -> None:
    assert any(trip.endswith("/pixel6pro") for trip in DEFAULT_VD_FACTOR_RESIDUAL_TRIPS)


def test_guard_reason_prefers_doppler_then_tdcp_thresholds() -> None:
    assert (
        _guard_reason(doppler_count=20, doppler_rms=8.1, tdcp_count=20, tdcp_rms=100.0)
        == "doppler"
    )
    assert _guard_reason(doppler_count=19, doppler_rms=100.0, tdcp_count=20, tdcp_rms=50.1) == "tdcp"
    assert _guard_reason(doppler_count=20, doppler_rms=8.0, tdcp_count=20, tdcp_rms=50.0) == ""
