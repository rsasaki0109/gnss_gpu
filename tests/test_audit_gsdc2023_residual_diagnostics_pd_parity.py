from __future__ import annotations

from pathlib import Path

import pandas as pd

from experiments.audit_gsdc2023_residual_diagnostics_pd_parity import (
    residual_diagnostics_pd_parity_audit,
)


def test_residual_diagnostics_pd_parity_audit_summarizes_trips_and_exports(tmp_path: Path) -> None:
    def fake_compare(
        trip_dir: Path,
        *,
        max_epochs: int,
        multi_gnss: bool,
        apply_observation_mask: bool,
        include_inactive_observations: bool,
        max_abs_delta_threshold: float,
    ):
        assert max_epochs == 50
        assert multi_gnss is False
        assert apply_observation_mask is True
        assert include_inactive_observations is True
        assert max_abs_delta_threshold == 1.0e-4
        trip_delta = 2.0e-5 if trip_dir.name == "phone-a" else 5.0e-5
        merged = pd.DataFrame(
            [
                {
                    "field": "P",
                    "diagnostics_column": "p_residual_m",
                    "freq": "L1",
                    "epoch_index": 1,
                    "utcTimeMillis": 1000,
                    "sys": 1,
                    "svid": 3,
                    "matlab_value": 10.0,
                    "bridge_value": 10.0 + trip_delta,
                    "side": "both",
                    "delta": trip_delta,
                },
            ],
        )
        column_summary = pd.DataFrame(
            [
                {
                    "field": "P",
                    "diagnostics_column": "p_residual_m",
                    "freq": "L1",
                    "matlab_count": 1,
                    "bridge_count": 1,
                    "matched_count": 1,
                    "matlab_only": 0,
                    "bridge_only": 0,
                    "max_abs_delta": trip_delta,
                },
            ],
        )
        bridge_values = pd.DataFrame(
            [
                {
                    "field": "P",
                    "diagnostics_column": "p_residual_m",
                    "freq": "L1",
                    "epoch_index": 1,
                    "utcTimeMillis": 1000,
                    "sys": 1,
                    "svid": 3,
                    "bridge_value": 10.0 + trip_delta,
                },
            ],
        )
        payload = {
            "total_matlab_count": 1,
            "total_bridge_count": 1,
            "total_matched_count": 1,
            "total_matlab_only": 0,
            "total_bridge_only": 0,
            "median_abs_delta": trip_delta,
            "p95_abs_delta": trip_delta,
            "max_abs_delta": trip_delta,
            "passed": True,
        }
        return merged, column_summary, bridge_values, payload

    export_dir = tmp_path / "exports"
    (
        trip_summary,
        column_summary,
        side_only,
        export_summary,
        wide_trip_summary,
        wide_column_summary,
        wide_side_only,
        wide_export_summary,
        payload,
    ) = residual_diagnostics_pd_parity_audit(
        tmp_path,
        ["train/course/phone-a", "train/course/phone-b"],
        max_epochs=50,
        multi_gnss=False,
        bridge_subset_export_dir=export_dir,
        compare_fn=fake_compare,
    )

    assert trip_summary["trip"].tolist() == ["train/course/phone-a", "train/course/phone-b"]
    assert payload["passed"] is True
    assert payload["completed_trip_count"] == 2
    assert payload["total_matched_count"] == 2
    assert payload["overall_max_abs_delta"] == 5.0e-5
    assert payload["worst_trip"] == "train/course/phone-b"
    assert side_only.empty
    assert len(column_summary) == 2
    assert len(export_summary) == 2
    assert wide_trip_summary.empty
    assert wide_column_summary.empty
    assert wide_side_only.empty
    assert wide_export_summary.empty
    assert payload["bridge_subset_export_total_rows"] == 2
    assert (export_dir / "train/course/phone-a/phone_data_residual_diagnostics_pd_subset.csv").is_file()


def test_residual_diagnostics_pd_parity_audit_fails_on_side_only_rows(tmp_path: Path) -> None:
    def fake_compare(
        _trip_dir: Path,
        *,
        max_epochs: int,
        multi_gnss: bool,
        apply_observation_mask: bool,
        include_inactive_observations: bool,
        max_abs_delta_threshold: float,
    ):
        merged = pd.DataFrame(
            [
                {
                    "field": "D",
                    "diagnostics_column": "d_residual_mps",
                    "freq": "L1",
                    "epoch_index": 1,
                    "utcTimeMillis": 1000,
                    "sys": 1,
                    "svid": 5,
                    "matlab_value": 1.0,
                    "side": "matlab_only",
                },
            ],
        )
        return (
            merged,
            pd.DataFrame(),
            pd.DataFrame(),
            {
                "total_matlab_count": 1,
                "total_bridge_count": 0,
                "total_matched_count": 0,
                "total_matlab_only": 1,
                "total_bridge_only": 0,
                "max_abs_delta": float("nan"),
                "passed": False,
            },
        )

    (
        _trip_summary,
        _column_summary,
        side_only,
        _export_summary,
        _wide_trip_summary,
        _wide_column_summary,
        _wide_side_only,
        _wide_export_summary,
        payload,
    ) = residual_diagnostics_pd_parity_audit(
        tmp_path,
        ["train/course/phone"],
        max_epochs=0,
        multi_gnss=False,
        compare_fn=fake_compare,
    )

    assert payload["passed"] is False
    assert payload["total_matlab_only"] == 1
    assert payload["top_side_only"][0]["diagnostics_column"] == "d_residual_mps"
    assert side_only.iloc[0]["trip"] == "train/course/phone"


def test_residual_diagnostics_pd_parity_audit_summarizes_wide_components(tmp_path: Path) -> None:
    def fake_compare(
        _trip_dir: Path,
        *,
        max_epochs: int,
        multi_gnss: bool,
        apply_observation_mask: bool,
        include_inactive_observations: bool,
        max_abs_delta_threshold: float,
    ):
        return (
            pd.DataFrame(
                [
                    {
                        "field": "P",
                        "diagnostics_column": "p_residual_m",
                        "freq": "L1",
                        "epoch_index": 1,
                        "utcTimeMillis": 1000,
                        "sys": 1,
                        "svid": 3,
                        "matlab_value": 10.0,
                        "bridge_value": 10.0,
                        "side": "both",
                        "delta": 0.0,
                    },
                ],
            ),
            pd.DataFrame(),
            pd.DataFrame(),
            {
                "total_matlab_count": 1,
                "total_bridge_count": 1,
                "total_matched_count": 1,
                "total_matlab_only": 0,
                "total_bridge_only": 0,
                "max_abs_delta": 0.0,
                "passed": True,
            },
        )

    def fake_wide_compare(
        trip_dir: Path,
        *,
        max_epochs: int,
        multi_gnss: bool,
        apply_observation_mask: bool,
        include_inactive_observations: bool,
        max_abs_delta_threshold: float,
    ):
        assert max_epochs == 25
        assert multi_gnss is False
        assert apply_observation_mask is True
        assert include_inactive_observations is True
        assert max_abs_delta_threshold == 5.0e-3
        delta = 0.001 if trip_dir.name == "phone-a" else 0.002
        merged = pd.DataFrame(
            [
                {
                    "diagnostics_column": "sat_rate_mps",
                    "freq": "L1",
                    "epoch_index": 1,
                    "utcTimeMillis": 1000,
                    "sys": 1,
                    "svid": 3,
                    "matlab_value": 20.0,
                    "bridge_value": 20.0 + delta,
                    "side": "both",
                    "delta": delta,
                },
            ],
        )
        summary = pd.DataFrame(
            [
                {
                    "diagnostics_column": "sat_rate_mps",
                    "matlab_count": 1,
                    "bridge_count": 1,
                    "matched_count": 1,
                    "matlab_only": 0,
                    "bridge_only": 0,
                    "max_abs_delta": delta,
                },
            ],
        )
        bridge_wide = pd.DataFrame([{"freq": "L1", "epoch_index": 1, "sat_rate_mps": 20.0 + delta}])
        payload = {
            "total_matlab_count": 1,
            "total_bridge_count": 1,
            "total_matched_count": 1,
            "total_matlab_only": 0,
            "total_bridge_only": 0,
            "median_abs_delta": delta,
            "p95_abs_delta": delta,
            "max_abs_delta": delta,
            "sat_col_mismatch_count": 0,
            "matlab_wide_row_count": 1,
            "bridge_wide_row_count": 1,
            "passed": True,
        }
        return merged, summary, bridge_wide, payload

    wide_export_dir = tmp_path / "wide_exports"
    (
        _trip_summary,
        _column_summary,
        _side_only,
        _export_summary,
        wide_trip_summary,
        wide_column_summary,
        wide_side_only,
        wide_export_summary,
        payload,
    ) = residual_diagnostics_pd_parity_audit(
        tmp_path,
        ["train/course/phone-a", "train/course/phone-b"],
        max_epochs=25,
        multi_gnss=False,
        run_wide_audit=True,
        compare_fn=fake_compare,
        wide_compare_fn=fake_wide_compare,
        bridge_wide_subset_export_dir=wide_export_dir,
    )

    assert payload["passed"] is True
    assert payload["wide_passed"] is True
    assert payload["wide_completed_trip_count"] == 2
    assert payload["wide_overall_max_abs_delta"] == 0.002
    assert payload["wide_sat_col_mismatch_count"] == 0
    assert payload["bridge_wide_subset_export_total_rows"] == 2
    assert len(wide_trip_summary) == 2
    assert len(wide_column_summary) == 2
    assert wide_side_only.empty
    assert len(wide_export_summary) == 2
    assert (wide_export_dir / "train/course/phone-a/phone_data_residual_diagnostics_pd_wide_subset.csv").is_file()
