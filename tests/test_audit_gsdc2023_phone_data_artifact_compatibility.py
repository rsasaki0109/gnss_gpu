from __future__ import annotations

import json
from pathlib import Path

from experiments.audit_gsdc2023_matlab_equivalence_gate import DEFAULT_EQUIVALENCE_TRIPS
from experiments.audit_gsdc2023_phone_data_artifact_compatibility import (
    phone_data_artifact_compatibility_report,
)
from experiments.gsdc2023_raw_bridge import DEFAULT_ROOT


def _summary_payload() -> dict[str, object]:
    return {
        "passed": True,
        "equivalence_claim": "matlab_equivalent",
        "data_root": str(Path(DEFAULT_ROOT).resolve()),
        "trips": list(DEFAULT_EQUIVALENCE_TRIPS),
        "trip_count": len(DEFAULT_EQUIVALENCE_TRIPS),
        "max_epochs": 0,
        "count_max_epochs": 0,
        "factor_multi_gnss": False,
        "residual_multi_gnss": False,
        "residual_observation_mask": True,
        "residual_include_inactive_observations": True,
        "count_multi_gnss": False,
        "asset_datasets": ["train"],
        "quick_assets": True,
        "strict_ref_height": False,
        "gates": {
            "assets": {"passed": True},
            "factor_mask": {
                "passed": True,
                "side_only_failure_count": 0,
                "total_matlab_only": 0,
                "total_bridge_only": 0,
            },
            "raw_bridge_counts": {
                "passed": True,
                "count_delta_failure_count": 0,
                "missing_bridge_count_rows": 0,
                "missing_phone_count_rows": 6,
                "matched_abs_delta_total": 0,
            },
            "residual_values": {
                "passed": True,
                "total_matlab_only": 0,
                "total_bridge_only": 0,
            },
            "residual_diagnostics_writer": {
                "passed": True,
                "bridge_residual_diagnostics_export_enabled": True,
                "bridge_residual_diagnostics_export_count": 12,
                "bridge_residual_diagnostics_export_total_rows": 258537,
                "bridge_residual_diagnostics_export_column_count_min": 44,
                "bridge_residual_diagnostics_export_column_count_max": 44,
                "bridge_residual_diagnostics_export_column_mismatch_count": 0,
                "writer_regression_checked": True,
                "writer_regression_passed": True,
                "writer_regression_mismatch_count": 0,
            },
        },
    }


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_phone_data_artifact_report_defers_mat_struct_writer(tmp_path: Path) -> None:
    summary_path = _write_json(tmp_path / "summary.json", _summary_payload())

    report = phone_data_artifact_compatibility_report(summary_path)

    assert report["passed"] is True
    assert report["cached_summary_validation_passed"] is True
    rows = {row["artifact"]: row for row in report["artifacts"]}
    assert rows["phone_data_factor_counts.csv"]["passed"] is True
    assert rows["phone_data_factor_mask.csv"]["passed"] is True
    assert rows["phone_data_residual_diagnostics.csv"]["status"] == "schema_and_value_equivalent_regression_locked"
    assert rows["phone_data.mat"]["passed"] is True
    assert rows["phone_data.mat"]["python_writer_available"] is False
    assert rows["phone_data.mat"]["required_for_submit_ready"] is False
    assert report["phone_data_mat_decision"] == "defer"


def test_phone_data_artifact_report_can_require_writer_export_summaries(tmp_path: Path) -> None:
    summary_path = _write_json(tmp_path / "summary.json", _summary_payload())

    report = phone_data_artifact_compatibility_report(
        summary_path,
        require_csv_writer_exports=True,
    )

    assert report["passed"] is False
    rows = {row["artifact"]: row for row in report["artifacts"]}
    assert rows["phone_data_factor_counts.csv"]["status"] == "writer_export_not_checked"
    assert rows["phone_data_factor_mask.csv"]["status"] == "writer_export_not_checked"


def test_phone_data_artifact_report_accepts_writer_export_summaries(tmp_path: Path) -> None:
    summary_path = _write_json(tmp_path / "summary.json", _summary_payload())
    count_summary = _write_json(
        tmp_path / "count_summary.json",
        {
            "bridge_factor_count_exports_written": 1,
            "count_delta_failure_count": 0,
            "matched_abs_delta_total": 0,
            "missing_bridge_count_rows": 0,
        },
    )
    mask_summary = _write_json(
        tmp_path / "mask_summary.json",
        {
            "bridge_factor_mask_export_count": 1,
            "bridge_factor_mask_export_failure_count": 0,
            "side_only_failure_count": 0,
            "total_matlab_only": 0,
            "total_bridge_only": 0,
        },
    )

    report = phone_data_artifact_compatibility_report(
        summary_path,
        factor_count_summary=count_summary,
        factor_mask_summary=mask_summary,
        require_csv_writer_exports=True,
    )

    assert report["passed"] is True
    rows = {row["artifact"]: row for row in report["artifacts"]}
    assert rows["phone_data_factor_counts.csv"]["writer_export_checked"] is True
    assert rows["phone_data_factor_mask.csv"]["writer_export_checked"] is True


def test_phone_data_artifact_report_fails_if_mat_struct_is_required(tmp_path: Path) -> None:
    summary_path = _write_json(tmp_path / "summary.json", _summary_payload())

    report = phone_data_artifact_compatibility_report(
        summary_path,
        require_phone_data_mat=True,
    )

    assert report["passed"] is False
    row = {item["artifact"]: item for item in report["artifacts"]}["phone_data.mat"]
    assert row["status"] == "required_but_not_generated"
