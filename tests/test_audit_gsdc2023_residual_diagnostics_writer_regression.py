from __future__ import annotations

from pathlib import Path

import pytest

from experiments.audit_gsdc2023_residual_diagnostics_writer_regression import (
    assert_writer_regression_manifest,
    build_writer_regression_manifest,
    writer_regression_mismatches,
)
from experiments.compare_gsdc2023_residual_diagnostics_pd import PD_WIDE_FRAME_COLUMNS


def _write_writer_csv(path: Path, rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ",".join(PD_WIDE_FRAME_COLUMNS)
    lines = [header]
    for row in rows:
        lines.append(",".join(str(value) for value in row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_build_writer_regression_manifest_records_schema_rows_and_hashes(tmp_path: Path) -> None:
    row = ["L1", 1, 1000, 1, 3, *([0] * (len(PD_WIDE_FRAME_COLUMNS) - 5))]
    _write_writer_csv(tmp_path / "train/course-a/phone-a/phone_data_residual_diagnostics.csv", [row, row])
    _write_writer_csv(tmp_path / "train/course-b/phone-b/phone_data_residual_diagnostics.csv", [row])

    manifest = build_writer_regression_manifest(tmp_path)

    assert manifest["manifest_version"] == 1
    assert manifest["expected_column_count"] == 44
    assert manifest["expected_columns"] == list(PD_WIDE_FRAME_COLUMNS)
    assert manifest["file_count"] == 2
    assert manifest["total_rows"] == 3
    assert [row["trip"] for row in manifest["files"]] == [
        "train/course-a/phone-a",
        "train/course-b/phone-b",
    ]
    assert {row["column_count"] for row in manifest["files"]} == {44}
    assert all(len(str(row["sha256"])) == 64 for row in manifest["files"])


def test_writer_regression_mismatches_reports_changed_file_hash(tmp_path: Path) -> None:
    row = ["L1", 1, 1000, 1, 3, *([0] * (len(PD_WIDE_FRAME_COLUMNS) - 5))]
    path = tmp_path / "train/course/phone/phone_data_residual_diagnostics.csv"
    _write_writer_csv(path, [row])
    expected = build_writer_regression_manifest(tmp_path)

    changed_row = ["L1", 1, 1000, 1, 4, *([0] * (len(PD_WIDE_FRAME_COLUMNS) - 5))]
    _write_writer_csv(path, [changed_row])
    actual = build_writer_regression_manifest(tmp_path)

    mismatches = writer_regression_mismatches(actual, expected)
    assert any("sha256" in item for item in mismatches)
    with pytest.raises(SystemExit, match="residual diagnostics writer regression mismatch"):
        assert_writer_regression_manifest(actual, expected)
