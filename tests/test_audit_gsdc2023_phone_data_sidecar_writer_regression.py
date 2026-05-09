from __future__ import annotations

from pathlib import Path

import pytest

from experiments.audit_gsdc2023_phone_data_sidecar_writer_regression import (
    FACTOR_COUNT_COLUMNS,
    FACTOR_COUNT_WRITER_FILENAME,
    FACTOR_MASK_WRITER_FILENAME,
    assert_sidecar_writer_regression_manifest,
    build_artifact_writer_regression_manifest,
    build_sidecar_writer_regression_manifest,
    sidecar_writer_regression_mismatches,
)
from experiments.compare_gsdc2023_factor_masks import FACTOR_MASK_EXPORT_COLUMNS


def _write_csv(path: Path, header: list[str] | tuple[str, ...], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [",".join(header)]
    lines.extend(",".join(str(value) for value in row) for row in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_build_factor_count_manifest_records_schema_rows_and_hashes(tmp_path: Path) -> None:
    _write_csv(
        tmp_path / "train/course/phone" / FACTOR_COUNT_WRITER_FILENAME,
        FACTOR_COUNT_COLUMNS,
        [["L1", "P", 10], ["L5", "D", 2]],
    )

    manifest = build_artifact_writer_regression_manifest(tmp_path, "factor_counts")

    assert manifest["writer_filename"] == FACTOR_COUNT_WRITER_FILENAME
    assert manifest["expected_columns"] == list(FACTOR_COUNT_COLUMNS)
    assert manifest["file_count"] == 1
    assert manifest["total_rows"] == 2
    assert manifest["files"][0]["trip"] == "train/course/phone"
    assert len(manifest["files"][0]["sha256"]) == 64


def test_build_factor_mask_manifest_supports_root_level_exports(tmp_path: Path) -> None:
    _write_csv(
        tmp_path / FACTOR_MASK_WRITER_FILENAME,
        FACTOR_MASK_EXPORT_COLUMNS,
        [["P", "L1", 1, 1000, 0, 0, 1, 3, 1]],
    )

    manifest = build_artifact_writer_regression_manifest(tmp_path, "factor_mask")

    assert manifest["writer_filename"] == FACTOR_MASK_WRITER_FILENAME
    assert manifest["expected_columns"] == list(FACTOR_MASK_EXPORT_COLUMNS)
    assert manifest["files"][0]["trip"] == "."
    assert manifest["files"][0]["relative_path"] == FACTOR_MASK_WRITER_FILENAME


def test_sidecar_writer_mismatches_report_changed_hash(tmp_path: Path) -> None:
    path = tmp_path / "train/course/phone" / FACTOR_COUNT_WRITER_FILENAME
    _write_csv(path, FACTOR_COUNT_COLUMNS, [["L1", "P", 10]])
    expected = build_sidecar_writer_regression_manifest(
        tmp_path,
        writer_filename=FACTOR_COUNT_WRITER_FILENAME,
        expected_columns=FACTOR_COUNT_COLUMNS,
    )

    _write_csv(path, FACTOR_COUNT_COLUMNS, [["L1", "P", 11]])
    actual = build_sidecar_writer_regression_manifest(
        tmp_path,
        writer_filename=FACTOR_COUNT_WRITER_FILENAME,
        expected_columns=FACTOR_COUNT_COLUMNS,
    )

    mismatches = sidecar_writer_regression_mismatches(actual, expected)
    assert any("sha256" in item for item in mismatches)
    with pytest.raises(SystemExit, match="phone_data sidecar writer regression mismatch"):
        assert_sidecar_writer_regression_manifest(actual, expected)
