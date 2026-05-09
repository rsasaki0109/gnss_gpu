#!/usr/bin/env python3
"""Build or verify compact manifests for generated phone_data sidecar writers."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from experiments.compare_gsdc2023_factor_masks import FACTOR_MASK_EXPORT_COLUMNS  # noqa: E402


FACTOR_COUNT_WRITER_FILENAME = "phone_data_factor_counts.csv"
FACTOR_COUNT_COLUMNS = ("freq", "field", "count")
FACTOR_MASK_WRITER_FILENAME = "phone_data_factor_mask.csv"

DEFAULT_FACTOR_COUNT_MANIFEST = _REPO / "data/gsdc2023_factor_count_writer_regression_manifest.json"
DEFAULT_FACTOR_MASK_MANIFEST = _REPO / "data/gsdc2023_factor_mask_writer_regression_manifest.json"

ARTIFACT_CONFIGS: dict[str, dict[str, Any]] = {
    "factor_counts": {
        "writer_filename": FACTOR_COUNT_WRITER_FILENAME,
        "expected_columns": list(FACTOR_COUNT_COLUMNS),
        "default_manifest": DEFAULT_FACTOR_COUNT_MANIFEST,
    },
    "factor_mask": {
        "writer_filename": FACTOR_MASK_WRITER_FILENAME,
        "expected_columns": list(FACTOR_MASK_EXPORT_COLUMNS),
        "default_manifest": DEFAULT_FACTOR_MASK_MANIFEST,
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _csv_header_and_row_count(path: Path) -> tuple[list[str], int]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            return [], 0
        row_count = sum(1 for _row in reader)
    return header, row_count


def _trip_from_writer_path(export_dir: Path, path: Path) -> str:
    relative = path.relative_to(export_dir)
    parent = str(relative.parent)
    return "." if parent == "." else parent


def build_sidecar_writer_regression_manifest(
    export_dir: Path,
    *,
    writer_filename: str,
    expected_columns: list[str] | tuple[str, ...],
) -> dict[str, Any]:
    """Return a stable schema/row/hash manifest for generated sidecar CSVs."""

    export_dir = Path(export_dir)
    files: list[dict[str, Any]] = []
    for path in sorted(export_dir.rglob(writer_filename)):
        header, row_count = _csv_header_and_row_count(path)
        files.append(
            {
                "trip": _trip_from_writer_path(export_dir, path),
                "relative_path": str(path.relative_to(export_dir)),
                "row_count": int(row_count),
                "column_count": int(len(header)),
                "sha256": _sha256(path),
                "columns": header,
            },
        )
    expected_columns = list(expected_columns)
    return {
        "manifest_version": 1,
        "writer_filename": writer_filename,
        "expected_column_count": int(len(expected_columns)),
        "expected_columns": expected_columns,
        "file_count": int(len(files)),
        "total_rows": int(sum(int(row["row_count"]) for row in files)),
        "files": files,
    }


def build_artifact_writer_regression_manifest(export_dir: Path, artifact: str) -> dict[str, Any]:
    config = ARTIFACT_CONFIGS[artifact]
    return build_sidecar_writer_regression_manifest(
        export_dir,
        writer_filename=str(config["writer_filename"]),
        expected_columns=list(config["expected_columns"]),
    )


def sidecar_writer_regression_mismatches(actual: dict[str, Any], expected: dict[str, Any]) -> list[str]:
    """Return human-readable mismatches between two sidecar writer manifests."""

    mismatches: list[str] = []
    for key in ("manifest_version", "writer_filename", "expected_column_count", "expected_columns"):
        if actual.get(key) != expected.get(key):
            mismatches.append(f"{key}: actual={actual.get(key)!r} expected={expected.get(key)!r}")
    for key in ("file_count", "total_rows"):
        if int(actual.get(key, -1) or -1) != int(expected.get(key, -1) or -1):
            mismatches.append(f"{key}: actual={actual.get(key)!r} expected={expected.get(key)!r}")

    actual_files = {
        str(row.get("relative_path")): row for row in actual.get("files", []) if isinstance(row, dict)
    }
    expected_files = {
        str(row.get("relative_path")): row for row in expected.get("files", []) if isinstance(row, dict)
    }
    missing = sorted(set(expected_files) - set(actual_files))
    extra = sorted(set(actual_files) - set(expected_files))
    if missing:
        mismatches.append("missing files: " + ", ".join(missing))
    if extra:
        mismatches.append("extra files: " + ", ".join(extra))

    for relative_path in sorted(set(actual_files) & set(expected_files)):
        actual_row = actual_files[relative_path]
        expected_row = expected_files[relative_path]
        for key in ("trip", "row_count", "column_count", "sha256", "columns"):
            if actual_row.get(key) != expected_row.get(key):
                mismatches.append(
                    f"{relative_path} {key}: actual={actual_row.get(key)!r} expected={expected_row.get(key)!r}",
                )
    return mismatches


def assert_sidecar_writer_regression_manifest(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    mismatches = sidecar_writer_regression_mismatches(actual, expected)
    if mismatches:
        raise SystemExit("phone_data sidecar writer regression mismatch:\n" + "\n".join(mismatches[:20]))


def check_artifact_writer_regression(export_dir: Path, expected_manifest: Path, artifact: str) -> list[str]:
    actual = build_artifact_writer_regression_manifest(export_dir, artifact)
    expected = json.loads(Path(expected_manifest).read_text(encoding="utf-8"))
    return sidecar_writer_regression_mismatches(actual, expected)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", choices=sorted(ARTIFACT_CONFIGS), required=True)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--write-manifest", type=Path)
    parser.add_argument("--expect-manifest", type=Path)
    parser.add_argument("--check", action="store_true", help="compare --export-dir against --expect-manifest")
    args = parser.parse_args(argv)

    config = ARTIFACT_CONFIGS[args.artifact]
    expect_manifest = args.expect_manifest or Path(config["default_manifest"])
    actual = build_artifact_writer_regression_manifest(args.export_dir, args.artifact)
    if args.write_manifest is not None:
        _write_json(args.write_manifest, actual)
        print(f"saved: {args.write_manifest}")
    if args.check:
        expected = json.loads(expect_manifest.read_text(encoding="utf-8"))
        assert_sidecar_writer_regression_manifest(actual, expected)
        print(f"matched: {actual['file_count']} file(s), {actual['total_rows']} row(s)")
    if args.write_manifest is None and not args.check:
        print(json.dumps(actual, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
