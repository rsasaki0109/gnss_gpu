"""Tests for PPC dataset install helper."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from experiments.download_ppc_dataset import (
    EXPECTED_RUNS,
    _find_run_root,
    _validate_layout,
    install_from_zip,
)


def _write_minimal_ppc_zip(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for run in EXPECTED_RUNS:
            for name in ("rover.obs", "base.obs", "base.nav", "reference.csv", "imu.csv"):
                archive.writestr(f"PPC-Dataset/{run}/{name}", "stub\n")


def test_find_run_root_nested_layout(tmp_path: Path):
    root = tmp_path / "extract"
    run_dir = root / "PPC-Dataset" / "tokyo" / "run1"
    run_dir.mkdir(parents=True)
    (run_dir / "rover.obs").write_text("stub\n", encoding="utf-8")
    assert _find_run_root(root) == root / "PPC-Dataset"


def test_install_from_zip_rejects_tiny_html_like_archive(tmp_path: Path):
    bad_zip = tmp_path / "bad.zip"
    bad_zip.write_text("<html>login</html>", encoding="utf-8")
    with pytest.raises(ValueError, match="too small"):
        install_from_zip(bad_zip, tmp_path / "dest", force=True)


def test_install_from_zip_layout(tmp_path: Path):
    zip_path = tmp_path / "ppc.zip"
    _write_minimal_ppc_zip(zip_path)
    dest = tmp_path / "PPC-Dataset-data"
    install_from_zip(zip_path, dest, force=True, min_zip_bytes=1)
    missing = _validate_layout(dest)
    assert missing == []
