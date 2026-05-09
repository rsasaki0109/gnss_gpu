from __future__ import annotations

import json
import re
from pathlib import Path

from experiments.gsdc2023_audit_output import (
    print_summary_and_output_dir,
    timestamped_output_dir,
    unique_output_dir,
    write_summary_json,
)


def test_unique_output_dir_creates_suffix_when_stem_exists(tmp_path: Path) -> None:
    first = unique_output_dir(tmp_path, "audit")
    second = unique_output_dir(tmp_path, "audit")

    assert first == tmp_path / "audit"
    assert second == tmp_path / "audit_01"
    assert first.is_dir()
    assert second.is_dir()


def test_timestamped_output_dir_uses_prefix_and_creates_directory(tmp_path: Path) -> None:
    out_dir = timestamped_output_dir(tmp_path, "gsdc2023_test")

    assert out_dir.parent == tmp_path
    assert re.fullmatch(r"gsdc2023_test_\d{8}_\d{6}", out_dir.name)
    assert out_dir.is_dir()


def test_write_and_print_summary_json(tmp_path: Path, capsys) -> None:
    out_dir = unique_output_dir(tmp_path, "audit")
    payload = {"total": 3, "ok": True}

    path = write_summary_json(out_dir, payload)
    print_summary_and_output_dir(payload, out_dir)

    assert path == out_dir / "summary.json"
    assert json.loads(path.read_text(encoding="utf-8")) == payload
    captured = capsys.readouterr().out
    assert '"total": 3' in captured
    assert f"comparison_dir={out_dir}" in captured
