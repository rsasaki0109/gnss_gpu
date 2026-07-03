"""Smoke tests for PPC PF NLOS orchestrator."""

from __future__ import annotations

from pathlib import Path

import pytest

from experiments.run_pf_nlos_smoke import _read_official_pct, _run_dir_ok


def test_run_dir_ok_requires_full_ppc_layout(tmp_path: Path):
    run_dir = tmp_path / "tokyo" / "run1"
    run_dir.mkdir(parents=True)
    for name in ("rover.obs", "base.obs", "base.nav", "reference.csv", "imu.csv"):
        (run_dir / name).write_text("stub\n", encoding="utf-8")
    assert _run_dir_ok(tmp_path, "tokyo/run1") is True
    assert _run_dir_ok(tmp_path, "tokyo/run2") is False


def test_read_official_pct_parses_honest_ppc_pct(tmp_path: Path):
    csv_path = tmp_path / "runs.csv"
    csv_path.write_text(
        "method,honest_ppc_pct\n"
        "rbpf+dd,61.25\n",
        encoding="utf-8",
    )
    assert _read_official_pct(csv_path) == pytest.approx(61.25)
