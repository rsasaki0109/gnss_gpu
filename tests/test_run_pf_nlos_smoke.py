"""Smoke tests for PPC PF NLOS orchestrator."""

from __future__ import annotations

from pathlib import Path

import pytest

from experiments.run_pf_nlos_smoke import (
    METHOD_LABEL_ALIASES,
    _read_official_pct,
    _read_run_row,
    _resolve_methods,
    _run_dir_ok,
)


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
        "method,honest_ppc_pct,segment_ppc_pct,segment_pass_m,segment_total_m\n"
        "RBPF-velKF+DD,61.25,55.0,120.0,200.0\n",
        encoding="utf-8",
    )
    assert _read_official_pct(csv_path, "rbpf+dd") == pytest.approx(61.25)


def test_read_official_pct_accepts_zero_pct(tmp_path: Path):
    csv_path = tmp_path / "runs.csv"
    csv_path.write_text(
        "method,honest_ppc_pct\n"
        "RBPF-velKF+DD+gate,0.0\n",
        encoding="utf-8",
    )
    assert _read_official_pct(csv_path, "rbpf+dd+gate") == pytest.approx(0.0)


def test_read_run_row_extracts_continuous_metrics(tmp_path: Path):
    csv_path = tmp_path / "runs.csv"
    csv_path.write_text(
        "method,honest_ppc_pct,honest_pass_m,honest_total_m,segment_ppc_pct,"
        "segment_pass_m,segment_total_m,segment_epoch_pass_pct,coverage_pct,"
        "rbpf_kf_gate_active,rbpf_kf_applied,dd_epochs_applied,"
        "hybrid_applied,rtkdiag_pf_pu_applied\n"
        "RBPF-velKF+DD+gate,12.5,100.0,800.0,15.0,30.0,200.0,25.0,99.0,"
        "1,40,24,0,0\n",
        encoding="utf-8",
    )
    row = _read_run_row(csv_path, "rbpf+dd+gate")
    assert row["method"] == "RBPF-velKF+DD+gate"
    assert row["honest_ppc_pct"] == pytest.approx(12.5)
    assert row["segment_pass_m"] == pytest.approx(30.0)
    assert row["rbpf_kf_applied"] == pytest.approx(40.0)


def test_resolve_methods_profile_and_override():
    args = type("Args", (), {"profile": "gate", "methods": None})()
    assert _resolve_methods(args) == "rbpf+dd+gate"
    args.methods = "rbpf+dd"
    assert _resolve_methods(args) == "rbpf+dd"


def test_method_label_aliases_cover_gate_profile():
    assert "RBPF-velKF+DD+gate" in METHOD_LABEL_ALIASES["rbpf+dd+gate"]
