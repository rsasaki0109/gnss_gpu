"""Tests for UrbanNav external baseline reproduction gate."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

import eval_harness_lib as harness  # noqa: E402
from eval_harness_lib import compare_summary, run_has_core_files  # noqa: E402


def test_run_has_core_files(tmp_path: Path) -> None:
    run_dir = tmp_path / "Odaiba"
    run_dir.mkdir()
    for name in ("base.nav", "reference.csv", "rover_trimble.obs"):
        (run_dir / name).write_text("x", encoding="utf-8")
    assert run_has_core_files(run_dir)
    assert not run_has_core_files(tmp_path / "Missing")


def test_ensure_data_fetches_each_missing_run_to_requested_root(
    tmp_path: Path, monkeypatch
) -> None:
    commands: list[list[str]] = []

    def fake_run(command, **kwargs):
        commands.append(command)
        run = command[command.index("--run") + 1]
        output = Path(command[command.index("--output-dir") + 1]) / run
        output.mkdir(parents=True)
        for name in ("base.nav", "reference.csv", "rover_trimble.obs"):
            (output / name).write_text("x", encoding="utf-8")

    monkeypatch.setattr(harness.subprocess, "run", fake_run)

    harness.ensure_urbannav_data(tmp_path, ("Odaiba", "Shinjuku"), fetch=True)

    assert [command[command.index("--run") + 1] for command in commands] == [
        "Odaiba",
        "Shinjuku",
    ]
    assert all(command[command.index("--output-dir") + 1] == str(tmp_path) for command in commands)


def test_compare_summary_passes_on_reference_copy() -> None:
    rows = [
        {
            "method": "EKF",
            "mean_rms_2d": "93.248321",
            "mean_p95": "178.178684",
            "mean_outlier_rate_pct": "16.294663",
            "mean_catastrophic_rate_pct": "0.160676",
        },
        {
            "method": "PF+RobustClear-10K",
            "mean_rms_2d": "66.595016",
            "mean_p95": "98.526971",
            "mean_outlier_rate_pct": "4.800731",
            "mean_catastrophic_rate_pct": "0.000000",
        },
    ]
    checks, passed = compare_summary(
        rows,
        rows,
        methods=("EKF", "PF+RobustClear-10K"),
        rms_tol_m=0.75,
        p95_tol_m=1.0,
        rate_tol_pp=0.25,
    )
    assert passed
    assert all(item["status"] == "checked" for item in checks)


def test_compare_summary_fails_when_ekf_regresses() -> None:
    reference = [
        {
            "method": "EKF",
            "mean_rms_2d": "93.25",
            "mean_p95": "178.18",
            "mean_outlier_rate_pct": "16.29",
            "mean_catastrophic_rate_pct": "0.16",
        }
    ]
    reproduced = [
        {
            "method": "EKF",
            "mean_rms_2d": "120.00",
            "mean_p95": "178.18",
            "mean_outlier_rate_pct": "16.29",
            "mean_catastrophic_rate_pct": "0.16",
        }
    ]
    _, passed = compare_summary(
        reproduced,
        reference,
        methods=("EKF",),
        rms_tol_m=0.75,
        p95_tol_m=1.0,
        rate_tol_pp=0.25,
    )
    assert not passed
