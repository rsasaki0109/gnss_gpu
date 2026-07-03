"""Tests for the examples demo chooser."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_DEMO = REPO_ROOT / "examples" / "run_demo.py"


def _run_chooser(*args: str) -> subprocess.CompletedProcess[str]:
    env = {"PYTHONPATH": "python:.", "PYTHONIOENCODING": "utf-8"}
    return subprocess.run(
        [sys.executable, str(RUN_DEMO), *args],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_run_demo_list_includes_cpu_starter():
    result = _run_chooser("--list")
    assert result.returncode == 0
    assert "urban_canyon" in result.stdout
    assert "CPU-only" in result.stdout


def test_run_demo_help_mentions_list_and_starter():
    result = _run_chooser("--help")
    assert result.returncode == 0
    assert "--list" in result.stdout
    assert "urban_canyon" in result.stdout


def test_run_demo_unknown_name_is_user_friendly():
    result = _run_chooser("not_a_real_demo")
    assert result.returncode == 1
    assert "Unknown demo" in result.stderr
    assert "--list" in result.stderr


def test_run_demo_describe_includes_build_hint_for_gpu_demo():
    result = _run_chooser("--describe", "signal_sim")
    assert result.returncode == 0
    assert "signal_sim" in result.stdout
    assert "gpu" in result.stdout.lower()
    assert "CUDA" in result.stdout


def test_run_demo_missing_repo_root_fails_clearly(tmp_path: Path):
    result = subprocess.run(
        [sys.executable, str(RUN_DEMO), "--list", "--project-root", str(tmp_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "Expected a gnss_gpu repository" in result.stderr
