from __future__ import annotations

import json
import sys
from pathlib import Path

from gnss_gpu import cli


def test_readiness_distinguishes_build_and_runtime_states():
    base = [
        cli.Check("Python", "PASS", "ok"),
        cli.Check("NVIDIA GPU/driver", "PASS", "ok"),
        cli.Check("CUDA compiler", "PASS", "ok"),
        cli.Check("CMake", "PASS", "ok"),
    ]
    assert cli.readiness([*base, cli.Check("CUDA runtime round-trip", "WARN", "missing")]) == "READY TO BUILD"
    assert cli.readiness([*base, cli.Check("CUDA runtime round-trip", "PASS", "ok")]) == "READY TO RUN"
    assert cli.readiness([*base, cli.Check("Acquisition binding", "FAIL", "broken")]) == "NOT READY"


def test_build_command_targets_current_interpreter(tmp_path: Path):
    command = cli.build_command(tmp_path, "89", no_build_isolation=True)
    assert command[:4] == [sys.executable, "-m", "pip", "install"]
    assert "--verbose" in command
    assert str(tmp_path) in command
    assert "cmake.define.CMAKE_CUDA_ARCHITECTURES=89" in command
    assert command[-1] == "--no-build-isolation"


def test_gpu_preset_writes_manifest(monkeypatch, tmp_path: Path, capsys):
    monkeypatch.setattr(
        cli,
        "_gpu_roundtrip",
        lambda: {
            "preset": "signal-acquisition",
            "backend": "CUDA",
            "elapsed_ms": 12.5,
            "sample_count": 4092,
            "prn": 1,
            "doppler_hz": 750.0,
            "acquired": True,
        },
    )
    assert cli.main(["run", "--output-dir", str(tmp_path)]) == 0
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["backend"] == "CUDA"
    assert manifest["acquired"] is True
    assert "PASS" in capsys.readouterr().out


def test_doctor_json_report(monkeypatch, tmp_path: Path):
    checks = [
        cli.Check("Python", "PASS", "ok"),
        cli.Check("NVIDIA GPU/driver", "PASS", "ok"),
        cli.Check("CUDA compiler", "PASS", "ok"),
        cli.Check("CMake", "PASS", "ok"),
        cli.Check("CUDA runtime round-trip", "WARN", "not built"),
    ]
    monkeypatch.setattr(cli, "collect_diagnostics", lambda runtime_test=True: checks)
    output = tmp_path / "doctor.json"
    assert cli.main(["doctor", "--json", str(output)]) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["state"] == "READY TO BUILD"
