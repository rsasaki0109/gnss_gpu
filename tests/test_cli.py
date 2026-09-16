from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from gnss_gpu import cli


def test_source_cli_doctor_works_without_installed_package():
    """The documented pre-build doctor path must not import UrbanNav eagerly."""

    project_root = Path(__file__).resolve().parents[1]
    script = project_root / "python" / "gnss_gpu" / "cli.py"
    result = subprocess.run(
        [sys.executable, "-S", str(script), "doctor", "--skip-runtime"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    # ``-S`` removes site-packages (including NumPy and gnss_gpu), so the
    # native binding checks are expected to fail.  The CLI itself must still
    # render diagnostics instead of raising during module import.
    assert result.returncode == 1
    assert "gnss_gpu GPU doctor" in result.stdout
    assert "ModuleNotFoundError" not in result.stderr


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


def test_run_manifest_has_common_schema_and_hashes(monkeypatch, tmp_path: Path):
    source = tmp_path / "sample.gml"
    source.write_bytes(b"<CityModel>sample</CityModel>\n")
    artifact = tmp_path / "summary.json"
    artifact.write_text('{"ok": true}\n', encoding="utf-8")
    monkeypatch.setattr(
        cli,
        "_query_gpu_info",
        lambda: {"available": True, "name": "test GPU", "driver_version": "1.0"},
    )
    monkeypatch.setattr(cli, "_git_value", lambda root, *args: "deadbeef")
    manifest = cli.build_run_manifest(
        preset="plateau-nlos",
        result={"backend": "CUDA", "elapsed_ms": 42.0, "metrics": {"rms_m": 2.5}},
        parameters={"preset": "plateau-nlos", "seed": 20260606},
        input_paths=[source],
        artifact_paths={"summary": artifact},
        repo_root=tmp_path,
    )
    assert manifest["schema"] == cli.RUN_MANIFEST_SCHEMA
    assert manifest["schema_version"] == 1
    assert manifest["git_sha"] == "deadbeef"
    assert manifest["gpu"]["name"] == "test GPU"
    assert manifest["input_hashes"]
    assert manifest["artifacts"]["summary"]["sha256"]
    assert manifest["metrics"]["runtime_ms"] == 42.0


def test_platform_metadata_falls_back_without_wmi_or_platform_probe(monkeypatch):
    def fail_probe():
        raise RuntimeError("platform provider unavailable")

    # Exercise the individually guarded getters without changing os.name: on
    # Windows changing it would make pathlib instantiate PosixPath for a
    # Windows current working directory.
    assert cli._safe_platform_component(fail_probe) == "unknown"
    monkeypatch.setattr(cli.platform, "platform", fail_probe)
    monkeypatch.setattr(cli.platform, "system", fail_probe)
    monkeypatch.setattr(cli.platform, "release", fail_probe)
    monkeypatch.setattr(cli.platform, "machine", fail_probe)
    monkeypatch.delenv("PROCESSOR_ARCHITEW6432", raising=False)
    monkeypatch.delenv("PROCESSOR_ARCHITECTURE", raising=False)
    monkeypatch.setattr(cli.sys, "getwindowsversion", fail_probe)
    assert cli._safe_platform_info() == "Windows-unknown-unknown"
    manifest = cli.build_run_manifest(
        preset="signal-acquisition",
        result={"backend": "CUDA", "elapsed_ms": 1.0, "acquired": True},
        parameters={"preset": "signal-acquisition"},
        repo_root=Path.cwd(),
    )
    assert manifest["platform"] == "Windows-unknown-unknown"

    # The Windows branch must not call platform.platform()/machine(), which
    # may reach WMI on affected Python/Windows combinations.
    monkeypatch.setattr(cli.os, "name", "nt")
    monkeypatch.setattr(cli.sys, "platform", "win32")
    assert cli._safe_platform_info().startswith("Windows-")


def _write_test_manifest(path: Path, *, rms: float, runtime: float, schema: str = cli.RUN_MANIFEST_SCHEMA):
    path.mkdir(parents=True, exist_ok=True)
    (path / "manifest.json").write_text(
        json.dumps(
            {
                "schema": schema,
                "schema_version": 1,
                "preset": "plateau-nlos",
                "backend": "CUDA",
                "gpu": {},
                "input_hashes": {"sample.gml": "same"},
                "parameters": {"preset": "plateau-nlos"},
                "metrics": {"rms_m": rms, "runtime_ms": runtime},
                "artifacts": {},
            }
        ),
        encoding="utf-8",
    )


def test_compare_writes_markdown_and_reports_improvement(tmp_path: Path, capsys):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_test_manifest(baseline, rms=10.0, runtime=100.0)
    _write_test_manifest(candidate, rms=5.0, runtime=80.0)
    assert cli.main(["compare", str(baseline), str(candidate)]) == 0
    report = (candidate / "comparison.md").read_text(encoding="utf-8")
    output = capsys.readouterr().out
    assert "improved" in report
    assert "rms_m" in output
    assert "Report:" in output


def test_compare_json_output_is_machine_readable(tmp_path: Path, capsys):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_test_manifest(baseline, rms=10.0, runtime=100.0)
    _write_test_manifest(candidate, rms=12.0, runtime=80.0)
    output = tmp_path / "comparison.json"
    assert cli.main(["compare", str(baseline), str(candidate), "--json", str(output)]) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["schema"] == cli.RUN_COMPARISON_SCHEMA
    assert report["metrics"]["rms_m"]["status"] == "regressed"
    assert "runtime_ms" in capsys.readouterr().out


def test_compare_rejects_incompatible_schema(tmp_path: Path, capsys):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_test_manifest(baseline, rms=10.0, runtime=100.0)
    _write_test_manifest(candidate, rms=5.0, runtime=80.0, schema="other_schema")
    assert cli.main(["compare", str(baseline), str(candidate)]) == 2
    assert "unsupported manifest schema" in capsys.readouterr().err


def test_plateau_missing_data_prints_repair_hint(tmp_path: Path, capsys):
    missing = tmp_path / "missing.gml"
    assert cli.main(
        [
            "run",
            "--preset",
            "plateau-nlos",
            "--gml",
            str(missing),
            "--output-dir",
            str(tmp_path / "run"),
        ]
    ) == 1
    error = capsys.readouterr().err
    assert "PLATEAU mesh was not found" in error
    assert "fetch_plateau_subset.py" in error
