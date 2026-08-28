from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from gnss_gpu import cli
from gnss_gpu.io.rinex_writer import EpochRecord, RinexObsHeader, write_rinex_obs
from gnss_gpu.urbannav_cli import (
    InputInspection,
    UrbanNavRunError,
    inspect_input,
    run_urbannav_pf,
)


def _write_observation(path: Path, *, position: np.ndarray) -> None:
    time = datetime(2024, 7, 20, 9, 52, 30)
    sat_ids = ["G01", "G02", "G03", "G04"]
    write_rinex_obs(
        path,
        RinexObsHeader(
            marker_name=path.stem,
            approx_position_ecef=position,
            obs_types={"G": ["C1C", "S1C"]},
        ),
        [
            EpochRecord(
                time=time,
                sat_ids=sat_ids,
                obs={
                    "C1C": np.full(4, 20_200_000.0),
                    "S1C": np.full(4, 40.0),
                },
            )
        ],
    )


def _write_minimal_bundle(tmp_path: Path) -> Path:
    run_dir = tmp_path / "Odaiba"
    run_dir.mkdir()
    position = np.array([3_978_000.0, 3_350_000.0, 3_695_000.0])
    _write_observation(run_dir / "rover_ublox.obs", position=np.zeros(3))
    _write_observation(run_dir / "base_trimble.obs", position=position)
    (run_dir / "base.nav").write_text("RINEX navigation fixture\n", encoding="utf-8")
    (run_dir / "reference.csv").write_text(
        "GPS TOW (s),ECEF X (m),ECEF Y (m),ECEF Z (m)\n"
        "300000,3978000,3350000,3695000\n",
        encoding="utf-8",
    )
    return run_dir


def test_inspect_missing_input_is_actionable_and_does_not_fetch(tmp_path: Path):
    result = inspect_input(tmp_path / "missing")

    assert result.status == "INVALID"
    assert result.detected_format == "missing path"
    assert "rover observation" in result.missing_files
    assert any("fetch_urbannav_subset.py" in command for command in result.repair_commands)


def test_inspect_valid_urbannav_bundle_reports_contract(monkeypatch, tmp_path: Path):
    run_dir = _write_minimal_bundle(tmp_path)
    monkeypatch.setattr(
        "gnss_gpu.urbannav_cli._nav_contract",
        lambda path: (
            {"messages": 4, "systems": ["G"]},
            [],
            ["RINEX navigation file", "navigation messages: 4"],
        ),
    )

    result = inspect_input(run_dir)

    assert result.ready
    assert result.detected_format == "UrbanNav run (RINEX + reference)"
    assert result.suggested_presets == ["urbannav-pf"]
    assert result.files["rover observation"].name == "rover_ublox.obs"
    assert result.files["base observation"].name == "base_trimble.obs"
    assert result.metadata["ground_truth"]["rows"] == 1


def test_data_inspect_cli_writes_json_for_incomplete_input(tmp_path: Path, capsys):
    input_dir = tmp_path / "partial"
    input_dir.mkdir()
    (input_dir / "rover.obs").write_text("not rinex\n", encoding="utf-8")
    report = tmp_path / "inspect.json"

    assert cli.main(["data", "inspect", str(input_dir), "--json", str(report)]) == 1
    payload = json.loads(report.read_text(encoding="utf-8"))
    output = capsys.readouterr().out
    assert payload["status"] == "INCOMPLETE"
    assert "Missing" in output
    assert "No external data was downloaded" in output


def test_urbannav_pf_writes_compare_ready_artifacts_without_gpu(monkeypatch, tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    files = {
        "rover observation": run_dir / "rover.obs",
        "base observation": run_dir / "base.obs",
        "navigation": run_dir / "base.nav",
        "ground truth": run_dir / "reference.csv",
        "imu": None,
    }
    for path in files.values():
        if path is not None:
            path.write_text("fixture\n", encoding="utf-8")
    inspection = InputInspection(
        input_path=run_dir,
        resolved_path=run_dir,
        detected_format="UrbanNav run (RINEX + reference)",
        status="READY",
        run_dir=run_dir,
        files=files,
    )
    monkeypatch.setattr("gnss_gpu.urbannav_cli.inspect_input", lambda path: inspection)

    truth_point = np.array([3_978_000.0, 3_350_000.0, 3_695_000.0])
    truth = np.tile(truth_point, (3, 1))
    data = {
        "dataset_name": "UrbanNav fixture",
        "n_epochs": 3,
        "times": np.array([300000.0, 300001.0, 300002.0]),
        "ground_truth": truth,
        "sat_ecef": [np.ones((4, 3)) * 20_000_000.0 for _ in range(3)],
        "pseudoranges": [np.full(4, 20_200_000.0) for _ in range(3)],
        "weights": [np.ones(4) for _ in range(3)],
        "n_satellites": 4,
        "constellations": ("G",),
        "dt": 1.0,
    }

    class FakeLoader:
        def __init__(self, path):
            assert path == run_dir

        def load_experiment_data(self, **kwargs):
            assert kwargs["max_epochs"] == 3
            return data

    def fake_wls(sat, pr, weights, max_iter, tolerance):
        return np.r_[truth_point, 0.0], 0

    class FakePF:
        def __init__(self, **kwargs):
            assert kwargs["n_particles"] == 8

        def initialize(self, position, **kwargs):
            self.position = np.asarray(position)

        def predict(self, **kwargs):
            return None

        def update(self, *args, **kwargs):
            return None

        def estimate(self):
            return np.r_[self.position, 0.0]

    result = run_urbannav_pf(
        run_dir,
        tmp_path / "run-output",
        particles=8,
        max_epochs=3,
        loader_factory=FakeLoader,
        wls_solver=fake_wls,
        pf_factory=FakePF,
    )

    assert result["backend"] == "CUDA"
    assert result["metrics"]["pf_rms_2d_m"] == pytest.approx(0.0)
    for artifact in result["artifact_paths"].values():
        assert Path(artifact).is_file()
    assert json.loads(
        (tmp_path / "run-output" / "urbannav_pf_summary.json").read_text(encoding="utf-8")
    )["backend"] == "CUDA"


def test_urbannav_pf_fails_closed_when_cuda_pf_is_unavailable(monkeypatch, tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    inspection = InputInspection(
        input_path=run_dir,
        resolved_path=run_dir,
        detected_format="UrbanNav run (RINEX + reference)",
        status="READY",
        run_dir=run_dir,
        files={},
    )
    monkeypatch.setattr("gnss_gpu.urbannav_cli.inspect_input", lambda path: inspection)

    truth = np.tile(np.array([3_978_000.0, 3_350_000.0, 3_695_000.0]), (1, 1))
    data = {
        "dataset_name": "fixture",
        "n_epochs": 1,
        "times": np.array([1.0]),
        "ground_truth": truth,
        "sat_ecef": [np.ones((4, 3))],
        "pseudoranges": [np.ones(4)],
        "weights": [np.ones(4)],
        "dt": 1.0,
    }

    class Loader:
        def __init__(self, path):
            pass

        def load_experiment_data(self, **kwargs):
            return data

    def bad_pf(**kwargs):
        raise RuntimeError("CUDA unavailable")

    with pytest.raises(UrbanNavRunError, match="does not use CPU fallback"):
        run_urbannav_pf(
            run_dir,
            tmp_path / "output",
            loader_factory=Loader,
            wls_solver=lambda *args: (np.r_[truth[0], 0.0], 0),
            pf_factory=bad_pf,
        )


def test_cli_urbannav_pf_writes_common_manifest(monkeypatch, tmp_path: Path):
    input_file = tmp_path / "rover.obs"
    input_file.write_text("fixture\n", encoding="utf-8")
    output_dir = tmp_path / "output"
    artifact = output_dir / "summary.json"
    artifact.parent.mkdir()
    artifact.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        cli,
        "run_urbannav_pf",
        lambda *args, **kwargs: {
            "preset": "urbannav-pf",
            "backend": "CUDA",
            "elapsed_ms": 12.0,
            "dataset_name": "UrbanNav fixture",
            "n_epochs": 2,
            "metrics": {"pf_rms_2d_m": 1.5, "runtime_ms": 12.0},
            "parameters": {"preset": "urbannav-pf", "particles": 8},
            "input_paths": [input_file],
            "artifact_paths": {"summary": artifact},
        },
    )

    assert (
        cli.main(
            [
                "run",
                "--preset",
                "urbannav-pf",
                "--input",
                str(tmp_path),
                "--output-dir",
                str(output_dir),
                "--particles",
                "8",
            ]
        )
        == 0
    )
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["preset"] == "urbannav-pf"
    assert manifest["schema"] == cli.RUN_MANIFEST_SCHEMA
    assert manifest["backend"] == "CUDA"
    assert manifest["metrics"]["pf_rms_2d_m"] == pytest.approx(1.5)
    assert manifest["artifacts"]["summary"]["sha256"]
