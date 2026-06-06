"""Tests for the one-command PLATEAU NLOS replay suite."""

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_SUITE_PATH = REPO_ROOT / "experiments" / "run_plateau_nlos_demo_suite.py"
SAMPLE_GML = REPO_ROOT / "data" / "sample_plateau.gml"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_run_plateau_nlos_demo_suite_writes_combined_outputs(tmp_path):
    if not SAMPLE_GML.exists():
        pytest.skip("sample PLATEAU CityGML file not found")

    runner = _load_module("run_plateau_nlos_demo_suite", RUN_SUITE_PATH)
    outputs = {
        "mask_csv": tmp_path / "mask.csv",
        "mask_summary_json": tmp_path / "mask_summary.json",
        "spp_summary_json": tmp_path / "spp_summary.json",
        "pf_summary_json": tmp_path / "pf_summary.json",
        "fgo_summary_json": tmp_path / "fgo_summary.json",
        "suite_json": tmp_path / "suite.json",
        "suite_md": tmp_path / "suite.md",
        "suite_csv": tmp_path / "suite.csv",
    }

    result = runner.run_suite(**outputs, pf_particles=1500)
    suite = result["suite"]

    assert suite["mask"]["rows"] == 980
    assert suite["mask"]["epochs"] == 70
    assert suite["min_rms_gain_pct"] > 60.0
    assert suite["best_mask_soft_estimator"] == "FGO"

    rows_by_estimator = {row["estimator"]: row for row in suite["rows"]}
    assert set(rows_by_estimator) == {"SPP", "PF", "FGO"}
    assert rows_by_estimator["SPP"]["wins_fraction"] == "48/68"
    assert rows_by_estimator["PF"]["mask_soft_rms_m"] < rows_by_estimator["PF"]["baseline_rms_m"]
    assert rows_by_estimator["FGO"]["mask_soft_rms_m"] < 1.0

    for path in outputs.values():
        assert path.exists()

    saved = json.loads(outputs["suite_json"].read_text(encoding="utf-8"))
    assert saved["best_mask_soft_estimator"] == suite["best_mask_soft_estimator"]
    assert "PLATEAU NLOS Demo Suite" in outputs["suite_md"].read_text(encoding="utf-8")

    with outputs["suite_csv"].open(newline="", encoding="utf-8") as fh:
        csv_rows = list(csv.DictReader(fh))
    assert [row["estimator"] for row in csv_rows] == ["SPP", "PF", "FGO"]
