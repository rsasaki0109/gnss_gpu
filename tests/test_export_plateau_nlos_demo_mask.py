"""Tests for exporting the PLATEAU NLOS demo mask as experiment CSV."""

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "experiments" / "export_plateau_nlos_demo_mask.py"
SAMPLE_GML = REPO_ROOT / "data" / "sample_plateau.gml"


def _load_exporter():
    spec = importlib.util.spec_from_file_location("export_plateau_nlos_demo_mask", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_export_plateau_nlos_demo_mask_csv_contract(tmp_path):
    if not SAMPLE_GML.exists():
        pytest.skip("sample PLATEAU CityGML file not found")

    exporter = _load_exporter()
    out_csv = tmp_path / "mask.csv"
    summary_json = tmp_path / "mask_summary.json"
    summary = exporter.export_mask_csv(
        out_csv,
        summary_json=summary_json,
        start_tow=1000.0,
        epoch_dt=0.5,
    )

    assert summary["epochs"] == 70
    assert summary["satellites"] == 14
    assert summary["rows"] == 980
    assert 0.40 < summary["nlos_frac"] < 0.80
    assert out_csv.exists()
    assert json.loads(summary_json.read_text(encoding="utf-8"))["rows"] == 980

    with out_csv.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        assert reader.fieldnames[:4] == ["tow", "epoch_idx", "prn", "is_los"]
        rows = list(reader)

    assert len(rows) == 980
    assert rows[0]["tow"] == "1000.000"
    assert rows[0]["epoch_idx"] == "0"
    assert rows[0]["prn"] == "G01"
    assert rows[14]["tow"] == "1000.500"

    los_rows = [row for row in rows if row["is_los"] == "1"]
    nlos_rows = [row for row in rows if row["is_los"] == "0"]
    assert los_rows
    assert nlos_rows
    assert all(float(row["nlos_expected_bias_m"]) == 0.0 for row in los_rows)
    assert all(float(row["nlos_expected_bias_m"]) > 0.0 for row in nlos_rows)

    loaded = {}
    for row in nlos_rows:
        loaded.setdefault(int(row["epoch_idx"]), set()).add(row["prn"])
    assert loaded
    assert sum(len(prns) for prns in loaded.values()) == summary["nlos"]
