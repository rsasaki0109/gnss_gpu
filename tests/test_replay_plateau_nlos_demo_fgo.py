"""Tests for replaying the PLATEAU NLOS demo mask through local FGO."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
EXPORTER_PATH = REPO_ROOT / "experiments" / "export_plateau_nlos_demo_mask.py"
REPLAY_PATH = REPO_ROOT / "experiments" / "replay_plateau_nlos_demo_fgo.py"
SAMPLE_GML = REPO_ROOT / "data" / "sample_plateau.gml"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_replay_plateau_nlos_demo_fgo_consumes_mask_csv(tmp_path):
    if not SAMPLE_GML.exists():
        pytest.skip("sample PLATEAU CityGML file not found")

    exporter = _load_module("export_plateau_nlos_demo_mask", EXPORTER_PATH)
    replay = _load_module("replay_plateau_nlos_demo_fgo", REPLAY_PATH)

    mask_csv = tmp_path / "mask.csv"
    exporter.export_mask_csv(mask_csv)
    summary_json = tmp_path / "fgo_replay_summary.json"

    summary = replay.replay_fgo(mask_csv, summary_json=summary_json)

    assert summary["n_complete_mask_epochs"] == 70
    assert summary["n_solved_epochs"] == 70
    assert 0.40 < summary["nlos_frac"] < 0.80

    assert summary["mask_soft_fgo"]["p50_m"] < 0.25 * summary["naive_fgo"]["p50_m"]
    assert summary["mask_soft_fgo"]["rms_m"] < 0.40 * summary["naive_fgo"]["rms_m"]
    assert summary["mask_soft_wins"] >= 60
    assert summary["rms_gain_vs_naive_pct"] > 70.0

    factors = summary["diagnostics"]["mask_soft_fgo"]["factor_counts"]
    assert factors["undiff_pseudorange"] == 980
    assert factors["between"] == 69

    saved = json.loads(summary_json.read_text(encoding="utf-8"))
    assert saved["mask_csv"] == str(mask_csv)
    assert saved["mask_soft_fgo"]["rms_m"] == pytest.approx(
        summary["mask_soft_fgo"]["rms_m"]
    )
