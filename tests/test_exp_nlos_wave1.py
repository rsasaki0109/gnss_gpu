"""Smoke tests for NLOS soft-weight and GMM evaluation experiments."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_GML = REPO_ROOT / "data" / "sample_plateau.gml"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mask_csv(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if not SAMPLE_GML.exists():
        pytest.skip("sample PLATEAU CityGML file not found")
    exporter = _load_module(
        "export_plateau_nlos_demo_mask",
        REPO_ROOT / "experiments" / "export_plateau_nlos_demo_mask.py",
    )
    path = tmp_path_factory.mktemp("nlos") / "mask.csv"
    exporter.export_mask_csv(path)
    return path


def test_nlos_soft_weight_sweep_finds_best_row(mask_csv: Path):
    sweep = _load_module(
        "exp_nlos_soft_weight_sweep",
        REPO_ROOT / "experiments" / "exp_nlos_soft_weight_sweep.py",
    )
    summary = sweep.sweep_soft_weights(
        mask_csv,
        residual_thresholds=(10.0,),
        pr_accel_thresholds=(5.0,),
        n_particles=800,
    )
    assert len(summary["rows"]) == 4
    assert float(summary["best"]["rms_m"]) > 0.0
    assert float(summary["best"]["n_epochs"]) == 70.0


def test_gmm_nlos_eval_beats_naive_or_reports_metrics(mask_csv: Path):
    gmm_eval = _load_module(
        "exp_gmm_nlos_eval",
        REPO_ROOT / "experiments" / "exp_gmm_nlos_eval.py",
    )
    summary = gmm_eval.evaluate_gmm_configs(
        mask_csv,
        n_particles=800,
        configs=((0.7, 15.0, 30.0),),
    )
    assert summary["configs"]
    assert float(summary["best"]["rms_m"]) > 0.0
    assert float(summary["mask_soft_baseline_rms_m"]) < float(summary["naive_baseline_rms_m"])
