"""Smoke tests for NLOS Wave 3: PF eval and DD weight scaling."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from gnss_gpu.nlos_mask import (
    NlosMaskTables,
    dd_pair_nlos_factors,
    scale_dd_result_weights_by_nlos_mask,
)

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


def test_scale_dd_result_weights_by_nlos_mask_mutates_weights():
    from gnss_gpu.dd_pseudorange import DDPseudorangeResult

    result = DDPseudorangeResult(
        dd_pseudorange_m=np.array([1.0], dtype=np.float64),
        sat_ecef_k=np.array([[1.0, 0.0, 0.0]], dtype=np.float64),
        sat_ecef_ref=np.array([[2.0, 0.0, 0.0]], dtype=np.float64),
        base_range_k=np.array([1.0], dtype=np.float64),
        base_range_ref=np.array([2.0], dtype=np.float64),
        dd_weights=np.array([2.0], dtype=np.float64),
        ref_sat_ids=("G01",),
        n_dd=1,
        sat_ids=("G02",),
    )
    tables = NlosMaskTables(weak={5: {"G02"}}, strong={})
    scale_dd_result_weights_by_nlos_mask(result, 5, tables, k_weak=2.0)
    assert result.dd_weights[0] == pytest.approx(1.0)


def test_compute_dd_skips_pair_rescale_when_rover_weights_already_masked():
    from gnss_gpu.dd_pseudorange import DDPseudorangeResult
    from gnss_gpu.dd_pseudorange_observation import compute_dd_pseudorange_observation
    from gnss_gpu.pf_smoother_config import DDPseudorangeConfig

    class _FakeComputer:
        def compute_dd(self, tow, measurements, pf_estimate, rover_weights=None):
            return DDPseudorangeResult(
                dd_pseudorange_m=np.array([1.0], dtype=np.float64),
                sat_ecef_k=np.array([[1.0, 0.0, 0.0]], dtype=np.float64),
                sat_ecef_ref=np.array([[2.0, 0.0, 0.0]], dtype=np.float64),
                base_range_k=np.array([1.0], dtype=np.float64),
                base_range_ref=np.array([2.0], dtype=np.float64),
                dd_weights=np.array([2.0], dtype=np.float64),
                ref_sat_ids=("G01",),
                n_dd=1,
                sat_ids=("G02",),
            )

    tables = NlosMaskTables(weak={5: {"G02"}}, strong={})
    decision = compute_dd_pseudorange_observation(
        _FakeComputer(),
        100.0,
        [],
        None,
        np.array([0.5], dtype=np.float64),
        DDPseudorangeConfig(enabled=True),
        nlos_tables=tables,
        epoch_idx=5,
        nlos_mask_applied_to_rover_weights=True,
    )
    assert decision.result is not None
    assert decision.result.dd_weights[0] == pytest.approx(2.0)


def test_ppc_pf_nlos_eval_mask_soft_beats_naive(mask_csv: Path):
    eval_mod = _load_module(
        "exp_ppc_pf_nlos_eval",
        REPO_ROOT / "experiments" / "exp_ppc_pf_nlos_eval.py",
    )
    summary = eval_mod.evaluate_pf_nlos_mask_path(mask_csv, n_particles=800)
    assert float(summary["mask_soft_rms_m"]) < float(summary["naive_rms_m"])
    assert float(summary["wins_mask_over_naive"]) >= 50.0
