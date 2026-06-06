"""Regression tests for the PLATEAU-backed NLOS simulation demo."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
DEMO_PATH = REPO_ROOT / "examples" / "demo_plateau_nlos_simulation.py"
SAMPLE_GML = REPO_ROOT / "data" / "sample_plateau.gml"


def _load_demo():
    spec = importlib.util.spec_from_file_location("demo_plateau_nlos_simulation", DEMO_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_plateau_demo_runs_and_ray_mask_improves_spp():
    if not SAMPLE_GML.exists():
        pytest.skip("sample PLATEAU CityGML file not found")

    demo = _load_demo()
    result = demo.main()

    assert result["n_triangles"] == 36
    assert result["n_satellites"] == 14
    assert result["n_epochs"] >= 60
    assert result["ray_source"] in {"native BVH", "CPU triangle ray-cast"}

    assert 0.40 < result["nlos_fraction"] < 0.80
    assert result["los_cn0_mean_dbhz"] - result["nlos_cn0_mean_dbhz"] > 10.0
    assert result["nlos_bias_mean_m"] > 20.0

    assert result["plateau_p50_m"] < result["naive_p50_m"]
    assert result["plateau_rms_m"] < 0.50 * result["naive_rms_m"]
    assert result["plateau_wins"] >= 45


def test_cpu_triangle_ray_check_blocks_simple_plate():
    demo = _load_demo()
    rx = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    sats = np.array(
        [
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
        ],
        dtype=np.float64,
    )
    triangles = np.array(
        [
            [
                [5.0, -1.0, -1.0],
                [5.0, 1.0, -1.0],
                [5.0, 0.0, 1.0],
            ]
        ],
        dtype=np.float64,
    )

    is_los = demo._check_los_cpu(rx, sats, triangles)

    assert not bool(is_los[0])
    assert bool(is_los[1])
