"""Regression tests for the CPU-only NLOS simulation demo."""

import importlib.util
import sys
from pathlib import Path

import numpy as np


DEMO_PATH = Path(__file__).resolve().parent.parent / "examples" / "demo_nlos_simulation.py"


def _load_demo():
    spec = importlib.util.spec_from_file_location("demo_nlos_simulation", DEMO_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_demo_runs_and_geometry_aware_solver_beats_naive():
    demo = _load_demo()
    result = demo.main()

    assert result["n_epochs"] == 80
    assert result["n_satellites"] == 12
    assert result["n_buildings"] == 12
    assert 0.30 < result["nlos_fraction"] < 0.70

    # The simulated RF-quality metadata should separate clean and blocked paths.
    assert result["los_cn0_mean_dbhz"] - result["nlos_cn0_mean_dbhz"] > 15.0
    assert result["nlos_bias_mean_m"] > 30.0

    # This demo's point is 3D geometry: ray-mask correction should clearly
    # recover accuracy when NLOS errors are correlated by the canyon.
    assert result["geometry_p50_m"] < result["naive_p50_m"]
    assert result["geometry_rms_m"] < 0.65 * result["naive_rms_m"]
    assert result["geometry_wins"] >= 60


def test_local_ray_classifier_blocks_low_elevation_building_path():
    demo = _load_demo()
    building = demo.BoxBuilding(
        center_e_m=20.0,
        center_n_m=0.0,
        width_e_m=10.0,
        depth_n_m=10.0,
        height_m=20.0,
    )
    rx = np.array([0.0, 0.0, 1.8], dtype=np.float64)

    low_east, high_east = demo.classify_los_nlos(
        rx,
        [(90.0, 5.0), (90.0, 80.0)],
        [building],
    )

    assert not bool(low_east)
    assert bool(high_east)
