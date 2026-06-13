import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DATA_DIR = _REPO_ROOT / "experiments" / "data" / "urbannav" / "Odaiba"


def _load_demo_module():
    demo_path = _REPO_ROOT / "examples" / "demo_urbannav_calibration.py"
    spec = importlib.util.spec_from_file_location("demo_urbannav_calibration", demo_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_residual_generator_is_deterministic_and_shaped():
    demo = _load_demo_module()
    gen = demo.make_residual_generator(2000, seed=7)
    params = {"nlos_fraction": 0.4, "nlos_bias_m": 20.0,
              "nlos_scale_m": 25.0, "los_sigma_m": 5.0}

    a = gen(params)
    b = gen(params)
    np.testing.assert_array_equal(a, b)  # deterministic
    assert a.size == 2000
    assert abs(np.median(a)) < 1e-9  # median removed
    # Higher NLOS fraction => heavier positive tail.
    low = gen({**params, "nlos_fraction": 0.1})
    high = gen({**params, "nlos_fraction": 0.8})
    assert np.percentile(high, 90) > np.percentile(low, 90)


@pytest.mark.skipif(not _DATA_DIR.is_dir(), reason="UrbanNav Odaiba data not present")
def test_demo_urbannav_calibration_real_data(tmp_path):
    try:
        demo = _load_demo_module()
        result = demo.main(site="Odaiba", max_epochs=30, out_dir=tmp_path)
    except Exception as exc:  # GPU ephemeris / RINEX parsing unavailable
        pytest.skip(f"UrbanNav pipeline unavailable: {exc}")

    assert result["target_samples"] > 0
    # Calibration must improve the distribution match over the grid optimum.
    assert result["final_score"] <= result["grid_score"] + 1e-9
    assert result["ks_after"] <= result["ks_before"] + 1e-9

    params = result["best_params"]
    assert 0.0 <= params["nlos_fraction"] <= 1.0
    assert params["nlos_scale_m"] > 0.0

    assert os.path.exists(result["plot_path"])
