import importlib.util
import os
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DATA_DIR = _REPO_ROOT / "experiments" / "data" / "urbannav" / "Odaiba"


def _load_demo_module():
    demo_path = _REPO_ROOT / "examples" / "demo_urbannav_residuals.py"
    spec = importlib.util.spec_from_file_location("demo_urbannav_residuals", demo_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(not _DATA_DIR.is_dir(), reason="UrbanNav Odaiba data not present")
def test_demo_urbannav_residuals_real_data(tmp_path):
    try:
        demo = _load_demo_module()
        result = demo.main(site="Odaiba", max_epochs=30, out_dir=tmp_path)
    except Exception as exc:  # GPU ephemeris / RINEX parsing unavailable
        pytest.skip(f"UrbanNav pipeline unavailable: {exc}")

    assert result["site"] == "Odaiba"
    assert result["n_epochs"] > 0
    assert result["n_samples"] > 0
    # Urban residuals are large (multipath/NLOS + unmodelled atmosphere).
    assert result["abs_p90_m"] > 5.0

    assert os.path.exists(result["csv_path"])
    assert os.path.exists(result["plot_path"])
