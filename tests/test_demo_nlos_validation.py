import importlib.util
import os
from pathlib import Path


def _load_demo_module():
    repo_root = Path(__file__).resolve().parents[1]
    demo_path = repo_root / "examples" / "demo_nlos_validation.py"

    spec = importlib.util.spec_from_file_location("demo_nlos_validation", demo_path)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_demo_nlos_validation_cpu_only(tmp_path):
    demo = _load_demo_module()
    result = demo.main(out_dir=tmp_path)

    assert result["n_epochs"] == 80
    assert result["n_satellites"] == 12
    assert result["n_buildings"] == 10

    assert 0.1 < result["nlos_fraction"] < 0.9
    assert result["nlos_p50_m"] > result["los_p50_m"]
    assert result["nlos_bias_mean_m"] > 5.0

    compare = result["compare"]
    assert "wasserstein" in compare
    assert "ks" in compare
    assert "p50_delta" in compare

    assert os.path.exists(result["csv_path"])
    assert os.path.exists(result["plot_path"])
