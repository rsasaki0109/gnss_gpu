import importlib.util
import os
from pathlib import Path


def _load_demo_module():
    repo_root = Path(__file__).resolve().parents[1]
    demo_path = repo_root / "examples" / "demo_nlos_diffraction.py"

    spec = importlib.util.spec_from_file_location("demo_nlos_diffraction", demo_path)
    assert spec is not None and spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_demo_nlos_diffraction_cpu_only(tmp_path):
    demo = _load_demo_module()
    result = demo.main(out_dir=tmp_path)

    assert result["n_epochs"] == 80
    assert result["n_buildings"] == 10
    assert result["n_edges"] > 0

    # Diffraction rescues otherwise-lost shadowed satellites.
    assert result["n_rescued"] > 0
    assert result["n_samples_on"] > result["n_samples_off"]
    assert result["nlos_count_on"] > result["nlos_count_off"]

    compare = result["compare_on_vs_off"]
    assert "wasserstein" in compare
    assert "ks" in compare

    assert os.path.exists(result["csv_path"])
    assert os.path.exists(result["plot_path"])
