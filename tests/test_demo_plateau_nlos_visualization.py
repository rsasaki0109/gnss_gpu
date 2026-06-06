"""Tests for the standalone PLATEAU NLOS visualization demo."""

import importlib.util
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
DEMO_PATH = REPO_ROOT / "examples" / "demo_plateau_nlos_visualization.py"
SAMPLE_GML = REPO_ROOT / "data" / "sample_plateau.gml"


def _load_demo():
    spec = importlib.util.spec_from_file_location("demo_plateau_nlos_visualization", DEMO_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_plateau_nlos_visualization_writes_standalone_html(tmp_path):
    if not SAMPLE_GML.exists():
        pytest.skip("sample PLATEAU CityGML file not found")

    demo = _load_demo()
    output = tmp_path / "plateau_nlos.html"
    path, result = demo.render_visualization(output)

    assert path == output
    assert path.exists()
    assert result["plateau_rms_m"] < result["naive_rms_m"]

    html = path.read_text(encoding="utf-8")
    assert "PLATEAU NLOS Visualization" in html
    assert "PLATEAU-aware SPP" in html
    assert "Sky plot at worst naive epoch" in html
    assert "<svg" in html
    assert "http" not in html
