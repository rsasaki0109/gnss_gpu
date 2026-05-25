"""Smoke + regression test for the CPU-only urban-canyon getting-started demo.

This demo is featured as the recommended first run in the README, so it must keep
working without a GPU/CUDA build and must keep demonstrating that robust SPP beats
naive least squares under NLOS multipath.
"""

import importlib.util
from pathlib import Path

DEMO_PATH = Path(__file__).resolve().parent.parent / "examples" / "demo_urban_canyon_sim.py"


def _load_demo():
    spec = importlib.util.spec_from_file_location("demo_urban_canyon_sim", DEMO_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_demo_runs_cpu_only_and_robust_beats_naive():
    demo = _load_demo()
    result = demo.main()

    # Deterministic scene (seeded): robust should clearly beat naive WLS.
    assert result["n_epochs"] == 60
    assert result["robust_p50_m"] < result["naive_p50_m"]
    assert result["robust_rms_m"] < result["naive_rms_m"]
    # Robust should win the large majority of epochs and land in the few-metre range.
    assert result["robust_wins"] >= 55
    assert result["robust_p50_m"] < 5.0
    assert result["naive_p50_m"] > 7.0
