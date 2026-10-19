"""Sparse valid rays exercise all new kernels and chunk tails under checking."""

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

if not os.environ.get("GNSS_BVH_TILED_EXTENSION") or os.name != "nt":
    pytest.skip("Explicit Windows tiled probe required", allow_module_level=True)
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "bvh_tiled_sanitizer_frozen_cases", ROOT / "tests/test_bvh_tiled_geometry.py"
)
OLD = importlib.util.module_from_spec(spec)
spec.loader.exec_module(OLD)
modules = OLD.modules
pytestmark = OLD.pytestmark


@pytest.mark.parametrize(
    "count,epochs,satellites", [(4096, 64, 8), (4097, 513, 32), (8193, 64, 16)]
)
def test_new_kernels_cover_last_tile_and_epoch_group(
    modules, count, epochs, satellites
):
    mesh = np.zeros((count, 3, 3))
    mesh[:13] = np.random.default_rng(74).uniform(-30.0, 30.0, (13, 3, 3))
    mesh[-1] = [[1.0, -20.0, -10.0], [1.0, 20.0, -10.0], [1.0, 0.0, 20.0]]
    rx = np.zeros((epochs, 3))
    sat = np.full((epochs, satellites, 3), np.nan)
    for epoch in {0, epochs - 1, epochs // 2, 127, 128, 129, 255, 256}:
        if epoch < epochs:
            sat[epoch, : min(3, satellites)] = [
                [0.0, 0.0, 10.0],
                [300.0, -21.0, 40.0],
                [-300.0, 10.0, 40.0],
            ][: min(3, satellites)]
    result = OLD.compare(modules, OLD.leaf(mesh), mesh, rx, sat, True)
    assert np.count_nonzero(result[2]) > 0
