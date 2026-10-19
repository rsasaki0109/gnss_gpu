"""Sparse current-ray inputs exercise the real cache/chunk and cap paths."""

import os
from pathlib import Path
import struct
import sys

import numpy as np
import pytest

if (
    os.name != "nt"
    or struct.calcsize("P") != 8
    or sys.version_info[:2] != (3, 12)
    or not os.environ.get("GNSS_BVH_LAZY_GROUP_EXTENSION")
):
    pytest.skip(
        "Explicit Windows x64 grouped-direction probe required", allow_module_level=True
    )

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
from bench_bvh_multipath_compare import _build, _digest, _load_extension, _rays  # noqa: E402
from bench_raytrace import _make_building_mesh  # noqa: E402


@pytest.fixture(scope="module")
def modules():
    return (
        _load_extension(
            Path(os.environ["GNSS_BVH_BASELINE_EXTENSION"]), "grouped_checker_old._bvh"
        ),
        _load_extension(
            Path(os.environ["GNSS_BVH_LAZY_GROUP_EXTENSION"]),
            "grouped_checker_new._bvh",
        ),
    )


@pytest.mark.parametrize(
    "epochs,satellites,active", [(64, 8, True), (16, 32, True), (1, 512, True)]
)
def test_sparse_group_chunk_cap_close(modules, epochs, satellites, active):
    old, new = modules
    nodes, mesh = _build(old, _make_building_mesh(1000))
    rx, sat = _rays(epochs, satellites, seed=733)
    sat[:, 1:] = np.nan
    rx[1:] = np.nan
    before = _digest(nodes, mesh, rx, sat)
    a, b = (
        old.DeviceBVH(nodes, mesh),
        new.DeviceBVH(nodes, mesh),
    )
    try:
        expected = a.trace_batch(rx, sat)
        assert b.group_preparation_count == b.group_preparation_scratch_bytes == 0
        b.check_los_batch(rx, sat)
        assert b.group_preparation_count == b.group_preparation_scratch_bytes == 0
        b.set_grouped_candidates_enabled(True)
        actual = b.trace_batch(rx, sat)
        assert b.group_preparation_count == 1
        assert b.group_preparation_scratch_bytes == 24 * len(mesh)
        for x, y in zip(expected, actual, strict=True):
            assert (
                x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes()
            )
        assert b.grouped_candidates_active is active
        assert 0 < b.group_count * 10 <= len(mesh) * 7
        assert b.group_metadata_bytes <= 128 * 1024 * 1024
        assert _digest(nodes, mesh, rx, sat) == before
    finally:
        a.close()
        b.close()
        assert a.allocated_bytes == b.allocated_bytes == 0
