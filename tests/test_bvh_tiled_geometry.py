"""Explicit native tiled geometry: exact original/off/on and resource checks."""

import os
from pathlib import Path
import struct
import sys

import numpy as np
import pytest

if (
    os.name != "nt"
    or sys.version_info[:2] != (3, 12)
    or struct.calcsize("P") != 8
    or not os.environ.get("GNSS_BVH_TILED_EXTENSION")
):
    pytest.skip(
        "Explicit Windows x64 BVH layout probe required", allow_module_level=True
    )

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
from bench_bvh_multipath_compare import (  # noqa: E402
    _build,
    _digest,
    _load_extension,
    _random_mesh,
    _rays,
    _validation_cases,
)

pytestmark = [pytest.mark.gpu, pytest.mark.cuda]


def bits(a, b):
    for x, y in zip(a, b, strict=True):
        assert x.dtype == y.dtype and x.shape == y.shape
        np.testing.assert_array_equal(x.view(np.uint8), y.view(np.uint8))


@pytest.fixture(scope="module")
def modules():
    return (
        _load_extension(
            Path(os.environ["GNSS_BVH_BASELINE_EXTENSION"]), "tiled_baseline._bvh"
        ),
        _load_extension(
            Path(os.environ["GNSS_BVH_TILED_EXTENSION"]), "tiled_candidate._bvh"
        ),
    )


def leaf(mesh):
    bound = np.concatenate((mesh.min(axis=(0, 1)), mesh.max(axis=(0, 1))))
    return np.array([[*bound, -1.0, -1.0, 0.0, len(mesh)]])


def compare(modules, nodes, mesh, rx, sat, active):
    old, new = modules
    input_hash = _digest(nodes, mesh, rx, sat)
    scenes = [
        old.DeviceBVH(nodes, mesh),
        new.DeviceBVH(nodes, mesh),
        new.DeviceBVH(nodes, mesh),
    ]
    scenes[-1].set_tiled_reflections(True)
    try:
        expected = scenes[0].trace_batch(rx, sat)
        for index, scene in enumerate(scenes[1:]):
            value = scene.trace_batch(rx, sat)
            bits(expected, value)
            assert scene.tiled_reflections_active is (active if index else False)
            assert (scene.tiled_geometry_bytes > 0) is (active if index else False)
            allocation = scene.allocated_bytes
            bits(value, scene.trace_batch(rx, sat))
            bits(expected[1:], scene.multipath_batch(rx, sat))
            bits(expected[:1], (scene.check_los_batch(rx, sat),))
            assert scene.allocated_bytes == allocation
        if active:
            tile_bytes = ((len(mesh) + 31) // 32) * 3200
            assert scenes[-1].tiled_geometry_bytes == tile_bytes
            assert (
                scenes[-1].allocated_bytes - scenes[1].allocated_bytes
                == tile_bytes - len(mesh) * 32
            )
        assert _digest(nodes, mesh, rx, sat) == input_hash
        return [v.copy() for v in expected]
    finally:
        for scene in scenes:
            scene.close()
            assert scene.allocated_bytes == 0


@pytest.mark.parametrize(
    "count,epochs,satellites,active",
    [
        (4095, 64, 8, False),
        (4096, 73, 7, False),
        (4096, 64, 8, True),
        (4097, 57, 9, True),
        (4127, 43, 12, True),
        (4128, 43, 12, True),
        (4129, 43, 12, True),
        (8191, 32, 16, True),
        (8192, 32, 16, True),
        (8193, 32, 16, True),
        (12000, 513, 32, True),
    ],
)
def test_dispatch_and_partial_tile_boundaries_are_exact(
    modules, count, epochs, satellites, active
):
    nodes, mesh = _build(modules[0], _random_mesh(count))
    rx, sat = _rays(epochs, satellites, near=True)
    compare(modules, nodes, mesh, rx, sat, active)


@pytest.mark.parametrize("scale", [1e-100, 1e-8, 3e-8, 1.0, 1e50, 1e100])
def test_extreme_and_degenerate_vertices_keep_original_results(modules, scale):
    mesh = _random_mesh(4097) * scale
    nodes = leaf(mesh)
    rx, sat = _rays(64, 8, near=True)
    compare(modules, nodes, mesh, rx * scale, sat * scale, True)


@pytest.mark.parametrize("reverse", [False, True])
def test_equal_delays_preserve_traversal_order_and_negative_zero(modules, reverse):
    mesh = np.repeat(
        np.array(
            [
                [[-1.0, -20.0, -10.0], [-1.0, 20.0, -10.0], [-1.0, 0.0, 20.0]],
                [[1.0, -20.0, -10.0], [1.0, 20.0, -10.0], [1.0, 0.0, 20.0]],
            ]
        ),
        2048,
        axis=0,
    )
    mesh[:, 2, 1] = -0.0
    nodes = np.array(
        [
            [-1.0, -20.0, -10.0, 1.0, 20.0, 20.0, 1.0, 2.0, 0.0, 0.0],
            [-1.0, -20.0, -10.0, -1.0, 20.0, 20.0, -1.0, -1.0, 0.0, 2048.0],
            [1.0, -20.0, -10.0, 1.0, 20.0, 20.0, -1.0, -1.0, 2048.0, 2048.0],
        ]
    )
    if reverse:
        nodes[0, 6:8] = [2.0, 1.0]
    rx = np.zeros((64, 3))
    sat = np.broadcast_to([0.0, 0.0, 10.0], (64, 8, 3)).copy()
    value = compare(modules, nodes, mesh, rx, sat, True)
    assert np.count_nonzero(value[2]) > 0


def test_ecef_padding_and_returned_arrays_survive_layout_changes(modules):
    old, new = modules
    shift = np.array([6378137.0, -4500000.0, 3500000.0])
    nodes, mesh = _build(old, _random_mesh(8193) + shift)
    before, after = old.DeviceBVH(nodes, mesh), new.DeviceBVH(nodes, mesh)
    # A later lazy tile preparation must read the owned GPU snapshot.
    nodes.fill(0.0)
    mesh.fill(0.0)
    saved = []
    try:
        for i, (enabled, epochs, satellites) in enumerate(
            [
                (True, 64, 8),
                (True, 1, 12),
                (False, 64, 8),
                (True, 64, 16),
                (True, 513, 32),
            ]
        ):
            after.set_tiled_reflections(enabled)
            rx, sat = _rays(epochs, satellites, seed=125 + i, near=True, shift=shift)
            rx[-1] = np.nan
            sat[0, 0] = np.inf
            a, b = before.trace_batch(rx, sat), after.trace_batch(rx, sat)
            bits(a, b)
            assert after.tiled_reflections_active is (
                enabled and epochs * satellites >= 512
            )
            saved.append((b, [v.copy() for v in b]))
            for value, original in saved:
                bits(value, original)
    finally:
        before.close()
        after.close()
    assert after.allocated_bytes == after.tiled_geometry_bytes == 0
    with pytest.raises(RuntimeError, match="closed"):
        after.set_tiled_reflections(True)


def test_overlapping_custom_traversal_keeps_legacy_fallback(modules):
    mesh = _random_mesh(4096)
    bounds = leaf(mesh)[0, :6]
    nodes = np.array(
        [[*bounds, 1.0, 1.0, 0.0, 0.0], [*bounds, -1.0, -1.0, 0.0, 4096.0]]
    )
    rx, sat = _rays(64, 8, near=True)
    compare(modules, nodes, mesh, rx, sat, False)


@pytest.mark.parametrize(
    "area", [np.nextafter(1e-15, 0.0), 1e-15, np.nextafter(1e-15, np.inf)]
)
@pytest.mark.parametrize("orientation", [(0, 1, 2), (0, 2, 1), (1, 2, 0)])
def test_strict_normal_cutoff_matches_at_tiled_dispatch(modules, area, orientation):
    mesh = np.repeat(
        np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, area, 0.0]]]), 4096, axis=0
    )
    mesh[::2, :, 2] = -0.0
    mesh = mesh[:, orientation].copy()
    rx = np.broadcast_to([0.25, area / 4.0, 1.0], (64, 3)).copy()
    sat = np.broadcast_to([0.25, area / 4.0, 10.0], (64, 8, 3)).copy()
    compare(modules, leaf(mesh), mesh, rx, sat, True)


@pytest.mark.parametrize("extra", [0, 1])
def test_actual_allocation_cap_boundary(modules, extra):
    count = ((128 * 1024 * 1024) // 3200) * 32 + extra
    mesh = np.zeros((count, 3, 3))
    # Padding exits before triangle evaluation; this tests the real allocation
    # boundary without an unnecessary enormous ray-triangle workload.
    rx = np.full((64, 3), np.nan)
    sat = np.full((64, 8, 3), np.nan)
    compare(modules, leaf(mesh), mesh, rx, sat, not extra)


@pytest.mark.parametrize("count", [1, 64, 4096, 4097, 8192])
def test_rebuilt_cpu_builder_preserves_nodes_and_permutation(modules, count):
    old, new = modules
    for mesh in (_random_mesh(count), np.zeros((count, 3, 3))):
        bits(old.bvh_build(mesh), new.bvh_build(mesh))


def test_legacy_validation_corpus_matches_current_installed_binary(modules):
    for _, (rx, sat, nodes, mesh) in _validation_cases(modules[0]):
        active = len(mesh) >= 4096 and len(rx) * sat.shape[1] >= 512
        compare(modules, nodes, mesh, rx, sat, active)
