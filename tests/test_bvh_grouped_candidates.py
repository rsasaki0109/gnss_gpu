"""Original/off/on direction reuse, immutable inputs, dispatch and lifetime."""

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
    or not os.environ.get("GNSS_BVH_GROUP_CANDIDATE_EXTENSION")
):
    pytest.skip(
        "Explicit Windows x64 grouped-direction probe required", allow_module_level=True
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
from bench_raytrace import _make_building_mesh  # noqa: E402

pytestmark = [pytest.mark.gpu, pytest.mark.cuda]


def bits(a, b):
    for x, y in zip(a, b, strict=True):
        assert x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes()


@pytest.fixture(scope="module")
def modules():
    return (
        _load_extension(
            Path(os.environ["GNSS_BVH_BASELINE_EXTENSION"]),
            "group_direction_original._bvh",
        ),
        _load_extension(
            Path(os.environ["GNSS_BVH_GROUP_CANDIDATE_EXTENSION"]),
            "group_direction_candidate._bvh",
        ),
    )


def compare(modules, nodes, mesh, rx, sat, active, change=True):
    old, new = modules
    before = _digest(nodes, mesh, rx, sat)
    scenes = [
        old.DeviceBVH(nodes, mesh),
        new.DeviceBVH(nodes, mesh),
        new.DeviceBVH(nodes, mesh, grouped_candidates=True),
    ]
    try:
        expected = scenes[0].trace_batch(rx, sat)
        for index, scene in enumerate(scenes[1:]):
            actual = scene.trace_batch(rx, sat)
            bits(expected, actual)
            assert scene.grouped_candidates_active is (active if index else False)
            assert (scene.group_metadata_bytes > 0) is (active if index else False)
            assert scene.group_metadata_bytes <= 128 * 1024 * 1024
            allocated = scene.allocated_bytes
            bits(actual, scene.trace_batch(rx, sat))
            bits(expected[1:], scene.multipath_batch(rx, sat))
            bits(expected[:1], (scene.check_los_batch(rx, sat),))
            assert scene.allocated_bytes == allocated
        if active:
            assert 0 < scenes[2].group_count * 10 <= len(mesh) * 7
        if change:
            rx2, sat2 = rx + 0.375, sat - 0.25
            bits(scenes[0].trace_batch(rx2, sat2), scenes[2].trace_batch(rx2, sat2))
        scenes[2].set_grouped_candidates_enabled(False)
        bits(expected, scenes[2].trace_batch(rx, sat))
        assert scenes[2].grouped_candidates_active is False
        scenes[2].set_grouped_candidates_enabled(True)
        bits(expected, scenes[2].trace_batch(rx, sat))
        assert scenes[2].grouped_candidates_active is active
        assert _digest(nodes, mesh, rx, sat) == before
        return [a.copy() for a in expected]
    finally:
        for scene in scenes:
            scene.close()
            assert scene.allocated_bytes == 0


@pytest.mark.parametrize(
    "count,epochs,satellites,active",
    [
        (12000, 64, 8, True),
        (12000, 32, 32, True),
        (120000, 16, 32, True),
        (4080, 64, 8, False),
        (12000, 1, 12, False),
        (12000, 128, 4, False),
        (12000, 1, 512, True),
        (12000, 1025, 8, True),
    ],
)
def test_bulk_fallback_chunks_and_toggles_without_ray_cache(
    modules, count, epochs, satellites, active
):
    nodes, mesh = _build(modules[0], _make_building_mesh(count // 12))
    rx, sat = _rays(epochs, satellites, seed=271)
    if count > 12000 or epochs > 1000 or satellites > 128:
        rx[1:] = np.nan
        sat[:, 2:] = np.nan
    compare(modules, nodes, mesh, rx, sat, active)


@pytest.mark.parametrize("scale", [1e-8, 1.0, 1e8, 1e100])
def test_scale_and_ecef_keep_original_intersection_and_normals(modules, scale):
    shift = np.array([-3957199.0, 3310205.0, 3737911.0]) if scale == 1 else np.zeros(3)
    nodes, mesh = _build(modules[0], _make_building_mesh(1000) * scale + shift)
    rx, sat = _rays(64, 8, seed=317, shift=shift)
    if scale != 1:
        rx *= scale
        sat *= scale
    # Extreme normals can retain many distinct signed zeros: output rules stay
    # exact regardless of whether enough groups pass the reuse threshold.
    before = _digest(nodes, mesh, rx, sat)
    old, new = modules
    a, b = (
        old.DeviceBVH(nodes, mesh),
        new.DeviceBVH(nodes, mesh, grouped_candidates=True),
    )
    try:
        bits(a.trace_batch(rx, sat), b.trace_batch(rx, sat))
        assert _digest(nodes, mesh, rx, sat) == before
        assert b.group_metadata_bytes <= 128 * 1024 * 1024
    finally:
        a.close()
        b.close()


def test_distinct_geometry_has_original_fallback(modules):
    nodes, mesh = _build(modules[0], _random_mesh(4096))
    rx, sat = _rays(64, 8, near=True)
    compare(modules, nodes, mesh, rx, sat, False)


def test_owned_snapshot_survives_source_mutation(modules):
    nodes, mesh = _build(modules[0], _make_building_mesh(1000))
    rx, sat = _rays(64, 8)
    old, new = modules
    a, b = (
        old.DeviceBVH(nodes, mesh),
        new.DeviceBVH(nodes, mesh, grouped_candidates=True),
    )
    try:
        expected = a.trace_batch(rx, sat)
        mesh[:] = -77
        nodes[:] = 99
        bits(expected, b.trace_batch(rx, sat))
        assert b.grouped_candidates_active
    finally:
        a.close()
        b.close()


def test_legacy_geometry_corpus(modules):
    for name, arguments in _validation_cases(modules[0]):
        rx, sat, nodes, mesh = arguments
        compare(modules, nodes, mesh, rx, sat, False, change=False)


@pytest.mark.parametrize("count", [4096, 4097, 8192])
def test_identical_groups_are_split_into_at_most_eight_members(modules, count):
    triangle = np.array([[1.0, -20.0, -10.0], [1.0, 20.0, -10.0], [1.0, 0.0, 20.0]])
    mesh = np.repeat(triangle[None], count, axis=0)
    bounds = np.concatenate((triangle.min(axis=0), triangle.max(axis=0)))
    nodes = np.array([[*bounds, -1.0, -1.0, 0.0, count]])
    rx, sat = _rays(64, 8, near=True)
    sat[:, 1:] = np.nan
    old, new = modules
    a, b = (
        old.DeviceBVH(nodes, mesh),
        new.DeviceBVH(nodes, mesh, grouped_candidates=True),
    )
    try:
        bits(a.trace_batch(rx, sat), b.trace_batch(rx, sat))
        assert b.grouped_candidates_active
        assert b.group_count == (count + 7) // 8
        assert b.group_metadata_bytes == 4 * (count + b.group_count + 1)
    finally:
        a.close()
        b.close()
        assert a.allocated_bytes == b.allocated_bytes == 0
