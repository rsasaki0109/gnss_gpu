"""Resident reflection normals retain uncached FP64 results and scene lifetime."""

import numpy as np
import pytest

from gnss_gpu.bvh import BVHAccelerator

native = pytest.importorskip('gnss_gpu._bvh')
pytestmark = [pytest.mark.gpu, pytest.mark.cuda]


def _mesh(count):
    rng = np.random.default_rng(count)
    mesh = rng.uniform(-100., 100., (count, 3, 3))
    mesh[::17, 2] = mesh[::17, 0]
    nodes, order = native.bvh_build(mesh)
    return nodes, mesh[order].copy()


def _rays(count, seed=13):
    rng = np.random.default_rng(seed)
    return rng.uniform(-30., 30., (1, 3)), rng.uniform(-300., 300., (1, count, 3))


def _leaf(mesh):
    bounds = np.concatenate((mesh.min(axis=(0, 1)), mesh.max(axis=(0, 1))))
    return np.array([[*bounds, -1., -1., 0., len(mesh)]])


def _assert_bits(actual, expected):
    for new, old in zip(actual, expected, strict=True):
        assert new.dtype == old.dtype and new.shape == old.shape
        np.testing.assert_array_equal(new.view(np.uint8), old.view(np.uint8))


def _reference(rx, sat, nodes, mesh):
    # The stateless API does not use the resident normal preparation kernel.
    return (native.raytrace_los_check_bvh_batch(rx, sat, nodes, mesh),
            *native.raytrace_multipath_bvh_batch(rx, sat, nodes, mesh))


@pytest.mark.parametrize('count', [511, 512, 513, 1025, 4097])
@pytest.mark.parametrize('rays', [1, 12, 257, 4097])
def test_mesh_and_batch_boundaries_match_uncached(count, rays):
    nodes, mesh = _mesh(count)
    rx, sat = _rays(rays)
    expected = _reference(rx, sat, nodes, mesh)
    scene = native.DeviceBVH(nodes, mesh)
    try:
        for _ in range(2):
            _assert_bits(scene.trace_batch(rx, sat), expected)
            _assert_bits(scene.multipath_batch(rx, sat), expected[1:])
        _assert_bits(scene.trace(rx[0], sat[0]), tuple(array[0] for array in expected))
    finally:
        scene.close()
    assert scene.allocated_bytes == 0


@pytest.mark.parametrize('scale', [1e-100, 1e-8, 3e-8, 1e-7, 1., 1e50, 1e100])
def test_normal_underflow_cutoff_and_large_geometry_match(scale):
    mesh = np.random.default_rng(137).uniform(-1., 1., (513, 3, 3)) * scale
    mesh[::17, 2] = mesh[::17, 0]
    nodes = _leaf(mesh)
    rx, sat = _rays(17)
    rx *= scale
    sat *= scale
    scene = native.DeviceBVH(nodes, mesh)
    try:
        expected = _reference(rx, sat, nodes, mesh)
        _assert_bits(scene.trace_batch(rx, sat), expected)
        _assert_bits(scene.trace_batch(rx, sat), expected)
    finally:
        scene.close()


@pytest.mark.parametrize('area', [np.nextafter(1e-15, 0.), 1e-15, np.nextafter(1e-15, np.inf)])
@pytest.mark.parametrize('orientation', [(0, 1, 2), (0, 2, 1), (1, 2, 0)])
def test_strict_normal_cutoff_and_winding_match(area, orientation):
    mesh = np.repeat(np.array([[[0., 0., 0.], [1., 0., 0.], [0., area, 0.]]]), 513, axis=0)
    mesh[::2, :, 2] = -0.
    mesh = mesh[:, orientation].copy()
    nodes = _leaf(mesh)
    rx = np.array([[.25, area / 4., 1.]])
    sat = np.array([[[.25, area / 4., 10.], [.5, area / 4., -10.]]])
    scene = native.DeviceBVH(nodes, mesh)
    try:
        _assert_bits(scene.trace_batch(rx, sat), _reference(rx, sat, nodes, mesh))
    finally:
        scene.close()


@pytest.mark.parametrize('reverse', [False, True])
def test_equal_delays_follow_bvh_order_after_normal_preparation(reverse):
    mesh = np.repeat(np.array([
        [[-1., -20., -10.], [-1., 20., -10.], [-1., 0., 20.]],
        [[1., -20., -10.], [1., 20., -10.], [1., 0., 20.]],
    ]), 256, axis=0)
    nodes = np.array([
        [-1., -20., -10., 1., 20., 20., 1., 2., 0., 0.],
        [-1., -20., -10., -1., 20., 20., -1., -1., 0., 256.],
        [1., -20., -10., 1., 20., 20., -1., -1., 256., 256.],
    ])
    if reverse:
        nodes[0, [6, 7]] = nodes[0, [7, 6]]
    rx = np.zeros((1, 3))
    sat = np.array([[[0., 0., 10.]]])
    scene = native.DeviceBVH(nodes, mesh)
    try:
        expected = _reference(rx, sat, nodes, mesh)
        for _ in range(2):
            _assert_bits(scene.trace_batch(rx, sat), expected)
        np.testing.assert_allclose(expected[1][0, 0], [-1. if reverse else 1., 0., 5.],
                                   atol=1e-14, rtol=0.)
    finally:
        scene.close()


def test_invalid_first_batch_does_not_poison_later_reflections():
    nodes, mesh = _mesh(513)
    rx, sat = _rays(17)
    scene = native.DeviceBVH(nodes, mesh)
    try:
        invalid = np.full_like(sat, np.nan)
        _assert_bits(scene.trace_batch(rx, invalid), _reference(rx, invalid, nodes, mesh))
        sat[0, 0] = rx[0]
        sat[0, 1] = np.inf
        _assert_bits(scene.trace_batch(rx, sat), _reference(rx, sat, nodes, mesh))
    finally:
        scene.close()


def test_reused_snapshot_growing_rays_and_returned_array_ownership():
    nodes, mesh = _mesh(513)
    scene = native.DeviceBVH(nodes, mesh)
    references = []
    for index, rays in enumerate([4097, 1, 257, 4096, 12, 1]):
        rx, sat = _rays(rays, seed=30 + index)
        references.append((rx, sat, _reference(rx, sat, nodes, mesh)))
    nodes.fill(0.)
    mesh.fill(0.)
    retained = []
    try:
        for rx, sat, expected in references:
            actual = scene.trace_batch(rx, sat)
            _assert_bits(actual, expected)
            retained.append((actual, tuple(array.copy() for array in actual)))
            allocated = scene.allocated_bytes
            _assert_bits(scene.trace_batch(rx, sat), expected)
            assert scene.allocated_bytes == allocated
            for output, snapshot in retained:
                _assert_bits(output, snapshot)
    finally:
        scene.close()
    assert scene.allocated_bytes == 0


@pytest.mark.parametrize('count', [511, 513])
def test_los_only_has_no_eager_reflection_cache_and_reflection_is_lazy(count):
    nodes, mesh = _mesh(count)
    scene = native.DeviceBVH(nodes, mesh)
    try:
        geometry_bytes = len(nodes) * 64 + mesh.nbytes
        assert scene.allocated_bytes == geometry_bytes
        rx, sat = _rays(4097)
        scene.check_los_batch(rx, sat)
        assert scene.allocated_bytes == geometry_bytes + rx.nbytes + sat.nbytes + 4097 * 4
        scene.multipath_batch(rx, sat)
        assert scene.allocated_bytes == (
            geometry_bytes + rx.nbytes + sat.nbytes + 4097 * (4 + 3 * 8 + 8)
            + ((36 * count + 40 * 4097) if count >= 64 else 0))
    finally:
        scene.close()


def test_mesh_update_release_and_reupload_replace_cached_normals():
    _, mesh = _mesh(513)
    model = BVHAccelerator(mesh)
    rx, sat = _rays(12)
    retained = []
    try:
        for transform in (np.eye(3), np.diag([1., -1., 1.]), np.diag([2., 3., 4.])):
            model.update_triangles(mesh @ transform)
            args = rx, sat, model._nodes_flat, model._sorted_tris
            los, points, delays = _reference(*args)
            expected = los, delays, points
            for _ in range(2):
                _assert_bits(model.trace_batch(rx, sat), expected)
            output = model.trace_batch(rx, sat)
            retained.append((output, tuple(array.copy() for array in output)))
            model.release_gpu()
            _assert_bits(model.trace_batch(rx, sat), expected)
            for old, snapshot in retained:
                _assert_bits(old, snapshot)
    finally:
        model.release_gpu()
