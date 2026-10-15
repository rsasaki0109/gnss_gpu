"""Receiver mirrors refresh per call and retain uncached reflection bits."""

import numpy as np
import pytest

from gnss_gpu.bvh import BVHAccelerator
from test_bvh_normal_cache import _assert_bits, _leaf, _mesh, _reference

native = pytest.importorskip('gnss_gpu._bvh')
pytestmark = [pytest.mark.gpu, pytest.mark.cuda]


def _rays(epochs, satellites, seed=47):
    rng = np.random.default_rng(seed)
    return rng.uniform(-30., 30., (epochs, 3)), rng.uniform(-300., 300., (epochs, satellites, 3))


@pytest.mark.parametrize('count', [8191, 8192, 8193])
@pytest.mark.parametrize('satellites', [15, 16, 17])
def test_mesh_and_satellite_dispatch_boundaries_match_uncached(count, satellites):
    nodes, mesh = _mesh(count)
    rx, sat = _rays(2, satellites)
    scene = native.DeviceBVH(nodes, mesh)
    try:
        expected = _reference(rx, sat, nodes, mesh)
        for _ in range(2):
            _assert_bits(scene.trace_batch(rx, sat), expected)
            _assert_bits(scene.multipath_batch(rx, sat), expected[1:])
        _assert_bits(scene.trace(rx[0], sat[0]), tuple(array[0] for array in expected))
    finally:
        scene.close()
    assert scene.allocated_bytes == 0


@pytest.mark.parametrize('epochs', [1, 2, 4, 17])
@pytest.mark.parametrize('satellites', [16, 32])
def test_epoch_stride_and_changing_receiver_values_match(epochs, satellites):
    nodes, mesh = _mesh(8193)
    rx, sat = _rays(epochs, satellites)
    scene = native.DeviceBVH(nodes, mesh)
    try:
        for shift in (0., 30., -30.):
            rx += shift
            sat -= shift
            expected = _reference(rx, sat, nodes, mesh)
            _assert_bits(scene.trace_batch(rx, sat), expected)
    finally:
        scene.close()


@pytest.mark.parametrize('epochs', [170, 171])
def test_mirror_scratch_cap_on_both_sides(epochs):
    mesh = np.random.default_rng(79).uniform(-100., 100., (8192, 3, 3))
    nodes = _leaf(mesh)
    rx, sat = _rays(epochs, 16)
    scene = native.DeviceBVH(nodes, mesh)
    try:
        # Use multipath alone so the expected non-mirror resident allocations
        # do not include a LOS buffer. One tile per ray for this large batch.
        expected = _reference(rx, sat, nodes, mesh)[1:]
        _assert_bits(scene.multipath_batch(rx, sat), expected)
        base = (len(nodes) * 64 + mesh.nbytes + len(mesh) * (4 + 32)
                + rx.nbytes + sat.nbytes + epochs * 16 * (4 * 8 + 40))
        extra = scene.allocated_bytes - base
        assert extra == min(epochs, (32 * 1024 * 1024) // 24 // len(mesh)) * len(mesh) * 24
        assert 0 <= extra <= 32 * 1024 * 1024
        allocated = scene.allocated_bytes
        _assert_bits(scene.multipath_batch(rx, sat), expected)
        assert scene.allocated_bytes == allocated
    finally:
        scene.close()
    assert scene.allocated_bytes == 0


def test_nonfinite_first_batch_and_coincident_rays_do_not_poison_next_call():
    nodes, mesh = _mesh(8193)
    rx, sat = _rays(3, 17)
    scene = native.DeviceBVH(nodes, mesh)
    try:
        _assert_bits(scene.trace_batch(np.full_like(rx, np.nan), sat),
                     _reference(np.full_like(rx, np.nan), sat, nodes, mesh))
        sat[0, 0] = rx[0]
        sat[0, 1] = np.inf
        sat[1, 1] = np.nan
        rx[2] = np.nan
        _assert_bits(scene.trace_batch(rx, sat), _reference(rx, sat, nodes, mesh))
        rx[2] = [1., 2., 3.]
        _assert_bits(scene.trace_batch(rx, sat), _reference(rx, sat, nodes, mesh))
    finally:
        scene.close()


@pytest.mark.parametrize('reverse', [False, True])
def test_tied_reflections_preserve_original_visit_order(reverse):
    mesh = np.repeat(np.array([
        [[-1., -20., -10.], [-1., 20., -10.], [-1., 0., 20.]],
        [[1., -20., -10.], [1., 20., -10.], [1., 0., 20.]],
    ]), 4096, axis=0)
    nodes = np.array([
        [-1., -20., -10., 1., 20., 20., 1., 2., 0., 0.],
        [-1., -20., -10., -1., 20., 20., -1., -1., 0., 4096.],
        [1., -20., -10., 1., 20., 20., -1., -1., 4096., 4096.],
    ])
    if reverse:
        nodes[0, [6, 7]] = nodes[0, [7, 6]]
    rx = np.zeros((1, 3))
    sat = np.repeat(np.array([[[0., 0., 10.]]]), 16, axis=1)
    scene = native.DeviceBVH(nodes, mesh)
    try:
        expected = _reference(rx, sat, nodes, mesh)
        _assert_bits(scene.trace_batch(rx, sat), expected)
        np.testing.assert_allclose(expected[1][0, 0], [-1. if reverse else 1., 0., 5.],
                                   rtol=0., atol=1e-14)
    finally:
        scene.close()


def test_snapshot_ownership_scratch_reuse_and_fallback_transitions():
    nodes, mesh = _mesh(8193)
    inputs = []
    for index, shape in enumerate(((1, 32), (4, 16), (1, 15), (1, 4097), (2, 17), (1, 1))):
        rx, sat = _rays(*shape, seed=71 + index)
        inputs.append((rx, sat, _reference(rx, sat, nodes, mesh)))
    scene = native.DeviceBVH(nodes, mesh)
    nodes.fill(0.)
    mesh.fill(0.)
    retained = []
    try:
        for rx, sat, expected in inputs:
            output = scene.trace_batch(rx, sat)
            _assert_bits(output, expected)
            retained.append((output, tuple(array.copy() for array in output)))
            allocated = scene.allocated_bytes
            _assert_bits(scene.trace_batch(rx, sat), expected)
            assert scene.allocated_bytes == allocated
            for old, snapshot in retained:
                _assert_bits(old, snapshot)
    finally:
        scene.close()
    assert scene.allocated_bytes == 0


def test_los_only_does_not_prepare_receiver_mirrors():
    nodes, mesh = _mesh(8193)
    rx, sat = _rays(3, 32)
    scene = native.DeviceBVH(nodes, mesh)
    try:
        geometry = len(nodes) * 64 + mesh.nbytes
        assert scene.allocated_bytes == geometry
        scene.check_los_batch(rx, sat)
        assert scene.allocated_bytes == geometry + rx.nbytes + sat.nbytes + 3 * 32 * 4
    finally:
        scene.close()


def test_scene_update_release_and_reupload_refresh_mirrors():
    _, mesh = _mesh(8193)
    rx, sat = _rays(2, 17)
    model = BVHAccelerator(mesh)
    try:
        for transform in (np.eye(3), np.diag([1., -1., 1.]), np.diag([2., 3., 4.])):
            model.update_triangles(mesh @ transform)
            los, points, delays = _reference(rx, sat, model._nodes_flat, model._sorted_tris)
            expected = los, delays, points
            _assert_bits(model.trace_batch(rx, sat), expected)
            model.release_gpu()
            _assert_bits(model.trace_batch(rx, sat), expected)
    finally:
        model.release_gpu()
