"""Bounded large candidate batches retain the independent stateless GPU outputs."""

import numpy as np
import pytest

from test_bvh_normal_cache import _assert_bits, _leaf, _reference, native

pytestmark = [pytest.mark.gpu, pytest.mark.cuda]


def _mesh(count, scale=1.):
    mesh = np.zeros((count, 3, 3))
    mesh[:13] = np.random.default_rng(74).uniform(-30., 30., (13, 3, 3))
    mesh[-1] = [[1., -20., -10.], [1., 20., -10.], [1., 0., 20.]]
    mesh *= scale
    return _leaf(mesh), mesh


def _rays(epochs, satellites, seed=83):
    rng = np.random.default_rng(seed)
    rx = rng.uniform(-5., 5., (epochs, 3))
    sat = np.full((epochs, satellites, 3), np.nan)
    for epoch in sorted({0, epochs-1, epochs//2}):
        sat[epoch, :3] = rng.uniform(-300., 300., (3, 3))
        sat[epoch, 3] = rx[epoch]
    return rx, sat


@pytest.mark.parametrize('count', [8191, 8192, 12000])
@pytest.mark.parametrize('shape', [(1, 4097), (129, 32), (257, 32)])
def test_mesh_boundary_and_large_batches_match_uncached_gpu(count, shape):
    nodes, mesh = _mesh(count)
    rx, sat = _rays(*shape)
    expected = _reference(rx, sat, nodes, mesh)
    scene = native.DeviceBVH(nodes, mesh)
    try:
        _assert_bits(scene.trace_batch(rx, sat), expected)
        allocated = scene.allocated_bytes
        _assert_bits(scene.trace_batch(rx, sat), expected)
        _assert_bits(scene.multipath_batch(rx, sat), expected[1:])
        assert scene.allocated_bytes == allocated
    finally:
        scene.close()
    assert scene.allocated_bytes == 0


@pytest.mark.parametrize('shape', [(1, 4096), (1, 4097), (512, 32), (513, 32), (1, 16385)])
def test_dispatch_and_winner_capacity_boundaries(shape):
    nodes, mesh = _mesh(12000)
    rx, sat = _rays(*shape)
    scene = native.DeviceBVH(nodes, mesh)
    try:
        _assert_bits(scene.trace_batch(rx, sat), _reference(rx, sat, nodes, mesh))
    finally:
        scene.close()


@pytest.mark.parametrize('scale', [1e-100, 1e-8, 3e-8, 1., 1e50, 1e100])
def test_parallel_extreme_geometry_keeps_exact_bits(scale):
    nodes, mesh = _mesh(8192, scale=scale)
    rx, sat = _rays(129, 32)
    rx *= scale
    sat *= scale
    scene = native.DeviceBVH(nodes, mesh)
    try:
        _assert_bits(scene.trace_batch(rx, sat), _reference(rx, sat, nodes, mesh))
    finally:
        scene.close()


@pytest.mark.parametrize('reverse', [False, True])
def test_large_equal_delays_keep_right_first_order(reverse):
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
    rx = np.zeros((129, 3))
    sat = np.full((129, 32, 3), np.nan)
    sat[[0, -1], 0] = [0., 0., 10.]
    scene = native.DeviceBVH(nodes, mesh)
    try:
        expected = _reference(rx, sat, nodes, mesh)
        _assert_bits(scene.trace_batch(rx, sat), expected)
        assert np.all(expected[2][[0, -1], 0] > 0.)
        assert np.all((expected[1][[0, -1], 0, 0] < 0.) == reverse)
    finally:
        scene.close()


def test_ecef_nonfinite_receiver_refresh_and_parallel_serial_transitions():
    offset = np.array([6378137., -4500000., 3500000.])
    nodes, mesh = _mesh(12000)
    nodes[:, :3] += offset
    nodes[:, 3:6] += offset
    mesh += offset
    scene = native.DeviceBVH(nodes, mesh)
    retained = []
    try:
        for index, shape in enumerate(((129, 32), (1, 16), (512, 32), (513, 32), (1, 4097))):
            rx, sat = _rays(*shape, seed=index+91)
            rx += offset
            sat += offset
            if shape[0] > 1:
                rx[1] = np.nan
                sat[-1, 0] = np.inf
            output = scene.trace_batch(rx, sat)
            _assert_bits(output, _reference(rx, sat, nodes, mesh))
            retained.append((output, tuple(v.copy() for v in output)))
            for previous, snapshot in retained:
                _assert_bits(previous, snapshot)
    finally:
        scene.close()


def test_winner_scratch_is_lazy_bounded_and_released():
    nodes, mesh = _mesh(12000)
    scene = native.DeviceBVH(nodes, mesh)
    try:
        geometry = len(nodes)*64 + mesh.nbytes
        assert scene.allocated_bytes == geometry
        rx, sat = _rays(512, 32)
        rays = 16384
        scene.check_los_batch(rx, sat)
        base = geometry + rx.nbytes + sat.nbytes + rays*4
        assert scene.allocated_bytes == base
        scene.multipath_batch(rx, sat)
        winner_bytes = rays*40
        assert winner_bytes == 640*1024
        mirror_epochs = (32*1024*1024)//24//12000
        mirror_bytes = mirror_epochs*12000*24
        assert mirror_bytes <= 32*1024*1024
        assert scene.allocated_bytes == base + rays*32 + 12000*36 + winner_bytes + mirror_bytes
    finally:
        scene.close()
    assert scene.allocated_bytes == 0
