"""Medium-mesh candidate dispatch matches the independent stateless GPU oracle."""

import numpy as np
import pytest

from test_bvh_large_parallel import _mesh, _rays
from test_bvh_normal_cache import _assert_bits, _leaf, _reference, native

pytestmark = [pytest.mark.gpu, pytest.mark.cuda]


@pytest.mark.parametrize('count', [511, 512, 513, 2047, 2048, 4095, 4096, 6144])
@pytest.mark.parametrize('shape', [(1, 4097), (129, 32), (257, 32)])
def test_medium_boundary_and_large_rays_match_original_gpu(count, shape):
    nodes, mesh = _mesh(count)
    rx, sat = _rays(*shape)
    scene = native.DeviceBVH(nodes, mesh)
    try:
        expected = _reference(rx, sat, nodes, mesh)
        _assert_bits(scene.trace_batch(rx, sat), expected)
        allocated = scene.allocated_bytes
        _assert_bits(scene.multipath_batch(rx, sat), expected[1:])
        _assert_bits(scene.trace_batch(rx, sat), expected)
        assert scene.allocated_bytes == allocated
    finally:
        scene.close()
    assert scene.allocated_bytes == 0


@pytest.mark.parametrize('count', [512, 4096])
@pytest.mark.parametrize('shape', [(1, 4096), (1, 4097), (512, 32), (1, 16385), (513, 32)])
def test_medium_ray_count_and_winner_capacity_boundaries(count, shape):
    nodes, mesh = _mesh(count)
    rx, sat = _rays(*shape)
    scene = native.DeviceBVH(nodes, mesh)
    try:
        _assert_bits(scene.trace_batch(rx, sat), _reference(rx, sat, nodes, mesh))
    finally:
        scene.close()


@pytest.mark.parametrize('scale', [1e-100, 1e-8, 3e-8, 1., 1e50, 1e100])
def test_medium_all_normals_and_intersections_preserve_extreme_bits(scale):
    mesh = np.random.default_rng(931).uniform(-30., 30., (4096, 3, 3))*scale
    mesh[::17, 2] = mesh[::17, 0]
    nodes = _leaf(mesh)
    rx, sat = _rays(129, 32)
    rx *= scale
    sat *= scale
    scene = native.DeviceBVH(nodes, mesh)
    try:
        _assert_bits(scene.trace_batch(rx, sat), _reference(rx, sat, nodes, mesh))
    finally:
        scene.close()


@pytest.mark.parametrize('reverse', [False, True])
def test_medium_equal_delays_preserve_first_visit(reverse):
    mesh = np.repeat(np.array([
        [[-1., -20., -10.], [-1., 20., -10.], [-1., 0., 20.]],
        [[1., -20., -10.], [1., 20., -10.], [1., 0., 20.]],
    ]), 2048, axis=0)
    nodes = np.array([
        [-1., -20., -10., 1., 20., 20., 1., 2., 0., 0.],
        [-1., -20., -10., -1., 20., 20., -1., -1., 0., 2048.],
        [1., -20., -10., 1., 20., 20., -1., -1., 2048., 2048.],
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


def test_medium_ecef_receiver_refresh_and_parallel_serial_transitions():
    nodes, mesh = _mesh(6144)
    offset = np.array([6378137., -4500000., 3500000.])
    nodes[:, :3] += offset
    nodes[:, 3:6] += offset
    mesh += offset
    scene = native.DeviceBVH(nodes, mesh)
    retained = []
    try:
        for index, shape in enumerate(((129, 32), (1, 16), (512, 32), (513, 32), (257, 32))):
            rx, sat = _rays(*shape, seed=index+731)
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


@pytest.mark.parametrize('count', [512, 4096])
def test_medium_winner_workspace_is_lazy_bounded_and_released(count):
    nodes, mesh = _mesh(count)
    scene = native.DeviceBVH(nodes, mesh)
    try:
        geometry = len(nodes)*64+mesh.nbytes
        assert scene.allocated_bytes == geometry
        rx, sat = _rays(512, 32)
        rays = 16384
        scene.check_los_batch(rx, sat)
        base = geometry+rx.nbytes+sat.nbytes+rays*4
        assert scene.allocated_bytes == base
        scene.multipath_batch(rx, sat)
        assert scene.allocated_bytes == base+rays*32+count*36+rays*40
        assert rays*40 == 640*1024
    finally:
        scene.close()
    assert scene.allocated_bytes == 0
