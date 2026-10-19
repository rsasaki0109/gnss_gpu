"""Ordinary-satellite mirror groups retain independent original GPU outputs."""

import numpy as np
import pytest

from test_bvh_normal_cache import _assert_bits, _reference, native
from test_bvh_ray_chunks import _mesh, _rays

pytestmark = [pytest.mark.cuda, pytest.mark.gpu]


@pytest.mark.parametrize('count', [8191, 8192, 12000])
@pytest.mark.parametrize('satellites', [7, 8, 9, 12, 15, 16])
def test_satellite_and_mesh_boundaries_match_stateless_gpu(count, satellites):
    nodes, mesh = _mesh(count)
    rx, sat = _rays(16384 // satellites + 1, satellites)
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


@pytest.mark.parametrize('satellites,last_count', [(8, 43690), (12, 63550), (15, 77672)])
@pytest.mark.parametrize('extra', [0, 1])
def test_large_mesh_budget_boundary_and_group_tails(satellites, last_count, extra):
    count = last_count + extra
    nodes, mesh = _mesh(count)
    rx, sat = _rays(16384 // satellites + 1, satellites)
    mirror_epochs = (32*1024*1024) // (24*count)
    for epoch in (mirror_epochs-1, mirror_epochs, mirror_epochs+1,
                  2*mirror_epochs-1, 2*mirror_epochs, len(rx)-1):
        sat[epoch, 0] = [300., -21., 40.]
    expected = _reference(rx, sat, nodes, mesh)
    scene = native.DeviceBVH(nodes, mesh)
    try:
        _assert_bits(scene.trace_batch(rx, sat), expected)
        _assert_bits(scene.multipath_batch(rx, sat), expected[1:])
    finally:
        scene.close()
    assert scene.allocated_bytes == 0


@pytest.mark.parametrize('scale', [1e-100, 1e-8, 3e-8, 1., 1e50, 1e100])
def test_original_normal_cutoffs_and_extreme_geometry(scale):
    nodes, mesh = _mesh(8192, scale)
    rx, sat = _rays(2049, 8)
    rx *= scale
    sat *= scale
    scene = native.DeviceBVH(nodes, mesh)
    try:
        _assert_bits(scene.trace_batch(rx, sat), _reference(rx, sat, nodes, mesh))
    finally:
        scene.close()


@pytest.mark.parametrize('satellites', [8, 12])
def test_ecef_receiver_refresh_and_nonfinite_padding(satellites):
    nodes, mesh = _mesh(12000)
    offset = np.array([6378137., -4500000., 3500000.])
    mesh += offset
    nodes[:, :3] += offset
    nodes[:, 3:6] += offset
    rx, sat = _rays(16384 // satellites + 1, satellites)
    rx += offset
    sat += offset
    scene = native.DeviceBVH(nodes, mesh)
    try:
        for step in range(2):
            rx += step * .3
            rx[1] = np.nan
            sat[-1, :3] = np.inf
            _assert_bits(scene.trace_batch(rx, sat), _reference(rx, sat, nodes, mesh))
    finally:
        scene.close()


@pytest.mark.parametrize('reverse', [False, True])
def test_equal_positive_delays_preserve_original_order(reverse):
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
        nodes[0, 6:8] = [2., 1.]
    rx, sat = _rays(1366, 12)
    valid = np.isfinite(sat).all(axis=2)
    rx[:] = 0.
    sat[valid] = [0., 0., 10.]
    scene = native.DeviceBVH(nodes, mesh)
    try:
        actual = scene.trace_batch(rx, sat)
        _assert_bits(actual, _reference(rx, sat, nodes, mesh))
        assert np.count_nonzero(actual[2]) > 0
    finally:
        scene.close()


def test_receiver_and_dispatch_changes_preserve_returned_array_ownership():
    nodes, mesh = _mesh(12000)
    scene = native.DeviceBVH(nodes, mesh)
    saved = []
    try:
        for index, satellites in enumerate((8, 7, 12, 15, 16, 8)):
            rx, sat = _rays(16384 // satellites + 1, satellites, seed=177+index)
            rx += index * .13
            actual = scene.trace_batch(rx, sat)
            _assert_bits(actual, _reference(rx, sat, nodes, mesh))
            saved.append((actual, [array.copy() for array in actual]))
            for original, copy in saved:
                _assert_bits(original, copy)
    finally:
        scene.close()
    assert scene.allocated_bytes == 0


@pytest.mark.parametrize('satellites', [8, 12, 15])
def test_scratch_is_bounded_reused_and_released(satellites):
    nodes, mesh = _mesh(12000)
    epochs = 16384 // satellites + 1
    rx, sat = _rays(epochs, satellites)
    scene = native.DeviceBVH(nodes, mesh)
    try:
        scene.check_los_batch(rx, sat)
        before = scene.allocated_bytes
        scene.trace_batch(rx, sat)
        allocated = scene.allocated_bytes
        payload = len(mesh)*36 + epochs*satellites*32
        assert 0 < allocated-before-payload <= 32*1024*1024 + 640*1024
        scene.trace_batch(rx, sat)
        assert scene.allocated_bytes == allocated
    finally:
        scene.close()
    assert scene.allocated_bytes == 0
