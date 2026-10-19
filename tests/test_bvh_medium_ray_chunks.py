"""Medium-mesh epoch groups retain the independent original GPU producer."""

import numpy as np
import pytest

from test_bvh_normal_cache import _assert_bits, _reference, native
from test_bvh_ray_chunks import _mesh, _rays

pytestmark = [pytest.mark.cuda, pytest.mark.gpu]


@pytest.mark.parametrize('count', [1023, 1024, 4096, 8191])
@pytest.mark.parametrize('shape', [(2049, 8), (1366, 12), (513, 32),
                                  (16385, 1), (2, 16384), (1, 16385)])
def test_mesh_satellite_and_group_boundaries_match_original_gpu(count, shape):
    nodes, mesh = _mesh(count)
    rx, sat = _rays(*shape)
    epochs, satellites = shape
    if count >= 1024 and 8 <= satellites <= 16384:
        group = min(max(1, 4096 // satellites), (32*1024*1024) // (24*count))
        for epoch in (group-1, group, group+1, 2*group-1, 2*group, epochs-1):
            if 0 <= epoch < epochs:
                sat[epoch, 0] = [300., -21., 40.]
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


@pytest.mark.parametrize('shape', [(512, 32), (513, 32), (1365, 12), (1366, 12)])
def test_total_ray_dispatch_boundary(shape):
    nodes, mesh = _mesh(1024)
    rx, sat = _rays(*shape)
    scene = native.DeviceBVH(nodes, mesh)
    try:
        _assert_bits(scene.trace_batch(rx, sat), _reference(rx, sat, nodes, mesh))
    finally:
        scene.close()


@pytest.mark.parametrize('scale', [1e-100, 1e-8, 3e-8, 1., 1e50, 1e100])
def test_extreme_geometry_and_original_normal_cutoffs(scale):
    nodes, mesh = _mesh(1024, scale)
    rx, sat = _rays(1366, 12)
    rx *= scale
    sat *= scale
    scene = native.DeviceBVH(nodes, mesh)
    try:
        _assert_bits(scene.trace_batch(rx, sat), _reference(rx, sat, nodes, mesh))
    finally:
        scene.close()


@pytest.mark.parametrize('satellites', [8, 12])
def test_ecef_receiver_changes_and_nonfinite_padding(satellites):
    nodes, mesh = _mesh(4096)
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
def test_equal_positive_delays_keep_original_visit_order(reverse):
    mesh = np.repeat(np.array([
        [[-1., -20., -10.], [-1., 20., -10.], [-1., 0., 20.]],
        [[1., -20., -10.], [1., 20., -10.], [1., 0., 20.]],
    ]), 512, axis=0)
    nodes = np.array([
        [-1., -20., -10., 1., 20., 20., 1., 2., 0., 0.],
        [-1., -20., -10., -1., 20., 20., -1., -1., 0., 512.],
        [1., -20., -10., 1., 20., 20., -1., -1., 512., 512.],
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


def test_owner_transitions_receiver_refresh_and_returned_array_ownership():
    nodes, mesh = _mesh(1024)
    scene = native.DeviceBVH(nodes, mesh)
    saved = []
    try:
        for index, shape in enumerate([(512, 32), (513, 32), (2341, 7),
                                       (1366, 12), (1, 16385), (1, 12)]):
            rx, sat = _rays(*shape, seed=217+index)
            rx += index * .13
            actual = scene.trace_batch(rx, sat)
            _assert_bits(actual, _reference(rx, sat, nodes, mesh))
            saved.append((actual, [array.copy() for array in actual]))
            for original, copies in saved:
                _assert_bits(original, copies)
    finally:
        scene.close()
    assert scene.allocated_bytes == 0


@pytest.mark.parametrize('satellites', [1, 7, 8, 12, 32, 16384])
def test_scratch_is_bounded_reused_and_released(satellites):
    nodes, mesh = _mesh(4096)
    epochs = 16384 // satellites + 1
    rx, sat = _rays(epochs, satellites)
    scene = native.DeviceBVH(nodes, mesh)
    try:
        scene.check_los_batch(rx, sat)
        before = scene.allocated_bytes
        scene.trace_batch(rx, sat)
        allocated = scene.allocated_bytes
        payload = len(mesh)*36 + epochs*satellites*32
        budget = 640*1024 + (32*1024*1024 if satellites >= 8 else 0)
        scratch = allocated-before-payload
        if satellites < 8:
            assert scratch == 0
        else:
            assert 0 < scratch <= budget
        scene.trace_batch(rx, sat)
        assert scene.allocated_bytes == allocated
    finally:
        scene.close()
    assert scene.allocated_bytes == 0
