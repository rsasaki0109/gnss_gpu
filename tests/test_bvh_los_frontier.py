"""Independent original GPU traversal remains the LOS frontier oracle."""

import numpy as np
import pytest

from test_bvh_normal_cache import _assert_bits, _reference, native

pytestmark = [pytest.mark.cuda, pytest.mark.gpu]


def _tree(height=4, reverse=False):
    mesh = np.zeros((8192, 3, 3))
    mesh[:1024] = [[1., -20., -10.], [1., 20., -10.], [1., 0., 20.]]
    nodes = []
    for index in range(15):
        tail = [2*index+1, 2*index+2, 0, 0] if index < 7 else [-1, -1, (index-7)*1024, 1024]
        nodes.append([-1000., -1000., -1000., 1000., 1000., 1000.]+tail)
    leaf = 7
    for _ in range(height-4):
        child = len(nodes)
        nodes.append(nodes[leaf].copy())
        nodes.append([-1000., -1000., -1000., 1000., 1000., 1000., -1., -1., 0., 0.])
        nodes[leaf][6:] = [child, child+1, 0, 0] if reverse else [child+1, child, 0, 0]
        leaf = child
    return np.asarray(nodes), mesh


def _rays(epochs=1, satellites=12, seed=210):
    rng = np.random.default_rng(seed)
    rx = rng.uniform(-.25, .25, (epochs, 3))
    sat = rng.uniform(-300., 300., (epochs, satellites, 3))
    sat[:, 0] = [10., 0., 0.]
    return rx, sat


def _built(count):
    rng = np.random.default_rng(count)
    mesh = rng.uniform(-30., 30., (count, 3, 3))
    nodes, order = native.bvh_build(mesh)
    return nodes, np.ascontiguousarray(mesh[order])


@pytest.mark.parametrize('count', [8191, 8192, 12000])
@pytest.mark.parametrize('shape', [(1, 1), (1, 12), (1, 32), (1, 33), (2, 12), (2, 16), (3, 11)])
def test_mesh_and_ray_dispatch_boundaries(count, shape):
    nodes, mesh = _built(count)
    rx, sat = _rays(*shape)
    expected = _reference(rx, sat, nodes, mesh)
    scene = native.DeviceBVH(nodes, mesh)
    try:
        _assert_bits(scene.trace_batch(rx, sat), expected)
        _assert_bits(scene.check_los_batch(rx, sat), expected[0])
        if len(rx) == 1:
            _assert_bits(scene.check_los(rx[0], sat[0]), expected[0][0])
    finally:
        scene.close()
    assert scene.allocated_bytes == 0


@pytest.mark.parametrize('height', [4, 31, 32, 33, 64, 70])
@pytest.mark.parametrize('reverse', [False, True])
def test_height_guard_preserves_legacy_stack_truncation(height, reverse):
    nodes, mesh = _tree(height, reverse)
    rx, sat = _rays()
    expected = _reference(rx, sat, nodes, mesh)
    scene = native.DeviceBVH(nodes, mesh)
    try:
        _assert_bits(scene.trace_batch(rx, sat), expected)
        _assert_bits(scene.check_los(rx[0], sat[0]), expected[0][0])
    finally:
        scene.close()


@pytest.mark.parametrize('ancestor', [0, 1, 3])
def test_ancestor_culling_is_retained_even_with_noncontaining_bounds(ancestor):
    nodes, mesh = _tree()
    nodes[ancestor, :6] = [100., 100., 100., 101., 101., 101.]
    rx, sat = np.zeros((1, 3)), np.array([[[10., 0., 0.]]])
    expected = _reference(rx, sat, nodes, mesh)
    assert expected[0].all()
    scene = native.DeviceBVH(nodes, mesh)
    try:
        _assert_bits(scene.trace_batch(rx, sat), expected)
    finally:
        scene.close()


@pytest.mark.parametrize('scale', [1e-100, 1e-8, 3e-8, 1., 1e50, 1e100])
def test_extreme_geometry_keeps_original_arithmetic(scale):
    nodes, mesh = _tree()
    nodes[:, :6] *= scale
    mesh *= scale
    rx, sat = _rays()
    rx *= scale
    sat *= scale
    scene = native.DeviceBVH(nodes, mesh)
    try:
        _assert_bits(scene.trace_batch(rx, sat), _reference(rx, sat, nodes, mesh))
    finally:
        scene.close()


def test_ecef_and_nonfinite_batch_padding():
    nodes, mesh = _tree()
    offset = np.array([6378137., -4500000., 3500000.])
    nodes[:, :3] += offset
    nodes[:, 3:6] += offset
    mesh += offset
    rx, sat = _rays(2, 16)
    rx += offset
    sat += offset
    rx[1] = np.nan
    sat[0, 1] = np.inf
    scene = native.DeviceBVH(nodes, mesh)
    try:
        _assert_bits(scene.trace_batch(rx, sat), _reference(rx, sat, nodes, mesh))
    finally:
        scene.close()


@pytest.mark.parametrize('exclude', [False, True])
def test_shared_leaf_with_distinct_ancestor_paths(exclude):
    nodes, mesh = _tree()
    nodes[4, 6] = 7
    if exclude:
        nodes[1, :6] = [100., 100., 100., 101., 101., 101.]
    rx, sat = _rays()
    scene = native.DeviceBVH(nodes, mesh)
    try:
        _assert_bits(scene.trace_batch(rx, sat), _reference(rx, sat, nodes, mesh))
    finally:
        scene.close()


def test_near_zero_distance_axis_inverse_cutoff_and_large_finite_rays():
    nodes, mesh = _tree()
    rx = np.zeros((1, 3))
    sat = np.array([[[0., 0., 0.], [np.nextafter(1e-15, 0.), 0., 0.],
                     [np.nextafter(1e-15, np.inf), 0., 0.], [10., 1e-19, 0.],
                     [10., np.nextafter(1e-19, 0.), 0.], [1e150, 0., 0.]]])
    expected = _reference(rx, sat, nodes, mesh)
    scene = native.DeviceBVH(nodes, mesh)
    try:
        _assert_bits(scene.trace_batch(rx, sat), expected)
        _assert_bits(scene.check_los(rx[0], sat[0]), expected[0][0])
    finally:
        scene.close()


def test_cpu_metadata_snapshot_receiver_refresh_and_returned_ownership():
    nodes, mesh = _tree()
    frozen_nodes, frozen_mesh = nodes.copy(), mesh.copy()
    scene = native.DeviceBVH(nodes, mesh)
    nodes[:] = np.nan
    mesh += 1000.
    saved = []
    try:
        for index, shape in enumerate([(1, 12), (1, 33), (2, 16), (1, 12)]):
            rx, sat = _rays(*shape, seed=210+index)
            actual = scene.trace_batch(rx, sat)
            _assert_bits(actual, _reference(rx, sat, frozen_nodes, frozen_mesh))
            saved.append((actual, [a.copy() for a in actual]))
            for retained, copies in saved:
                _assert_bits(retained, copies)
    finally:
        scene.close()


def test_frontier_adds_no_persistent_gpu_allocation_and_reuses_storage():
    nodes, mesh = _tree()
    rx, sat = _rays()
    geometry = len(nodes)*64+mesh.nbytes
    scene = native.DeviceBVH(nodes, mesh)
    try:
        assert scene.allocated_bytes == geometry
        scene.check_los_batch(rx, sat)
        expected = geometry+rx.nbytes+sat.nbytes+12*4
        assert scene.allocated_bytes == expected
        scene.check_los(rx[0], sat[0])
        assert scene.allocated_bytes == expected
    finally:
        scene.close()
    assert scene.allocated_bytes == 0
