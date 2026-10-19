"""Large resident ray groups preserve the independent stateless GPU producer."""

import numpy as np
import pytest

from test_bvh_normal_cache import _assert_bits, _leaf, _reference, native

pytestmark = [pytest.mark.cuda, pytest.mark.gpu]


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
    chunk = max(1, 4096 // satellites)
    interesting = {0, epochs-1, epochs//2, chunk-1, chunk, chunk+1,
                   2*chunk-1, 2*chunk, 115, 116, 117, 127, 128, 129, 511, 512}
    for epoch in sorted(e for e in interesting if 0 <= e < epochs):
        count = min(3, satellites)
        sat[epoch, :count] = rng.uniform(-300., 300., (count, 3))
        if satellites > 3:
            sat[epoch, 3] = rx[epoch]
    return rx, sat


@pytest.mark.parametrize('count', [8191, 8192, 12000])
@pytest.mark.parametrize('shape', [(513, 32), (1025, 32), (1366, 12), (1093, 15),
                                  (2, 8193), (2, 16384), (1, 16385), (16385, 1)])
def test_large_groups_and_fallbacks_match_original_gpu(count, shape):
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


@pytest.mark.parametrize('scale', [1e-100, 1e-8, 3e-8, 1., 1e50, 1e100])
def test_extreme_geometry_across_epoch_groups(scale):
    nodes, mesh = _mesh(8192, scale)
    rx, sat = _rays(513, 32)
    rx *= scale
    sat *= scale
    scene = native.DeviceBVH(nodes, mesh)
    try:
        _assert_bits(scene.trace_batch(rx, sat), _reference(rx, sat, nodes, mesh))
    finally:
        scene.close()


@pytest.mark.parametrize('satellites', [12, 32, 8193, 16384])
def test_winner_and_mirror_workspaces_are_bounded_and_reused(satellites):
    nodes, mesh = _mesh(8192)
    epochs = 16384 // satellites + 1
    rx, sat = _rays(epochs, satellites)
    scene = native.DeviceBVH(nodes, mesh)
    try:
        scene.check_los_batch(rx, sat)
        before = scene.allocated_bytes
        scene.trace_batch(rx, sat)
        after = scene.allocated_bytes
        # New order/normals and output payloads are distinct from bounded scratch.
        payload = len(mesh)*36 + epochs*satellites*32
        assert 0 < after-before-payload <= 32*1024*1024 + 640*1024
        scene.trace_batch(rx, sat)
        assert scene.allocated_bytes == after
    finally:
        scene.close()
    assert scene.allocated_bytes == 0


@pytest.mark.parametrize('reverse', [False, True])
def test_equal_delays_keep_original_visit_order_across_groups(reverse):
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
    rx, sat = _rays(513, 32)
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


def test_owner_transitions_receiver_changes_and_returned_array_ownership():
    nodes, mesh = _mesh(12000)
    scene = native.DeviceBVH(nodes, mesh)
    saved = []
    try:
        for index, shape in enumerate([(512, 32), (513, 32), (1, 16385),
                                       (1093, 15), (1025, 32), (1, 16)]):
            rx, sat = _rays(*shape, seed=93+index)
            rx += index * .3
            expected = _reference(rx, sat, nodes, mesh)
            actual = scene.trace_batch(rx, sat)
            _assert_bits(actual, expected)
            saved.append((actual, [array.copy() for array in actual]))
            for retained, copies in saved:
                _assert_bits(retained, copies)
    finally:
        scene.close()
    assert scene.allocated_bytes == 0
