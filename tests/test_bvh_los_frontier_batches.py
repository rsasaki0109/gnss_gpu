"""Compare extended resident LOS batches against the original stateless GPU."""

import numpy as np
import pytest

from test_bvh_los_frontier import _built, _rays, _tree
from test_bvh_normal_cache import _assert_bits, _reference, native

pytestmark = [pytest.mark.cuda, pytest.mark.gpu]


@pytest.mark.parametrize('count', [8191, 8192, 12000])
@pytest.mark.parametrize('shape', [(1, 33), (3, 12), (2, 32), (8, 16), (17, 15), (16, 16), (1, 257), (17, 16)])
def test_medium_batches_match_original_stateless_traversal(count, shape):
    nodes, mesh = _built(count)
    rx, sat = _rays(*shape, seed=count+sum(shape))
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


@pytest.mark.parametrize('height', [31, 32, 33, 64, 70])
@pytest.mark.parametrize('reverse', [False, True])
def test_medium_batch_height_guard_keeps_original_stack_behavior(height, reverse):
    nodes, mesh = _tree(height, reverse)
    rx, sat = _rays(8, 32)
    expected = _reference(rx, sat, nodes, mesh)
    scene = native.DeviceBVH(nodes, mesh)
    try:
        _assert_bits(scene.trace_batch(rx, sat), expected)
    finally:
        scene.close()


@pytest.mark.parametrize('excluded_ancestor', [None, 0, 1, 3])
def test_medium_batch_ancestor_culling_and_padding(excluded_ancestor):
    nodes, mesh = _tree()
    nodes[4, 6] = 7
    if excluded_ancestor is not None:
        nodes[excluded_ancestor, :6] = [100., 100., 100., 101., 101., 101.]
    offset = np.array([6378137., -4500000., 3500000.])
    nodes[:, :3] += offset
    nodes[:, 3:6] += offset
    mesh += offset
    rx, sat = _rays(16, 16)
    rx += offset
    sat += offset
    rx[-1] = np.nan
    sat[0, 0] = np.inf
    sat[1, 0] = rx[1]
    scene = native.DeviceBVH(nodes, mesh)
    try:
        _assert_bits(scene.trace_batch(rx, sat), _reference(rx, sat, nodes, mesh))
    finally:
        scene.close()


def test_reused_owner_refreshes_rays_preserves_outputs_and_adds_no_workspace():
    nodes, mesh = _tree()
    scene = native.DeviceBVH(nodes, mesh)
    reference_owner = native.DeviceBVH(nodes, mesh)
    retained = []
    try:
        for seed, shape in enumerate(((16, 16), (1, 33), (8, 32), (17, 16), (3, 12))):
            rx, sat = _rays(*shape, seed=seed)
            reference_owner.check_los_batch(rx, sat)
            reference_owner.multipath_batch(rx, sat)
            result = scene.trace_batch(rx, sat)
            _assert_bits(result, _reference(rx, sat, nodes, mesh))
            assert scene.allocated_bytes == reference_owner.allocated_bytes
            retained.append((result, tuple(array.copy() for array in result)))
            for output, snapshot in retained:
                _assert_bits(output, snapshot)
    finally:
        scene.close()
        reference_owner.close()


@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
def test_single_api_keeps_rejecting_nonfinite_values(bad):
    nodes, mesh = _tree()
    rx, sat = _rays(1, 256)
    sat[0, 20, 1] = bad
    scene = native.DeviceBVH(nodes, mesh)
    try:
        with pytest.raises(RuntimeError, match='sat_ecef must be finite'):
            native.raytrace_los_check_bvh(rx[0], sat[0], nodes, mesh)
        with pytest.raises(RuntimeError, match='sat_ecef must be finite'):
            scene.check_los(rx[0], sat[0])
    finally:
        scene.close()
