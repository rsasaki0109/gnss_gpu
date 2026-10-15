"""Bounded epoch mirrors preserve the independent stateless GPU results."""

import numpy as np
import pytest

from test_bvh_large_parallel import _mesh, _rays
from test_bvh_normal_cache import _assert_bits, _leaf, _reference, native

pytestmark = [pytest.mark.gpu, pytest.mark.cuda]
MIRROR_LIMIT = 32 * 1024 * 1024


def _chunk_rays(epochs, satellites, chunk, seed=517):
    rx, sat = _rays(epochs, satellites, seed=seed)
    rng = np.random.default_rng(seed + epochs)
    for epoch in sorted({0, epochs - 1, chunk - 1, chunk, 2 * chunk - 1, 2 * chunk}):
        if 0 <= epoch < epochs:
            sat[epoch, :3] = rng.uniform(-300., 300., (3, 3))
    rx[1] = np.nan
    sat[-1, 3] = rx[-1]
    return rx, sat


@pytest.mark.parametrize('count', [8191, 8192, 12000, 120000])
@pytest.mark.parametrize('satellites', [15, 16, 32])
@pytest.mark.parametrize('boundary', ['cap', 'after', 'two_tail'])
def test_chunk_boundaries_match_stateless_gpu(count, satellites, boundary):
    chunk = MIRROR_LIMIT // 24 // count
    epochs = {'cap': chunk, 'after': chunk + 1, 'two_tail': 2 * chunk + 1}[boundary]
    nodes, mesh = _mesh(count)
    rx, sat = _chunk_rays(epochs, satellites, chunk)
    expected = _reference(rx, sat, nodes, mesh)
    scene = native.DeviceBVH(nodes, mesh)
    try:
        _assert_bits(scene.trace_batch(rx, sat), expected)
        allocated = scene.allocated_bytes
        _assert_bits(scene.multipath_batch(rx, sat), expected[1:])
        _assert_bits(scene.trace_batch(rx, sat), expected)
        assert scene.allocated_bytes == allocated
    finally:
        scene.close()
    assert scene.allocated_bytes == 0


@pytest.mark.parametrize('scale', [1e-100, 1e-8, 3e-8, 1e-7, 1., 1e50, 1e100])
def test_chunked_extreme_geometry_preserves_all_bits(scale):
    mesh = np.random.default_rng(538).uniform(-30., 30., (8192, 3, 3)) * scale
    mesh[::17, 2] = mesh[::17, 0]
    nodes = _leaf(mesh)
    rx, sat = _chunk_rays(171, 16, 170)
    rx *= scale
    sat *= scale
    scene = native.DeviceBVH(nodes, mesh)
    try:
        _assert_bits(scene.trace_batch(rx, sat), _reference(rx, sat, nodes, mesh))
    finally:
        scene.close()


@pytest.mark.parametrize('area', [np.nextafter(1e-15, 0.), 1e-15, np.nextafter(1e-15, np.inf)])
def test_chunked_normal_cutoff_keeps_original_gpu_bits(area):
    mesh = np.repeat(np.array([[[0., 0., 0.], [1., 0., 0.], [0., area, 0.]]]), 8192, axis=0)
    mesh[::2, :, 2] = -0.
    nodes = _leaf(mesh)
    rx, sat = _chunk_rays(171, 16, 170)
    scene = native.DeviceBVH(nodes, mesh)
    try:
        _assert_bits(scene.trace_batch(rx, sat), _reference(rx, sat, nodes, mesh))
    finally:
        scene.close()


@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('satellites', [16, 17])
def test_positive_equal_delays_keep_right_first_order_across_chunks(reverse, satellites):
    mesh = np.repeat(np.array([
        [[-1., -20., -10.], [-1., 20., -10.], [-1., 0., 20.]],
        [[1., -20., -10.], [1., 20., -10.], [1., 0., 20.]],
    ]), 6000, axis=0)
    nodes = np.array([
        [-1., -20., -10., 1., 20., 20., 1., 2., 0., 0.],
        [-1., -20., -10., -1., 20., 20., -1., -1., 0., 6000.],
        [1., -20., -10., 1., 20., 20., -1., -1., 6000., 6000.],
    ])
    if reverse:
        nodes[0, [6, 7]] = nodes[0, [7, 6]]
    rx = np.zeros((117, 3))
    sat = np.full((117, satellites, 3), np.nan)
    sat[[0, 115, 116], 0] = [0., 0., 10.]
    expected = _reference(rx, sat, nodes, mesh)
    scene = native.DeviceBVH(nodes, mesh)
    try:
        _assert_bits(scene.trace_batch(rx, sat), expected)
        assert np.all(expected[2][[0, 115, 116], 0] > 0.)
        assert np.all((expected[1][[0, 115, 116], 0, 0] < 0.) == reverse)
    finally:
        scene.close()


@pytest.mark.parametrize('count', [8192, 12000, 120000])
@pytest.mark.parametrize('satellites', [15, 16])
def test_capacity_is_lazy_exact_bounded_reused_and_released(count, satellites):
    nodes, mesh = _mesh(count)
    geometry = len(nodes) * 64 + mesh.nbytes
    chunk = MIRROR_LIMIT // 24 // count
    scene = native.DeviceBVH(nodes, mesh)
    try:
        assert scene.allocated_bytes == geometry
        rx, sat = _chunk_rays(chunk - 1, satellites, chunk)
        scene.check_los_batch(rx, sat)
        assert scene.allocated_bytes == geometry + rx.nbytes + sat.nbytes + len(rx) * satellites * 4
        epoch_capacity = ray_capacity = winner_capacity = mirror_capacity = 0
        for epochs in (chunk - 1, chunk, chunk + 1, 2 * chunk + 1, 1):
            rx, sat = _rays(epochs, satellites)
            rays = epochs * satellites
            epoch_capacity = max(epoch_capacity, epochs)
            ray_capacity = max(ray_capacity, rays)
            tiles = min(32, (count + 127) // 128, max(1, 256 // rays))
            winner_capacity = max(winner_capacity, rays * tiles)
            if satellites >= 16:
                mirror_capacity = max(mirror_capacity, min(epochs, chunk) * count * 24)
            scene.trace_batch(rx, sat)
            expected = (geometry + epoch_capacity * 24 + ray_capacity * (24 + 4 + 32)
                        + count * 36 + winner_capacity * 40 + mirror_capacity)
            assert scene.allocated_bytes == expected
            assert mirror_capacity <= MIRROR_LIMIT
            assert winner_capacity * 40 <= 640 * 1024
    finally:
        scene.close()
    assert scene.allocated_bytes == 0


@pytest.mark.parametrize('count', [8192, 12000])
def test_serial_transition_refreshes_receivers_and_keeps_returned_ownership(count):
    nodes, mesh = _mesh(count)
    chunk = MIRROR_LIMIT // 24 // count
    scene = native.DeviceBVH(nodes, mesh)
    retained = []
    try:
        for index, shape in enumerate(((chunk + 1, 32), (513, 32), (1, 16), (2 * chunk + 1, 32), (1, 15))):
            rx, sat = _rays(*shape, seed=731 + index)
            actual = scene.trace_batch(rx, sat)
            _assert_bits(actual, _reference(rx, sat, nodes, mesh))
            retained.append((actual, tuple(v.copy() for v in actual)))
            for previous, snapshot in retained:
                _assert_bits(previous, snapshot)
    finally:
        scene.close()
    assert scene.allocated_bytes == 0


def test_ecef_receiver_updates_and_partial_chunk_tail_match():
    nodes, mesh = _mesh(12000)
    offset = np.array([6378137., -4500000., 3500000.])
    nodes[:, :3] += offset
    nodes[:, 3:6] += offset
    mesh += offset
    scene = native.DeviceBVH(nodes, mesh)
    try:
        for index, epochs in enumerate((117, 116, 257, 129)):
            rx, sat = _chunk_rays(epochs, 32, 116, seed=987 + index)
            rx += offset + index * np.array([.25, -.125, .0625])
            sat += offset
            _assert_bits(scene.trace_batch(rx, sat), _reference(rx, sat, nodes, mesh))
    finally:
        scene.close()


def test_single_epoch_larger_than_mirror_budget_uses_original_fallback():
    count = MIRROR_LIMIT // 24 + 1
    nodes, mesh = _mesh(count)
    rx, sat = _rays(1, 16)
    scene = native.DeviceBVH(nodes, mesh)
    try:
        _assert_bits(scene.trace_batch(rx, sat), _reference(rx, sat, nodes, mesh))
        tiles = min(32, (count + 127) // 128, 256 // 16)
        expected = len(nodes) * 64 + mesh.nbytes + rx.nbytes + sat.nbytes + 16 * (4 + 32) + count * 36 + 16 * tiles * 40
        assert scene.allocated_bytes == expected
    finally:
        scene.close()
    assert scene.allocated_bytes == 0


@pytest.mark.parametrize('count', [300000, 600000])
def test_multiple_candidate_tiles_keep_chunk_offsets_and_exact_capacity(count):
    chunk = MIRROR_LIMIT // 24 // count
    nodes, mesh = _mesh(count)
    rx, sat = _chunk_rays(chunk + 1, 16, chunk)
    rays = len(rx) * 16
    tiles = min(32, (count + 127) // 128, max(1, 256 // rays))
    assert tiles > 1
    scene = native.DeviceBVH(nodes, mesh)
    try:
        _assert_bits(scene.trace_batch(rx, sat), _reference(rx, sat, nodes, mesh))
        expected = (len(nodes) * 64 + mesh.nbytes + rx.nbytes + sat.nbytes
                    + rays * (4 + 32) + count * 36 + rays * tiles * 40 + chunk * count * 24)
        assert scene.allocated_bytes == expected
        allocated = scene.allocated_bytes
        _assert_bits(scene.multipath_batch(rx, sat), _reference(rx, sat, nodes, mesh)[1:])
        assert scene.allocated_bytes == allocated
    finally:
        scene.close()
    assert scene.allocated_bytes == 0
