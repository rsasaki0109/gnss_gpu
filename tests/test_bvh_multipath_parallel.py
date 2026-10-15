"""Parallel reflection evaluation agrees with the serial traversal contract."""

import numpy as np
import pytest

try:
    from gnss_gpu import _bvh as native
    HAS_GPU = True
except ImportError:
    HAS_GPU = False

pytestmark = pytest.mark.skipif(not HAS_GPU, reason="CUDA BVH module not available")


def _mesh(count):
    rng = np.random.default_rng(count)
    triangles = rng.uniform(-100., 100., (count, 3, 3))
    triangles[::17, 2] = triangles[::17, 0]
    nodes, indices = native.bvh_build(triangles)
    return nodes, triangles[indices].copy()


def _serial_reference(rx, sat, nodes, triangles):
    """The unchanged path for >4096 rays provides independent GPU evaluation."""
    epochs, satellites = sat.shape[:2]
    indices = np.arange(max(4097, epochs * satellites)) % (epochs * satellites)
    extended_rx = np.repeat(rx, satellites, axis=0)[indices].copy()
    extended_sat = sat.reshape(-1, 3)[indices, None].copy()
    points, delays = native.raytrace_multipath_bvh_batch(
        extended_rx, extended_sat, nodes, triangles
    )
    return (points[:epochs * satellites].reshape(epochs, satellites, 3),
            delays[:epochs * satellites].reshape(epochs, satellites))


@pytest.mark.parametrize("count", [511, 512, 513, 1025, 4097])
def test_parallel_reflections_match_serial_for_mesh_tails(count):
    nodes, triangles = _mesh(count)
    rng = np.random.default_rng(23)
    rx = rng.uniform(-30., 30., (3, 3))
    sat = rng.uniform(-300., 300., (3, 7, 3))
    expected = _serial_reference(rx, sat, nodes, triangles)
    actual = native.raytrace_multipath_bvh_batch(rx, sat, nodes, triangles)
    for before, after in zip(expected, actual, strict=True):
        np.testing.assert_array_equal(after.view(np.uint64), before.view(np.uint64))


@pytest.mark.parametrize("rays", [1, 127, 128, 129, 255, 256, 257, 4096, 4097])
def test_parallel_reflections_match_serial_for_ray_boundaries(rays):
    nodes, triangles = _mesh(512)
    rng = np.random.default_rng(23)
    rx = np.array([[3., -7., 8.]])
    sat = rng.uniform(-300., 300., (1, rays, 3))
    expected = _serial_reference(rx, sat, nodes, triangles)
    actual = native.raytrace_multipath_bvh_batch(rx, sat, nodes, triangles)
    for before, after in zip(expected, actual, strict=True):
        np.testing.assert_array_equal(after.view(np.uint64), before.view(np.uint64))


@pytest.mark.parametrize("reverse_root", [False, True])
def test_equal_delays_select_first_bvh_visit_across_tiles(reverse_root):
    """Distinct reflection points tie; array index order differs from visit order."""
    triangles = np.repeat(np.array([
        [[-1., -20., -10.], [-1., 20., -10.], [-1., 0., 20.]],
        [[1., -20., -10.], [1., 20., -10.], [1., 0., 20.]],
    ]), 256, axis=0)
    nodes = np.array([
        [-1., -20., -10., 1., 20., 20., 1., 2., 0., 0.],
        [-1., -20., -10., -1., 20., 20., -1., -1., 0., 256.],
        [1., -20., -10., 1., 20., 20., -1., -1., 256., 256.],
    ])
    if reverse_root:
        nodes[0, [6, 7]] = nodes[0, [7, 6]]
    rx = np.zeros((1, 3))
    sat = np.array([[[0., 0., 10.]]])
    points, delays = native.raytrace_multipath_bvh_batch(rx, sat, nodes, triangles)
    np.testing.assert_allclose(points[0, 0], [-1. if reverse_root else 1., 0., 5.],
                               rtol=0., atol=1e-14)
    np.testing.assert_allclose(delays[0, 0], np.sqrt(104.) - 10., rtol=0., atol=1e-14)


def test_parallel_nonfinite_padding_returns_zero():
    nodes, triangles = _mesh(512)
    rx = np.array([[0., 0., 0.], [np.nan, 0., 0.]])
    sat = np.array([[[np.inf, 0., 0.], [np.nan, 0., 0.]],
                    [[10., 20., 30.], [40., 50., 60.]]])
    points, delays = native.raytrace_multipath_bvh_batch(rx, sat, nodes, triangles)
    np.testing.assert_array_equal(points, np.zeros((2, 2, 3)))
    np.testing.assert_array_equal(delays, np.zeros((2, 2)))


def test_overlapping_custom_leaves_keep_serial_results():
    nodes, triangles = _mesh(512)
    root = nodes[0].copy()
    root[6:10] = [1, 2, 0, 0]
    leaf = root.copy()
    leaf[6:10] = [-1, -1, 0, 512]
    nodes = np.stack([root, leaf, leaf])
    rx = np.zeros((1, 3))
    sat = np.array([[[300., -200., 150.]]])
    expected = _serial_reference(rx, sat, nodes, triangles)
    actual = native.raytrace_multipath_bvh_batch(rx, sat, nodes, triangles)
    for before, after in zip(expected, actual, strict=True):
        np.testing.assert_array_equal(after.view(np.uint64), before.view(np.uint64))


def test_parallel_degenerate_triangles_have_no_reflections():
    nodes, indices = native.bvh_build(np.zeros((512, 3, 3)))
    triangles = np.zeros((512, 3, 3))[indices].copy()
    points, delays = native.raytrace_multipath_bvh_batch(
        np.zeros((1, 3)), np.array([[[1., 2., 3.]]]), nodes, triangles
    )
    np.testing.assert_array_equal(points, np.zeros((1, 1, 3)))
    np.testing.assert_array_equal(delays, np.zeros((1, 1)))
