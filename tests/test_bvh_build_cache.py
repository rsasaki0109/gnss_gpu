"""Build geometry, scratch lifetime, and changed-mesh ray results."""

import numpy as np
import pytest

native = pytest.importorskip("gnss_gpu._bvh")


def _mesh(count, kind="random"):
    mesh = np.random.default_rng(count).uniform(-100., 100., (count, 3, 3))
    if kind == "coplanar":
        mesh[:, :, 2] = 0.
    elif kind == "points":
        mesh[:, 1:] = mesh[:, :1]
    elif kind == "same_centroid":
        mesh[:, 1] = -mesh[:, 0]
        mesh[:, 2] = 0.
    return mesh


def _assert_tree_covers_mesh(mesh, nodes, indices):
    assert nodes.dtype == np.float64 and nodes.shape[1] == 10
    assert np.all(np.isfinite(nodes))
    np.testing.assert_array_equal(np.sort(indices), np.arange(len(mesh)))
    sorted_mesh = mesh[indices]
    np.testing.assert_array_equal(nodes[0, :3], mesh.min(axis=(0, 1)))
    np.testing.assert_array_equal(nodes[0, 3:6], mesh.max(axis=(0, 1)))
    covered = np.zeros(len(mesh), dtype=int)
    visited = set()
    pending = [0]
    while pending:
        index = pending.pop()
        assert 0 <= index < len(nodes) and index not in visited
        visited.add(index)
        node = nodes[index]
        left, right, start, count = node[6:].astype(np.int64)
        if left == -1:
            assert right == -1 and 1 <= count <= 4
            assert 0 <= start < start + count <= len(mesh)
            covered[start:start + count] += 1
            triangles = sorted_mesh[start:start + count]
            np.testing.assert_array_equal(node[:3], triangles.min(axis=(0, 1)))
            np.testing.assert_array_equal(node[3:6], triangles.max(axis=(0, 1)))
        else:
            assert start == count == 0 and left != right
            assert index < left < len(nodes) and index < right < len(nodes)
            np.testing.assert_array_equal(node[:3], np.minimum(nodes[left, :3], nodes[right, :3]))
            np.testing.assert_array_equal(node[3:6], np.maximum(nodes[left, 3:6], nodes[right, 3:6]))
            pending.extend((int(right), int(left)))
    assert len(visited) == len(nodes)
    np.testing.assert_array_equal(covered, 1)


@pytest.mark.parametrize("count", [1, 4, 5, 31, 127, 257])
@pytest.mark.parametrize("kind", ["random", "coplanar", "points", "same_centroid"])
def test_build_partitions_cover_every_triangle_and_exact_bounds(count, kind):
    mesh = _mesh(count, kind)
    previous = mesh.copy()
    nodes, indices = native.bvh_build(mesh)
    _assert_tree_covers_mesh(mesh, nodes, indices)
    np.testing.assert_array_equal(mesh.view(np.uint64), previous.view(np.uint64))


@pytest.mark.parametrize("count", [4, 67])
@pytest.mark.parametrize("scale", [1e-100, 1e-16, 1e-14, 1., 1e50, 1e100])
def test_fp64_bounds_preserve_small_and_large_finite_coordinates(count, scale):
    mesh = _mesh(count) * scale
    _assert_tree_covers_mesh(mesh, *native.bvh_build(mesh))


@pytest.mark.parametrize("layout", ["c", "fortran", "strided"])
def test_changed_input_storage_and_interleaved_builds_keep_outputs_owned(layout):
    storage = _mesh(257)
    mesh = {"c": storage, "fortran": np.asfortranarray(storage), "strided": storage[::2, ::-1]}[layout]
    original = mesh.copy()
    retained = []
    for shift in (0., 1000., -2000., 1000.):
        mesh += shift
        output = native.bvh_build(mesh)
        _assert_tree_covers_mesh(mesh, *output)
        retained.append((output, [array.copy() for array in output]))
        native.bvh_build(_mesh(5))
        native.bvh_build(_mesh(1025))
        for arrays, snapshots in retained:
            for array, snapshot in zip(arrays, snapshots, strict=True):
                np.testing.assert_array_equal(array.view(np.uint8), snapshot.view(np.uint8))
    # Reusing the original values must also reproduce the original partition.
    mesh[...] = original
    for array, snapshot in zip(native.bvh_build(mesh), retained[0][1], strict=True):
        np.testing.assert_array_equal(array.view(np.uint8), snapshot.view(np.uint8))


def test_signed_zero_vertices_keep_stable_tree_bits():
    mesh = np.zeros((33, 3, 3))
    mesh[::2, ::2, ::2] = -0.
    expected = native.bvh_build(mesh)
    _assert_tree_covers_mesh(mesh, *expected)
    native.bvh_build(_mesh(513))
    for old, new in zip(expected, native.bvh_build(mesh), strict=True):
        np.testing.assert_array_equal(old.view(np.uint8), new.view(np.uint8))


@pytest.mark.gpu
@pytest.mark.cuda
@pytest.mark.parametrize("count", [5, 33, 257, 513])
def test_rebuilt_mesh_rays_match_independent_linear_traversal(count):
    from gnss_gpu.bvh import BVHAccelerator
    from gnss_gpu.raytrace import BuildingModel

    mesh = _mesh(count)
    model = BVHAccelerator(mesh)
    rng = np.random.default_rng(13)
    rx = rng.uniform(-50., 50., 3)
    sat = rng.uniform(-300., 300., (31, 3))
    retained = []
    try:
        for offset in (0., 1000., -1000., 0.):
            changed = mesh + offset
            model.update_triangles(changed)
            # Linear traversal supplies a separate intersection implementation.
            expected = BuildingModel(changed).check_los(rx, sat)
            actual = model.check_los(rx, sat)
            np.testing.assert_array_equal(actual, expected)
            retained.append((actual, actual.copy()))
            changed.fill(0.)
            np.testing.assert_array_equal(model.check_los(rx, sat), expected)
            for output, snapshot in retained:
                np.testing.assert_array_equal(output, snapshot)
    finally:
        model.release_gpu()
