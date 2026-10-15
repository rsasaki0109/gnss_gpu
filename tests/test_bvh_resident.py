"""Resident BVH ownership, buffer reuse, updates, and exact legacy outputs."""

import copy
import pickle

import numpy as np
import pytest

from gnss_gpu.bvh import BVHAccelerator
from gnss_gpu.raytrace import BuildingModel

native = pytest.importorskip("gnss_gpu._bvh")
pytestmark = [pytest.mark.gpu, pytest.mark.cuda]


def _mesh(count):
    rng = np.random.default_rng(count)
    triangles = rng.uniform(-100., 100., (count, 3, 3))
    nodes, indices = native.bvh_build(triangles)
    return nodes, triangles[indices].copy()


def _rays(epochs, satellites, seed=17):
    rng = np.random.default_rng(seed)
    return rng.uniform(-30., 30., (epochs, 3)), rng.uniform(-300., 300., (epochs, satellites, 3))


def _assert_bits(actual, expected):
    for new, old in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(new.view(np.uint64), old.view(np.uint64))


@pytest.mark.parametrize("triangles", [1, 511, 512, 513, 1025])
@pytest.mark.parametrize("rays", [1, 127, 257, 4096, 4097])
def test_resident_matches_legacy(triangles, rays):
    nodes, mesh = _mesh(triangles)
    rx, sat = _rays(1, rays)
    scene = native.DeviceBVH(nodes, mesh)
    try:
        _assert_bits(scene.multipath_batch(rx, sat),
                     native.raytrace_multipath_bvh_batch(rx, sat, nodes, mesh))
        np.testing.assert_array_equal(scene.check_los_batch(rx, sat),
                                      native.raytrace_los_check_bvh_batch(rx, sat, nodes, mesh))
        np.testing.assert_array_equal(scene.check_los(rx[0], sat[0]),
                                      native.raytrace_los_check_bvh(rx[0], sat[0], nodes, mesh))
    finally:
        scene.close()


def test_growing_and_shrinking_buffers_keep_outputs_owned():
    nodes, mesh = _mesh(512)
    scene = native.DeviceBVH(nodes, mesh)
    retained = []
    try:
        highwater = 0
        for epochs, satellites in ((1, 1), (3, 7), (1, 257), (16, 256), (1, 4097), (2, 5), (1, 1)):
            rx, sat = _rays(epochs, satellites, seed=epochs + satellites)
            outputs = scene.multipath_batch(rx, sat)
            _assert_bits(outputs, native.raytrace_multipath_bvh_batch(rx, sat, nodes, mesh))
            flags = scene.check_los_batch(rx, sat)
            np.testing.assert_array_equal(flags,
                                          native.raytrace_los_check_bvh_batch(rx, sat, nodes, mesh))
            retained.append((outputs, tuple(array.copy() for array in outputs), flags, flags.copy()))
            assert scene.allocated_bytes >= highwater
            highwater = scene.allocated_bytes
            for old_outputs, copies, old_flags, copied_flags in retained:
                _assert_bits(old_outputs, copies)
                np.testing.assert_array_equal(old_flags, copied_flags)
            scene.multipath_batch(rx, sat)
            scene.check_los_batch(rx, sat)
            assert scene.allocated_bytes == highwater
    finally:
        scene.close()
    assert scene.allocated_bytes == 0


def test_device_snapshot_survives_input_mutation_and_release():
    nodes, mesh = _mesh(512)
    rx, sat = _rays(2, 7)
    expected = native.raytrace_multipath_bvh_batch(rx, sat, nodes, mesh)
    expected_los = native.raytrace_los_check_bvh_batch(rx, sat, nodes, mesh)
    scene = native.DeviceBVH(nodes, mesh)
    nodes[:] = 0.
    mesh[:] = 0.
    del nodes, mesh
    _assert_bits(scene.multipath_batch(rx, sat), expected)
    np.testing.assert_array_equal(scene.check_los_batch(rx, sat), expected_los)
    assert scene.allocated_bytes > 0
    scene.close()
    scene.close()
    assert scene.allocated_bytes == 0
    for method in (scene.check_los_batch, scene.multipath_batch):
        with pytest.raises(RuntimeError, match="closed"):
            method(rx, sat)


def test_nonfinite_padding_and_coincident_rays():
    nodes, mesh = _mesh(512)
    rx, sat = _rays(3, 7)
    sat[0, 0] = rx[0]
    sat[0, 1] = np.nan
    sat[1, 0] = np.inf
    rx[2] = np.nan
    scene = native.DeviceBVH(nodes, mesh)
    try:
        _assert_bits(scene.multipath_batch(rx, sat),
                     native.raytrace_multipath_bvh_batch(rx, sat, nodes, mesh))
        np.testing.assert_array_equal(scene.check_los_batch(rx, sat),
                                      native.raytrace_los_check_bvh_batch(rx, sat, nodes, mesh))
    finally:
        scene.close()


def test_los_only_does_not_allocate_reflection_work():
    nodes, mesh = _mesh(512)
    rx, sat = _rays(1, 12)
    scene = native.DeviceBVH(nodes, mesh)
    try:
        # BVHNode contains six doubles and four int32 fields (64 bytes).
        geometry_bytes = len(nodes) * 64 + mesh.nbytes
        assert scene.allocated_bytes == geometry_bytes
        scene.check_los_batch(rx, sat)
        assert scene.allocated_bytes == geometry_bytes + rx.nbytes + sat.nbytes + 12 * 4
        scene.multipath_batch(rx, sat)
        after_reflection = scene.allocated_bytes
        assert after_reflection > geometry_bytes + rx.nbytes + sat.nbytes + 12 * 4
        scene.multipath_batch(rx, sat)
        assert scene.allocated_bytes == after_reflection
    finally:
        scene.close()


@pytest.mark.parametrize("field,value,message", [
    (6, .5, "int32 integers"), (6, 2**40, "int32 integers"),
    (6, 10000., "child index"), (6, 0., "cycle"),
])
def test_bad_device_nodes_are_rejected(field, value, message):
    nodes, mesh = _mesh(512)
    nodes[0, field] = value
    with pytest.raises((RuntimeError, ValueError), match=message):
        native.DeviceBVH(nodes, mesh)


def test_bad_leaf_range_is_rejected():
    nodes, mesh = _mesh(512)
    leaf = np.flatnonzero(nodes[:, 6] == -1)[0]
    nodes[leaf, 9] = 513
    with pytest.raises(ValueError, match="triangle range"):
        native.DeviceBVH(nodes, mesh)


@pytest.mark.parametrize("triangles", [np.zeros((0, 3, 3)), np.zeros((1, 2, 3)),
                                      np.full((1, 3, 3), np.nan)])
def test_failed_mesh_update_keeps_current_scene(triangles):
    model = BVHAccelerator.create_box([100., 0., 25.], 20., 20., 50.)
    rx = np.zeros((1, 3))
    sat = np.array([[[200., 0., 25.]]])
    expected = model.compute_multipath_batch(rx, sat)
    scene = model._device_scene
    with pytest.raises(ValueError):
        model.update_triangles(triangles)
    assert model._device_scene is scene
    _assert_bits(model.compute_multipath_batch(rx, sat), expected)
    assert model.n_triangles == 12
    model.release_gpu()


def test_wrapper_snapshot_update_and_explicit_gpu_release():
    triangles = BuildingModel.create_box([100., 0., 25.], 20., 20., 50.).triangles
    model = BVHAccelerator(triangles)
    assert model._device_scene is None  # CPU build does not allocate CUDA memory.
    rx = np.zeros((1, 3))
    sat = np.array([[[200., 0., 25.], [0., 0., 2e7]]])
    assert not model.check_los_batch(rx, sat)[0, 0]
    original_scene = model._device_scene
    triangles[:] += 10000.
    assert not model.check_los_batch(rx, sat)[0, 0]
    with pytest.raises(ValueError, match="read-only"):
        model.triangles[0, 0, 0] = 0.
    model.update_triangles(triangles)
    assert original_scene.allocated_bytes == 0
    assert model._device_scene is None
    assert np.all(model.check_los_batch(rx, sat))
    expected = model.compute_multipath_batch(rx, sat)
    updated_scene = model._device_scene
    model.release_gpu()
    model.release_gpu()
    assert updated_scene.allocated_bytes == 0
    _assert_bits(model.compute_multipath_batch(rx, sat), expected)
    assert model._device_scene is not updated_scene
    model.release_gpu()


@pytest.mark.parametrize("clone", [copy.deepcopy, lambda model: pickle.loads(pickle.dumps(model))])
def test_mesh_serialization_excludes_gpu_resources(clone):
    model = BVHAccelerator.create_box([100., 0., 25.], 20., 20., 50.)
    rx, sat = _rays(2, 7)
    expected = model.compute_multipath_batch(rx, sat)
    restored = clone(model)
    assert restored._device_scene is None
    assert not restored.triangles.flags.writeable
    _assert_bits(restored.compute_multipath_batch(rx, sat), expected)
    assert restored._device_scene is not model._device_scene
    restored.release_gpu()
    _assert_bits(model.compute_multipath_batch(rx, sat), expected)
    model.release_gpu()


def test_two_scenes_remain_independent_when_one_is_updated():
    first = BVHAccelerator.create_box([100., 0., 25.], 20., 20., 50.)
    second = BVHAccelerator(first.triangles)
    rx, sat = _rays(2, 7)
    expected = second.compute_multipath_batch(rx, sat)
    first.check_los_batch(rx, sat)
    first.update_triangles(first.triangles + 10000.)
    first.check_los_batch(rx, sat)
    _assert_bits(second.compute_multipath_batch(rx, sat), expected)
    first.release_gpu()
    second.release_gpu()
