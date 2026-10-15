"""Combined LOS/reflection results, buffer lifetime, and urban integration."""

import numpy as np
import pytest

from gnss_gpu.bvh import BVHAccelerator
from gnss_gpu.urban_signal_sim import UrbanSignalSimulator

native = pytest.importorskip("gnss_gpu._bvh")
pytestmark = [pytest.mark.gpu, pytest.mark.cuda]


def _scene(count=512):
    rng = np.random.default_rng(count)
    triangles = rng.uniform(-100., 100., (count, 3, 3))
    nodes, indices = native.bvh_build(triangles)
    return native.DeviceBVH(nodes, triangles[indices]), nodes, triangles[indices]


def _rays(epochs, satellites, seed=13):
    rng = np.random.default_rng(seed)
    return rng.uniform(-30., 30., (epochs, 3)), rng.uniform(-300., 300., (epochs, satellites, 3))


def _assert_native(actual, expected):
    np.testing.assert_array_equal(actual[0], expected[0])
    for new, old in zip(actual[1:], expected[1:], strict=True):
        np.testing.assert_array_equal(new.view(np.uint64), old.view(np.uint64))


def _separate(scene, rx, sat):
    return (scene.check_los_batch(rx, sat), *scene.multipath_batch(rx, sat))


@pytest.mark.parametrize("triangles", [12, 511, 512, 513, 1200])
@pytest.mark.parametrize("rays", [1, 257, 4096, 4097])
def test_combined_matches_separate_and_stateless(triangles, rays):
    scene, nodes, mesh = _scene(triangles)
    rx, sat = _rays(1, rays)
    try:
        # Start with combined to cover first-use allocation and order upload.
        actual = scene.trace_batch(rx, sat)
        expected = (native.raytrace_los_check_bvh_batch(rx, sat, nodes, mesh),
                    *native.raytrace_multipath_bvh_batch(rx, sat, nodes, mesh))
        _assert_native(actual, expected)
        _assert_native(_separate(scene, rx, sat), expected)
        single = scene.trace(rx[0], sat[0])
        _assert_native(single, (scene.check_los(rx[0], sat[0]),
                                expected[1][0], expected[2][0]))
    finally:
        scene.close()


def test_mixed_queries_grow_shrink_and_keep_outputs_owned():
    scene, nodes, mesh = _scene()
    retained = []
    try:
        for epochs, satellites in ((1, 4097), (1, 12), (3, 7), (16, 256), (1, 257), (1, 1)):
            rx, sat = _rays(epochs, satellites, epochs + satellites)
            actual = scene.trace_batch(rx, sat)
            expected = (native.raytrace_los_check_bvh_batch(rx, sat, nodes, mesh),
                        *native.raytrace_multipath_bvh_batch(rx, sat, nodes, mesh))
            _assert_native(actual, expected)
            highwater = scene.allocated_bytes
            _assert_native(_separate(scene, rx, sat), expected)
            _assert_native(scene.trace_batch(rx, sat), expected)
            assert scene.allocated_bytes == highwater
            retained.append((actual, tuple(array.copy() for array in actual)))
            rx[:] = 1234.
            sat[:] = 5678.
            for old, copies in retained:
                _assert_native(old, copies)
    finally:
        scene.close()
    scene.close()
    assert scene.allocated_bytes == 0
    rx, sat = _rays(1, 1)
    with pytest.raises(RuntimeError, match="closed"):
        scene.trace_batch(rx, sat)
    with pytest.raises(RuntimeError, match="closed"):
        scene.trace(rx[0], sat[0])


def test_padding_coincident_and_noncontiguous_rays():
    scene, _, _ = _scene()
    rx, sat = _rays(3, 14)
    sat[0, 0] = rx[0]
    sat[0, 2] = np.nan
    sat[1, 0] = np.inf
    rx[2] = np.nan
    rx, sat = rx[::-1], sat[::-1, ::2]
    try:
        actual = scene.trace_batch(rx, sat)
        _assert_native(actual, _separate(scene, rx, sat))
        assert not np.any(actual[0][0])
        assert not actual[0][1, 0]
        assert actual[0][2, 0]
        assert not actual[0][2, 1]
        for array in actual[1:]:
            assert np.all(array[0] == 0.)
    finally:
        scene.close()


@pytest.mark.parametrize("rx,sat,message", [
    (np.zeros(3), np.zeros((1, 2, 3)), "rx_ecef must have shape"),
    (np.zeros((1, 3)), np.zeros((2, 3)), "sat_ecef must have shape"),
    (np.zeros((2, 3)), np.zeros((1, 2, 3)), "leading N"),
    (np.zeros((0, 3)), np.zeros((0, 2, 3)), "n_epoch"),
    (np.zeros((1, 3)), np.zeros((1, 0, 3)), "n_sat"),
])
def test_combined_rejects_bad_batch_without_allocating(rx, sat, message):
    scene, _, _ = _scene()
    try:
        allocated = scene.allocated_bytes
        with pytest.raises(RuntimeError, match=message):
            scene.trace_batch(rx, sat)
        assert scene.allocated_bytes == allocated
    finally:
        scene.close()


@pytest.mark.parametrize("rx,sat", [
    (np.full(3, np.nan), np.zeros((1, 3))),
    (np.zeros(3), np.full((1, 3), np.inf)),
    (np.zeros(2), np.zeros((1, 3))),
    (np.zeros(3), np.zeros((0, 3))),
])
def test_single_combined_validation_matches_los(rx, sat):
    model = BVHAccelerator.create_box([100., 0., 25.], 20., 20., 50.)
    for method in (model.trace, model.check_los):
        with pytest.raises(ValueError):
            method(rx, sat)
    assert model._device_scene is None


def test_public_output_order_and_mesh_update_release():
    model = BVHAccelerator.create_box([100., 0., 25.], 20., 20., 50.)
    rx, sat = _rays(2, 7)
    try:
        los, delays, points = model.trace_batch(rx, sat)
        assert los.shape == delays.shape == (2, 7)
        assert points.shape == (2, 7, 3)
        np.testing.assert_array_equal(los, model.check_los_batch(rx, sat))
        expected = model.compute_multipath_batch(rx, sat)
        for new, old in zip((delays, points), expected, strict=True):
            np.testing.assert_array_equal(new.view(np.uint64), old.view(np.uint64))
        old_scene = model._device_scene
        model.release_gpu()
        assert old_scene.allocated_bytes == 0
        reloaded = model.trace_batch(rx, sat)
        _assert_native((reloaded[0], reloaded[2], reloaded[1]), (los, points, delays))
        model.update_triangles(model.triangles + 10000.)
        single = model.trace(rx[0].tolist(), sat[0].tolist())
        expected_delays, expected_points = model.compute_multipath(rx[0], sat[0])
        _assert_native((single[0], single[2], single[1]),
                       (model.check_los(rx[0], sat[0]), expected_points, expected_delays))
    finally:
        model.release_gpu()


class _SeparateModel:
    def __init__(self, model):
        self.model = model

    def check_los(self, rx, sat):
        return self.model.check_los(rx, sat)

    def compute_multipath(self, rx, sat):
        return self.model.compute_multipath(rx, sat)


@pytest.mark.parametrize("samples", [32, 2600, 26000])
def test_urban_iq_matches_separate_queries_and_uses_combined(monkeypatch, samples):
    pytest.importorskip("gnss_gpu._gnss_gpu_signal_sim")
    rx = np.array([6378137., 0., 0.])
    model = BVHAccelerator.create_box(rx + [100., 0., 25.], 20., 20., 50.)
    sats = rx + np.array([[200., 0., 25.], [2e7, 1e7, 0.], [-2e7, 0., 0.]])
    old = UrbanSignalSimulator(_SeparateModel(model), elevation_mask_deg=0.)
    new = UrbanSignalSimulator(model, elevation_mask_deg=0.)
    old.sim.noise_seed = new.sim.noise_seed = 37
    try:
        expected = old.compute_epoch(rx, sats, n_samples=samples)
        calls = []
        original_trace = model.trace

        def trace(receiver, satellites):
            calls.append(satellites.copy())
            return original_trace(receiver, satellites)

        def unexpected(*args):
            pytest.fail("urban simulator repeated a separate LOS/reflection query")

        monkeypatch.setattr(model, "trace", trace)
        monkeypatch.setattr(model, "check_los", unexpected)
        monkeypatch.setattr(model, "compute_multipath", unexpected)
        actual = new.compute_epoch(rx, sats, n_samples=samples)
        assert len(calls) == 1
        np.testing.assert_array_equal(calls[0], sats[:2])
        for key, value in expected.items():
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(actual[key].view(np.uint8), value.view(np.uint8))
            else:
                assert actual[key] == value
    finally:
        model.release_gpu()


def test_urban_detailed_reflection_paths_do_not_invoke_combined():
    class Model:
        def check_los(self, rx, sat):
            return np.ones(len(sat), dtype=bool)

        def compute_reflection_paths(self, rx, sat, max_paths):
            return [[] for _ in sat]

        def trace(self, *args):
            pytest.fail("combined first-order approximation replaced detailed paths")

    class Generator:
        sampling_freq = 2.6e6

        def generate_epoch(self, channels, n_samples):
            return np.zeros(2 * n_samples, dtype=np.float32)

    simulator = UrbanSignalSimulator(Model(), max_reflection_paths=2)
    simulator.sim = Generator()
    rx = np.array([6378137., 0., 0.])
    result = simulator.compute_epoch(rx, np.array([rx + [2e7, 0., 0.]]), n_samples=32)
    assert result["n_reflection_paths"] == 0
    assert result["is_los"][0]


def test_urban_no_visible_satellites_skip_combined():
    class Model:
        def trace(self, *args):
            pytest.fail("queried a model without visible satellites")

    class Generator:
        sampling_freq = 2.6e6

        def generate_epoch(self, channels, n_samples):
            assert channels == []
            return np.zeros(2 * n_samples, dtype=np.float32)

    simulator = UrbanSignalSimulator(Model())
    simulator.sim = Generator()
    rx = np.array([6378137., 0., 0.])
    result = simulator.compute_epoch(rx, np.array([rx - [2e7, 0., 0.]]), n_samples=32)
    assert not result["visible"][0]
