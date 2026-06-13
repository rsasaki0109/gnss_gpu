import math

import numpy as np

from gnss_gpu.raytrace import BuildingModel, horizontal_ground_plane


def _model(triangles):
    return BuildingModel(np.asarray(triangles, dtype=np.float64))


def _box_triangles(min_corner, max_corner):
    x0, y0, z0 = min_corner
    x1, y1, z1 = max_corner
    v = np.array([
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
    ], dtype=np.float64)
    faces = [
        (0, 1, 2), (0, 2, 3),
        (4, 6, 5), (4, 7, 6),
        (0, 4, 5), (0, 5, 1),
        (1, 5, 6), (1, 6, 2),
        (2, 6, 7), (2, 7, 3),
        (3, 7, 4), (3, 4, 0),
    ]
    return np.array([[v[i], v[j], v[k]] for i, j, k in faces], dtype=np.float64)


def _far_box():
    return _box_triangles(
        [1.0e6, 1.0e6, 1.0e6],
        [1.0e6 + 10.0, 1.0e6 + 10.0, 1.0e6 + 10.0],
    )


def _ground_path(paths):
    return next(path for path in paths if path.triangle_id == -1)


def test_horizontal_ground_reflection_matches_image_solution():
    model = _model(_far_box())
    rx = np.array([0.0, 0.0, 10.0])
    sat = np.array([3.0, 0.0, 1000.0])

    paths = model.compute_reflection_paths(
        rx, sat, max_paths=4, ground_plane=horizontal_ground_plane(0.0),
    )[0]

    assert len(paths) == 1
    path = paths[0]
    assert path.triangle_id == -1
    np.testing.assert_allclose(path.normal, [0.0, 0.0, 1.0], atol=1e-12)

    rx_img = np.array([0.0, 0.0, -10.0])
    t = (0.0 - rx_img[2]) / (sat[2] - rx_img[2])
    reflection_point = rx_img + t * (sat - rx_img)
    expected_excess = (
        np.linalg.norm(rx - reflection_point)
        + np.linalg.norm(reflection_point - sat)
        - np.linalg.norm(rx - sat)
    )

    np.testing.assert_allclose(path.reflection_point, reflection_point, atol=1e-9)
    assert math.isclose(path.excess_delay, expected_excess, rel_tol=0.0, abs_tol=1e-9)


def test_ground_incidence_is_larger_for_low_elevation_satellite():
    model = _model(_far_box())
    rx = np.array([0.0, 0.0, 10.0])
    high_sat = np.array([3.0, 0.0, 1000.0])
    low_sat = np.array([1000.0, 0.0, 20.0])

    paths = model.compute_reflection_paths(
        rx, np.vstack([high_sat, low_sat]), max_paths=4,
        ground_plane=horizontal_ground_plane(0.0),
    )

    high_incidence = _ground_path(paths[0]).incidence_angle
    low_incidence = _ground_path(paths[1]).incidence_angle

    assert low_incidence > high_incidence
    assert low_incidence > math.radians(80.0)
    assert high_incidence < math.radians(5.0)


def test_ground_reflection_is_rejected_when_building_blocks_rx_leg():
    blocker = _box_triangles([2.4, -1.0, 3.0], [2.6, 1.0, 7.0])
    model = _model(np.vstack([_far_box(), blocker]))

    rx = np.array([0.0, 0.0, 10.0])
    sat = np.array([20.0, 0.0, 30.0])

    paths = model.compute_reflection_paths(
        rx, sat, max_paths=4, ground_plane=horizontal_ground_plane(0.0),
    )[0]

    assert all(path.triangle_id != -1 for path in paths)


def test_wall_and_ground_paths_are_merged_and_sorted_by_excess_delay():
    wall = np.array([
        [[0.0, -20.0, 0.0], [0.0, 20.0, 0.0], [0.0, 20.0, 20.0]],
        [[0.0, -20.0, 0.0], [0.0, 20.0, 20.0], [0.0, -20.0, 20.0]],
    ], dtype=np.float64)
    model = _model(np.vstack([wall, _far_box()]))

    rx = np.array([10.0, 0.0, 10.0])
    sat = np.array([100.0, 0.0, 10.0])

    paths = model.compute_reflection_paths(
        rx, sat, max_paths=4, ground_plane=horizontal_ground_plane(0.0),
    )[0]

    assert any(path.triangle_id == -1 for path in paths)
    assert any(path.triangle_id != -1 for path in paths)

    delays = [path.excess_delay for path in paths]
    assert delays == sorted(delays)


def test_default_ground_plane_none_preserves_no_ground_behavior():
    model = _model(_far_box())
    rx = np.array([0.0, 0.0, 10.0])
    sat = np.array([3.0, 0.0, 1000.0])

    paths = model.compute_reflection_paths(rx, sat, max_paths=4)[0]

    assert all(path.triangle_id != -1 for path in paths)


def test_urban_ground_reflection_uses_ground_material():
    """ground_reflection adds a ground path whose amplitude uses ground_material."""
    from gnss_gpu.urban_signal_sim import UrbanSignalSimulator
    from gnss_gpu.fresnel import reflection_coefficient

    class _CaptureSignalGenerator:
        def __init__(self, sampling_freq=2.6e6):
            self.sampling_freq = sampling_freq
            self.channels = None

        def generate_epoch(self, channels, n_samples=None):
            self.channels = [dict(ch) for ch in channels]
            count = int(n_samples) if n_samples is not None else int(self.sampling_freq * 1e-3)
            return np.zeros(2 * count, dtype=np.float32)

    class _StubBuildingModel:
        def __init__(self, mesh):
            self._mesh = mesh

        def check_los(self, rx, sats):
            sats = np.asarray(sats).reshape(-1, 3)
            return np.ones(sats.shape[0], dtype=bool)

        def compute_reflection_paths(self, rx, sats, max_paths=4, ground_plane=None):
            return self._mesh.compute_reflection_paths(
                rx, sats, max_paths=max_paths, ground_plane=ground_plane)

    # ECEF-like receiver so ecef_to_lla yields a sane local up; a thin far box
    # keeps the mesh non-empty without adding building reflections.
    rx = np.array([6378137.0 + 30.0, 0.0, 0.0])
    sat = np.array([[6378137.0 + 20200000.0, 5.0e6, 0.0]])
    far = _box_triangles(
        [rx[0] + 1.0e6, 1.0e6, 1.0e6],
        [rx[0] + 1.0e6 + 10.0, 1.0e6 + 10.0, 1.0e6 + 10.0],
    )
    mesh = _model(far)

    usim = UrbanSignalSimulator(
        building_model=_StubBuildingModel(mesh),
        elevation_mask_deg=-90.0,
        max_reflection_paths=4,
        reflector_material="concrete",
        ground_material="wet_ground",
        ground_reflection=True,
        ground_height_m=30.0,
    )
    usim.sim = _CaptureSignalGenerator()

    result = usim.compute_epoch(rx, sat, prn_list=[3], n_samples=16)

    ground_paths = [p for p in result["reflection_paths"][0] if p.triangle_id == -1]
    assert len(ground_paths) >= 1

    gpath = ground_paths[0]
    expected = reflection_coefficient(
        gpath.incidence_angle, "wet_ground", polarization="rhcp")
    # The replica channel for the ground path should use wet_ground Fresnel.
    replica_amps = [c["amplitude"] for c in result["channels"][1:]]
    assert any(abs(a - float(expected)) < 1e-9 for a in replica_amps)
