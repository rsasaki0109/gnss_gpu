import numpy as np
import pytest

from gnss_gpu.raytrace import BuildingModel


def _combine_models(*models):
    return BuildingModel(np.concatenate([model.triangles for model in models], axis=0))


def test_single_vertical_wall_reflection_matches_analytic_delay():
    model = BuildingModel.create_box(
        center=[0.05, 0.0, 0.0],
        width=0.1,
        depth=20.0,
        height=20.0,
    )

    rx = np.array([-10.0, -3.0, 0.0])
    sat = np.array([-20.0, 9.0, 0.0])

    paths_by_sat = model.compute_reflection_paths(rx, sat, max_paths=4)

    assert len(paths_by_sat) == 1
    assert len(paths_by_sat[0]) == 1

    path = paths_by_sat[0][0]
    expected_reflection = np.array([0.0, 1.0, 0.0])
    expected_excess = (
        np.linalg.norm(rx - expected_reflection)
        + np.linalg.norm(sat - expected_reflection)
        - np.linalg.norm(rx - sat)
    )

    np.testing.assert_allclose(path.reflection_point, expected_reflection, atol=1e-8)
    np.testing.assert_allclose(path.excess_delay, expected_excess, rtol=1e-12, atol=1e-9)
    assert 0.0 <= path.incidence_angle <= 0.5 * np.pi
    np.testing.assert_allclose(np.linalg.norm(path.normal), 1.0, rtol=1e-12, atol=1e-12)


def test_reflection_path_occluded_by_second_box_is_rejected():
    wall = BuildingModel.create_box(
        center=[0.05, 0.0, 0.0],
        width=0.1,
        depth=20.0,
        height=20.0,
    )
    blocker = BuildingModel.create_box(
        center=[-5.0, 3.0, 0.0],
        width=0.2,
        depth=0.2,
        height=2.0,
    )
    model = _combine_models(wall, blocker)

    rx = np.array([-10.0, -3.0, 0.0])
    sat = np.array([[-20.0, 9.0, 0.0]])

    paths_by_sat = model.compute_reflection_paths(rx, sat, max_paths=4)

    assert paths_by_sat == [[]]


def test_two_walls_return_max_paths_sorted_by_excess_delay():
    wall_x = BuildingModel.create_box(
        center=[0.05, -8.5, 0.0],
        width=0.1,
        depth=8.0,
        height=20.0,
    )
    wall_y = BuildingModel.create_box(
        center=[-14.0, 0.05, 0.0],
        width=10.0,
        depth=0.1,
        height=20.0,
    )
    model = _combine_models(wall_x, wall_y)

    rx = np.array([-10.0, -6.0, 0.0])
    sat = np.array([[-20.0, -12.0, 0.0]])

    paths = model.compute_reflection_paths(rx, sat, max_paths=2)[0]

    assert len(paths) == 2
    delays = np.array([path.excess_delay for path in paths])
    assert np.all(delays[:-1] <= delays[1:])

    reflection_x_wall = np.array([0.0, -8.0, 0.0])
    reflection_y_wall = np.array([-40.0 / 3.0, 0.0, 0.0])
    sat0 = sat[0]
    expected_delays = sorted([
        np.linalg.norm(rx - reflection_x_wall)
        + np.linalg.norm(sat0 - reflection_x_wall)
        - np.linalg.norm(rx - sat0),
        np.linalg.norm(rx - reflection_y_wall)
        + np.linalg.norm(sat0 - reflection_y_wall)
        - np.linalg.norm(rx - sat0),
    ])

    np.testing.assert_allclose(delays, expected_delays, rtol=1e-12, atol=1e-9)

    paths_again = model.compute_reflection_paths(rx, sat, max_paths=2)[0]
    np.testing.assert_allclose(
        [path.excess_delay for path in paths_again],
        [path.excess_delay for path in paths],
        rtol=0.0,
        atol=0.0,
    )
    assert [path.triangle_id for path in paths_again] == [path.triangle_id for path in paths]


def test_zenith_satellite_has_no_vertical_wall_reflection():
    model = BuildingModel.create_box(
        center=[0.05, 0.0, 0.0],
        width=0.1,
        depth=20.0,
        height=20.0,
    )

    rx = np.array([-10.0, -3.0, 0.0])
    sat = np.array([[-10.0, -3.0, 100.0]])

    paths_by_sat = model.compute_reflection_paths(rx, sat, max_paths=4)

    assert paths_by_sat == [[]]


def test_near_miss_specular_point_recovered_with_tolerance():
    # Wall whose finite extent ends at y=0.25, while the analytic specular point
    # for this rx/sat lands at y=1.0 -> a 0.75 m miss past the triangle edge,
    # exactly the tessellation gap that drops reflections on real meshes.
    model = BuildingModel.create_box(
        center=[0.05, -9.75, 0.0],  # +x left face at x=0, y-extent [-19.75, 0.25]
        width=0.1,
        depth=20.0,
        height=20.0,
    )
    rx = np.array([-10.0, -3.0, 0.0])
    sat = np.array([-20.0, 9.0, 0.0])

    # Strict containment rejects the near-miss reflection.
    assert model.compute_reflection_paths(rx, sat, max_paths=4) == [[]]

    # A 1 m tolerance recovers it, snapping the point onto the facade edge.
    tol = model.compute_reflection_paths(
        rx, sat, max_paths=4, reflection_point_tol_m=1.0)
    assert len(tol[0]) == 1
    path = tol[0][0]
    assert path.reflection_point[1] == pytest.approx(0.25, abs=1e-6)  # snapped to edge
    assert path.excess_delay > 0.0
    # A tolerance smaller than the 0.75 m miss still rejects it.
    assert model.compute_reflection_paths(
        rx, sat, max_paths=4, reflection_point_tol_m=0.5) == [[]]


def test_in_triangle_reflection_unaffected_by_tolerance():
    # When the specular point is squarely inside the facet, adding a tolerance
    # must not change the result (same point, same delay).
    model = BuildingModel.create_box(center=[0.05, 0.0, 0.0],
                                     width=0.1, depth=20.0, height=20.0)
    rx = np.array([-10.0, -3.0, 0.0])
    sat = np.array([-20.0, 9.0, 0.0])
    strict = model.compute_reflection_paths(rx, sat, max_paths=4)
    # Tolerance below the inter-triangle spacing must leave a solidly-contained
    # reflection identical (a larger tolerance can legitimately add duplicate
    # hits from adjacent co-planar facets of the same wall, which is harmless for
    # strongest-replica tracking but would change the count here).
    relaxed = model.compute_reflection_paths(
        rx, sat, max_paths=4, reflection_point_tol_m=0.1)
    assert len(strict[0]) == len(relaxed[0]) == 1
    np.testing.assert_allclose(strict[0][0].reflection_point,
                               relaxed[0][0].reflection_point, atol=1e-9)
    np.testing.assert_allclose(strict[0][0].excess_delay,
                               relaxed[0][0].excess_delay, rtol=1e-12, atol=1e-9)
