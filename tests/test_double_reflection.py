import numpy as np

from gnss_gpu.double_reflection import compute_double_reflection_paths


def _corner_triangles():
    return np.array(
        [
            [
                [5.0, -10.0, -5.0],
                [5.0, 10.0, -5.0],
                [5.0, 0.0, 5.0],
            ],
            [
                [-10.0, 5.0, -5.0],
                [10.0, 5.0, -5.0],
                [0.0, 5.0, 5.0],
            ],
        ],
        dtype=float,
    )


def _rx_sat():
    return np.array([0.0, 0.0, 0.0]), np.array([0.0, 2.0, 0.0])


def _blocking_triangle():
    return np.array(
        [
            [
                [4.3, 4.4, -0.2],
                [4.3, 4.8, -0.2],
                [4.3, 4.6, 0.2],
            ],
        ],
        dtype=float,
    )


def test_corner_double_reflection_finds_path():
    triangles = _corner_triangles()
    rx, sat = _rx_sat()

    result = compute_double_reflection_paths(triangles, rx, sat, max_paths=4)

    assert len(result) == 1
    paths = result[0]
    assert paths

    delays = [path.excess_delay for path in paths]
    assert delays == sorted(delays)

    path = paths[0]
    assert path.excess_delay > 0.0
    assert len(path.points) == 2
    assert all(np.asarray(point).shape == (3,) for point in path.points)
    assert path.triangle_ids[0] != path.triangle_ids[1]

    assert len(path.normals) == 2
    assert all(np.isclose(np.linalg.norm(normal), 1.0) for normal in path.normals)

    for angle in path.incidence_angles:
        assert np.isfinite(angle)
        assert 0.0 <= angle <= 0.5 * np.pi + 1e-12


def test_blocking_triangle_removes_original_path():
    triangles = _corner_triangles()
    rx, sat = _rx_sat()

    base_paths = compute_double_reflection_paths(triangles, rx, sat, max_paths=8)[0]
    assert base_paths

    blocked_triangles = np.concatenate([triangles, _blocking_triangle()], axis=0)
    blocked_paths = compute_double_reflection_paths(
        blocked_triangles,
        rx,
        sat,
        max_paths=8,
    )[0]

    base_ids = {path.triangle_ids for path in base_paths}
    blocked_ids = {path.triangle_ids for path in blocked_paths}

    assert len(blocked_paths) < len(base_paths)
    assert base_ids.isdisjoint(blocked_ids)


def test_empty_inputs_and_max_paths_zero():
    triangles = _corner_triangles()
    rx, sat = _rx_sat()

    assert compute_double_reflection_paths(np.empty((0, 3, 3)), rx, sat) == [[]]
    assert compute_double_reflection_paths(triangles, rx, np.empty((0, 3))) == []
    assert compute_double_reflection_paths(triangles, rx, sat, max_paths=0) == [[]]


def test_max_paths_limits_results():
    triangles = np.concatenate([_corner_triangles() for _ in range(4)], axis=0)
    rx, sat = _rx_sat()

    paths = compute_double_reflection_paths(triangles, rx, sat, max_paths=2)[0]

    assert 0 < len(paths) <= 2
    delays = [path.excess_delay for path in paths]
    assert delays == sorted(delays)


def test_multi_satellite_output_shape():
    triangles = _corner_triangles()
    rx, sat = _rx_sat()
    sats = np.vstack([sat, sat + np.array([0.0, 0.5, 0.0])])

    result = compute_double_reflection_paths(triangles, rx, sats, max_paths=4)

    assert len(result) == 2
    assert result[0]
    assert all(isinstance(paths, list) for paths in result)
