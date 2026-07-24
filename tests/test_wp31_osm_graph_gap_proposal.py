import numpy as np

from experiments.build_wp31_osm_graph_gap_proposal import (
    build_road_graph,
    resample_offset_path,
    shortest_road_paths,
)


def test_shortest_road_paths_reports_distinct_runner_gap():
    lines = [
        [[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]],
        [[0.0, 0.0], [10.0, 5.0], [20.0, 0.0]],
    ]
    graph = build_road_graph(lines)
    paths, diagnostics = shortest_road_paths(graph, np.array([0.0, 0.0]), np.array([20.0, 0.0]))
    assert len(paths) == 2
    assert diagnostics["path_lengths_m"][0] == 20.0
    assert diagnostics["second_path_length_gap_m"] > 2.0


def test_resample_offset_path_closes_exactly_at_boundaries():
    centerline = np.array([[0.0, 0.0], [10.0, 0.0]])
    route = resample_offset_path(
        centerline,
        np.array([0.0, 2.0]),
        np.array([10.0, 4.0]),
        np.ones(4),
    )
    np.testing.assert_allclose(route[0], [0.0, 2.0])
    np.testing.assert_allclose(route[-1], [10.0, 4.0])
    np.testing.assert_allclose(route[2], [5.0, 3.0])
