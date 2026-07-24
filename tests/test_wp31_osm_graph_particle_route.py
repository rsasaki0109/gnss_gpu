import networkx as nx
import numpy as np

from experiments.build_wp31_osm_graph_particle_route import (
    advance_particle,
    edge_bearing,
    endpoint_node_distances,
    nearby_edge_states,
    state_endpoint_distance,
)


def _graph():
    graph = nx.Graph()
    graph.add_node((0, 0), xy=np.array([0.0, 0.0]))
    graph.add_node((1, 0), xy=np.array([10.0, 0.0]))
    graph.add_node((2, 0), xy=np.array([20.0, 0.0]))
    graph.add_node((1, 1), xy=np.array([10.0, 10.0]))
    graph.add_edge((0, 0), (1, 0), weight=10.0)
    graph.add_edge((1, 0), (2, 0), weight=10.0)
    graph.add_edge((1, 0), (1, 1), weight=10.0)
    return graph


def test_edge_bearing_uses_east_north_convention():
    assert np.isclose(edge_bearing(np.array([0.0, 0.0]), np.array([10.0, 0.0])), np.pi / 2)
    assert np.isclose(edge_bearing(np.array([0.0, 0.0]), np.array([0.0, 10.0])), 0.0)


def test_advance_particle_takes_heading_consistent_branch():
    graph = _graph()
    result = advance_particle(
        graph, (0, 0), (1, 0), 9.0, 5.0, np.pi / 2,
        np.random.default_rng(3), np.deg2rad(1.0),
    )
    assert result[0:2] == ((1, 0), (2, 0))
    assert np.isclose(result[2], 4.0)


def test_state_endpoint_distance_uses_both_ends_of_edge():
    graph = _graph()
    distances = endpoint_node_distances(graph, (1, 0), (2, 0), 5.0)
    assert np.isclose(state_endpoint_distance(graph, distances, (0, 0), (1, 0), 4.0), 11.0)


def test_nearby_edge_states_preserves_multiple_start_hypotheses():
    states = nearby_edge_states(_graph(), np.array([10.0, 1.0]), max_distance_m=2.0, limit=8)
    assert len(states) >= 3
    assert states[0][4] <= states[-1][4]
