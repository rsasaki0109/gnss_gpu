import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

from utd_edge_features import (  # noqa: E402
    DiffractionEdgeSet,
    epoch_utd_summary,
    extract_diffraction_edges,
    per_sat_utd_candidates,
)


def test_extract_diffraction_edges_filters_coplanar_fan_diagonal():
    tris = np.array(
        [
            [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 10.0, 0.0]],
            [[0.0, 0.0, 0.0], [10.0, 10.0, 0.0], [0.0, 10.0, 0.0]],
        ],
        dtype=np.float64,
    )

    edges = extract_diffraction_edges(
        tris,
        include_boundary_edges=False,
        min_edge_length_m=1.0,
        min_dihedral_deg=20.0,
    )

    assert edges.size == 0


def test_extract_diffraction_edges_keeps_welded_box_edges():
    from gnss_gpu.raytrace import BuildingModel

    model = BuildingModel.create_box(center=[0.0, 0.0, 5.0], width=10.0, depth=10.0, height=10.0)
    edges = extract_diffraction_edges(
        model.triangles,
        include_boundary_edges=False,
        min_edge_length_m=1.0,
        min_dihedral_deg=20.0,
    )

    assert edges.size >= 12
    assert not np.any(edges.is_boundary)
    assert np.nanmax(edges.dihedral_deg) >= 80.0


def test_per_sat_utd_candidates_detects_edge_on_ray():
    edges = DiffractionEdgeSet(
        start=np.array([[100.0, 0.0, 0.0]], dtype=np.float64),
        end=np.array([[100.0, 0.0, 50.0]], dtype=np.float64),
        midpoint=np.array([[100.0, 0.0, 25.0]], dtype=np.float64),
        length_m=np.array([50.0], dtype=np.float64),
        dihedral_deg=np.array([90.0], dtype=np.float64),
        is_boundary=np.array([False]),
    )
    rx = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    sat = np.array([[200.0, 0.0, 20.0]], dtype=np.float64)

    out = per_sat_utd_candidates(
        rx,
        sat,
        edges,
        max_edge_range_m=150.0,
        max_ray_edge_distance_m=1.0,
        max_excess_path_m=1.0,
    )

    assert out["candidate_count"][0] == 1
    assert out["min_edge_distance_m"][0] < 1e-9
    assert out["min_excess_path_m"][0] < 1e-9
    assert out["score"][0] > 0.9


def test_epoch_utd_summary_tracks_nlos_candidate_counts():
    edges = DiffractionEdgeSet(
        start=np.array([[100.0, 0.0, 0.0]], dtype=np.float64),
        end=np.array([[100.0, 0.0, 50.0]], dtype=np.float64),
        midpoint=np.array([[100.0, 0.0, 25.0]], dtype=np.float64),
        length_m=np.array([50.0], dtype=np.float64),
        dihedral_deg=np.array([90.0], dtype=np.float64),
        is_boundary=np.array([False]),
    )
    rx = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    sats = np.array(
        [
            [200.0, 0.0, 20.0],
            [0.0, 200.0, 20.0],
        ],
        dtype=np.float64,
    )
    is_los = np.array([False, True])

    summary = epoch_utd_summary(
        rx,
        sats,
        is_los,
        edges,
        max_edge_range_m=150.0,
        max_ray_edge_distance_m=1.0,
        max_excess_path_m=1.0,
    )

    assert summary["sat_count"] == 2
    assert summary["utd_candidate_sat_count"] == 1
    assert summary["utd_candidate_nlos_sat_count"] == 1
    assert summary["utd_candidate_count_nlos"] == 1
    assert summary["utd_score_nlos_sum"] > 0.9
