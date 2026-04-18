import json
import math

import numpy as np
import pytest

from gnss_gpu.osm_constraint import (
    OSMRoadNetwork,
    RoadSegmentArrays,
    huber_road_log_weight,
    lla_deg_to_ecef,
    nearest_distances_enu,
)


def _write_geojson(path):
    data = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"highway": "residential"},
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[0.0, 0.0], [0.001, 0.0]],
                },
            },
            {
                "type": "Feature",
                "properties": {"highway": "motorway"},
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[0.0, 0.001], [0.001, 0.001]],
                },
            },
            {
                "type": "Feature",
                "properties": {"highway": "footway"},
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[0.0, 0.002], [0.001, 0.002]],
                },
            },
        ],
    }
    path.write_text(json.dumps(data), encoding="utf-8")


def test_nearest_distances_enu_returns_distance_and_segment_scale():
    segments = RoadSegmentArrays(
        start_enu_m=np.array([[0.0, 0.0], [0.0, 10.0]]),
        end_enu_m=np.array([[10.0, 0.0], [10.0, 10.0]]),
        sigma_scales=np.array([1.0, 3.0]),
        highway_types=("residential", "motorway"),
    )

    distances, scales = nearest_distances_enu(
        np.array([[5.0, 2.0], [5.0, 8.0], [-3.0, 0.0]]),
        segments,
    )

    np.testing.assert_allclose(distances, [2.0, 2.0, 3.0])
    np.testing.assert_allclose(scales, [1.0, 3.0, 1.0])


def test_huber_road_log_weight_is_soft_and_rejects_too_tight_sigma():
    weights = huber_road_log_weight(
        np.array([1.0, 10.0]),
        sigma_m=2.0,
        huber_k=2.0,
    )

    np.testing.assert_allclose(weights, [-0.125, -8.0])
    with pytest.raises(ValueError, match="sigma_m must be >= 1.0m"):
        huber_road_log_weight(np.array([0.0]), sigma_m=0.5)


def test_from_geojson_filters_non_drivable_and_marks_limited_access(tmp_path):
    path = tmp_path / "roads.geojson"
    _write_geojson(path)

    network = OSMRoadNetwork.from_geojson(
        path,
        origin_lat_deg=0.0,
        origin_lon_deg=0.0,
        limited_access_sigma_scale=4.0,
    )

    assert network.segments.n_segments == 2
    assert network.segments.highway_types == ("residential", "motorway")
    np.testing.assert_allclose(network.segments.sigma_scales, [1.0, 4.0])


def test_nearest_distances_ecef_uses_horizontal_road_distance(tmp_path):
    path = tmp_path / "roads.geojson"
    _write_geojson(path)
    network = OSMRoadNetwork.from_geojson(path, origin_lat_deg=0.0, origin_lon_deg=0.0)

    point = lla_deg_to_ecef(0.0001, 0.0005, 50.0)
    distances, scales = network.nearest_distances_ecef(point.reshape(1, 3))

    assert math.isclose(float(distances[0]), 11.057, rel_tol=0.0, abs_tol=0.10)
    assert float(scales[0]) == 1.0


def test_candidate_kernel_array_limits_segments_near_center(tmp_path):
    path = tmp_path / "roads.geojson"
    _write_geojson(path)
    network = OSMRoadNetwork.from_geojson(path, origin_lat_deg=0.0, origin_lon_deg=0.0)

    center = lla_deg_to_ecef(0.0, 0.0005, 0.0)
    arr = network.candidate_kernel_array(center, radius_m=20.0, max_segments=4)

    assert arr.shape == (1, 5)
    np.testing.assert_allclose(arr[0, 4], 1.0)
