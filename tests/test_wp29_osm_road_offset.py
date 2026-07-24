from __future__ import annotations

import pytest
from shapely.geometry import LineString, Point

from experiments.analyze_wp29_static_grid_osm_shadow import road_offset_vector


def test_road_offset_vector_points_from_road_to_candidate() -> None:
    east, north, distance = road_offset_vector(
        Point(3.0, 4.0), LineString([(0.0, 0.0), (10.0, 0.0)])
    )
    assert east == pytest.approx(0.0)
    assert north == pytest.approx(4.0)
    assert distance == pytest.approx(4.0)
