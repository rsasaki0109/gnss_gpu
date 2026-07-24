from __future__ import annotations

import numpy as np
from shapely.geometry import MultiLineString

from experiments.analyze_wp70_road_translation_observability import (
    evaluate_road_translation_observability,
)


def test_l_shaped_route_has_unique_zero_width_translation() -> None:
    road = MultiLineString([[(0.0, 0.0), (0.0, 4.0)], [(0.0, 4.0), (4.0, 4.0)]])
    route = np.asarray([[0.0, 0.0], [0.0, 2.0], [0.0, 4.0], [2.0, 4.0], [4.0, 4.0]])

    result = evaluate_road_translation_observability(
        route,
        road,
        radius_m=1.0,
        step_m=0.5,
        lower_m=0.0,
        upper_m=0.0,
        max_equivalent_cells=1,
        max_equivalent_extent_m=0.0,
        min_runner_margin=0.2,
    )

    assert result["accepted"]
    assert result["winner"]["translation_xy_m"] == [0.0, 0.0]
    assert result["equivalent_cell_count"] == 1


def test_straight_road_rejects_along_road_translation_plateau() -> None:
    road = MultiLineString([[(-10.0, -20.0), (-10.0, 20.0)]])
    route = np.asarray([[-10.0, -2.0], [-10.0, 0.0], [-10.0, 2.0]])

    result = evaluate_road_translation_observability(
        route,
        road,
        radius_m=2.0,
        step_m=1.0,
        lower_m=0.0,
        upper_m=0.0,
        max_equivalent_cells=1,
        max_equivalent_extent_m=0.0,
        min_runner_margin=0.2,
    )

    assert not result["accepted"]
    assert result["reason"] == "road_translation_posterior_unobservable"
    assert result["equivalent_cell_count"] == 5
    assert result["equivalent_extent_m"] == 4.0
