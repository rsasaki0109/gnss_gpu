from __future__ import annotations

import sys
from pathlib import Path

import pytest

_EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
if str(_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS))

from analyze_wp29_osm_basin_shadow import _rank_epoch  # noqa: E402


def test_centerline_prior_can_select_nearest_basin() -> None:
    rows = [
        {"log_weight": 0.0, "road_distance_m": 3.0},
        {"log_weight": -0.1, "road_distance_m": 0.2},
    ]

    selected, gamma, triggered = _rank_epoch(
        rows,
        sigma_m=1.0,
        trigger_distance_m=2.5,
        corridor_half_width_m=0.0,
    )

    assert selected == 1
    assert gamma > 0.98
    assert triggered


def test_corridor_prior_does_not_distinguish_points_inside_road_width() -> None:
    rows = [
        {"log_weight": 0.0, "road_distance_m": 3.0},
        {"log_weight": -0.1, "road_distance_m": 0.2},
    ]

    selected, gamma, triggered = _rank_epoch(
        rows,
        sigma_m=1.0,
        trigger_distance_m=2.5,
        corridor_half_width_m=4.0,
    )

    assert selected == 0
    assert gamma == pytest.approx(0.5249791875)
    assert triggered


def test_prior_abstains_when_map_basin_is_close_to_road() -> None:
    rows = [
        {"log_weight": 0.0, "road_distance_m": 1.0},
        {"log_weight": -0.2, "road_distance_m": 0.0},
    ]

    selected, _gamma, triggered = _rank_epoch(
        rows,
        sigma_m=0.1,
        trigger_distance_m=2.5,
        corridor_half_width_m=0.0,
    )

    assert selected == 0
    assert not triggered
