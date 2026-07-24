from __future__ import annotations

import numpy as np
from shapely.geometry import LineString

from experiments.build_wp31_osm_particle_route_bridge import (
    particle_route,
    road_band_log_likelihood,
    systematic_resample,
)


def test_road_band_likelihood_is_flat_inside_calibration() -> None:
    values = road_band_log_likelihood(
        np.asarray([0.0, 0.2, 2.0, 5.0, 7.0]),
        lower_m=0.2,
        upper_m=5.0,
        sigma_m=2.0,
    )
    np.testing.assert_allclose(values[1:4], 0.0)
    assert values[0] < 0.0
    assert values[4] < values[0]


def test_systematic_resample_is_deterministic_for_seed() -> None:
    weights = np.asarray([0.05, 0.15, 0.3, 0.5])
    first = systematic_resample(weights, np.random.default_rng(7))
    second = systematic_resample(weights, np.random.default_rng(7))
    np.testing.assert_array_equal(first, second)
    assert np.all(np.diff(first) >= 0)


def test_particle_route_closes_simple_road_segment() -> None:
    route, metrics = particle_route(
        start_xy=np.asarray([0.0, 0.0]),
        end_xy=np.asarray([0.0, 20.0]),
        step_lengths_m=np.ones(20),
        gyro_increments_rad=np.zeros(20),
        dt_s=np.full(20, 0.2),
        initial_heading_rad=0.0,
        road_geometry=LineString([(0.0, -10.0), (0.0, 30.0)]),
        particles=256,
        random_seed=11,
        road_lower_m=0.0,
        road_upper_m=1.0,
        road_sigma_m=1.0,
        heading_sigma_deg=3.0,
        gyro_bias_sigma_dps=0.05,
        turn_noise_deg=0.05,
        scale_lower=0.98,
        scale_upper=1.02,
    )

    np.testing.assert_allclose(route[0], [0.0, 0.0])
    np.testing.assert_allclose(route[-1], [0.0, 20.0], atol=1e-9)
    assert metrics["road_distance_p95_m"] < 1.0
    assert 0.98 <= metrics["route_length_scale"] <= 1.02
