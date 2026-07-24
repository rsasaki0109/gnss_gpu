from __future__ import annotations

import numpy as np

from experiments.audit_wp76_affine_profile import apply_affine_profile
from experiments.select_wp76_affine_multibasis_road_carrier import (
    affine_baseline_route,
)
from experiments.apply_wp42_moving_block_offset import (
    apply_right_boundary_affine_profile,
)


def test_affine_profile_reaches_zero_at_right_boundary() -> None:
    positions = np.zeros((6, 3))

    output = apply_affine_profile(
        positions,
        start=1,
        end=5,
        boundary_epoch=5,
        reference_offset=np.asarray([4.0, 0.0, 0.0]),
    )

    np.testing.assert_allclose(output[:, 0], [0.0, 4.0, 3.0, 2.0, 1.0, 0.0])


def test_production_affine_profile_matches_shadow_implementation() -> None:
    positions = np.arange(18, dtype=np.float64).reshape(6, 3)
    arguments = {
        "start": 1,
        "end": 5,
        "boundary_epoch": 5,
        "reference_offset": np.asarray([4.0, -2.0, 1.0]),
    }

    shadow = apply_affine_profile(positions, **arguments)
    production = apply_right_boundary_affine_profile(positions, **arguments)

    np.testing.assert_allclose(production, shadow, atol=0.0)


def test_fixed_affine_selector_baseline_reaches_promoted_boundary() -> None:
    route = np.zeros((3, 3))
    scales = np.asarray([1.0, 0.5, 0.0])

    adjusted = affine_baseline_route(
        route,
        scales,
        {
            "mode": "right_boundary_affine_fixed",
            "boundary_offset_ecef_m": [2.0, 4.0, 6.0],
        },
    )

    np.testing.assert_allclose(adjusted[:, 0], [0.0, 1.0, 2.0])
    np.testing.assert_allclose(adjusted[-1], [2.0, 4.0, 6.0])
