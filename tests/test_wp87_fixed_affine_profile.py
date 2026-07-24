from __future__ import annotations

import numpy as np

from experiments.audit_wp87_fixed_affine_profile import apply_fixed_affine_profile
from experiments.apply_wp42_moving_block_offset import (
    apply_fixed_boundary_affine_profile,
)


def test_fixed_affine_profile_connects_to_existing_boundary() -> None:
    positions = np.zeros((6, 3))

    output = apply_fixed_affine_profile(
        positions,
        start=1,
        end=5,
        start_offset=np.asarray([4.0, 0.0, 0.0]),
        boundary_offset=np.asarray([0.0, 4.0, 0.0]),
    )

    np.testing.assert_allclose(output[1], [4.0, 0.0, 0.0])
    np.testing.assert_allclose(output[4], [1.0, 3.0, 0.0])
    np.testing.assert_allclose(output[5], [0.0, 0.0, 0.0])


def test_production_fixed_affine_matches_shadow() -> None:
    positions = np.arange(18, dtype=np.float64).reshape(6, 3)
    arguments = {
        "start": 1,
        "end": 5,
        "start_offset": np.asarray([4.0, -2.0, 1.0]),
        "boundary_offset": np.asarray([1.0, 2.0, 3.0]),
    }
    shadow = apply_fixed_affine_profile(positions, **arguments)
    production = apply_fixed_boundary_affine_profile(
        positions,
        start=arguments["start"],
        end=arguments["end"],
        reference_offset=arguments["start_offset"],
        boundary_offset=arguments["boundary_offset"],
    )

    np.testing.assert_allclose(production, shadow, atol=0.0)
