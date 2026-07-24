from __future__ import annotations

import numpy as np

from experiments.build_wp75_affine_horizontal_seed_grid import build_grid


def test_affine_grid_preserves_gsi_up_and_enumerates_horizontal_cells() -> None:
    source = {
        "production_input_truth": False,
        "segment": [0, 2],
        "offset_model": {"mode": "right_boundary_affine_zero"},
        "float_ambiguity_diagnostics": {"float_offset_ecef_m": [1.0, 2.0, 3.0]},
        "gsi_height_prior": {"affine_reference_up_prior_center_m": -0.5},
    }
    trajectory = [
        {"epoch": "0", "ecef_x": "6378137", "ecef_y": "0", "ecef_z": "0"},
        {"epoch": "1", "ecef_x": "6378137", "ecef_y": "1", "ecef_z": "0"},
    ]

    result = build_grid(source, trajectory, radius_m=1.0, step_m=1.0)

    assert result["grid"]["seed_count"] == 9
    up = np.asarray(result["local_basis_ecef"]["up"])
    for seed in result["seeds"]:
        assert abs(float(np.dot(seed["offset_ecef_m"], up)) + 0.5) < 1e-12


def test_affine_grid_accepts_fixed_promoted_boundary_model() -> None:
    source = {
        "production_input_truth": False,
        "segment": [0, 1],
        "offset_model": {
            "mode": "right_boundary_affine_fixed",
            "boundary_offset_ecef_m": [1.0, 2.0, 3.0],
        },
        "float_ambiguity_diagnostics": {"float_offset_ecef_m": [0.0, 0.0, 0.0]},
        "gsi_height_prior": {"affine_reference_up_prior_center_m": 0.0},
    }
    trajectory = [{"epoch": "0", "ecef_x": "6378137", "ecef_y": "0", "ecef_z": "0"}]

    result = build_grid(source, trajectory, radius_m=0.5, step_m=0.5)

    assert result["offset_model"]["mode"] == "right_boundary_affine_fixed"
    assert result["grid"]["seed_count"] == 9


def test_grid_accepts_constant_model_and_uses_constant_gsi_center() -> None:
    source = {
        "production_input_truth": False,
        "segment": [0, 1],
        "offset_model": {"mode": "constant"},
        "float_ambiguity_diagnostics": {"float_offset_ecef_m": [1.0, 2.0, 3.0]},
        "gsi_height_prior": {
            "up_prior_center_m": -0.25,
            "affine_reference_up_prior_center_m": 99.0,
        },
    }
    trajectory = [{"epoch": "0", "ecef_x": "6378137", "ecef_y": "0", "ecef_z": "0"}]

    result = build_grid(source, trajectory, radius_m=0.5, step_m=0.5)

    assert result["gsi_reference_up_center_key"] == "up_prior_center_m"
    up = np.asarray(result["local_basis_ecef"]["up"])
    assert all(
        abs(float(np.dot(seed["offset_ecef_m"], up)) + 0.25) < 1e-12
        for seed in result["seeds"]
    )
