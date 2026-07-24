from __future__ import annotations

import pytest
from pyproj import Transformer

from experiments.build_wp31_gsi_height_corrected_shell import build_corrected_shell


def _ecef(height: float) -> list[float]:
    transform = Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True)
    return list(transform.transform(139.0, 35.0, height))


def test_build_corrected_shell_changes_only_height_at_center() -> None:
    source = {
        "segment": [10, 20],
        "candidates": [{"candidate_id": 7, "position_ecef": _ecef(42.0)}],
    }
    height = {
        "reason": "weak_absolute_height_match",
        "selected_candidate_id": None,
        "best_candidate_id": 7,
        "best_height_residual_m": 2.0,
        "runner_gap_m": 1.0,
        "predicted_antenna_ellipsoid_height_m": 40.0,
    }

    result = build_corrected_shell(source, height, (0.5,))

    to_lla = Transformer.from_crs("EPSG:4978", "EPSG:4979", always_xy=True)
    lon, lat, ellipsoid_height = to_lla.transform(*result["seed_center_ecef"])
    assert lon == pytest.approx(139.0)
    assert lat == pytest.approx(35.0)
    assert ellipsoid_height == pytest.approx(40.0, abs=1e-6)
    assert result["candidates"][0]["proposal_kind"] == "shell_center"
    assert len(result["candidates"]) == 27


def test_build_corrected_shell_rejects_unseparated_parent() -> None:
    source = {"segment": [10, 20], "candidates": [{"candidate_id": 7, "position_ecef": _ecef(42)}]}
    height = {
        "reason": "weak_absolute_height_match",
        "selected_candidate_id": None,
        "best_candidate_id": 7,
        "best_height_residual_m": 2.0,
        "runner_gap_m": 0.05,
        "predicted_antenna_ellipsoid_height_m": 40.0,
    }
    with pytest.raises(ValueError, match="not separated"):
        build_corrected_shell(source, height, (0.5,))
