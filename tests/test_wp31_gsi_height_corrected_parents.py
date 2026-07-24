from __future__ import annotations

import pytest
from pyproj import Transformer

from experiments.build_wp31_gsi_height_corrected_parents import build_corrected_parents


def _ecef(lon: float, height: float) -> list[float]:
    transform = Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True)
    return list(transform.transform(lon, 35.0, height))


def test_build_corrected_parents_preserves_horizontal_coordinates() -> None:
    source = {
        "segment": [10, 20],
        "candidates": [
            {"candidate_id": 0, "position_ecef": _ecef(139.0, 42.0)},
            {"candidate_id": 1, "position_ecef": _ecef(139.1, 38.0)},
        ],
    }
    height = {
        "reason": "weak_absolute_height_match",
        "predicted_antenna_ellipsoid_height_m": 40.0,
        "best_candidate_id": 0,
        "runner_candidate_id": 1,
        "runner_gap_m": 1.0,
    }

    result = build_corrected_parents(source, height)

    to_lla = Transformer.from_crs("EPSG:4978", "EPSG:4979", always_xy=True)
    coordinates = [to_lla.transform(*row["position_ecef"]) for row in result["candidates"]]
    assert coordinates[0][0] == pytest.approx(139.0)
    assert coordinates[1][0] == pytest.approx(139.1)
    assert coordinates[0][2] == pytest.approx(40.0, abs=1e-6)
    assert coordinates[1][2] == pytest.approx(40.0, abs=1e-6)
    assert result["candidates"][0]["absolute_height_correction_m"] == pytest.approx(2.0)


def test_build_corrected_parents_rejects_selected_result() -> None:
    with pytest.raises(ValueError, match="proposal"):
        build_corrected_parents(
            {"segment": [1, 2], "candidates": []},
            {"reason": "gsi_ground_height_calibrated"},
        )
