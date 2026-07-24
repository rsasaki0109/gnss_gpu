from __future__ import annotations

import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from select_wp31_static_gsi_height import (
    select_gsi_height_candidate,
    select_gsi_height_osm_candidate,
)
from experiments.run_wp29_tdcp_anchor_smoother import _load_static_position_override


def _ecef_at_height(height_m: float) -> list[float]:
    # At latitude=longitude=0 the WGS84 ellipsoid surface is x=6378137 m.
    return [6378137.0 + height_m, 0.0, 0.0]


def _calibration(height_m: float, elevation_m: float = 2.0) -> dict:
    return {
        "antenna_position_ecef": _ecef_at_height(height_m),
        "elevation_m": elevation_m,
        "geoid_height_m": 36.0,
        "geoid_model": "GSIGEO2011_Ver2.2",
        "dem_source": "1m（レーザ）",
    }


def _target() -> dict:
    return {
        "elevation_m": 3.0,
        "geoid_height_m": 36.0,
        "geoid_model": "GSIGEO2011_Ver2.2",
        "dem_source": "1m（レーザ）",
    }


def test_selects_separated_candidate_from_calibrated_ground_height() -> None:
    calibrations = [_calibration(39.5), _calibration(39.51)]
    candidates = [
        {"candidate_id": 7, "position_ecef": _ecef_at_height(40.505)},
        {"candidate_id": 8, "position_ecef": _ecef_at_height(40.75)},
    ]
    result = select_gsi_height_candidate(candidates, calibrations, _target())
    assert result["reason"] == "gsi_ground_height_calibrated"
    assert result["selected_candidate_id"] == 7


def test_rejects_inconsistent_reference_antenna_height() -> None:
    calibrations = [_calibration(39.5), _calibration(39.9)]
    result = select_gsi_height_candidate([], calibrations, _target())
    assert result["reason"] == "inconsistent_antenna_height_calibration"


def test_rejects_unseparated_height_winner() -> None:
    calibrations = [_calibration(39.5), _calibration(39.51)]
    candidates = [
        {"candidate_id": 7, "position_ecef": _ecef_at_height(40.50)},
        {"candidate_id": 8, "position_ecef": _ecef_at_height(40.58)},
    ]
    result = select_gsi_height_candidate(candidates, calibrations, _target())
    assert result["reason"] == "height_winner_not_separated"
    assert result["best_candidate_id"] == 7


def test_rejects_non_laser_dem_source() -> None:
    target = _target()
    target["dem_source"] = "10m"
    result = select_gsi_height_candidate([], [_calibration(39.5), _calibration(39.51)], target)
    assert result["reason"] == "unsupported_dem_source"


def test_smoother_accepts_gsi_height_artifact(tmp_path: Path) -> None:
    path = tmp_path / "height.json"
    path.write_text(
        json.dumps(
            {
                "selected_candidate_id": 18,
                "reason": "gsi_ground_height_calibrated",
                "segment": [1867, 2066],
                "position_ecef": [1.0, 2.0, 3.0],
            }
        ),
        encoding="utf-8",
    )
    start, end, position, candidate_id, reason = _load_static_position_override(path)
    assert (start, end, candidate_id, reason) == (
        1867,
        2066,
        18,
        "gsi_ground_height_calibrated",
    )
    assert position.tolist() == [1.0, 2.0, 3.0]


def test_height_osm_gate_accepts_exactly_one_candidate() -> None:
    calibrations = [_calibration(39.5), _calibration(39.51)]
    candidates = [
        {"candidate_id": 7, "position_ecef": _ecef_at_height(40.50)},
        {"candidate_id": 8, "position_ecef": _ecef_at_height(40.55)},
    ]
    road = [
        {"candidate_id": 7, "position_ecef": candidates[0]["position_ecef"], "road_distance_m": 0.8},
        {"candidate_id": 8, "position_ecef": candidates[1]["position_ecef"], "road_distance_m": 1.2},
    ]
    result = select_gsi_height_osm_candidate(candidates, road, calibrations, _target())
    assert result["reason"] == "gsi_height_osm_unique_gate"
    assert result["selected_candidate_id"] == 7


def test_height_osm_gate_rejects_multiple_candidates() -> None:
    calibrations = [_calibration(39.5), _calibration(39.51)]
    candidates = [
        {"candidate_id": 7, "position_ecef": _ecef_at_height(40.50)},
        {"candidate_id": 8, "position_ecef": _ecef_at_height(40.55)},
    ]
    road = [
        {"candidate_id": row["candidate_id"], "position_ecef": row["position_ecef"], "road_distance_m": 0.8}
        for row in candidates
    ]
    result = select_gsi_height_osm_candidate(candidates, road, calibrations, _target())
    assert result["reason"] == "height_osm_gate_not_unique"
    assert result["selected_candidate_id"] is None
