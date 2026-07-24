from __future__ import annotations

import csv
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "experiments"))

from acquire_wp50_gsi_moving_height_cache import (  # noqa: E402
    acquire_moving_cache,
    moving_candidate_samples,
    moving_candidate_source,
)


def _trajectory(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["epoch", "ecef_x", "ecef_y", "ecef_z"]
        )
        writer.writeheader()
        for epoch in range(4):
            writer.writerow(
                {
                    "epoch": epoch,
                    "ecef_x": 1 + epoch,
                    "ecef_y": 2 + epoch,
                    "ecef_z": 3 + epoch,
                }
            )


def test_moving_source_uses_contiguous_segment_median(tmp_path: Path) -> None:
    path = tmp_path / "trajectory.csv"
    _trajectory(path)
    source = moving_candidate_source(path, start=1, end=4)
    assert source["segment"] == [1, 4]
    assert source["candidates"][0]["position_ecef"] == [3.0, 4.0, 5.0]


def test_acquire_cache_uses_mocked_official_responses(tmp_path: Path) -> None:
    path = tmp_path / "trajectory.csv"
    _trajectory(path)
    calibration = {"calibration_points": [{"name": "accepted"}]}

    def fetch(url: str):
        if "getelevation" in url:
            return {"elevation": "1.5", "hsrc": "5m（レーザ）"}
        return {"OutputData": {"geoidHeight": "37.9"}}

    result = acquire_moving_cache(
        path,
        calibration,
        start=1,
        end=4,
        fetch_json=fetch,
        acquired_utc="2026-07-22T00:00:00Z",
    )
    assert result["production_input_truth"] is False
    assert result["schema"] == "wp50_gsi_moving_height_cache_v1"
    assert result["target_point"]["elevation_m"] == 1.5
    assert result["runtime_network_required"] is False


def test_moving_samples_are_fixed_and_evenly_spaced(tmp_path: Path) -> None:
    path = tmp_path / "trajectory.csv"
    _trajectory(path)
    samples = moving_candidate_samples(path, start=0, end=4, sample_count=3)
    assert [row["epoch"] for row in samples] == [0, 2, 3]


def test_acquire_cache_can_freeze_multiple_target_points(tmp_path: Path) -> None:
    path = tmp_path / "trajectory.csv"
    _trajectory(path)
    calibration = {"calibration_points": [{"name": "accepted"}]}

    def fetch(url: str):
        if "getelevation" in url:
            return {"elevation": "1.5", "hsrc": "1m（レーザ）"}
        return {"OutputData": {"geoidHeight": "37.9"}}

    result = acquire_moving_cache(
        path,
        calibration,
        start=0,
        end=4,
        sample_count=3,
        fetch_json=fetch,
        acquired_utc="2026-07-22T00:00:00Z",
    )
    assert [point["epoch"] for point in result["target_points"]] == [0, 2, 3]
    assert result["target_sampling"]["requested_count"] == 3
