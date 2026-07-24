#!/usr/bin/env python3
"""Acquire cached GSI heights at fixed points in a moving segment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "experiments"))

from acquire_wp31_gsi_height_cache import _fetch_json, build_cache  # noqa: E402


def _fetch_validated_gsi_json(url: str) -> dict[str, Any]:
    """Retry boundedly when an official GSI endpoint returns an incomplete body."""

    last_payload: dict[str, Any] = {}
    for attempt in range(4):
        payload = _fetch_json(url)
        last_payload = payload
        valid_dem = "getelevation" in url and {"elevation", "hsrc"} <= payload.keys()
        valid_geoid = "geoidcalc" in url and isinstance(payload.get("OutputData"), dict)
        if valid_dem or valid_geoid:
            return payload
        if attempt < 3:
            time.sleep(2**attempt)
    raise ValueError(f"GSI returned incomplete response after retries: {last_payload}")


def moving_candidate_source(
    trajectory: Path, *, start: int, end: int
) -> dict[str, Any]:
    if not 0 <= start < end:
        raise ValueError("moving GSI segment is invalid")
    with trajectory.open(newline="", encoding="utf-8-sig") as fh:
        rows = [
            row
            for row in csv.DictReader(fh)
            if start <= int(row["epoch"]) < end
        ]
    epochs = [int(row["epoch"]) for row in rows]
    if epochs != list(range(start, end)):
        raise ValueError("moving GSI trajectory segment is not contiguous")
    positions = np.asarray(
        [
            [float(row["ecef_x"]), float(row["ecef_y"]), float(row["ecef_z"])]
            for row in rows
        ]
    )
    center = np.median(positions, axis=0)
    return {
        "segment": [start, end],
        "candidates": [{"candidate_id": 0, "position_ecef": center.tolist()}],
    }


def moving_candidate_samples(
    trajectory: Path, *, start: int, end: int, sample_count: int
) -> list[dict[str, Any]]:
    """Return deterministic, evenly spaced trajectory samples."""

    if sample_count < 1:
        raise ValueError("moving GSI sample count must be positive")
    with trajectory.open(newline="", encoding="utf-8-sig") as fh:
        rows = {
            int(row["epoch"]): row
            for row in csv.DictReader(fh)
            if start <= int(row["epoch"]) < end
        }
    if sorted(rows) != list(range(start, end)):
        raise ValueError("moving GSI trajectory segment is not contiguous")
    count = min(sample_count, end - start)
    epochs = sorted(
        {
            int(round(value))
            for value in np.linspace(start, end - 1, num=count, dtype=np.float64)
        }
    )
    return [
        {
            "epoch": epoch,
            "position_ecef": [
                float(rows[epoch]["ecef_x"]),
                float(rows[epoch]["ecef_y"]),
                float(rows[epoch]["ecef_z"]),
            ],
        }
        for epoch in epochs
    ]


def acquire_moving_cache(
    trajectory: Path,
    calibration_cache: dict[str, Any],
    *,
    start: int,
    end: int,
    sample_count: int = 1,
    fetch_json: Callable[[str], dict[str, Any]],
    acquired_utc: str | None = None,
) -> dict[str, Any]:
    source = moving_candidate_source(trajectory, start=start, end=end)
    result = build_cache(
        source,
        calibration_cache,
        query_basis="trajectory_segment_median_independent_of_ambiguity_candidate",
        fetch_json=fetch_json,
        acquired_utc=acquired_utc,
    )
    result["schema"] = "wp50_gsi_moving_height_cache_v1"
    result["production_input_truth"] = False
    result["segment"] = [start, end]
    result["target_point"]["name"] = f"moving_{start}_{end}_trajectory_median"
    result["target_point"]["trajectory_source"] = str(trajectory).replace("\\", "/")
    result["target_point"]["trajectory_source_sha256"] = hashlib.sha256(
        trajectory.read_bytes()
    ).hexdigest()
    if sample_count > 1:
        samples = moving_candidate_samples(
            trajectory, start=start, end=end, sample_count=sample_count
        )
        target_points = []
        for sample in samples:
            sample_cache = build_cache(
                {
                    "segment": [start, end],
                    "candidates": [
                        {
                            "candidate_id": sample["epoch"],
                            "position_ecef": sample["position_ecef"],
                        }
                    ],
                },
                calibration_cache,
                query_basis="fixed_evenly_spaced_trajectory_epoch",
                fetch_json=fetch_json,
                acquired_utc=acquired_utc,
            )
            point = dict(sample_cache["target_point"])
            point["name"] = f"moving_{start}_{end}_epoch_{sample['epoch']}"
            point["epoch"] = sample["epoch"]
            target_points.append(point)
        result["target_points"] = target_points
        result["target_sampling"] = {
            "method": "fixed_evenly_spaced_trajectory_epochs",
            "requested_count": sample_count,
            "sampled_epochs": [point["epoch"] for point in target_points],
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("--calibration-cache", type=Path, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--sample-count", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    calibration_bytes = args.calibration_cache.read_bytes()
    calibration = json.loads(calibration_bytes.decode("utf-8"))
    result = acquire_moving_cache(
        args.trajectory,
        calibration,
        start=args.start,
        end=args.end,
        sample_count=args.sample_count,
        fetch_json=_fetch_validated_gsi_json,
    )
    result["calibration_cache"] = str(args.calibration_cache).replace("\\", "/")
    result["calibration_cache_sha256"] = hashlib.sha256(calibration_bytes).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
