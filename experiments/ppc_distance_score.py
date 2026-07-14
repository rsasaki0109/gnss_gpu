"""Official PPC distance-weighted threshold score helpers."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def reference_step_distances(
    reference_path: Path,
    *,
    start_epoch: int = 0,
    end_epoch: int | None = None,
) -> np.ndarray:
    """Return reference path increments for one contiguous evaluation slice."""

    positions: list[list[float]] = []
    with Path(reference_path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        next(reader)
        for row in reader:
            positions.append([float(row[5]), float(row[6]), float(row[7])])
    stop = len(positions) if end_epoch is None else min(int(end_epoch), len(positions))
    start = max(0, int(start_epoch))
    selected = np.asarray(positions[start:stop], dtype=np.float64)
    distances = np.zeros(selected.shape[0], dtype=np.float64)
    if selected.shape[0] > 1:
        distances[1:] = np.linalg.norm(np.diff(selected, axis=0), axis=1)
    return distances


def honest_ppc_distance_score(
    errors_by_epoch: dict[int, float],
    reference_path: Path,
    *,
    start_epoch: int = 0,
    end_epoch: int | None = None,
    threshold_m: float = 0.5,
) -> dict[str, float]:
    """Score emitted errors; missing/non-finite epochs receive no pass distance."""

    distances = reference_step_distances(
        reference_path, start_epoch=start_epoch, end_epoch=end_epoch
    )
    passed = 0.0
    threshold = float(threshold_m)
    for local_index, distance in enumerate(distances):
        error = float(errors_by_epoch.get(int(start_epoch) + local_index, float("inf")))
        if np.isfinite(error) and error <= threshold:
            passed += float(distance)
    total = float(np.sum(distances))
    return {
        "honest_ppc_score_pct": 100.0 * passed / total if total > 0.0 else 0.0,
        "pass_distance_m": passed,
        "total_distance_m": total,
    }
