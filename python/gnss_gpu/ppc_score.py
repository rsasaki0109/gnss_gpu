"""PPC2024 scoring helpers.

The PPC2024 challenge ranks submissions by the traveled-distance ratio whose
3D positioning error is within 0.5 m.  This module keeps that metric separate
from the repository's generic RMS/P95 metrics so optimization targets the
competition score directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PPCScore:
    """Distance-weighted PPC score summary."""

    score_pct: float
    pass_distance_m: float
    total_distance_m: float
    epoch_pass_pct: float
    threshold_m: float
    n_epochs: int
    fallback_epoch_weighted: bool
    errors_3d: np.ndarray
    segment_distances_m: np.ndarray
    pass_mask: np.ndarray


def _as_positions(name: str, value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"{name} must have shape (N, 3+) ECEF positions")
    return arr[:, :3]


def ppc_3d_errors(estimated_ecef: np.ndarray, reference_ecef: np.ndarray) -> np.ndarray:
    """Return per-epoch ECEF 3D errors in meters."""
    est = _as_positions("estimated_ecef", estimated_ecef)
    ref = _as_positions("reference_ecef", reference_ecef)
    if est.shape[0] != ref.shape[0]:
        raise ValueError("estimated_ecef and reference_ecef must have the same length")
    return np.linalg.norm(est - ref, axis=1)


def ppc_segment_distances(reference_ecef: np.ndarray) -> np.ndarray:
    """Return per-epoch traveled-distance weights in meters.

    Weight at epoch ``i`` is the reference trajectory distance from ``i - 1`` to
    ``i``.  The first epoch has no preceding traveled segment and receives zero
    weight.  This makes the weighted pass ratio a distance ratio instead of an
    epoch count ratio when the official speed-derived weights are unavailable.
    """
    ref = _as_positions("reference_ecef", reference_ecef)
    weights = np.zeros(ref.shape[0], dtype=np.float64)
    if ref.shape[0] > 1:
        weights[1:] = np.linalg.norm(np.diff(ref, axis=0), axis=1)
    return weights


def _as_segment_distances(name: str, value: np.ndarray, length: int) -> np.ndarray:
    distances = np.asarray(value, dtype=np.float64).reshape(-1)
    if distances.shape[0] != length:
        raise ValueError(f"{name} must have length {length}")
    finite = np.isfinite(distances)
    if np.any(distances[finite] < 0.0):
        raise ValueError(f"{name} must contain non-negative distances")
    return distances


def score_ppc2024(
    estimated_ecef: np.ndarray,
    reference_ecef: np.ndarray,
    threshold_m: float = 0.5,
    segment_distances_m: np.ndarray | None = None,
) -> PPCScore:
    """Compute the PPC2024 distance-weighted 3D pass score.

    The official challenge metric is the traveled-distance percentage for which
    3D error is at most 50 cm.  Pass ``segment_distances_m`` to use official
    speed-derived per-epoch distance weights; otherwise adjacent reference ECEF
    displacements are used.  When a fixture or smoke segment has no traveled
    distance, this function falls back to epoch-uniform weights so the result is
    still usable in tests and static scenarios.
    """
    if threshold_m <= 0.0:
        raise ValueError("threshold_m must be positive")

    errors = ppc_3d_errors(estimated_ecef, reference_ecef)
    distances = (
        ppc_segment_distances(reference_ecef)
        if segment_distances_m is None
        else _as_segment_distances("segment_distances_m", segment_distances_m, errors.size)
    )
    finite = np.isfinite(errors) & np.isfinite(distances)
    pass_mask = finite & (errors <= float(threshold_m))

    total_distance = float(np.sum(distances[finite]))
    fallback_epoch_weighted = total_distance <= 0.0
    if fallback_epoch_weighted:
        weights = finite.astype(np.float64)
        total_weight = float(np.sum(weights))
    else:
        weights = np.where(finite, distances, 0.0)
        total_weight = total_distance

    pass_distance = float(np.sum(weights[pass_mask]))
    score_pct = 100.0 * pass_distance / total_weight if total_weight > 0.0 else 0.0
    epoch_total = int(np.sum(np.isfinite(errors)))
    epoch_pass_pct = 100.0 * float(np.sum(pass_mask)) / epoch_total if epoch_total else 0.0

    return PPCScore(
        score_pct=float(score_pct),
        pass_distance_m=pass_distance,
        total_distance_m=total_weight,
        epoch_pass_pct=float(epoch_pass_pct),
        threshold_m=float(threshold_m),
        n_epochs=int(errors.size),
        fallback_epoch_weighted=bool(fallback_epoch_weighted),
        errors_3d=errors,
        segment_distances_m=distances,
        pass_mask=pass_mask,
    )


def ppc_score_dict(
    estimated_ecef: np.ndarray,
    reference_ecef: np.ndarray,
    threshold_m: float = 0.5,
    segment_distances_m: np.ndarray | None = None,
) -> dict[str, float | int | bool]:
    """Return scalar PPC score fields suitable for CSV rows."""
    score = score_ppc2024(
        estimated_ecef,
        reference_ecef,
        threshold_m=threshold_m,
        segment_distances_m=segment_distances_m,
    )
    return {
        "ppc_score_pct": score.score_pct,
        "ppc_pass_distance_m": score.pass_distance_m,
        "ppc_total_distance_m": score.total_distance_m,
        "ppc_epoch_pass_pct": score.epoch_pass_pct,
        "ppc_threshold_m": score.threshold_m,
        "ppc_n_epochs": score.n_epochs,
        "ppc_fallback_epoch_weighted": score.fallback_epoch_weighted,
    }
