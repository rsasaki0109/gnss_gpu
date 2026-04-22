"""Generic motion-based position-update decisions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MotionPositionUpdateDecision:
    apply_update: bool
    predicted_position: np.ndarray | None
    reason: str


def evaluate_motion_position_update(
    prev_estimate: np.ndarray | None,
    velocity: np.ndarray | None,
    dt: float,
) -> MotionPositionUpdateDecision:
    if prev_estimate is None:
        return _skip("no_previous_estimate")
    if velocity is None:
        return _skip("no_velocity")
    if dt <= 0:
        return _skip("invalid_dt")

    prev = np.asarray(prev_estimate, dtype=np.float64).ravel()[:3]
    vel = np.asarray(velocity, dtype=np.float64).ravel()[:3]
    if prev.shape[0] != 3 or not np.isfinite(prev).all():
        return _skip("invalid_previous_estimate")
    if vel.shape[0] != 3 or not np.isfinite(vel).all():
        return _skip("invalid_velocity")

    predicted = prev + vel * float(dt)
    if not np.isfinite(predicted).all():
        return _skip("invalid_predicted_position")
    return MotionPositionUpdateDecision(
        apply_update=True,
        predicted_position=predicted,
        reason="ok",
    )


def _skip(reason: str) -> MotionPositionUpdateDecision:
    return MotionPositionUpdateDecision(
        apply_update=False,
        predicted_position=None,
        reason=reason,
    )
