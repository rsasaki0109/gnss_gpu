"""IMU tight-coupling position-update decisions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ImuTightPositionUpdateDecision:
    apply_update: bool
    predicted_position: np.ndarray | None
    sigma_pos: float | None
    residual_rms: float | None
    reason: str


def evaluate_imu_tight_position_update(
    prev_pf_estimate: np.ndarray | None,
    imu_velocity: np.ndarray | None,
    dt: float,
    sat_ecef: np.ndarray,
    pseudoranges: np.ndarray,
    spp_position: np.ndarray,
    *,
    n_measurements: int,
) -> ImuTightPositionUpdateDecision:
    if prev_pf_estimate is None:
        return _skip("no_previous_pf_estimate")
    if imu_velocity is None:
        return _skip("no_imu_velocity")
    if dt <= 0:
        return _skip("invalid_dt")

    velocity = np.asarray(imu_velocity, dtype=np.float64).ravel()[:3]
    if velocity.shape[0] != 3 or not np.isfinite(velocity).all():
        return _skip("invalid_imu_velocity")

    prev_position = np.asarray(prev_pf_estimate, dtype=np.float64).ravel()[:3]
    predicted_position = prev_position + velocity * float(dt)
    residual_rms = _spp_residual_rms(sat_ecef, pseudoranges, spp_position)
    sigma_pos = _imu_tight_sigma(int(n_measurements), residual_rms)

    if predicted_position.shape[0] != 3 or not np.isfinite(predicted_position).all():
        return ImuTightPositionUpdateDecision(
            apply_update=False,
            predicted_position=None,
            sigma_pos=sigma_pos,
            residual_rms=residual_rms,
            reason="invalid_predicted_position",
        )

    return ImuTightPositionUpdateDecision(
        apply_update=True,
        predicted_position=predicted_position,
        sigma_pos=sigma_pos,
        residual_rms=residual_rms,
        reason="ok",
    )


def _skip(reason: str) -> ImuTightPositionUpdateDecision:
    return ImuTightPositionUpdateDecision(
        apply_update=False,
        predicted_position=None,
        sigma_pos=None,
        residual_rms=None,
        reason=reason,
    )


def _spp_residual_rms(
    sat_ecef: np.ndarray,
    pseudoranges: np.ndarray,
    spp_position: np.ndarray,
) -> float:
    sat = np.asarray(sat_ecef, dtype=np.float64).reshape(-1, 3)
    pr = np.asarray(pseudoranges, dtype=np.float64).ravel()
    spp = np.asarray(spp_position, dtype=np.float64).ravel()[:3]
    if sat.shape[0] == 0 or pr.shape[0] == 0:
        return float("inf")
    n = min(sat.shape[0], pr.shape[0])
    ranges = np.linalg.norm(sat[:n] - spp, axis=1)
    valid_mask = np.isfinite(ranges) & np.isfinite(pr[:n])
    if not np.any(valid_mask):
        return float("inf")
    cb_est = float(np.median((pr[:n] - ranges)[valid_mask]))
    residuals = np.abs((pr[:n] - ranges - cb_est)[valid_mask])
    return float(np.sqrt(np.mean(residuals**2)))


def _imu_tight_sigma(n_sats: int, residual_rms: float) -> float:
    if int(n_sats) < 6 or residual_rms > 20.0:
        return 3.0
    if int(n_sats) < 8 or residual_rms > 10.0:
        return 8.0
    return 30.0
