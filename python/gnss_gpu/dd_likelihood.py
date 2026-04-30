"""DD-pseudorange likelihood helpers for particle transport experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DDLikelihoodGradient:
    """DD-pseudorange log-likelihood gradient at one position."""

    residuals_m: np.ndarray
    design: np.ndarray
    gradient: np.ndarray
    robust_weights: np.ndarray
    robust_rms_m: float
    n_dd: int


def dd_pseudorange_residual_and_design(dd_result, position_ecef: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return DD residuals and geometry design matrix for one receiver position.

    Residual convention is ``observed_dd - expected_dd(position)``.  The design
    matrix is ``d expected_dd / d position`` for each DD pair.
    """
    pos = np.asarray(position_ecef, dtype=np.float64).reshape(3)
    sat_k = np.asarray(dd_result.sat_ecef_k, dtype=np.float64).reshape(-1, 3)
    sat_ref = np.asarray(dd_result.sat_ecef_ref, dtype=np.float64).reshape(-1, 3)
    if sat_k.shape != sat_ref.shape:
        raise ValueError("sat_ecef_k and sat_ecef_ref must have matching shapes")

    range_k = np.linalg.norm(sat_k - pos, axis=1)
    range_ref = np.linalg.norm(sat_ref - pos, axis=1)
    expected = (
        range_k
        - range_ref
        - np.asarray(dd_result.base_range_k, dtype=np.float64).reshape(-1)
        + np.asarray(dd_result.base_range_ref, dtype=np.float64).reshape(-1)
    )
    residuals = np.asarray(dd_result.dd_pseudorange_m, dtype=np.float64).reshape(-1) - expected
    unit_k = (sat_k - pos) / np.maximum(range_k[:, np.newaxis], 1.0)
    unit_ref = (sat_ref - pos) / np.maximum(range_ref[:, np.newaxis], 1.0)
    design = -unit_k + unit_ref
    if residuals.shape[0] != design.shape[0]:
        raise ValueError("DD arrays must have one row per DD pair")
    return residuals, design


def dd_log_likelihood_gradient(
    dd_result,
    position_ecef: np.ndarray,
    *,
    sigma_m: float = 1.0,
    huber_k_m: float | None = None,
) -> DDLikelihoodGradient:
    """Return the gradient of Gaussian DD log likelihood with optional Huber weights."""
    if sigma_m <= 0.0:
        raise ValueError("sigma_m must be positive")

    residuals, design = dd_pseudorange_residual_and_design(dd_result, position_ecef)
    abs_res = np.abs(residuals)
    if huber_k_m is not None and huber_k_m > 0.0:
        robust = np.minimum(1.0, float(huber_k_m) / np.maximum(abs_res, 1.0e-12))
    else:
        robust = np.ones_like(residuals)
    dd_weights = np.asarray(getattr(dd_result, "dd_weights", np.ones_like(residuals)), dtype=np.float64).reshape(-1)
    if dd_weights.shape[0] != residuals.shape[0]:
        raise ValueError("dd_weights must have one value per DD pair")
    weights = np.clip(dd_weights, 1.0e-12, None) * robust
    gradient = np.sum((weights * residuals)[:, np.newaxis] * design, axis=0) / (float(sigma_m) ** 2)
    robust_rms = float(np.sqrt(np.mean(np.square(residuals) * weights))) if residuals.size else float("nan")
    return DDLikelihoodGradient(
        residuals_m=residuals,
        design=design,
        gradient=gradient,
        robust_weights=weights,
        robust_rms_m=robust_rms,
        n_dd=int(residuals.size),
    )


def dd_log_likelihood_gradients(
    dd_result,
    particles_ecef: np.ndarray,
    *,
    sigma_m: float = 1.0,
    huber_k_m: float | None = None,
) -> np.ndarray:
    """Vectorized convenience wrapper for per-particle DD likelihood gradients."""
    particles = np.asarray(particles_ecef, dtype=np.float64)
    if particles.ndim != 2 or particles.shape[1] < 3:
        raise ValueError("particles_ecef must have shape (N, 3+)")
    out = np.zeros((particles.shape[0], 3), dtype=np.float64)
    for i, particle in enumerate(particles):
        out[i] = dd_log_likelihood_gradient(
            dd_result,
            particle[:3],
            sigma_m=sigma_m,
            huber_k_m=huber_k_m,
        ).gradient
    return out
