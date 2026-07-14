"""Ambiguity-eliminated window carrier-phase (WCP) factors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WCPProjection:
    residual: np.ndarray
    jacobian: np.ndarray
    left_nullspace: np.ndarray
    rank_ambiguity: int


def left_nullspace_project(
    residual: np.ndarray,
    jacobian: np.ndarray,
    ambiguity_design: np.ndarray,
    covariance: np.ndarray,
    *,
    rcond: float = 1.0e-10,
) -> WCPProjection:
    """Whiten and project a carrier window into the left nullspace of A.

    The input model is ``residual ~= J dx + A dN + noise``.  The returned
    factor contains no ambiguity variable and preserves the whitened
    least-squares information orthogonal to ``A``.
    """

    r = np.asarray(residual, dtype=np.float64).reshape(-1)
    j = np.asarray(jacobian, dtype=np.float64)
    a = np.asarray(ambiguity_design, dtype=np.float64)
    cov = np.asarray(covariance, dtype=np.float64)
    m = r.size
    if j.ndim != 2 or j.shape[0] != m:
        raise ValueError("jacobian rows must match residual")
    if a.ndim != 2 or a.shape[0] != m:
        raise ValueError("ambiguity_design rows must match residual")
    if cov.shape != (m, m):
        raise ValueError("covariance must be square and match residual")
    if not (np.isfinite(r).all() and np.isfinite(j).all() and np.isfinite(a).all() and np.isfinite(cov).all()):
        raise ValueError("WCP inputs must be finite")
    chol = np.linalg.cholesky(0.5 * (cov + cov.T))
    rw = np.linalg.solve(chol, r)
    jw = np.linalg.solve(chol, j)
    aw = np.linalg.solve(chol, a)
    u, singular, _vh = np.linalg.svd(aw, full_matrices=True)
    scale = float(singular[0]) if singular.size else 0.0
    rank = int(np.sum(singular > max(float(rcond) * scale, float(rcond))))
    null = u[:, rank:]
    return WCPProjection(
        residual=null.T @ rw,
        jacobian=null.T @ jw,
        left_nullspace=null,
        rank_ambiguity=rank,
    )


def single_arc_wcp(
    residual_cycles: np.ndarray,
    jacobian_cycles: np.ndarray,
    sigma_cycles: float | np.ndarray,
) -> WCPProjection:
    """Project one continuous slip-free ambiguity arc spanning >=2 epochs."""

    residual = np.asarray(residual_cycles, dtype=np.float64).reshape(-1)
    if residual.size < 2:
        raise ValueError("a WCP arc needs at least two carrier observations")
    sigma = np.asarray(sigma_cycles, dtype=np.float64)
    if sigma.ndim == 0:
        sigma = np.full(residual.size, float(sigma), dtype=np.float64)
    sigma = sigma.reshape(-1)
    if sigma.size != residual.size or np.any(sigma <= 0.0):
        raise ValueError("sigma_cycles must be positive and match the arc")
    return left_nullspace_project(
        residual,
        jacobian_cycles,
        np.ones((residual.size, 1), dtype=np.float64),
        np.diag(sigma * sigma),
    )
