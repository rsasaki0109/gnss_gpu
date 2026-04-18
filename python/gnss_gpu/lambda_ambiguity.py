"""Integer ambiguity resolution for carrier-phase GNSS.

The public API mirrors the LAMBDA workflow:

1. validate/decorrelate the float ambiguities,
2. solve the integer least-squares problem,
3. accept the best integer vector only when the ratio test passes.

The search routine solves the ILS objective directly from the covariance matrix
using a Schnorr-Euchner style depth-first enumeration over the Cholesky factor
of the information matrix.  This keeps the implementation small and avoids
external dependencies while preserving the strict ratio-test semantics needed
to avoid wrong fixes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LambdaSolution:
    """Best integer candidates and their squared normalized residuals."""

    candidates: np.ndarray
    residuals: np.ndarray

    @property
    def ratio(self) -> float:
        if self.residuals.size < 2 or not np.isfinite(self.residuals[:2]).all():
            return 0.0
        if self.residuals[0] <= 0.0:
            return float("inf") if self.residuals[1] > 0.0 else 0.0
        return float(self.residuals[1] / self.residuals[0])


def decorrelate_ambiguities(
    float_amb: np.ndarray,
    cov: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate and return ambiguities in the search domain.

    The current implementation leaves the integer basis unchanged.  The
    subsequent search uses the full covariance, so the returned values still
    produce the exact ILS optimum for the supplied covariance.  ``Z`` is the
    unimodular integer transform from the original ambiguity vector to the
    returned search vector; here it is identity.
    """

    amb, q = _validate_ambiguity_problem(float_amb, cov)
    z = np.eye(amb.size, dtype=np.int64)
    return amb.copy(), z, q.copy()


def integer_search(
    decorrelated_amb: np.ndarray,
    tz_cov: np.ndarray,
    n_candidates: int = 2,
    *,
    max_expansions: int = 16,
    max_nodes: int = 250_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the best integer candidates for the ILS objective.

    Parameters
    ----------
    decorrelated_amb
        Float ambiguity vector.
    tz_cov
        Positive-definite covariance matrix for ``decorrelated_amb``.
    n_candidates
        Number of candidates to return.  Ratio validation requires at least two.
    max_expansions
        Number of times the initial search radius may be expanded if too few
        candidates are found.
    max_nodes
        Hard cap on recursive nodes per radius to avoid pathological runtimes.
    """

    amb, q = _validate_ambiguity_problem(decorrelated_amb, tz_cov)
    want = max(1, int(n_candidates))
    if amb.size == 0:
        return np.empty((0, 0), dtype=np.int64), np.empty(0, dtype=np.float64)

    information = np.linalg.inv(q)
    try:
        upper = np.linalg.cholesky(information).T
    except np.linalg.LinAlgError:
        jitter = max(1e-12, 1e-10 * float(np.trace(q)) / max(1, q.shape[0]))
        information = np.linalg.inv(q + np.eye(q.shape[0], dtype=np.float64) * jitter)
        upper = np.linalg.cholesky(information).T

    rounded = np.rint(amb).astype(np.int64)
    radius = max(_objective(rounded, amb, information), 1.0)
    best: list[tuple[float, np.ndarray]] = []
    for _ in range(max(1, int(max_expansions))):
        best = _enumerate_with_radius(
            amb,
            upper,
            radius=float(radius),
            want=want,
            max_nodes=int(max_nodes),
        )
        if len(best) >= want:
            break
        radius *= 4.0

    if len(best) < want:
        fallback = _axis_fallback_candidates(amb, information, want)
        seen = {tuple(cand.tolist()) for _, cand in best}
        for dist, cand in fallback:
            key = tuple(cand.tolist())
            if key in seen:
                continue
            best.append((dist, cand))
            seen.add(key)
            if len(best) >= want:
                break

    best.sort(key=lambda item: item[0])
    best = best[:want]
    candidates = np.vstack([cand for _, cand in best]).astype(np.int64, copy=False)
    residuals = np.asarray([dist for dist, _ in best], dtype=np.float64)
    return candidates, residuals


def ratio_test(
    candidates: np.ndarray,
    residuals: np.ndarray,
    threshold: float = 3.0,
) -> tuple[np.ndarray | None, bool]:
    """Validate an integer fix with the standard ambiguity ratio test."""

    cand = np.asarray(candidates, dtype=np.int64)
    res = np.asarray(residuals, dtype=np.float64).ravel()
    if cand.ndim != 2 or cand.shape[0] < 2 or res.size < 2:
        return None, False
    if not np.isfinite(res[:2]).all() or res[0] < 0.0 or res[1] < 0.0:
        return None, False
    if res[0] <= 0.0:
        ratio = float("inf") if res[1] > 0.0 else 0.0
    else:
        ratio = float(res[1] / res[0])
    if ratio < float(threshold):
        return None, False
    return cand[0].copy(), True


def solve_lambda(
    float_amb: np.ndarray,
    cov: np.ndarray,
    *,
    ratio_threshold: float = 3.0,
    n_candidates: int = 2,
) -> tuple[np.ndarray | None, bool, LambdaSolution]:
    """Convenience wrapper for decorrelation, search, and ratio validation."""

    z_float, _z, z_cov = decorrelate_ambiguities(float_amb, cov)
    candidates, residuals = integer_search(z_float, z_cov, n_candidates=n_candidates)
    fixed, ok = ratio_test(candidates, residuals, threshold=ratio_threshold)
    return fixed, ok, LambdaSolution(candidates=candidates, residuals=residuals)


def _validate_ambiguity_problem(
    float_amb: np.ndarray,
    cov: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    amb = np.asarray(float_amb, dtype=np.float64).ravel()
    q = np.asarray(cov, dtype=np.float64)
    if q.ndim != 2 or q.shape[0] != q.shape[1] or q.shape[0] != amb.size:
        raise ValueError("covariance must be square and match ambiguity length")
    if not np.isfinite(amb).all() or not np.isfinite(q).all():
        raise ValueError("ambiguities and covariance must be finite")
    q = 0.5 * (q + q.T)
    try:
        np.linalg.cholesky(q)
    except np.linalg.LinAlgError as exc:
        raise ValueError("covariance must be positive definite") from exc
    return amb, q


def _objective(z: np.ndarray, amb: np.ndarray, information: np.ndarray) -> float:
    dz = np.asarray(z, dtype=np.float64).ravel() - amb
    return float(dz @ information @ dz)


def _enumerate_with_radius(
    amb: np.ndarray,
    upper: np.ndarray,
    *,
    radius: float,
    want: int,
    max_nodes: int,
) -> list[tuple[float, np.ndarray]]:
    n = amb.size
    z = np.zeros(n, dtype=np.int64)
    dz = np.zeros(n, dtype=np.float64)
    best: list[tuple[float, np.ndarray]] = []
    nodes = 0

    def add_candidate(dist: float) -> None:
        cand = z.copy()
        best.append((float(dist), cand))
        best.sort(key=lambda item: item[0])
        del best[max(want, 1) :]

    def recurse(k: int, dist_future: float) -> None:
        nonlocal nodes
        if nodes >= max_nodes:
            return
        nodes += 1
        if k < 0:
            add_candidate(dist_future)
            return
        if len(best) >= want and dist_future >= best[-1][0]:
            return
        remaining = radius - dist_future
        if len(best) >= want:
            remaining = min(remaining, best[-1][0] - dist_future)
        if remaining < -1e-12:
            return
        diag = float(upper[k, k])
        if not np.isfinite(diag) or abs(diag) <= 0.0:
            return
        tail = 0.0
        if k + 1 < n:
            tail = float(upper[k, k + 1 :] @ dz[k + 1 :])
        center = float(amb[k] - tail / diag)
        half_width = float(np.sqrt(max(0.0, remaining)) / abs(diag))
        lo = int(np.ceil(center - half_width - 1e-12))
        hi = int(np.floor(center + half_width + 1e-12))
        if hi < lo:
            return
        values = list(range(lo, hi + 1))
        values.sort(key=lambda value: (abs(float(value) - center), value))
        for value in values:
            z[k] = int(value)
            dz[k] = float(value) - float(amb[k])
            row_residual = diag * dz[k] + tail
            dist = dist_future + row_residual * row_residual
            if dist <= radius + 1e-10:
                recurse(k - 1, dist)
            if nodes >= max_nodes:
                break

    recurse(n - 1, 0.0)
    best.sort(key=lambda item: item[0])
    return best


def _axis_fallback_candidates(
    amb: np.ndarray,
    information: np.ndarray,
    want: int,
) -> list[tuple[float, np.ndarray]]:
    base = np.rint(amb).astype(np.int64)
    candidates = [base]
    for axis in range(amb.size):
        for step in (-1, 1):
            cand = base.copy()
            cand[axis] += step
            candidates.append(cand)
    scored = [(_objective(cand, amb, information), cand) for cand in candidates]
    scored.sort(key=lambda item: item[0])
    return scored[: max(int(want), 1)]
