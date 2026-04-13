"""Adaptive per-satellite weighting based on pseudorange residuals.

Implements a vote-based scheme inspired by Gupta & Gao (2021):
each satellite's weight is downscaled based on how consistent its
pseudorange is with the current position estimate across the particle
population (or a reference position).

This provides soft NLOS rejection without hard thresholds.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2


def compute_adaptive_weights(
    sat_ecef: np.ndarray,
    pseudoranges: np.ndarray,
    reference_position: np.ndarray,
    reference_clock_bias: float = 0.0,
    sigma_pr: float = 10.0,
    base_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Compute per-satellite adaptive weights using residual voting.

    Parameters
    ----------
    sat_ecef : (K, 3)
        Satellite ECEF positions.
    pseudoranges : (K,)
        Observed pseudoranges.
    reference_position : (3,) or (4,)
        Reference position (and optional clock bias) for residual computation.
        Typically from EKF or PF estimate.
    reference_clock_bias : float
        Receiver clock bias [m]. Ignored if reference_position has 4 elements.
    sigma_pr : float
        Expected pseudorange noise standard deviation [m].
    base_weights : (K,), optional
        Original per-satellite weights (e.g., from C/N0). Multiplied with
        adaptive weights. Defaults to ones.

    Returns
    -------
    weights : (K,)
        Adaptive per-satellite weights. Consistent satellites get weight ~1,
        NLOS/faulty satellites get downweighted toward 0.
    """
    sat = np.asarray(sat_ecef, dtype=np.float64).reshape(-1, 3)
    pr = np.asarray(pseudoranges, dtype=np.float64).ravel()
    ref = np.asarray(reference_position, dtype=np.float64).ravel()

    if ref.size >= 4:
        pos = ref[:3]
        cb = ref[3]
    else:
        pos = ref[:3]
        cb = reference_clock_bias

    # Compute predicted pseudoranges from reference position
    dx = sat - pos[np.newaxis, :]
    ranges = np.sqrt(np.sum(dx ** 2, axis=1))
    predicted_pr = ranges + cb

    # Normalized residuals
    residuals = pr - predicted_pr
    normalized_r2 = (residuals / sigma_pr) ** 2

    # Vote: probability that a chi-squared(1) random variable is <= r^2
    # High vote = consistent (small residual), low vote = NLOS (large residual)
    # We want weight = 1 - CDF, so consistent sats get high weight
    votes = 1.0 - chi2.cdf(normalized_r2, df=1)

    # Clamp minimum weight to avoid complete rejection
    votes = np.clip(votes, 0.01, 1.0)

    if base_weights is not None:
        votes *= np.asarray(base_weights, dtype=np.float64).ravel()

    return votes
