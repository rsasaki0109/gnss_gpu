"""Closed-form switchable constraints for robust GNSS factors."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class ReducedSwitchFactor:
    residual: np.ndarray
    jacobian: np.ndarray
    switches: np.ndarray


def optimal_switch(normalized_residual: np.ndarray, prior_strength: float = 1.0) -> np.ndarray:
    """Return argmin_s ``(s*r)^2 + prior_strength*(1-s)^2``."""

    r = np.asarray(normalized_residual, dtype=np.float64)
    strength = float(prior_strength)
    if not math.isfinite(strength) or strength <= 0.0:
        raise ValueError("prior_strength must be positive")
    return strength / (strength + r * r)


def reduce_switchable_factor(
    normalized_residual: np.ndarray,
    normalized_jacobian: np.ndarray,
    prior_strength: float = 1.0,
) -> ReducedSwitchFactor:
    """Analytically eliminate independent switch variables from a factor.

    The signed equivalent residual has exactly the minimized switch-factor
    cost, while its Jacobian is the exact derivative of that reduced cost.
    """

    r = np.asarray(normalized_residual, dtype=np.float64).reshape(-1)
    j = np.asarray(normalized_jacobian, dtype=np.float64)
    if j.ndim != 2 or j.shape[0] != r.size:
        raise ValueError("jacobian rows must match residual")
    strength = float(prior_strength)
    switches = optimal_switch(r, strength)
    denom = strength + r * r
    reduced_r = math.sqrt(strength) * r / np.sqrt(denom)
    derivative = (strength ** 1.5) / np.power(denom, 1.5)
    return ReducedSwitchFactor(
        residual=reduced_r,
        jacobian=derivative[:, None] * j,
        switches=switches,
    )
