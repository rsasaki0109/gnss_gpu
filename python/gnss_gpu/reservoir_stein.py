"""Reservoir + Stein particle utilities for realtime PF experiments.

This module is intentionally CPU/numpy-only.  It keeps the algorithmic unit small
enough to test before moving the same selection and transport steps into GPU PF
experiments.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ReservoirSteinConfig:
    """Configuration for a bounded reservoir Stein update."""

    reservoir_size: int
    elite_fraction: float = 0.25
    stein_steps: int = 1
    stein_step_size: float = 0.1
    bandwidth: float | None = None
    repulsion_scale: float = 1.0
    seed: int = 0


@dataclass(frozen=True)
class ReservoirSteinResult:
    """Result of selecting and transporting a particle reservoir."""

    particles: np.ndarray
    source_indices: np.ndarray
    weights: np.ndarray
    ess_before: float
    bandwidths: tuple[float, ...]


def normalize_log_weights(log_weights: np.ndarray) -> np.ndarray:
    """Return stable normalized weights from unnormalized log weights."""
    logw = np.asarray(log_weights, dtype=np.float64).reshape(-1)
    if logw.size == 0:
        raise ValueError("log_weights must not be empty")
    finite = np.isfinite(logw)
    if not np.any(finite):
        return np.full(logw.shape, 1.0 / logw.size, dtype=np.float64)

    shifted = np.where(finite, logw - np.max(logw[finite]), -np.inf)
    weights = np.exp(shifted)
    total = float(np.sum(weights))
    if total <= 0.0 or not np.isfinite(total):
        return np.full(logw.shape, 1.0 / logw.size, dtype=np.float64)
    return weights / total


def effective_sample_size(weights: np.ndarray) -> float:
    """Compute ESS from normalized or unnormalized non-negative weights."""
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if w.size == 0:
        raise ValueError("weights must not be empty")
    w = np.where(np.isfinite(w) & (w > 0.0), w, 0.0)
    total = float(np.sum(w))
    if total <= 0.0:
        return 0.0
    normalized = w / total
    return float(1.0 / np.sum(normalized * normalized))


def _elite_count(reservoir_size: int, elite_fraction: float) -> int:
    if reservoir_size <= 0:
        raise ValueError("reservoir_size must be positive")
    fraction = float(np.clip(elite_fraction, 0.0, 1.0))
    if fraction <= 0.0:
        return 0
    return min(reservoir_size, max(1, int(round(reservoir_size * fraction))))


def weighted_reservoir_indices(
    log_weights: np.ndarray,
    reservoir_size: int,
    *,
    elite_fraction: float = 0.25,
    seed: int | None = None,
) -> np.ndarray:
    """Select unique particle indices for a bounded weighted reservoir.

    The heaviest particles are pinned as elites.  The remaining slots use
    Efraimidis-Spirakis weighted reservoir sampling without replacement.
    """
    weights = normalize_log_weights(log_weights)
    n_particles = weights.size
    if reservoir_size <= 0:
        raise ValueError("reservoir_size must be positive")
    if reservoir_size >= n_particles:
        return np.arange(n_particles, dtype=np.int64)

    n_elite = _elite_count(reservoir_size, elite_fraction)
    elite = (
        np.argsort(weights, kind="stable")[-n_elite:][::-1]
        if n_elite > 0
        else np.empty(0, dtype=np.int64)
    )
    remaining_slots = reservoir_size - elite.size
    if remaining_slots <= 0:
        return elite.astype(np.int64)

    mask = np.ones(n_particles, dtype=bool)
    mask[elite] = False
    candidate_idx = np.nonzero(mask & (weights > 0.0))[0]
    if candidate_idx.size <= remaining_slots:
        filler = np.nonzero(mask)[0]
        chosen = filler[:remaining_slots]
        return np.concatenate([elite, chosen]).astype(np.int64)

    rng = np.random.default_rng(seed)
    u = np.clip(rng.random(candidate_idx.size), np.finfo(np.float64).tiny, 1.0)
    keys = np.log(u) / weights[candidate_idx]
    selected = candidate_idx[np.argsort(keys, kind="stable")[-remaining_slots:][::-1]]
    return np.concatenate([elite, selected]).astype(np.int64)


def rbf_median_bandwidth(
    particles: np.ndarray,
    *,
    min_bandwidth: float = 1.0e-9,
) -> float:
    """Estimate an RBF bandwidth using the SVGD median heuristic."""
    x = np.asarray(particles, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] == 0:
        raise ValueError("particles must have shape (N, D) with N > 0")
    if x.shape[0] == 1:
        return float(max(min_bandwidth, 1.0))

    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    sqdist = np.sum(diff * diff, axis=2)
    upper = sqdist[np.triu_indices(x.shape[0], k=1)]
    positive = upper[upper > 0.0]
    if positive.size == 0:
        return float(max(min_bandwidth, 1.0))

    bandwidth = float(np.median(positive) / np.log(x.shape[0] + 1.0))
    return float(max(min_bandwidth, bandwidth))


def stein_rejuvenate_particles(
    particles: np.ndarray,
    score_gradients: np.ndarray,
    *,
    step_size: float,
    bandwidth: float | None = None,
    repulsion_scale: float = 1.0,
) -> tuple[np.ndarray, float]:
    """Apply one SVGD-style transport step to a particle cloud."""
    x = np.asarray(particles, dtype=np.float64)
    grad = np.asarray(score_gradients, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("particles must have shape (N, D)")
    if grad.shape != x.shape:
        raise ValueError("score_gradients must have the same shape as particles")
    if x.shape[0] == 0:
        raise ValueError("particles must not be empty")

    h = rbf_median_bandwidth(x) if bandwidth is None else float(bandwidth)
    h = max(h, 1.0e-12)
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    sqdist = np.sum(diff * diff, axis=2)
    kernel = np.exp(-sqdist / h)
    n_particles = float(x.shape[0])

    attraction = kernel @ grad / n_particles
    repulsion = (2.0 / h) * np.sum(kernel[:, :, np.newaxis] * diff, axis=1) / n_particles
    velocity = attraction + float(repulsion_scale) * repulsion
    return x + float(step_size) * velocity, h


def reservoir_stein_update(
    particles: np.ndarray,
    log_weights: np.ndarray,
    score_gradients: np.ndarray,
    config: ReservoirSteinConfig,
) -> ReservoirSteinResult:
    """Select a bounded reservoir and run Stein rejuvenation on it."""
    x = np.asarray(particles, dtype=np.float64)
    grad = np.asarray(score_gradients, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("particles must have shape (N, D)")
    if grad.shape != x.shape:
        raise ValueError("score_gradients must have the same shape as particles")
    if np.asarray(log_weights).reshape(-1).size != x.shape[0]:
        raise ValueError("log_weights must have one entry per particle")

    weights = normalize_log_weights(log_weights)
    source_indices = weighted_reservoir_indices(
        log_weights,
        config.reservoir_size,
        elite_fraction=config.elite_fraction,
        seed=config.seed,
    )
    reservoir = x[source_indices].copy()
    reservoir_grad = grad[source_indices].copy()
    reservoir_weights = weights[source_indices].copy()
    reservoir_weights /= np.sum(reservoir_weights)

    bandwidths: list[float] = []
    for _ in range(int(max(0, config.stein_steps))):
        reservoir, bandwidth = stein_rejuvenate_particles(
            reservoir,
            reservoir_grad,
            step_size=config.stein_step_size,
            bandwidth=config.bandwidth,
            repulsion_scale=config.repulsion_scale,
        )
        bandwidths.append(float(bandwidth))

    return ReservoirSteinResult(
        particles=reservoir,
        source_indices=source_indices,
        weights=reservoir_weights,
        ess_before=effective_sample_size(weights),
        bandwidths=tuple(bandwidths),
    )
