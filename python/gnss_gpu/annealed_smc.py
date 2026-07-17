"""Annealed SMC utilities for sharp, staged particle-filter likelihoods.

Unlike a one-shot ESS guard, :func:`annealed_smc_update` never throws away the
unconsumed part of an observation likelihood.  It advances the likelihood
power from zero to one, resampling and re-evaluating the observation on the
new particles whenever another tempering increment is required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class AnnealedSMCResult:
    """Diagnostics from one fully-consumed observation likelihood."""

    beta_increments: tuple[float, ...]
    beta_consumed: float
    log_evidence: float
    initial_ess_ratio: float
    untempered_ess_ratio: float
    final_ess_ratio: float
    resample_count: int
    likelihood_evaluations: int


def ess_ratio_from_log_weights(log_weights: np.ndarray) -> float:
    """Return ESS/N without requiring normalized log weights."""

    values = np.asarray(log_weights, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.all(np.isfinite(values)):
        return float("nan")
    shifted = values - float(np.max(values))
    weights = np.exp(shifted)
    sw = float(np.sum(weights))
    sw2 = float(np.sum(weights * weights))
    if sw <= 0.0 or sw2 <= 0.0:
        return float("nan")
    return float((sw * sw) / (sw2 * values.size))


def _logsumexp(values: np.ndarray) -> float:
    vmax = float(np.max(values))
    return vmax + float(np.log(np.sum(np.exp(values - vmax))))


def _log_evidence_increment(
    pre_log_weights: np.ndarray,
    log_likelihood: np.ndarray,
    delta_beta: float,
) -> float:
    """SMC normalizing-constant increment under the current particle weights."""

    pre = np.asarray(pre_log_weights, dtype=np.float64)
    delta = np.asarray(log_likelihood, dtype=np.float64)
    return _logsumexp(pre + float(delta_beta) * delta) - _logsumexp(pre)


def _largest_safe_increment(
    pre_log_weights: np.ndarray,
    log_likelihood: np.ndarray,
    *,
    remaining_beta: float,
    target_ess_ratio: float,
    max_iters: int,
) -> tuple[float, float]:
    """Largest increment in ``[0, remaining_beta]`` meeting the ESS target."""

    pre = np.asarray(pre_log_weights, dtype=np.float64)
    delta = np.asarray(log_likelihood, dtype=np.float64)
    remaining = float(remaining_beta)
    full_ratio = ess_ratio_from_log_weights(pre + remaining * delta)
    if np.isfinite(full_ratio) and full_ratio >= float(target_ess_ratio):
        return remaining, full_ratio

    lo = 0.0
    hi = remaining
    best_ratio = ess_ratio_from_log_weights(pre)
    for _ in range(max(1, int(max_iters))):
        mid = 0.5 * (lo + hi)
        ratio = ess_ratio_from_log_weights(pre + mid * delta)
        if np.isfinite(ratio) and ratio >= float(target_ess_ratio):
            lo = mid
            best_ratio = ratio
        else:
            hi = mid
    return float(lo), float(best_ratio)


def annealed_smc_update(
    pf,
    apply_likelihood: Callable[[], None],
    *,
    target_ess_ratio: float,
    max_bisection_iters: int = 20,
    max_tempering_steps: int = 64,
    resample_before: bool = False,
    resample_at_end: bool = True,
) -> AnnealedSMCResult:
    """Apply one observation likelihood completely using annealed SMC.

    ``apply_likelihood`` must add the *full* log likelihood to the PF weights
    without resampling.  It may be called repeatedly because particles change
    after an intermediate resample.  The PF object must provide
    ``get_log_weights()``, ``set_log_weights()``, ``resample()`` and
    ``resample_if_needed()``.

    Raises
    ------
    RuntimeError
        If finite increments cannot be obtained, an entering cloud cannot be
        rejuvenated above the target ESS, or beta=1 is not reached within the
        configured step limit.  A staged update is never silently reverted.
    """

    target = float(target_ess_ratio)
    if not (0.0 < target <= 1.0):
        raise ValueError("target_ess_ratio must be in (0, 1]")
    if int(max_tempering_steps) < 1:
        raise ValueError("max_tempering_steps must be positive")

    resample_count = 0
    initial = np.asarray(pf.get_log_weights(), dtype=np.float64)
    initial_ratio = ess_ratio_from_log_weights(initial)
    if not np.isfinite(initial_ratio):
        raise RuntimeError("non-finite entering ESS for annealed SMC update")

    if bool(resample_before) or initial_ratio < target:
        pf.resample()
        resample_count += 1
        refreshed_ratio = ess_ratio_from_log_weights(pf.get_log_weights())
        if not np.isfinite(refreshed_ratio) or refreshed_ratio < target:
            raise RuntimeError("resampling did not restore ESS above target")

    remaining = 1.0
    increments: list[float] = []
    log_evidence = 0.0
    likelihood_evaluations = 0
    untempered_ratio = float("nan")

    for _ in range(int(max_tempering_steps)):
        pre = np.asarray(pf.get_log_weights(), dtype=np.float64)
        pre_ratio = ess_ratio_from_log_weights(pre)
        if not np.isfinite(pre_ratio) or pre_ratio < target:
            raise RuntimeError("annealed SMC increment entered below ESS target")

        apply_likelihood()
        likelihood_evaluations += 1
        post = np.asarray(pf.get_log_weights(), dtype=np.float64)
        if post.shape != pre.shape:
            raise RuntimeError("likelihood update changed the particle count")
        log_likelihood = post - pre
        pf.set_log_weights(pre)
        if not np.all(np.isfinite(log_likelihood)):
            raise RuntimeError("non-finite log-likelihood increment")

        raw_ratio = ess_ratio_from_log_weights(pre + remaining * log_likelihood)
        if not increments:
            untempered_ratio = raw_ratio
        delta_beta, _ = _largest_safe_increment(
            pre,
            log_likelihood,
            remaining_beta=remaining,
            target_ess_ratio=target,
            max_iters=int(max_bisection_iters),
        )
        min_progress = max(1.0e-12, remaining * (0.5 ** max(1, int(max_bisection_iters))))
        if not np.isfinite(delta_beta) or delta_beta < min_progress:
            raise RuntimeError("annealed SMC could not make positive beta progress")

        log_evidence += _log_evidence_increment(pre, log_likelihood, delta_beta)
        pf.set_log_weights(pre + delta_beta * log_likelihood)
        increments.append(float(delta_beta))
        remaining = max(0.0, remaining - float(delta_beta))

        if remaining <= 1.0e-12:
            remaining = 0.0
            break

        # The next increment must evaluate the same observation on the newly
        # selected particles.  Forced resampling is intentional even when the
        # PF's ordinary resampling threshold is below the annealing target.
        pf.resample()
        resample_count += 1
    else:
        raise RuntimeError(
            f"annealed SMC exhausted {max_tempering_steps} steps with "
            f"beta={1.0 - remaining:.12f}"
        )

    if resample_at_end and bool(pf.resample_if_needed()):
        resample_count += 1

    final_ratio = ess_ratio_from_log_weights(pf.get_log_weights())
    return AnnealedSMCResult(
        beta_increments=tuple(increments),
        beta_consumed=float(sum(increments)),
        log_evidence=float(log_evidence),
        initial_ess_ratio=float(initial_ratio),
        untempered_ess_ratio=float(untempered_ratio),
        final_ess_ratio=float(final_ratio),
        resample_count=int(resample_count),
        likelihood_evaluations=int(likelihood_evaluations),
    )
