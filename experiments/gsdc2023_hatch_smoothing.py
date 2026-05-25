"""Hatch carrier-phase smoothing for GSDC2023 pseudorange observations.

The Hatch filter combines noisy pseudorange ``P_k`` and smooth carrier-phase
``L_k`` (in metres) into a smoothed pseudorange ``\tilde{P}_k`` that retains
the unbiased mean of ``P`` while inheriting the low-noise short-term
trajectory of ``L``:

    \tilde{P}_k = (1/n) * P_k + ((n-1)/n) * (\tilde{P}_{k-1} + (L_k - L_{k-1}))

with ``n = min(k_in_arc, N)`` where ``N`` is the smoothing window.  Standard
GNSS practice uses ``N=100`` (= 100 seconds at 1 Hz).

Arc handling
------------

A carrier-phase arc breaks whenever any of the following happens:

  * ``adr_state`` does not have ``ADR_STATE_VALID`` set, **or**
  * ``adr_state`` has either ``ADR_STATE_RESET`` or ``ADR_STATE_CYCLE_SLIP``,
  * the satellite has no observation at this epoch (e.g. set lost).

When an arc breaks the smoothed PR resets to the raw PR (= no smoothing
benefit until the arc accumulates at least one valid carrier delta).

Why GSDC2023?
-------------

This is the same Hatch implementation as
``experiments/build_hatch_smoothed_spp_candidate.py`` (PPC2024 path) but
adapted for the dense ``(T, S)`` matrix layout used by the GSDC2023 raw
bridge.  Smoothing reduces multipath / receiver-noise variance on each PR
by roughly ``1/sqrt(N)`` while leaving slowly-varying bias intact, which
helps the WLS/FGO solvers fit cleaner residuals → smaller mse_pr → less
gate pressure on outlier rejection.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


HATCH_SMOOTHING_N_DEFAULT = 100
ADR_STATE_VALID = 1
ADR_STATE_RESET = 2
ADR_STATE_CYCLE_SLIP = 4


@dataclass(frozen=True)
class HatchSmoothingStats:
    epochs_total: int
    obs_total: int
    obs_smoothed: int
    arcs_total: int
    mean_arc_length: float
    mean_smoothing_n: float


def _valid_adr_state(state: int) -> bool:
    return bool(state & ADR_STATE_VALID) and not bool(
        state & (ADR_STATE_RESET | ADR_STATE_CYCLE_SLIP)
    )


def apply_hatch_smoothing(
    pseudorange: np.ndarray,
    adr: np.ndarray | None,
    adr_state: np.ndarray | None,
    *,
    smoothing_n: int = HATCH_SMOOTHING_N_DEFAULT,
) -> tuple[np.ndarray, HatchSmoothingStats]:
    """Return Hatch-smoothed pseudorange + per-satellite arc statistics.

    Parameters
    ----------
    pseudorange : ``(T, S)`` raw pseudorange (m).  NaN entries are passed
        through unchanged.
    adr : ``(T, S)`` carrier-phase ADR in metres, or ``None`` to disable
        smoothing entirely (= identity passthrough).
    adr_state : ``(T, S)`` int state codes (bits VALID/RESET/CYCLE_SLIP).
        Required when ``adr`` is provided.
    smoothing_n : Hatch window length.  Larger = smoother but slower to
        track real range changes.  ``100`` matches the Google-Earth-Engine
        suggestion for 1 Hz GNSS data.

    Returns
    -------
    smoothed : ``(T, S)`` smoothed pseudorange.  NaN entries are preserved.
    stats : per-trip diagnostics.
    """
    pr = np.asarray(pseudorange, dtype=np.float64).copy()
    n_epoch, n_sat = pr.shape
    if adr is None or adr_state is None:
        return pr, HatchSmoothingStats(
            epochs_total=n_epoch,
            obs_total=int(np.count_nonzero(np.isfinite(pr))),
            obs_smoothed=0,
            arcs_total=0,
            mean_arc_length=0.0,
            mean_smoothing_n=0.0,
        )
    adr_m = np.asarray(adr, dtype=np.float64)
    state = np.asarray(adr_state, dtype=np.int32)
    n = max(2, int(smoothing_n))

    arc_lengths: list[int] = []
    smoothing_ns: list[int] = []
    obs_smoothed = 0
    arcs_total = 0

    for s in range(n_sat):
        p_prev = np.nan
        l_prev = np.nan
        k_in_arc = 0
        for t in range(n_epoch):
            pr_t = pr[t, s]
            adr_t = adr_m[t, s]
            st_t = int(state[t, s])
            if not (np.isfinite(pr_t) and np.isfinite(adr_t) and _valid_adr_state(st_t)):
                if k_in_arc > 0:
                    arc_lengths.append(k_in_arc)
                p_prev = np.nan
                l_prev = np.nan
                k_in_arc = 0
                continue
            k_in_arc += 1
            if k_in_arc == 1:
                arcs_total += 1
                pr[t, s] = pr_t  # arc start: smoothed = raw
                p_prev = pr_t
                l_prev = adr_t
                continue
            n_eff = min(k_in_arc, n)
            smoothing_ns.append(n_eff)
            p_smooth = (1.0 / n_eff) * pr_t + ((n_eff - 1) / n_eff) * (
                p_prev + (adr_t - l_prev)
            )
            pr[t, s] = p_smooth
            p_prev = p_smooth
            l_prev = adr_t
            obs_smoothed += 1
        if k_in_arc > 0:
            arc_lengths.append(k_in_arc)

    mean_arc = float(np.mean(arc_lengths)) if arc_lengths else 0.0
    mean_n = float(np.mean(smoothing_ns)) if smoothing_ns else 0.0
    return pr, HatchSmoothingStats(
        epochs_total=n_epoch,
        obs_total=int(np.count_nonzero(np.isfinite(pr))),
        obs_smoothed=obs_smoothed,
        arcs_total=arcs_total,
        mean_arc_length=mean_arc,
        mean_smoothing_n=mean_n,
    )


__all__ = [
    "HATCH_SMOOTHING_N_DEFAULT",
    "HatchSmoothingStats",
    "apply_hatch_smoothing",
]
