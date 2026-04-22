"""Melbourne-Wübbena wide-lane integer ambiguity resolution.

Resolves the wide-lane ambiguity N_wl per satellite using the
Melbourne-Wübbena combination of dual-frequency code and carrier-phase
observables.  Once N_wl is fixed to an integer, produces a wide-lane
carrier-phase pseudorange with sub-meter noise (lambda_wl ≈ 0.862 m).

Only GPS L1/L2 is supported.  The narrow-lane (N1) ambiguity is NOT
resolved here — that requires external geometry and is left to the caller.

References
----------
Melbourne (1985) / Wübbena (1985) wide-lane combination.
"""

from __future__ import annotations

from collections import defaultdict
from typing import NamedTuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

C_LIGHT = 299792458.0
L1_FREQ = 1575.42e6  # Hz
L2_FREQ = 1227.60e6  # Hz
LAMBDA_1 = C_LIGHT / L1_FREQ  # ~0.1903 m
LAMBDA_2 = C_LIGHT / L2_FREQ  # ~0.2442 m
LAMBDA_WL = C_LIGHT / (L1_FREQ - L2_FREQ)  # ~0.862 m


# ---------------------------------------------------------------------------
# Single-epoch float N_wl
# ---------------------------------------------------------------------------


def compute_n_wl_float(
    L1_cycles: float,
    L2_cycles: float,
    P1_m: float,
    P2_m: float,
) -> float:
    """Compute single-epoch float wide-lane ambiguity (Melbourne-Wübbena).

    Parameters
    ----------
    L1_cycles : float
        L1 carrier-phase observation in **cycles**.
    L2_cycles : float
        L2 carrier-phase observation in **cycles**.
    P1_m : float
        L1 code (pseudorange) observation in **meters**.
    P2_m : float
        L2 code (pseudorange) observation in **meters**.

    Returns
    -------
    float
        Float-valued wide-lane ambiguity estimate N_wl.
    """
    P_nl = (L1_FREQ * P1_m + L2_FREQ * P2_m) / (L1_FREQ + L2_FREQ)
    return L1_cycles - L2_cycles - P_nl / LAMBDA_WL


# ---------------------------------------------------------------------------
# Accumulator for per-satellite convergence
# ---------------------------------------------------------------------------


class _SatAccumulator(NamedTuple):
    """Running statistics for N_wl estimates of one satellite."""

    estimates: list[float]


class WidelaneResolver:
    """Accumulate Melbourne-Wübbena N_wl estimates and fix to integer.

    Usage::

        wr = WidelaneResolver()
        for epoch in epochs:
            for prn, L1, L2, P1, P2 in observations:
                wr.update(prn, L1, L2, P1, P2)

            n_wl = wr.get_fixed_ambiguity(prn)
            if n_wl is not None:
                rho = wr.get_widelane_pseudorange(prn, L1, L2)

    Parameters
    ----------
    min_epochs : int
        Minimum number of accumulated epochs before attempting integer fix.
    max_std : float
        Maximum standard deviation (in cycles) of accumulated N_wl estimates
        for the fix to be accepted.
    """

    def __init__(self, min_epochs: int = 5, max_std: float = 0.4) -> None:
        self._min_epochs = min_epochs
        self._max_std = max_std
        self._accum: dict[int, list[float]] = defaultdict(list)
        self._fixed: dict[int, int] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        prn: int,
        L1_cycles: float,
        L2_cycles: float,
        P1_m: float,
        P2_m: float,
    ) -> None:
        """Accumulate one epoch of observables for *prn*."""
        n_wl = compute_n_wl_float(L1_cycles, L2_cycles, P1_m, P2_m)
        if not np.isfinite(n_wl):
            return
        self._accum[prn].append(n_wl)
        # Re-evaluate fix every update (cheap).
        self._try_fix(prn)

    def get_fixed_ambiguity(self, prn: int) -> int | None:
        """Return integer N_wl for *prn*, or ``None`` if not yet converged."""
        return self._fixed.get(prn)

    def get_widelane_pseudorange(
        self,
        prn: int,
        L1_cycles: float,
        L2_cycles: float,
    ) -> float | None:
        """Return wide-lane carrier-phase pseudorange [m] using fixed N_wl.

        The wide-lane phase observable (in cycles) is::

            phi_wl = (f1 * L1 - f2 * L2) / (f1 - f2)

        The corrected pseudorange is::

            rho_wl = (phi_wl - N_wl) * lambda_wl

        Returns ``None`` if N_wl is not yet fixed for *prn*.
        """
        n_wl = self._fixed.get(prn)
        if n_wl is None:
            return None
        phi_wl = (L1_FREQ * L1_cycles - L2_FREQ * L2_cycles) / (L1_FREQ - L2_FREQ)
        return (phi_wl - n_wl) * LAMBDA_WL

    def reset(self, prn: int | None = None) -> None:
        """Clear accumulated data (and any fix) for *prn*, or all if ``None``."""
        if prn is None:
            self._accum.clear()
            self._fixed.clear()
        else:
            self._accum.pop(prn, None)
            self._fixed.pop(prn, None)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _try_fix(self, prn: int) -> None:
        estimates = self._accum[prn]
        if len(estimates) < self._min_epochs:
            return
        arr = np.asarray(estimates, dtype=np.float64)
        if float(np.std(arr)) > self._max_std:
            return
        self._fixed[prn] = int(np.round(np.mean(arr)))
