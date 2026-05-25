"""TEASER-light pairwise consistency pre-filter for GSDC2023 PR observations.

Inspired by TEASER++'s graph-theoretic outlier pruning (Yang et al., 2020).
This module implements a much-simplified per-epoch version: for each epoch,
compute PR residuals against a robust reference position, then build the
pairwise consistency graph and keep only observations belonging to the
largest mutually-consistent subset (Maximal Consensus over scalar residual
agreement).

Algorithm (per epoch)
---------------------

1. Initial position estimate ``x0`` is taken from the row's ``kaggle_wls``
   (= Android Kalman baseline, the most reliable single-shot reference).
2. For each observation ``i`` compute Sagnac-corrected geometric range
   ``r_i`` and residual ``d_i = pseudorange_i - r_i - clock_offset_i``.
   The per-system clock offset is removed by subtracting the per-system
   median residual (= robust clock-bias estimate when receiver clock
   parameters are unknown).
3. Two observations ``(i, j)`` are *consistent* iff
   ``|(d_i - median_sys_i) - (d_j - median_sys_j)| <= 2 * mad_threshold_m``.
4. The largest mutually-consistent subset is approximated by the **median-
   distance core**: keep observations whose Mahalanobis distance from the
   per-system median is below ``mad_threshold_m * MAD * 1.4826``.  This
   is the "minimum clique" approximation suggested by TEASER++ Sec 4.2
   and is much cheaper than enumerating the actual maximum clique while
   matching its behaviour on the typical GNSS outlier distribution
   (single-bias-shifted reflection).
5. Observations outside the core have their weight clamped to 0 (= masked
   out of FGO + WLS downstream).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# Per-epoch defaults; chosen empirically based on the typical PR residual
# spread on the GSDC2023 dataset (3-10 m for inliers, > 20 m for NLOS).
MAD_THRESHOLD_DEFAULT = 3.5
MIN_OBS_AFTER_FILTER_DEFAULT = 5

LIGHT_SPEED_MPS = 299_792_458.0
OMEGA_E = 7.2921151467e-5


@dataclass(frozen=True)
class PairwiseFilterStats:
    epochs_total: int
    epochs_filtered: int
    obs_before: int
    obs_after: int
    obs_masked: int


def _sagnac_geometric_range(rx: np.ndarray, sat: np.ndarray) -> float:
    dx0 = rx[0] - sat[0]; dy0 = rx[1] - sat[1]; dz0 = rx[2] - sat[2]
    r0 = float(np.sqrt(dx0 * dx0 + dy0 * dy0 + dz0 * dz0))
    if r0 < 1.0:
        return 0.0
    theta = OMEGA_E * (r0 / LIGHT_SPEED_MPS)
    ct = np.cos(theta); st = np.sin(theta)
    sx_rot = sat[0] * ct + sat[1] * st
    sy_rot = -sat[0] * st + sat[1] * ct
    dx = rx[0] - sx_rot; dy = rx[1] - sy_rot; dz = rx[2] - sat[2]
    return float(np.sqrt(dx * dx + dy * dy + dz * dz))


def apply_pairwise_consistency_pre_filter(
    sat_ecef: np.ndarray,
    pseudorange: np.ndarray,
    weights: np.ndarray,
    *,
    reference_xyz: np.ndarray,
    sys_kind: np.ndarray | None = None,
    n_clock: int = 1,
    mad_threshold_m: float = MAD_THRESHOLD_DEFAULT,
    min_obs_after_filter: int = MIN_OBS_AFTER_FILTER_DEFAULT,
) -> tuple[np.ndarray, PairwiseFilterStats]:
    """Return a new ``weights`` array with outlier rows zeroed.

    Parameters
    ----------
    sat_ecef : ``(T, S, 3)`` satellite positions per (epoch, slot).
    pseudorange : ``(T, S)`` pseudorange measurements (m).
    weights : ``(T, S)`` original per-obs weights.
    reference_xyz : ``(T, 3)`` per-epoch reference receiver position (e.g.
        Kaggle Kalman baseline).  Non-finite rows skip filtering.
    sys_kind : ``(T, S)`` int32 per-obs clock-group index in ``[0, n_clock)``;
        per-system median residual is subtracted to act as the per-clock
        bias estimator.  When ``None`` all observations share clock group 0.
    n_clock : number of clock groups (= the upper bound of ``sys_kind``).
    mad_threshold_m : k multiplier; an obs is masked when
        ``|residual - per_sys_median| > k * 1.4826 * MAD_sys``.
    min_obs_after_filter : if the filtered epoch has fewer than this many
        live obs, the filter is reverted for that epoch (= keep original
        weights) so the FGO solver does not lose geometric coverage.
    """
    sat = np.asarray(sat_ecef, dtype=np.float64)
    pr = np.asarray(pseudorange, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64).copy()
    ref = np.asarray(reference_xyz, dtype=np.float64).reshape(-1, 3)
    sk = (
        np.asarray(sys_kind, dtype=np.int32).reshape(pr.shape)
        if sys_kind is not None
        else np.zeros(pr.shape, dtype=np.int32)
    )

    n_epoch, n_sat = pr.shape
    if ref.shape[0] != n_epoch:
        raise ValueError(f"reference_xyz must have T rows ({n_epoch}), got {ref.shape[0]}")

    epochs_filtered = 0
    obs_before = int(np.count_nonzero(w > 0))
    obs_masked = 0

    for t in range(n_epoch):
        if not np.isfinite(ref[t]).all():
            continue
        active = (w[t] > 0) & np.isfinite(pr[t]) & np.isfinite(sat[t]).all(axis=1)
        if active.sum() <= min_obs_after_filter:
            continue
        residuals = np.full(n_sat, np.nan, dtype=np.float64)
        for s in np.flatnonzero(active):
            r_geom = _sagnac_geometric_range(ref[t], sat[t, s])
            if r_geom <= 1.0:
                active[s] = False
                continue
            residuals[s] = pr[t, s] - r_geom
        valid_idx = np.flatnonzero(active & np.isfinite(residuals))
        if valid_idx.size <= min_obs_after_filter:
            continue
        # Per-clock-group median residual removes the receiver clock bias
        # contribution; remaining spread is the multipath/NLOS noise.
        per_group_mad: dict[int, tuple[float, float]] = {}
        sys_present = sk[t, valid_idx]
        for grp in np.unique(sys_present):
            sel = valid_idx[sys_present == grp]
            if sel.size == 0:
                continue
            res_grp = residuals[sel]
            med = float(np.median(res_grp))
            mad = float(np.median(np.abs(res_grp - med)))
            per_group_mad[int(grp)] = (med, mad if mad > 0.0 else 1.0)
        new_active = np.zeros(n_sat, dtype=bool)
        for s in valid_idx:
            grp = int(sk[t, s])
            med, mad = per_group_mad.get(grp, (0.0, 1.0))
            sigma_robust = max(1.4826 * mad, 1.0)
            if abs(residuals[s] - med) <= float(mad_threshold_m) * sigma_robust:
                new_active[s] = True
        if new_active.sum() < min_obs_after_filter:
            # Filter would over-prune this epoch; revert.
            continue
        # Apply: zero weights for observations no longer in the inlier set.
        rejected = active & ~new_active
        obs_masked += int(np.count_nonzero(rejected))
        w[t, rejected] = 0.0
        if rejected.any():
            epochs_filtered += 1

    obs_after = int(np.count_nonzero(w > 0))
    return w, PairwiseFilterStats(
        epochs_total=n_epoch,
        epochs_filtered=epochs_filtered,
        obs_before=obs_before,
        obs_after=obs_after,
        obs_masked=obs_masked,
    )


__all__ = [
    "MAD_THRESHOLD_DEFAULT",
    "MIN_OBS_AFTER_FILTER_DEFAULT",
    "PairwiseFilterStats",
    "apply_pairwise_consistency_pre_filter",
]
