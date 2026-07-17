"""Validate ray-traced LOS/NLOS + diffraction physics against measured C/N0.

Pseudorange residuals are one way to check the ray-traced geometry against
real UrbanNav data, but they are entangled with clock, atmosphere and
ephemeris error. The RINEX ``S1C`` observable (carrier-to-noise density,
dB-Hz) gives an independent channel: it is a direct measurement of received
signal power, and the ray tracer predicts both a discrete LOS/NLOS state per
satellite-epoch and, for NLOS satellites, a diffraction amplitude loss
(``attenuation_db``). If the physics is right, measured C/N0 should (a) be
systematically lower for predicted-NLOS satellites than predicted-LOS ones,
and (b) drop *in proportion* to the predicted diffraction attenuation among
the NLOS satellites, once the normal elevation-dependent C/N0 trend (higher
satellites read stronger even in the clear) is factored out via an
elevation-matched LOS baseline.

This module is pure numpy: it takes already-computed measured C/N0, ray-traced
LOS booleans and per-satellite predicted attenuation, and returns summary
statistics. It does not touch RINEX parsing, ray tracing, or plotting -- see
``examples/demo_cn0_validation.py`` for the end-to-end wiring.
"""

from __future__ import annotations

import numpy as np

from gnss_gpu.validation.reference_quality import auc_abs_residual_vs_nlos

_NAN = float("nan")

__all__ = [
    "cn0_los_nlos_separation",
    "elevation_binned_los_baseline",
    "baseline_at_elevation",
    "cn0_deficit",
    "attenuation_deficit_correlation",
]


def cn0_los_nlos_separation(cn0_dbhz, is_los) -> dict:
    """Compare measured C/N0 for predicted-LOS vs predicted-NLOS satellites.

    Parameters
    ----------
    cn0_dbhz:
        Measured C/N0 [dB-Hz] per satellite-epoch. NaN/non-finite entries are
        dropped before comparison.
    is_los:
        Ray-traced LOS boolean per satellite-epoch (same length).

    Returns a dict with per-class counts, means, medians, the mean/median gap
    (LOS - NLOS; positive means the physics prediction matches expectation),
    and ``auc`` -- the rank-based (Mann-Whitney) AUC of C/N0 as a classifier
    of the ray-traced LOS label (0.5 = C/N0 carries no LOS/NLOS information,
    1.0 = perfect separation, higher C/N0 always predicts LOS). This reuses
    the generic rank-AUC helper already used to grade the pseudorange-residual
    reference in :mod:`gnss_gpu.validation.reference_quality`.
    """
    cn0 = np.asarray(cn0_dbhz, dtype=float)
    los = np.asarray(is_los, dtype=bool)
    if cn0.shape != los.shape:
        raise ValueError("cn0_dbhz and is_los must have equal length")

    finite = np.isfinite(cn0)
    cn0 = cn0[finite]
    los = los[finite]

    n_los = int(los.sum())
    n_nlos = int((~los).sum())
    mean_los = float(np.mean(cn0[los])) if n_los else _NAN
    mean_nlos = float(np.mean(cn0[~los])) if n_nlos else _NAN
    median_los = float(np.median(cn0[los])) if n_los else _NAN
    median_nlos = float(np.median(cn0[~los])) if n_nlos else _NAN

    return {
        "n_los": n_los,
        "n_nlos": n_nlos,
        "mean_los_dbhz": mean_los,
        "mean_nlos_dbhz": mean_nlos,
        "median_los_dbhz": median_los,
        "median_nlos_dbhz": median_nlos,
        "mean_gap_dbhz": mean_los - mean_nlos if n_los and n_nlos else _NAN,
        "median_gap_dbhz": median_los - median_nlos if n_los and n_nlos else _NAN,
        # auc_abs_residual_vs_nlos(scores, positive_mask) ranks the
        # positive_mask == True group above the False group; passing is_los as
        # the mask makes this the AUC of C/N0 as an LOS classifier.
        "auc": auc_abs_residual_vs_nlos(cn0, los) if n_los and n_nlos else _NAN,
    }


def elevation_binned_los_baseline(
    elevation_deg,
    cn0_dbhz,
    is_los,
    bin_edges_deg=None,
) -> dict:
    """Median measured C/N0 of LOS satellites, binned by elevation.

    Clear-sky C/N0 rises with elevation (shorter atmospheric path, less
    multipath near the horizon), so a flat LOS mean is not a fair baseline for
    an NLOS satellite's C/N0 deficit -- the comparison must be elevation
    matched. ``bin_edges_deg`` defaults to 10-degree bins from 0 to 90.

    Returns a dict with ``bin_edges_deg`` (length n_bins+1), ``median_cn0_dbhz``
    and ``count`` (both length n_bins; NaN / 0 for empty bins).
    """
    edges = np.asarray(
        bin_edges_deg if bin_edges_deg is not None else np.arange(0.0, 95.0, 10.0),
        dtype=float,
    )
    if edges.size < 2:
        raise ValueError("bin_edges_deg must have at least 2 edges")
    if np.any(np.diff(edges) <= 0.0):
        raise ValueError("bin_edges_deg must be strictly increasing")

    elev = np.asarray(elevation_deg, dtype=float)
    cn0 = np.asarray(cn0_dbhz, dtype=float)
    los = np.asarray(is_los, dtype=bool)
    n = elev.size
    if not (cn0.size == n and los.size == n):
        raise ValueError("elevation_deg, cn0_dbhz, is_los must have equal length")

    n_bins = edges.size - 1
    median = np.full(n_bins, np.nan)
    count = np.zeros(n_bins, dtype=np.int64)

    valid = los & np.isfinite(cn0) & np.isfinite(elev)
    ev = elev[valid]
    cv = cn0[valid]
    if ev.size:
        bin_idx = np.searchsorted(edges, ev, side="right") - 1
        # Right edge of the last bin is closed (searchsorted with side="right"
        # would otherwise push elevation == edges[-1] one bin past the end).
        bin_idx = np.where(ev == edges[-1], n_bins - 1, bin_idx)
        for b in range(n_bins):
            sel = bin_idx == b
            if np.any(sel):
                vals = cv[sel]
                median[b] = float(np.median(vals))
                count[b] = int(vals.size)

    return {
        "bin_edges_deg": edges,
        "median_cn0_dbhz": median,
        "count": count,
    }


def baseline_at_elevation(baseline: dict, elevation_deg) -> np.ndarray:
    """Look up the elevation-binned LOS baseline C/N0 for each sample.

    ``baseline`` is the dict returned by :func:`elevation_binned_los_baseline`.
    Samples falling outside the binned range, or into an empty bin, get NaN.
    """
    edges = np.asarray(baseline["bin_edges_deg"], dtype=float)
    median = np.asarray(baseline["median_cn0_dbhz"], dtype=float)
    elev = np.asarray(elevation_deg, dtype=float)
    n_bins = median.size

    out = np.full(elev.shape, np.nan)
    finite = np.isfinite(elev)
    idx = np.full(elev.shape, -1, dtype=np.int64)
    idx[finite] = np.searchsorted(edges, elev[finite], side="right") - 1
    idx = np.where((elev == edges[-1]) & finite, n_bins - 1, idx)
    ok = finite & (idx >= 0) & (idx < n_bins)
    out[ok] = median[idx[ok]]
    return out


def cn0_deficit(elevation_deg, cn0_dbhz, baseline: dict) -> np.ndarray:
    """Measured C/N0 deficit = elevation-matched LOS baseline - measured C/N0.

    Positive values mean the satellite reads weaker than clear-sky satellites
    at the same elevation, as expected for a genuinely blocked/diffracted path.
    NaN where no baseline is available at that elevation.
    """
    base = baseline_at_elevation(baseline, elevation_deg)
    return base - np.asarray(cn0_dbhz, dtype=float)


def _rankdata(x: np.ndarray) -> np.ndarray:
    """Average ranks (1-based), ties get the mean rank (same scheme as
    :func:`gnss_gpu.validation.reference_quality.auc_abs_residual_vs_nlos`)."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(order.size, dtype=float)
    ranks[order] = np.arange(1, order.size + 1, dtype=float)
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    tie_sum = np.zeros(counts.size)
    np.add.at(tie_sum, inv, ranks)
    return (tie_sum / counts)[inv]


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return _NAN
    return float(np.corrcoef(x, y)[0, 1])


def attenuation_deficit_correlation(attenuation_db, cn0_deficit_db) -> dict:
    """Correlate predicted diffraction attenuation with the measured C/N0 deficit.

    Both arrays are per NLOS satellite-epoch: ``attenuation_db`` is the
    ray-traced diffraction amplitude loss, ``cn0_deficit_db`` the elevation-
    matched measured deficit (see :func:`cn0_deficit`). Non-finite pairs are
    dropped. If the physics is right, both correlations should be positive
    (more predicted attenuation -> a bigger measured C/N0 drop).

    Returns ``{"n": int, "pearson_r": float, "spearman_r": float}``; both
    correlations are NaN when fewer than 2 finite pairs remain.
    """
    a = np.asarray(attenuation_db, dtype=float)
    d = np.asarray(cn0_deficit_db, dtype=float)
    if a.shape != d.shape:
        raise ValueError("attenuation_db and cn0_deficit_db must have equal length")

    finite = np.isfinite(a) & np.isfinite(d)
    a = a[finite]
    d = d[finite]
    n = int(a.size)
    if n < 2:
        return {"n": n, "pearson_r": _NAN, "spearman_r": _NAN}

    pearson_r = _pearson(a, d)
    spearman_r = _pearson(_rankdata(a), _rankdata(d))
    return {"n": n, "pearson_r": pearson_r, "spearman_r": spearman_r}
