"""Per-satellite, per-epoch validation of a predicted diffraction/NLOS bias.

:mod:`gnss_gpu.validation.diffraction_benchmark` asks a *distribution-level*
question: is the simulated bias distribution close to the measured residual
distribution, in aggregate, across many satellite-epochs? That is necessary
but not sufficient for the claim this project actually wants to make -- that
the ray-traced physics predicts *which* satellite is biased, by *how much*,
at *each* epoch, well enough to be subtracted off as a measurement
correction.

This module answers that sharper question given already-aligned
per-(satellite, epoch) arrays: one predicted bias sample and one measured
residual sample per satellite-epoch, produced by the same diffraction model
run against the same real data (see
``examples/demo_per_satellite_validation.py`` for how those arrays are
built from UrbanNav + PLATEAU). It computes:

- Pearson and Spearman correlation between predicted and measured bias,
  overall and restricted to NLOS satellite-epochs.
- The sign-agreement rate on NLOS satellite-epochs whose |measured residual|
  exceeds a threshold (does the model get the *direction* of a real bias
  right, not just its rough size).
- The "correction gain": does subtracting the predicted bias from the
  measured residual reduce its RMS relative to not correcting at all? This
  is the direct measurement-correction usability test.
- A per-satellite breakdown table with all of the above computed within each
  satellite's own samples, so a model that works well on average but fails
  systematically on one SV is visible.

Every function here is a pure computation over numpy arrays: no I/O, no
GPU, no file access. NaNs in either ``predicted`` or ``measured`` mark a
satellite-epoch with no usable prediction (e.g. no trackable diffraction
path) and are dropped pairwise before any statistic is computed.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "pearson_correlation",
    "spearman_correlation",
    "sign_agreement_rate",
    "correction_gain",
    "per_satellite_table",
    "evaluate_predictions",
]


def _finite_pair(predicted, measured):
    """Return ``(predicted, measured)`` filtered to jointly-finite entries."""
    p = np.asarray(predicted, dtype=float)
    m = np.asarray(measured, dtype=float)
    mask = np.isfinite(p) & np.isfinite(m)
    return p[mask], m[mask]


def pearson_correlation(predicted, measured) -> float:
    """Pearson correlation between predicted and measured bias.

    NaN entries in either array are dropped pairwise. Returns NaN if fewer
    than two finite pairs remain or either series is constant (correlation
    undefined).
    """
    p, m = _finite_pair(predicted, measured)
    if p.size < 2 or np.std(p) == 0.0 or np.std(m) == 0.0:
        return float("nan")
    return float(np.corrcoef(p, m)[0, 1])


def _rankdata(x: np.ndarray) -> np.ndarray:
    """Average ranks (1-based), ties resolved by the mean of tied ranks."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(order.size, dtype=float)
    ranks[order] = np.arange(1, order.size + 1, dtype=float)
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    tie_sum = np.zeros(counts.size)
    np.add.at(tie_sum, inv, ranks)
    return (tie_sum / counts)[inv]


def spearman_correlation(predicted, measured) -> float:
    """Spearman rank correlation between predicted and measured bias.

    Computed as the Pearson correlation of the rank transforms (average
    ranks for ties). NaN entries are dropped pairwise; returns NaN if fewer
    than two finite pairs remain or ranks are constant.
    """
    p, m = _finite_pair(predicted, measured)
    if p.size < 2:
        return float("nan")
    rp, rm = _rankdata(p), _rankdata(m)
    if np.std(rp) == 0.0 or np.std(rm) == 0.0:
        return float("nan")
    return float(np.corrcoef(rp, rm)[0, 1])


def sign_agreement_rate(predicted, measured, *, is_nlos=None,
                         threshold_m: float = 1.0) -> dict:
    """Fraction of qualifying satellite-epochs where sign(predicted) == sign(measured).

    Restricts to NLOS satellite-epochs (via ``is_nlos``, if given) whose
    ``|measured|`` exceeds ``threshold_m`` -- this is the direction test: on
    satellite-epochs with a real, non-trivial bias, does the model predict
    the bias on the correct side of zero? Entries with NaN in either array
    are dropped first. Zero measured or predicted sign counts as a
    disagreement (``np.sign(0) == 0`` never equals +/-1).

    Returns ``{"n": <qualifying count>, "rate": <agreement fraction or NaN>}``.
    """
    p = np.asarray(predicted, dtype=float)
    m = np.asarray(measured, dtype=float)
    mask = np.isfinite(p) & np.isfinite(m)
    if is_nlos is not None:
        mask &= np.asarray(is_nlos, dtype=bool)
    mask &= np.abs(m) > float(threshold_m)
    n = int(mask.sum())
    if n == 0:
        return {"n": 0, "rate": float("nan")}
    agree = np.sign(p[mask]) == np.sign(m[mask])
    return {"n": n, "rate": float(np.mean(agree))}


def correction_gain(predicted, measured) -> dict:
    """Does subtracting ``predicted`` from ``measured`` reduce its RMS?

    Compares RMS(measured) [no correction] against RMS(measured - predicted)
    [subtract the model's prediction]. ``gain_m`` is the RMS reduction in
    metres (positive means the correction helps); ``gain_pct`` expresses it
    as a percentage of the uncorrected RMS. NaN entries are dropped pairwise
    first, so a satellite-epoch with no prediction contributes to neither
    RMS (it is simply excluded, not treated as a zero correction).
    """
    p, m = _finite_pair(predicted, measured)
    n = int(p.size)
    if n == 0:
        return {
            "n": 0,
            "rms_raw_m": float("nan"),
            "rms_corrected_m": float("nan"),
            "gain_m": float("nan"),
            "gain_pct": float("nan"),
        }
    rms_raw = float(np.sqrt(np.mean(m ** 2)))
    rms_corrected = float(np.sqrt(np.mean((m - p) ** 2)))
    gain_m = rms_raw - rms_corrected
    gain_pct = float(100.0 * gain_m / rms_raw) if rms_raw > 0.0 else float("nan")
    return {
        "n": n,
        "rms_raw_m": rms_raw,
        "rms_corrected_m": rms_corrected,
        "gain_m": gain_m,
        "gain_pct": gain_pct,
    }


def per_satellite_table(sat_ids, predicted, measured, *, is_nlos=None,
                         threshold_m: float = 1.0) -> list[dict]:
    """Per-satellite breakdown of the same metrics, one row per unique sat id.

    ``sat_ids`` (e.g. PRN strings such as ``"G01"``) aligns element-wise with
    ``predicted`` and ``measured``. Rows are sorted by ``sat_id`` (numpy's
    default sort of the id array). Each row reports the satellite's sample
    count, NLOS count, Pearson/Spearman correlation, sign-agreement rate,
    and correction gain, all computed within that satellite's own samples.
    """
    sat_ids = np.asarray(sat_ids)
    p = np.asarray(predicted, dtype=float)
    m = np.asarray(measured, dtype=float)
    if is_nlos is not None:
        nlos = np.asarray(is_nlos, dtype=bool)
    else:
        nlos = np.zeros(sat_ids.shape, dtype=bool)

    rows = []
    for sid in np.unique(sat_ids):
        sel = sat_ids == sid
        pp, mm, nn = p[sel], m[sel], nlos[sel]
        gain = correction_gain(pp, mm)
        sign = sign_agreement_rate(pp, mm, is_nlos=nn, threshold_m=threshold_m)
        rows.append({
            "sat_id": sid,
            "n": int(sel.sum()),
            "n_nlos": int(nn.sum()),
            "pearson": pearson_correlation(pp, mm),
            "spearman": spearman_correlation(pp, mm),
            "sign_agreement": sign["rate"],
            "sign_agreement_n": sign["n"],
            "rms_raw_m": gain["rms_raw_m"],
            "rms_corrected_m": gain["rms_corrected_m"],
            "gain_m": gain["gain_m"],
            "gain_pct": gain["gain_pct"],
        })
    return rows


def evaluate_predictions(predicted, measured, *, sat_ids=None, is_nlos=None,
                          nlos_threshold_m: float = 1.0) -> dict:
    """Full per-satellite/per-epoch validation summary for one model.

    ``predicted`` and ``measured`` are aligned 1-D arrays, one entry per
    satellite-epoch: ``predicted`` is the diffraction model's DLL code-bias
    prediction (metres), ``measured`` the tropo-purified, clock-removed
    pseudorange residual (metres) for the same satellite at the same epoch.
    NaN marks "no prediction available" and is dropped pairwise wherever a
    metric needs both arrays.

    ``is_nlos`` (optional boolean array, aligned) restricts the NLOS-only
    correlation and the sign-agreement test to true-NLOS satellite-epochs.
    ``sat_ids`` (optional, aligned) additionally requests the per-satellite
    breakdown table.

    Returns a dict with ``n``, ``pearson_all``, ``spearman_all``, ``n_nlos``,
    ``pearson_nlos``, ``spearman_nlos``, ``sign_agreement_nlos`` (+ ``_n``),
    ``rms_raw_m``, ``rms_corrected_m``, ``correction_gain_m``,
    ``correction_gain_pct``, and (if ``sat_ids`` given) ``per_satellite``.
    """
    p = np.asarray(predicted, dtype=float)
    m = np.asarray(measured, dtype=float)

    out: dict = {
        "n": int(np.sum(np.isfinite(p) & np.isfinite(m))),
        "pearson_all": pearson_correlation(p, m),
        "spearman_all": spearman_correlation(p, m),
    }

    if is_nlos is not None:
        nlos = np.asarray(is_nlos, dtype=bool)
        pn, mn = p[nlos], m[nlos]
        out["n_nlos"] = int(np.sum(np.isfinite(pn) & np.isfinite(mn)))
        out["pearson_nlos"] = pearson_correlation(pn, mn)
        out["spearman_nlos"] = spearman_correlation(pn, mn)
        sign = sign_agreement_rate(p, m, is_nlos=nlos, threshold_m=nlos_threshold_m)
        out["sign_agreement_nlos"] = sign["rate"]
        out["sign_agreement_nlos_n"] = sign["n"]
    else:
        nlos = None
        out["n_nlos"] = 0
        out["pearson_nlos"] = float("nan")
        out["spearman_nlos"] = float("nan")
        out["sign_agreement_nlos"] = float("nan")
        out["sign_agreement_nlos_n"] = 0

    gain = correction_gain(p, m)
    out["rms_raw_m"] = gain["rms_raw_m"]
    out["rms_corrected_m"] = gain["rms_corrected_m"]
    out["correction_gain_m"] = gain["gain_m"]
    out["correction_gain_pct"] = gain["gain_pct"]

    if sat_ids is not None:
        out["per_satellite"] = per_satellite_table(
            sat_ids, p, m, is_nlos=nlos, threshold_m=nlos_threshold_m)

    return out
