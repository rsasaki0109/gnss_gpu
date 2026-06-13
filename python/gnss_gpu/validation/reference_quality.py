"""Is the real-data residual a usable NLOS ground truth?

Benchmarking an NLOS/diffraction model against tropo-purified pseudorange
residuals only makes sense if those residuals actually reflect NLOS: clear
line-of-sight satellites should have small residuals and blocked ones large
residuals. In single-frequency urban GPS that is frequently *not* the case --
uncorrected ionosphere, ephemeris error, ground-truth error and pervasive
multipath inflate even high-elevation LOS residuals to tens of metres, so the
residual cannot discriminate LOS from NLOS at all.

This module quantifies that with a single rank statistic (the AUC of
``|residual|`` predicting the geometric NLOS label) plus summary spreads, and
returns a verdict on whether the residual reference is clean enough to validate
an NLOS model. It is a guard against drawing physics conclusions from a
contaminated reference.
"""

from __future__ import annotations

import numpy as np


def auc_abs_residual_vs_nlos(abs_residual, is_nlos) -> float:
    """Mann-Whitney AUC that ``|residual|`` ranks NLOS satellites above LOS ones.

    0.5 means the residual is useless for telling NLOS from LOS; 1.0 means a
    larger residual perfectly predicts the geometric NLOS label. Returns NaN if
    either class is empty.
    """
    r = np.asarray(abs_residual, dtype=float)
    nlos = np.asarray(is_nlos, dtype=bool)
    pos = r[nlos]
    neg = r[~nlos]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(order.size, dtype=float)
    ranks[order] = np.arange(1, order.size + 1, dtype=float)
    # average ranks for ties
    _, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    tie_sum = np.zeros(counts.size)
    np.add.at(tie_sum, inv, ranks)
    ranks = (tie_sum / counts)[inv]
    r_pos = ranks[: pos.size].sum()
    return float((r_pos - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size))


def residual_reference_quality(
    abs_residual,
    is_los,
    *,
    small_m: float = 5.0,
    large_m: float = 15.0,
    los_clean_median_m: float = 10.0,
    min_auc: float = 0.60,
) -> dict:
    """Summarise whether residuals can serve as an NLOS ground truth.

    ``abs_residual`` are tropo-purified |pseudorange residuals| (m), ``is_los``
    the geometric line-of-sight booleans for the same satellite-epochs. Returns a
    dict with per-class medians, small/large fractions, the discrimination AUC,
    and ``is_clean_reference`` -- True only if LOS residuals are small (median <=
    ``los_clean_median_m``) and the AUC clears ``min_auc``.
    """
    r = np.asarray(abs_residual, dtype=float)
    los = np.asarray(is_los, dtype=bool)
    nlos = ~los
    out = {
        "n": int(r.size),
        "nlos_fraction": float(nlos.mean()) if r.size else float("nan"),
        "los_median_m": float(np.median(r[los])) if los.any() else float("nan"),
        "nlos_median_m": float(np.median(r[nlos])) if nlos.any() else float("nan"),
        "los_frac_small": float((r[los] < small_m).mean()) if los.any() else float("nan"),
        "nlos_frac_large": float((r[nlos] > large_m).mean()) if nlos.any() else float("nan"),
        "auc": auc_abs_residual_vs_nlos(r, nlos),
    }
    auc = out["auc"]
    out["is_clean_reference"] = bool(
        np.isfinite(auc)
        and auc >= min_auc
        and np.isfinite(out["los_median_m"])
        and out["los_median_m"] <= los_clean_median_m
    )
    return out


def format_reference_quality(q: dict) -> str:
    """One-paragraph human summary of :func:`residual_reference_quality`."""
    verdict = (
        "CLEAN: residuals discriminate LOS/NLOS, usable as NLOS ground truth"
        if q["is_clean_reference"]
        else "CONTAMINATED: residuals do NOT cleanly reflect NLOS (uncorrected "
        "iono / ephemeris / ground-truth error dominate) -- NLOS model "
        "validation against them is unreliable"
    )
    return (
        f"residual reference quality (n={q['n']}, "
        f"NLOS frac={q['nlos_fraction']:.2f}):\n"
        f"  |residual| median  LOS={q['los_median_m']:.1f} m  "
        f"NLOS={q['nlos_median_m']:.1f} m\n"
        f"  LOS<5m frac={q['los_frac_small']:.2f}  NLOS>15m frac={q['nlos_frac_large']:.2f}\n"
        f"  AUC(|resid| -> NLOS)={q['auc']:.3f}  (0.5=useless, 1=perfect)\n"
        f"  -> {verdict}"
    )


__all__ = [
    "auc_abs_residual_vs_nlos",
    "residual_reference_quality",
    "format_reference_quality",
]
