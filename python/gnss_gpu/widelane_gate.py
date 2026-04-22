"""Wide-lane-derived DD pseudorange gating helpers."""

from __future__ import annotations

import numpy as np

from gnss_gpu.dd_quality import dd_pseudorange_residuals_m

def _filter_dd_pseudorange_pairs(dd_result, keep_mask: np.ndarray):
    keep = np.asarray(keep_mask, dtype=bool).ravel()
    if dd_result is None or keep.size != int(getattr(dd_result, "n_dd", 0)):
        return None
    if int(np.count_nonzero(keep)) <= 0:
        return None
    ref_sat_ids = tuple(
        sid for sid, use in zip(getattr(dd_result, "ref_sat_ids", ()), keep) if bool(use)
    )
    return dd_result.__class__(
        dd_pseudorange_m=np.asarray(dd_result.dd_pseudorange_m, dtype=np.float64)[keep],
        sat_ecef_k=np.asarray(dd_result.sat_ecef_k, dtype=np.float64)[keep],
        sat_ecef_ref=np.asarray(dd_result.sat_ecef_ref, dtype=np.float64)[keep],
        base_range_k=np.asarray(dd_result.base_range_k, dtype=np.float64)[keep],
        base_range_ref=np.asarray(dd_result.base_range_ref, dtype=np.float64)[keep],
        dd_weights=np.asarray(dd_result.dd_weights, dtype=np.float64)[keep],
        ref_sat_ids=ref_sat_ids,
        n_dd=int(np.count_nonzero(keep)),
    )


def _gate_widelane_pseudorange_result(
    wl_result,
    wl_stats,
    pf_est: np.ndarray | None,
    *,
    min_fixed_pairs: int | None,
    min_fix_rate: float | None,
    min_spread_m: float | None,
    spread_m: float | None,
    max_epoch_median_residual_m: float | None,
    max_pair_residual_m: float | None,
    min_pairs: int = 3,
):
    """Apply WL-specific epoch/pair guards before reusing DD-PR machinery."""

    info: dict[str, object] = {
        "reason": getattr(wl_stats, "reason", None),
        "pair_rejected": 0,
        "raw_abs_res_median_m": None,
        "raw_abs_res_max_m": None,
        "kept_abs_res_median_m": None,
        "kept_abs_res_max_m": None,
    }
    if wl_result is None or int(getattr(wl_result, "n_dd", 0)) <= 0:
        return None, info
    if min_fixed_pairs is not None and int(getattr(wl_stats, "n_fixed_pairs", 0)) < int(min_fixed_pairs):
        info["reason"] = "gate_min_fixed_pairs"
        return None, info
    if min_fix_rate is not None and float(getattr(wl_stats, "fix_rate", 0.0)) < float(min_fix_rate):
        info["reason"] = "gate_min_fix_rate"
        return None, info
    if min_spread_m is not None:
        if spread_m is None or not np.isfinite(float(spread_m)):
            info["reason"] = "gate_missing_spread"
            return None, info
        if float(spread_m) < float(min_spread_m):
            info["reason"] = "gate_min_spread"
            return None, info

    gated = wl_result
    needs_residuals = max_epoch_median_residual_m is not None or max_pair_residual_m is not None
    if needs_residuals:
        if pf_est is None or not np.isfinite(np.asarray(pf_est, dtype=np.float64)).all():
            info["reason"] = "gate_missing_pf_est"
            return None, info
        residuals = np.abs(dd_pseudorange_residuals_m(gated, np.asarray(pf_est, dtype=np.float64)))
        finite = residuals[np.isfinite(residuals)]
        if finite.size == 0:
            info["reason"] = "gate_no_finite_residuals"
            return None, info
        info["raw_abs_res_median_m"] = float(np.median(finite))
        info["raw_abs_res_max_m"] = float(np.max(finite))
        if max_pair_residual_m is not None:
            keep = residuals <= float(max_pair_residual_m)
            info["pair_rejected"] = int(len(residuals) - np.count_nonzero(keep))
            if int(np.count_nonzero(keep)) < int(min_pairs):
                info["reason"] = "gate_pair_residual"
                return None, info
            gated = _filter_dd_pseudorange_pairs(gated, keep)
            if gated is None:
                info["reason"] = "gate_pair_residual"
                return None, info
            residuals = np.abs(dd_pseudorange_residuals_m(gated, np.asarray(pf_est, dtype=np.float64)))
            finite = residuals[np.isfinite(residuals)]
            if finite.size == 0:
                info["reason"] = "gate_no_finite_residuals"
                return None, info
        info["kept_abs_res_median_m"] = float(np.median(finite))
        info["kept_abs_res_max_m"] = float(np.max(finite))
        if (
            max_epoch_median_residual_m is not None
            and float(info["kept_abs_res_median_m"]) > float(max_epoch_median_residual_m)
        ):
            info["reason"] = "gate_epoch_median_residual"
            return None, info

    info["reason"] = "ok"
    return gated, info
