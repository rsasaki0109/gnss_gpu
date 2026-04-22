"""Quality gates for double-differenced pseudorange and carrier observations."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass
class DDGateStats:
    """Diagnostics for a DD quality gate decision."""

    n_input_pairs: int
    n_kept_pairs: int
    n_pair_rejected: int
    metric_median: float
    metric_max: float
    pair_threshold: float
    threshold_scale: float
    rejected_by_epoch: bool


def _subset_dd_result(dd_result, mask: np.ndarray):
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    n_kept = int(np.count_nonzero(mask))
    updates = {
        "sat_ecef_k": np.asarray(dd_result.sat_ecef_k, dtype=np.float64)[mask],
        "sat_ecef_ref": np.asarray(dd_result.sat_ecef_ref, dtype=np.float64)[mask],
        "base_range_k": np.asarray(dd_result.base_range_k, dtype=np.float64)[mask],
        "base_range_ref": np.asarray(dd_result.base_range_ref, dtype=np.float64)[mask],
        "dd_weights": np.asarray(dd_result.dd_weights, dtype=np.float64)[mask],
        "ref_sat_ids": tuple(np.asarray(dd_result.ref_sat_ids, dtype=object)[mask].tolist()),
        "n_dd": n_kept,
    }
    if hasattr(dd_result, "sat_ids") and getattr(dd_result, "sat_ids"):
        updates["sat_ids"] = tuple(np.asarray(dd_result.sat_ids, dtype=object)[mask].tolist())
    if hasattr(dd_result, "dd_pseudorange_m"):
        updates["dd_pseudorange_m"] = np.asarray(dd_result.dd_pseudorange_m, dtype=np.float64)[mask]
    if hasattr(dd_result, "dd_carrier_cycles"):
        updates["dd_carrier_cycles"] = np.asarray(dd_result.dd_carrier_cycles, dtype=np.float64)[mask]
    if hasattr(dd_result, "wavelengths_m"):
        updates["wavelengths_m"] = np.asarray(dd_result.wavelengths_m, dtype=np.float64)[mask]
    return replace(dd_result, **updates)


def dd_pseudorange_residuals_m(dd_result, rover_position_ecef: np.ndarray) -> np.ndarray:
    """Compute DD pseudorange residuals [m] at a rover position."""

    rover_position_ecef = np.asarray(rover_position_ecef, dtype=np.float64).reshape(3)
    range_k = np.linalg.norm(np.asarray(dd_result.sat_ecef_k, dtype=np.float64) - rover_position_ecef, axis=1)
    range_ref = np.linalg.norm(
        np.asarray(dd_result.sat_ecef_ref, dtype=np.float64) - rover_position_ecef,
        axis=1,
    )
    expected = range_k - range_ref - dd_result.base_range_k + dd_result.base_range_ref
    return np.asarray(dd_result.dd_pseudorange_m, dtype=np.float64) - expected


def dd_carrier_afv_cycles(dd_result, rover_position_ecef: np.ndarray) -> np.ndarray:
    """Compute DD carrier AFV residuals [cycles] at a rover position."""

    rover_position_ecef = np.asarray(rover_position_ecef, dtype=np.float64).reshape(3)
    range_k = np.linalg.norm(np.asarray(dd_result.sat_ecef_k, dtype=np.float64) - rover_position_ecef, axis=1)
    range_ref = np.linalg.norm(
        np.asarray(dd_result.sat_ecef_ref, dtype=np.float64) - rover_position_ecef,
        axis=1,
    )
    expected_m = range_k - range_ref - dd_result.base_range_k + dd_result.base_range_ref
    residual_cycles = np.asarray(dd_result.dd_carrier_cycles, dtype=np.float64) - (
        expected_m / np.asarray(dd_result.wavelengths_m, dtype=np.float64)
    )
    return residual_cycles - np.round(residual_cycles)


def _adaptive_pair_threshold(
    abs_metric: np.ndarray,
    *,
    fixed_max: float | None = None,
    adaptive_floor: float | None = None,
    adaptive_mad_mult: float | None = None,
) -> float:
    threshold = float("inf")
    if fixed_max is not None:
        threshold = min(threshold, float(fixed_max))
    if adaptive_mad_mult is not None:
        metric_median = float(np.median(abs_metric))
        metric_mad = float(np.median(np.abs(abs_metric - metric_median)))
        adaptive_threshold = metric_median + float(adaptive_mad_mult) * metric_mad
        if adaptive_floor is not None:
            adaptive_threshold = max(adaptive_threshold, float(adaptive_floor))
        threshold = min(threshold, adaptive_threshold)
    elif adaptive_floor is not None:
        threshold = min(threshold, float(adaptive_floor))
    return threshold


def ess_gate_scale(
    ess_ratio: float,
    *,
    low_ratio: float = 0.15,
    high_ratio: float = 0.75,
    min_scale: float = 1.0,
    max_scale: float = 1.0,
) -> float:
    """Map ESS ratio to a smooth threshold scale.

    Lower ESS tightens the gate toward ``min_scale``; higher ESS relaxes it
    toward ``max_scale``. When min/max are both 1.0, scaling is disabled.
    """

    if not np.isfinite(ess_ratio):
        return 1.0
    if low_ratio >= high_ratio:
        raise ValueError("low_ratio must be smaller than high_ratio")
    t = (float(ess_ratio) - float(low_ratio)) / (float(high_ratio) - float(low_ratio))
    t = float(np.clip(t, 0.0, 1.0))
    return float(min_scale) + t * (float(max_scale) - float(min_scale))


def spread_gate_scale(
    spread_m: float,
    *,
    low_spread_m: float = 1.5,
    high_spread_m: float = 8.0,
    min_scale: float = 1.0,
    max_scale: float = 1.0,
) -> float:
    """Map particle spread to a smooth threshold scale."""

    if not np.isfinite(spread_m):
        return 1.0
    if low_spread_m >= high_spread_m:
        raise ValueError("low_spread_m must be smaller than high_spread_m")
    t = (float(spread_m) - float(low_spread_m)) / (float(high_spread_m) - float(low_spread_m))
    t = float(np.clip(t, 0.0, 1.0))
    return float(min_scale) + t * (float(max_scale) - float(min_scale))


def pair_count_sigma_scale(
    n_pairs: int,
    *,
    low_pairs: int,
    high_pairs: int,
    min_scale: float = 1.0,
    max_scale: float = 1.0,
) -> float:
    """Map DD pair count to a sigma multiplier.

    Lower pair counts relax the sigma toward ``max_scale``. Higher pair counts
    keep the sigma near ``min_scale``.
    """

    if low_pairs >= high_pairs:
        raise ValueError("low_pairs must be smaller than high_pairs")
    t = (float(high_pairs) - float(n_pairs)) / (float(high_pairs) - float(low_pairs))
    t = float(np.clip(t, 0.0, 1.0))
    return float(min_scale) + t * (float(max_scale) - float(min_scale))


def metric_sigma_scale(
    metric_value: float,
    *,
    good_value: float,
    bad_value: float,
    min_scale: float = 1.0,
    max_scale: float = 1.0,
) -> float:
    """Map a quality metric to a sigma multiplier.

    Metrics at or below ``good_value`` keep sigma near ``min_scale``. Metrics at
    or above ``bad_value`` relax sigma toward ``max_scale``.
    """

    if not np.isfinite(metric_value):
        return float(min_scale)
    if good_value >= bad_value:
        raise ValueError("good_value must be smaller than bad_value")
    t = (float(metric_value) - float(good_value)) / (float(bad_value) - float(good_value))
    t = float(np.clip(t, 0.0, 1.0))
    return float(min_scale) + t * (float(max_scale) - float(min_scale))


def combine_sigma_scales(*scales: float, max_scale: float | None = None) -> float:
    """Combine independent sigma multipliers and optionally clip the result."""

    combined = 1.0
    for scale in scales:
        if scale is None:
            continue
        combined *= float(scale)
    if max_scale is not None:
        combined = min(combined, float(max_scale))
    return combined


def gate_dd_pseudorange(
    dd_result,
    rover_position_ecef: np.ndarray,
    *,
    pair_residual_max_m: float | None = None,
    adaptive_pair_floor_m: float | None = None,
    adaptive_pair_mad_mult: float | None = None,
    epoch_median_residual_max_m: float | None = None,
    threshold_scale: float = 1.0,
    min_pairs: int = 3,
):
    """Filter DD pseudorange pairs/epochs using geometric residuals."""

    residuals = dd_pseudorange_residuals_m(dd_result, rover_position_ecef)
    abs_metric = np.abs(residuals)
    mask = np.ones(abs_metric.shape[0], dtype=bool)
    pair_threshold = _adaptive_pair_threshold(
        abs_metric,
        fixed_max=pair_residual_max_m,
        adaptive_floor=adaptive_pair_floor_m,
        adaptive_mad_mult=adaptive_pair_mad_mult,
    )
    if np.isfinite(pair_threshold):
        pair_threshold *= float(threshold_scale)
        mask &= abs_metric <= pair_threshold

    kept_metric = abs_metric[mask]
    metric_median = float(np.median(kept_metric)) if kept_metric.size else float("inf")
    metric_max = float(np.max(kept_metric)) if kept_metric.size else float("inf")
    stats = DDGateStats(
        n_input_pairs=int(abs_metric.size),
        n_kept_pairs=int(np.count_nonzero(mask)),
        n_pair_rejected=int(abs_metric.size - np.count_nonzero(mask)),
        metric_median=metric_median,
        metric_max=metric_max,
        pair_threshold=pair_threshold,
        threshold_scale=float(threshold_scale),
        rejected_by_epoch=False,
    )
    if stats.n_kept_pairs < int(min_pairs):
        stats.rejected_by_epoch = True
        return None, stats
    if (
        epoch_median_residual_max_m is not None
        and metric_median > float(epoch_median_residual_max_m) * float(threshold_scale)
    ):
        stats.rejected_by_epoch = True
        return None, stats
    return _subset_dd_result(dd_result, mask), stats


def gate_dd_carrier(
    dd_result,
    rover_position_ecef: np.ndarray,
    *,
    pair_afv_max_cycles: float | None = None,
    adaptive_pair_floor_cycles: float | None = None,
    adaptive_pair_mad_mult: float | None = None,
    epoch_median_afv_max_cycles: float | None = None,
    threshold_scale: float = 1.0,
    min_pairs: int = 3,
):
    """Filter DD carrier pairs/epochs using AFV residuals."""

    afv = dd_carrier_afv_cycles(dd_result, rover_position_ecef)
    abs_metric = np.abs(afv)
    mask = np.ones(abs_metric.shape[0], dtype=bool)
    pair_threshold = _adaptive_pair_threshold(
        abs_metric,
        fixed_max=pair_afv_max_cycles,
        adaptive_floor=adaptive_pair_floor_cycles,
        adaptive_mad_mult=adaptive_pair_mad_mult,
    )
    if np.isfinite(pair_threshold):
        pair_threshold *= float(threshold_scale)
        mask &= abs_metric <= pair_threshold

    kept_metric = abs_metric[mask]
    metric_median = float(np.median(kept_metric)) if kept_metric.size else float("inf")
    metric_max = float(np.max(kept_metric)) if kept_metric.size else float("inf")
    stats = DDGateStats(
        n_input_pairs=int(abs_metric.size),
        n_kept_pairs=int(np.count_nonzero(mask)),
        n_pair_rejected=int(abs_metric.size - np.count_nonzero(mask)),
        metric_median=metric_median,
        metric_max=metric_max,
        pair_threshold=pair_threshold,
        threshold_scale=float(threshold_scale),
        rejected_by_epoch=False,
    )
    if stats.n_kept_pairs < int(min_pairs):
        stats.rejected_by_epoch = True
        return None, stats
    if (
        epoch_median_afv_max_cycles is not None
        and metric_median > float(epoch_median_afv_max_cycles) * float(threshold_scale)
    ):
        stats.rejected_by_epoch = True
        return None, stats
    return _subset_dd_result(dd_result, mask), stats
