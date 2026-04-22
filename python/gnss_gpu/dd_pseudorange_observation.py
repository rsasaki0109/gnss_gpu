"""DD pseudorange observation compute/gate helper."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gnss_gpu.dd_quality import dd_pseudorange_residuals_m, gate_dd_pseudorange
from gnss_gpu.pf_smoother_config import DDPseudorangeConfig


@dataclass(frozen=True)
class DDPseudorangeObservationDecision:
    result: object | None
    gate_stats: object | None
    input_pairs: int
    raw_abs_res_median_m: float | None
    raw_abs_res_max_m: float | None
    gate_pairs_rejected: int
    gate_epoch_skipped: bool


def compute_dd_pseudorange_observation(
    dd_pr_computer,
    tow: float,
    measurements: list,
    pf_estimate: np.ndarray | None,
    rover_weights: np.ndarray,
    config: DDPseudorangeConfig,
    *,
    existing_result: object | None = None,
    existing_input_pairs: int = 0,
    collect_diagnostics: bool = False,
    raw_abs_res_median_m: float | None = None,
    raw_abs_res_max_m: float | None = None,
    gate_scale: float = 1.0,
    min_pairs: int = 3,
) -> DDPseudorangeObservationDecision:
    result = existing_result
    input_pairs = int(existing_input_pairs)
    raw_median = raw_abs_res_median_m
    raw_max = raw_abs_res_max_m

    if result is None and config.enabled and dd_pr_computer is not None:
        result = dd_pr_computer.compute_dd(
            tow,
            measurements,
            pf_estimate,
            rover_weights=rover_weights,
        )
        if result is not None:
            input_pairs = int(getattr(result, "n_dd", 0))
            if collect_diagnostics and int(getattr(result, "n_dd", 0)) > 0:
                raw_median, raw_max = _dd_pr_abs_residual_summary(result, pf_estimate)

    if result is not None and int(getattr(result, "n_dd", 0)) >= int(min_pairs):
        if collect_diagnostics and raw_median is None and pf_estimate is not None:
            raw_median, raw_max = _dd_pr_abs_residual_summary(result, pf_estimate)
        result, gate_stats = gate_dd_pseudorange(
            result,
            pf_estimate,
            pair_residual_max_m=config.gate_residual_m,
            adaptive_pair_floor_m=config.gate_adaptive_floor_m,
            adaptive_pair_mad_mult=config.gate_adaptive_mad_mult,
            epoch_median_residual_max_m=config.gate_epoch_median_m,
            threshold_scale=gate_scale,
            min_pairs=min_pairs,
        )
        return DDPseudorangeObservationDecision(
            result=result,
            gate_stats=gate_stats,
            input_pairs=input_pairs,
            raw_abs_res_median_m=raw_median,
            raw_abs_res_max_m=raw_max,
            gate_pairs_rejected=int(gate_stats.n_pair_rejected),
            gate_epoch_skipped=bool(gate_stats.rejected_by_epoch),
        )

    return DDPseudorangeObservationDecision(
        result=result,
        gate_stats=None,
        input_pairs=input_pairs,
        raw_abs_res_median_m=raw_median,
        raw_abs_res_max_m=raw_max,
        gate_pairs_rejected=0,
        gate_epoch_skipped=False,
    )


def _dd_pr_abs_residual_summary(
    result,
    pf_estimate: np.ndarray | None,
) -> tuple[float | None, float | None]:
    if pf_estimate is None:
        return None, None
    residuals = np.abs(dd_pseudorange_residuals_m(result, pf_estimate))
    finite = residuals[np.isfinite(residuals)]
    if finite.size == 0:
        return None, None
    return float(np.median(finite)), float(np.max(finite))
