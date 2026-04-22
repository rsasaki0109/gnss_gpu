"""DD carrier observation compute/gate helper."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gnss_gpu.carrier_rescue import _effective_dd_carrier_epoch_median_gate
from gnss_gpu.dd_quality import dd_carrier_afv_cycles, gate_dd_carrier
from gnss_gpu.pf_smoother_config import CarrierRescueConfig, DDCarrierConfig


@dataclass(frozen=True)
class DDCarrierObservationDecision:
    result: object | None
    gate_stats: object | None
    input_pairs: int
    raw_abs_afv_median_cycles: float | None
    raw_abs_afv_max_cycles: float | None
    gate_pairs_rejected: int
    gate_epoch_skipped: bool


def compute_dd_carrier_observation(
    dd_computer,
    tow: float,
    measurements: list,
    pf_estimate: np.ndarray | None,
    config: DDCarrierConfig,
    carrier_rescue: CarrierRescueConfig,
    *,
    dd_pseudorange_result: object | None,
    ess_ratio: float | None,
    spread_m: float | None,
    collect_diagnostics: bool = False,
    gate_scale: float = 1.0,
    min_pairs: int = 3,
) -> DDCarrierObservationDecision:
    result = None
    input_pairs = 0
    raw_median = None
    raw_max = None

    if config.enabled and dd_computer is not None:
        result = dd_computer.compute_dd(tow, measurements, pf_estimate)
        if result is not None:
            input_pairs = int(getattr(result, "n_dd", 0))
            if _needs_raw_afv_summary(config, carrier_rescue, collect_diagnostics) and input_pairs > 0:
                raw_median, raw_max = _dd_carrier_abs_afv_summary(result, pf_estimate)

    if result is not None and int(getattr(result, "n_dd", 0)) >= int(min_pairs):
        if _needs_raw_afv_summary(config, carrier_rescue, collect_diagnostics) and raw_median is None:
            raw_median, raw_max = _dd_carrier_abs_afv_summary(result, pf_estimate)
        epoch_median_cycles = _effective_dd_carrier_epoch_median_gate(
            dd_pseudorange_result,
            base_epoch_median_cycles=config.gate_epoch_median_cycles,
            ess_ratio=ess_ratio,
            spread_m=spread_m,
            low_ess_epoch_median_cycles=config.gate_low_ess_epoch_median_cycles,
            low_ess_max_ratio=config.gate_low_ess_max_ratio,
            low_ess_max_spread_m=config.gate_low_ess_max_spread_m,
            low_ess_require_no_dd_pr=config.gate_low_ess_require_no_dd_pr,
        )
        result, gate_stats = gate_dd_carrier(
            result,
            pf_estimate,
            pair_afv_max_cycles=config.gate_afv_cycles,
            adaptive_pair_floor_cycles=config.gate_adaptive_floor_cycles,
            adaptive_pair_mad_mult=config.gate_adaptive_mad_mult,
            epoch_median_afv_max_cycles=epoch_median_cycles,
            threshold_scale=gate_scale,
            min_pairs=min_pairs,
        )
        return DDCarrierObservationDecision(
            result=result,
            gate_stats=gate_stats,
            input_pairs=input_pairs,
            raw_abs_afv_median_cycles=raw_median,
            raw_abs_afv_max_cycles=raw_max,
            gate_pairs_rejected=int(gate_stats.n_pair_rejected),
            gate_epoch_skipped=bool(gate_stats.rejected_by_epoch),
        )

    return DDCarrierObservationDecision(
        result=result,
        gate_stats=None,
        input_pairs=input_pairs,
        raw_abs_afv_median_cycles=raw_median,
        raw_abs_afv_max_cycles=raw_max,
        gate_pairs_rejected=0,
        gate_epoch_skipped=False,
    )


def _needs_raw_afv_summary(
    config: DDCarrierConfig,
    carrier_rescue: CarrierRescueConfig,
    collect_diagnostics: bool,
) -> bool:
    return (
        bool(collect_diagnostics)
        or carrier_rescue.skip_low_support_min_raw_afv_median_cycles is not None
        or (
            config.sigma_afv_good_cycles is not None
            and config.sigma_afv_bad_cycles is not None
        )
    )


def _dd_carrier_abs_afv_summary(
    result,
    pf_estimate: np.ndarray | None,
) -> tuple[float | None, float | None]:
    if pf_estimate is None:
        return None, None
    afv = np.abs(dd_carrier_afv_cycles(result, pf_estimate))
    if afv.size == 0:
        return None, None
    return float(np.median(afv)), float(np.max(afv))
