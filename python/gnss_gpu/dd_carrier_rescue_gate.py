"""DD carrier rescue gating decisions."""

from __future__ import annotations

from dataclasses import dataclass

from gnss_gpu.carrier_rescue import (
    _should_replace_weak_dd_with_fallback,
    _should_skip_low_support_dd_carrier,
)
from gnss_gpu.pf_smoother_config import CarrierRescueConfig


@dataclass(frozen=True)
class DDCarrierRescueGateDecision:
    result: object | None
    support_skipped: bool = False
    replace_weak_with_fallback: bool = False


def evaluate_dd_carrier_rescue_gate(
    dd_result,
    dd_pseudorange_result,
    config: CarrierRescueConfig,
    *,
    ess_ratio: float | None,
    spread_m: float | None,
    raw_abs_afv_median_cycles: float | None,
    min_pairs: int = 3,
) -> DDCarrierRescueGateDecision:
    if dd_result is None or int(getattr(dd_result, "n_dd", 0)) < int(min_pairs):
        return DDCarrierRescueGateDecision(result=dd_result)

    if _should_skip_low_support_dd_carrier(
        dd_result,
        dd_pseudorange_result,
        ess_ratio=ess_ratio,
        spread_m=spread_m,
        raw_afv_median_cycles=raw_abs_afv_median_cycles,
        low_support_ess_ratio=config.skip_low_support_ess_ratio,
        low_support_max_pairs=config.skip_low_support_max_pairs,
        low_support_max_spread_m=config.skip_low_support_max_spread_m,
        low_support_min_raw_afv_median_cycles=(
            config.skip_low_support_min_raw_afv_median_cycles
        ),
        low_support_require_no_dd_pr=config.skip_low_support_require_no_dd_pr,
    ):
        return DDCarrierRescueGateDecision(result=None, support_skipped=True)

    replace_weak = _should_replace_weak_dd_with_fallback(
        dd_result,
        dd_pseudorange_result,
        raw_afv_median_cycles=raw_abs_afv_median_cycles,
        ess_ratio=ess_ratio,
        weak_dd_max_pairs=config.fallback_weak_dd_max_pairs,
        weak_dd_max_ess_ratio=config.fallback_weak_dd_max_ess_ratio,
        weak_dd_min_raw_afv_median_cycles=config.fallback_weak_dd_min_raw_afv_median_cycles,
        weak_dd_require_no_dd_pr=config.fallback_weak_dd_require_no_dd_pr,
    )
    return DDCarrierRescueGateDecision(
        result=dd_result,
        replace_weak_with_fallback=replace_weak,
    )
