"""DD carrier update parameter decision helper."""

from __future__ import annotations

from dataclasses import dataclass

from gnss_gpu.dd_quality import (
    combine_sigma_scales,
    ess_gate_scale,
    metric_sigma_scale,
    pair_count_sigma_scale,
)
from gnss_gpu.pf_smoother_config import DDCarrierConfig


@dataclass(frozen=True)
class DDCarrierUpdateDecision:
    apply_update: bool
    sigma_cycles: float | None = None
    sigma_support_scale: float = 1.0
    sigma_afv_scale: float = 1.0
    sigma_ess_scale: float = 1.0
    sigma_scale: float = 1.0
    sigma_relaxed: bool = False


def build_dd_carrier_update_decision(
    dd_result,
    config: DDCarrierConfig,
    *,
    raw_abs_afv_median_cycles: float | None,
    ess_ratio: float | None,
    min_pairs: int = 3,
) -> DDCarrierUpdateDecision:
    if (
        not config.enabled
        or dd_result is None
        or int(getattr(dd_result, "n_dd", 0)) < int(min_pairs)
    ):
        return DDCarrierUpdateDecision(apply_update=False)

    sigma_support_scale = 1.0
    sigma_afv_scale = 1.0
    sigma_ess_scale = 1.0

    if (
        config.sigma_support_low_pairs is not None
        and config.sigma_support_high_pairs is not None
        and config.sigma_support_max_scale > 1.0
    ):
        sigma_support_scale = pair_count_sigma_scale(
            int(getattr(dd_result, "n_dd", 0)),
            low_pairs=int(config.sigma_support_low_pairs),
            high_pairs=int(config.sigma_support_high_pairs),
            max_scale=config.sigma_support_max_scale,
        )

    if (
        config.sigma_afv_good_cycles is not None
        and config.sigma_afv_bad_cycles is not None
        and config.sigma_afv_max_scale > 1.0
        and raw_abs_afv_median_cycles is not None
    ):
        sigma_afv_scale = metric_sigma_scale(
            float(raw_abs_afv_median_cycles),
            good_value=float(config.sigma_afv_good_cycles),
            bad_value=float(config.sigma_afv_bad_cycles),
            max_scale=config.sigma_afv_max_scale,
        )

    if (
        config.sigma_ess_low_ratio is not None
        and config.sigma_ess_high_ratio is not None
        and config.sigma_ess_max_scale > 1.0
        and ess_ratio is not None
    ):
        sigma_ess_scale = ess_gate_scale(
            float(ess_ratio),
            low_ratio=float(config.sigma_ess_low_ratio),
            high_ratio=float(config.sigma_ess_high_ratio),
            min_scale=float(config.sigma_ess_max_scale),
            max_scale=1.0,
        )

    sigma_scale = combine_sigma_scales(
        sigma_support_scale,
        sigma_afv_scale,
        sigma_ess_scale,
        max_scale=config.sigma_max_scale,
    )
    sigma_cycles = float(config.sigma_cycles) * float(sigma_scale)
    return DDCarrierUpdateDecision(
        apply_update=True,
        sigma_cycles=sigma_cycles,
        sigma_support_scale=float(sigma_support_scale),
        sigma_afv_scale=float(sigma_afv_scale),
        sigma_ess_scale=float(sigma_ess_scale),
        sigma_scale=float(sigma_scale),
        sigma_relaxed=bool(sigma_scale > 1.0 + 1e-9),
    )
