"""Wide-lane DD pseudorange observation helper."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gnss_gpu.pf_smoother_config import WidelaneConfig
from gnss_gpu.widelane_gate import _gate_widelane_pseudorange_result


def _default_gate_info() -> dict[str, object]:
    return {
        "reason": None,
        "pair_rejected": 0,
        "raw_abs_res_median_m": None,
        "raw_abs_res_max_m": None,
        "kept_abs_res_median_m": None,
        "kept_abs_res_max_m": None,
    }


@dataclass(frozen=True)
class WidelaneObservationDecision:
    dd_pseudorange_result: object | None
    stats: object | None
    gate_info: dict[str, object]
    input_pairs: int = 0
    fixed_pairs: int = 0
    fix_rate: float | None = None
    dd_sigma_m: float | None = None
    used: bool = False
    skipped: bool = False
    gate_skipped: bool = False
    low_fix_rate: bool = False
    gate_pair_rejected: int = 0


def compute_widelane_observation(
    wl_computer,
    tow: float,
    measurements: list,
    pf_estimate: np.ndarray | None,
    rover_weights: np.ndarray,
    config: WidelaneConfig,
    *,
    spread_m: float | None,
    min_pairs: int = 3,
) -> WidelaneObservationDecision:
    if wl_computer is None or not config.enabled:
        return WidelaneObservationDecision(
            dd_pseudorange_result=None,
            stats=None,
            gate_info=_default_gate_info(),
        )

    wl_result, wl_stats = wl_computer.compute_dd(
        tow,
        measurements,
        pf_estimate,
        rover_weights=rover_weights,
        min_fix_rate=config.min_fix_rate,
    )
    input_pairs = int(getattr(wl_stats, "n_candidate_pairs", 0))
    fixed_pairs = int(getattr(wl_stats, "n_fixed_pairs", 0))
    fix_rate = float(getattr(wl_stats, "fix_rate", 0.0))
    low_fix_rate = getattr(wl_stats, "reason", None) == "low_fix_rate"

    if wl_result is None or int(getattr(wl_result, "n_dd", 0)) < int(min_pairs):
        return WidelaneObservationDecision(
            dd_pseudorange_result=None,
            stats=wl_stats,
            gate_info=_default_gate_info(),
            input_pairs=input_pairs,
            fixed_pairs=fixed_pairs,
            fix_rate=fix_rate,
            skipped=True,
            low_fix_rate=low_fix_rate,
        )

    gated_result, gate_info = _gate_widelane_pseudorange_result(
        wl_result,
        wl_stats,
        pf_estimate,
        min_fixed_pairs=config.gate_min_fixed_pairs,
        min_fix_rate=config.gate_min_fix_rate,
        min_spread_m=config.gate_min_spread_m,
        spread_m=spread_m,
        max_epoch_median_residual_m=config.gate_max_epoch_median_residual_m,
        max_pair_residual_m=config.gate_max_pair_residual_m,
        min_pairs=min_pairs,
    )
    gate_pair_rejected = int(gate_info.get("pair_rejected") or 0)
    if gated_result is None or int(getattr(gated_result, "n_dd", 0)) < int(min_pairs):
        return WidelaneObservationDecision(
            dd_pseudorange_result=None,
            stats=wl_stats,
            gate_info=gate_info,
            input_pairs=input_pairs,
            fixed_pairs=fixed_pairs,
            fix_rate=fix_rate,
            skipped=True,
            gate_skipped=True,
            low_fix_rate=low_fix_rate,
            gate_pair_rejected=gate_pair_rejected,
        )

    return WidelaneObservationDecision(
        dd_pseudorange_result=gated_result,
        stats=wl_stats,
        gate_info=gate_info,
        input_pairs=input_pairs,
        fixed_pairs=fixed_pairs,
        fix_rate=fix_rate,
        dd_sigma_m=float(config.dd_sigma),
        used=True,
        low_fix_rate=low_fix_rate,
        gate_pair_rejected=gate_pair_rejected,
    )
