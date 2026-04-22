"""Carrier AFV and DD carrier epoch update flow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from gnss_gpu.carrier_afv_observation import build_carrier_afv_observation
from gnss_gpu.carrier_rescue import CarrierBiasState
from gnss_gpu.dd_carrier_observation import compute_dd_carrier_observation
from gnss_gpu.dd_carrier_rescue_flow import (
    apply_post_dd_carrier_rescue,
    apply_weak_dd_carrier_fallback_replacement,
)
from gnss_gpu.dd_carrier_rescue_gate import evaluate_dd_carrier_rescue_gate
from gnss_gpu.dd_carrier_update import build_dd_carrier_update_decision


@dataclass(frozen=True)
class CarrierEpochUpdateResult:
    used_carrier_afv: bool
    used_dd_carrier: bool
    used_carrier_anchor: bool
    used_carrier_fallback: bool


def apply_carrier_epoch_update(
    pf: Any,
    epoch_state: Any,
    stats: Any,
    *,
    dd_computer: Any,
    carrier_bias_tracker: dict[tuple[int, int], CarrierBiasState],
    tow: float,
    measurements: Iterable[Any],
    sat_ecef: np.ndarray,
    pseudoranges: np.ndarray,
    spp_position_ecef: np.ndarray,
    gate_pf_estimate: np.ndarray | None,
    gate_ess_ratio: float | None,
    gate_spread_m: float | None,
    prev_pf_state: np.ndarray | None,
    velocity: np.ndarray | None,
    dt: float,
    observations_config: Any,
    collect_diagnostics: bool,
    min_pairs: int = 3,
) -> CarrierEpochUpdateResult:
    current_measurements = list(measurements)
    used_carrier_afv = False
    used_dd_carrier = False

    if observations_config.mupf.enabled:
        carrier_afv_obs = build_carrier_afv_observation(
            current_measurements,
            sat_ecef,
            pseudoranges,
            np.asarray(spp_position_ecef, dtype=np.float64),
            snr_min=observations_config.mupf.snr_min,
            elev_min=observations_config.mupf.elev_min,
            target_sigma_cycles=observations_config.mupf.sigma_cycles,
            min_sats=4,
            residual_max_m=30.0,
        )
        if carrier_afv_obs is not None:
            used_carrier_afv = True
            for sigma_cycles in carrier_afv_obs.sigma_sequence_cycles:
                pf.resample_if_needed()
                pf.update_carrier_afv(
                    carrier_afv_obs.sat_ecef,
                    carrier_afv_obs.carrier_phase_cycles,
                    weights=carrier_afv_obs.weights,
                    sigma_cycles=sigma_cycles,
                )

    if not (observations_config.dd_carrier.enabled and dd_computer is not None):
        return CarrierEpochUpdateResult(
            used_carrier_afv=used_carrier_afv,
            used_dd_carrier=False,
            used_carrier_anchor=bool(epoch_state.anchor_attempt.used),
            used_carrier_fallback=bool(epoch_state.fallback_attempt.used),
        )

    dd_cp_decision = compute_dd_carrier_observation(
        dd_computer,
        tow,
        current_measurements,
        gate_pf_estimate,
        observations_config.dd_carrier,
        observations_config.carrier_rescue,
        dd_pseudorange_result=epoch_state.dd_pr_result,
        ess_ratio=gate_ess_ratio,
        spread_m=gate_spread_m,
        collect_diagnostics=collect_diagnostics,
        gate_scale=epoch_state.dd_cp_gate_scale,
        min_pairs=min_pairs,
    )
    dd_result = dd_cp_decision.result
    epoch_state.dd_gate_stats = dd_cp_decision.gate_stats
    epoch_state.dd_cp_input_pairs = dd_cp_decision.input_pairs
    epoch_state.dd_cp_raw_abs_afv_median_cycles = (
        dd_cp_decision.raw_abs_afv_median_cycles
    )
    epoch_state.dd_cp_raw_abs_afv_max_cycles = dd_cp_decision.raw_abs_afv_max_cycles
    stats.n_dd_gate_pairs_rejected += dd_cp_decision.gate_pairs_rejected
    if dd_cp_decision.gate_epoch_skipped:
        stats.n_dd_gate_epoch_skip += 1

    rescue_gate = evaluate_dd_carrier_rescue_gate(
        dd_result,
        epoch_state.dd_pr_result,
        observations_config.carrier_rescue,
        ess_ratio=gate_ess_ratio,
        spread_m=gate_spread_m,
        raw_abs_afv_median_cycles=epoch_state.dd_cp_raw_abs_afv_median_cycles,
        min_pairs=min_pairs,
    )
    dd_result = rescue_gate.result
    if rescue_gate.support_skipped:
        epoch_state.dd_cp_support_skip = True
        stats.n_dd_skip_support_guard += 1
    if rescue_gate.replace_weak_with_fallback:
        replacement = apply_weak_dd_carrier_fallback_replacement(
            pf,
            current_measurements,
            sat_ecef,
            pseudoranges,
            np.asarray(spp_position_ecef, dtype=np.float64),
            carrier_bias_tracker,
            epoch_state.carrier_anchor_rows,
            np.asarray(pf.estimate(), dtype=np.float64),
            tow,
            dd_result,
            observations_config.mupf,
            observations_config.carrier_rescue,
        )
        epoch_state.fallback_attempt = replacement.fallback_attempt
        dd_result = replacement.dd_carrier_result

    if dd_result is not None and int(getattr(dd_result, "n_dd", 0)) >= int(min_pairs):
        dd_cp_update = build_dd_carrier_update_decision(
            dd_result,
            observations_config.dd_carrier,
            raw_abs_afv_median_cycles=epoch_state.dd_cp_raw_abs_afv_median_cycles,
            ess_ratio=gate_ess_ratio,
            min_pairs=min_pairs,
        )
        epoch_state.dd_cp_sigma_support_scale = dd_cp_update.sigma_support_scale
        epoch_state.dd_cp_sigma_afv_scale = dd_cp_update.sigma_afv_scale
        epoch_state.dd_cp_sigma_ess_scale = dd_cp_update.sigma_ess_scale
        epoch_state.dd_cp_sigma_scale = dd_cp_update.sigma_scale
        epoch_state.dd_cp_sigma_cycles = dd_cp_update.sigma_cycles
        if dd_cp_update.apply_update:
            pf.resample_if_needed()
            pf.update_dd_carrier_afv(
                dd_result,
                sigma_cycles=epoch_state.dd_cp_sigma_cycles,
            )
            epoch_state.dd_carrier_result = dd_result
            stats.n_dd_used += 1
            used_dd_carrier = True
            if dd_cp_update.sigma_relaxed:
                stats.record_dd_sigma_relaxed(epoch_state.dd_cp_sigma_scale)
        else:
            stats.n_dd_skip += 1
    else:
        stats.n_dd_skip += 1

    rescue_flow = apply_post_dd_carrier_rescue(
        pf,
        current_measurements,
        sat_ecef,
        pseudoranges,
        np.asarray(spp_position_ecef, dtype=np.float64),
        carrier_bias_tracker,
        epoch_state.carrier_anchor_rows,
        np.asarray(pf.estimate(), dtype=np.float64),
        prev_pf_state,
        velocity,
        dt,
        tow,
        epoch_state.dd_carrier_result,
        observations_config.mupf,
        observations_config.carrier_rescue,
        fallback_attempt=epoch_state.fallback_attempt,
    )
    epoch_state.anchor_attempt = rescue_flow.anchor_attempt
    epoch_state.fallback_attempt = rescue_flow.fallback_attempt
    if epoch_state.anchor_attempt.used:
        stats.n_carrier_anchor_used += 1

    if epoch_state.fallback_attempt.attempted_tracked:
        stats.n_dd_fallback_tracked_attempted += 1
    if epoch_state.fallback_attempt.used:
        stats.n_dd_fallback_undiff_used += 1
    if epoch_state.fallback_attempt.used_tracked:
        stats.n_dd_fallback_tracked_used += 1
    if (
        epoch_state.fallback_attempt.used
        and epoch_state.fallback_attempt.replaced_weak_dd
    ):
        stats.n_dd_fallback_weak_dd_replaced += 1

    return CarrierEpochUpdateResult(
        used_carrier_afv=used_carrier_afv,
        used_dd_carrier=used_dd_carrier,
        used_carrier_anchor=bool(epoch_state.anchor_attempt.used),
        used_carrier_fallback=bool(epoch_state.fallback_attempt.used),
    )
