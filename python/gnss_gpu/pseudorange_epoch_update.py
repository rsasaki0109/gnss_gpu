"""Widelane and DD pseudorange epoch update flow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from gnss_gpu.dd_pseudorange_observation import compute_dd_pseudorange_observation
from gnss_gpu.widelane_observation import compute_widelane_observation


@dataclass(frozen=True)
class PseudorangeEpochUpdateResult:
    used_dd_pseudorange: bool
    used_widelane: bool
    used_gmm_fallback: bool


def apply_widelane_dd_pseudorange_update(
    pf: Any,
    epoch_state: Any,
    stats: Any,
    *,
    dd_pr_computer: Any,
    wl_computer: Any,
    tow: float,
    measurements: Iterable[Any],
    sat_ecef: np.ndarray,
    pseudoranges: np.ndarray,
    rover_weights: np.ndarray,
    pf_estimate: np.ndarray | None,
    gate_pf_estimate: np.ndarray | None,
    gate_spread_m: float | None,
    observations_config: Any,
    collect_diagnostics: bool,
    min_pairs: int = 3,
) -> PseudorangeEpochUpdateResult:
    current_measurements = list(measurements)

    if observations_config.widelane.enabled and wl_computer is not None:
        wl_decision = compute_widelane_observation(
            wl_computer,
            tow,
            current_measurements,
            gate_pf_estimate,
            rover_weights,
            observations_config.widelane,
            spread_m=gate_spread_m,
            min_pairs=min_pairs,
        )
        epoch_state.wl_stats = wl_decision.stats
        epoch_state.wl_gate_info = wl_decision.gate_info
        epoch_state.wl_input_pairs = wl_decision.input_pairs
        epoch_state.wl_fixed_pairs = wl_decision.fixed_pairs
        epoch_state.wl_fix_rate = wl_decision.fix_rate
        stats.n_wl_candidate_pairs += epoch_state.wl_input_pairs
        stats.n_wl_fixed_pairs += epoch_state.wl_fixed_pairs
        if wl_decision.low_fix_rate:
            stats.n_wl_low_fix_rate += 1
        stats.n_wl_gate_pair_rejected += wl_decision.gate_pair_rejected
        if wl_decision.used:
            epoch_state.dd_pr_result = wl_decision.dd_pseudorange_result
            epoch_state.dd_pr_input_pairs = int(
                getattr(wl_decision.dd_pseudorange_result, "n_dd", 0)
            )
            epoch_state.dd_pr_sigma_epoch = float(wl_decision.dd_sigma_m)
            epoch_state.used_widelane_epoch = True
            stats.n_wl_used += 1
        elif wl_decision.skipped:
            stats.n_wl_skip += 1
            if wl_decision.gate_skipped:
                stats.n_wl_gate_skip += 1

    dd_pr_decision = compute_dd_pseudorange_observation(
        dd_pr_computer,
        tow,
        current_measurements,
        pf_estimate,
        rover_weights,
        observations_config.dd_pseudorange,
        existing_result=epoch_state.dd_pr_result,
        existing_input_pairs=epoch_state.dd_pr_input_pairs,
        collect_diagnostics=collect_diagnostics,
        raw_abs_res_median_m=epoch_state.dd_pr_raw_abs_res_median_m,
        raw_abs_res_max_m=epoch_state.dd_pr_raw_abs_res_max_m,
        gate_scale=epoch_state.dd_pr_gate_scale,
        min_pairs=min_pairs,
    )
    epoch_state.dd_pr_result = dd_pr_decision.result
    epoch_state.dd_pr_gate_stats = dd_pr_decision.gate_stats
    epoch_state.dd_pr_input_pairs = dd_pr_decision.input_pairs
    epoch_state.dd_pr_raw_abs_res_median_m = dd_pr_decision.raw_abs_res_median_m
    epoch_state.dd_pr_raw_abs_res_max_m = dd_pr_decision.raw_abs_res_max_m
    stats.n_dd_pr_gate_pairs_rejected += dd_pr_decision.gate_pairs_rejected
    if dd_pr_decision.gate_epoch_skipped:
        stats.n_dd_pr_gate_epoch_skip += 1

    if (
        epoch_state.dd_pr_result is not None
        and int(getattr(epoch_state.dd_pr_result, "n_dd", 0)) >= int(min_pairs)
    ):
        pf.update_dd_pseudorange(
            epoch_state.dd_pr_result,
            sigma_pr=epoch_state.dd_pr_sigma_epoch,
        )
        stats.n_dd_pr_used += 1
        return PseudorangeEpochUpdateResult(
            used_dd_pseudorange=True,
            used_widelane=bool(epoch_state.used_widelane_epoch),
            used_gmm_fallback=False,
        )

    if observations_config.dd_pseudorange.enabled:
        stats.n_dd_pr_skip += 1
    pf.correct_clock_bias(sat_ecef, pseudoranges)
    robust_config = observations_config.robust
    if robust_config.use_gmm:
        pf.update_gmm(
            sat_ecef,
            pseudoranges,
            weights=rover_weights,
            w_los=robust_config.gmm_w_los,
            mu_nlos=robust_config.gmm_mu_nlos,
            sigma_nlos=robust_config.gmm_sigma_nlos,
        )
        return PseudorangeEpochUpdateResult(
            used_dd_pseudorange=False,
            used_widelane=bool(epoch_state.used_widelane_epoch),
            used_gmm_fallback=True,
        )

    pf.update(sat_ecef, pseudoranges, weights=rover_weights)
    return PseudorangeEpochUpdateResult(
        used_dd_pseudorange=False,
        used_widelane=bool(epoch_state.used_widelane_epoch),
        used_gmm_fallback=False,
    )
