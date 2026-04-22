"""Position-update epoch flow for PF smoother forward passes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from gnss_gpu.imu_position_update import evaluate_imu_tight_position_update
from gnss_gpu.motion_position_update import evaluate_motion_position_update
from gnss_gpu.tdcp_motion import evaluate_tdcp_position_update


@dataclass(frozen=True)
class PositionEpochUpdateResult:
    used_spp_position_update: bool
    used_imu_tight: bool
    used_doppler_position_update: bool
    used_tdcp_position_update: bool
    tdcp_gate_reason: str | None


def apply_position_epoch_updates(
    pf: Any,
    epoch_state: Any,
    stats: Any,
    *,
    spp_position_ecef: np.ndarray,
    sat_ecef: np.ndarray,
    pseudoranges: np.ndarray,
    n_measurements: int,
    prev_estimate: np.ndarray | None,
    prev_pf_estimate: np.ndarray | None,
    dt: float,
    gate_ess_ratio: float | None,
    gate_spread_m: float | None,
    particle_filter_config: Any,
    motion_config: Any,
    doppler_config: Any,
    tdcp_position_config: Any,
) -> PositionEpochUpdateResult:
    spp_pos = np.asarray(spp_position_ecef, dtype=np.float64).ravel()[:3]
    used_spp = False
    used_imu_tight = False
    used_doppler_pu = False
    used_tdcp_pu = False

    if particle_filter_config.position_update_sigma is not None:
        if np.isfinite(spp_pos).all() and np.linalg.norm(spp_pos) > 1e6:
            pf.position_update(
                spp_pos,
                sigma_pos=particle_filter_config.position_update_sigma,
            )
            used_spp = True

    if motion_config.imu_tight_coupling:
        imu_tight_decision = evaluate_imu_tight_position_update(
            prev_pf_estimate,
            epoch_state.imu_velocity,
            dt,
            sat_ecef,
            pseudoranges,
            spp_pos,
            n_measurements=n_measurements,
        )
        if imu_tight_decision.apply_update:
            pf.position_update(
                imu_tight_decision.predicted_position,
                sigma_pos=imu_tight_decision.sigma_pos,
            )
            stats.n_imu_tight_used += 1
            epoch_state.used_imu_tight_epoch = True
            used_imu_tight = True
        else:
            stats.n_imu_tight_skip += 1

    if doppler_config.position_update:
        doppler_pu_decision = evaluate_motion_position_update(
            prev_estimate,
            epoch_state.velocity,
            dt,
        )
        if doppler_pu_decision.apply_update:
            pf.position_update(
                doppler_pu_decision.predicted_position,
                sigma_pos=doppler_config.pu_sigma,
            )
            used_doppler_pu = True

    if tdcp_position_config.enabled and prev_estimate is not None and dt > 0:
        tdcp_pu_decision = evaluate_tdcp_position_update(
            prev_estimate,
            epoch_state.tdcp_pu_velocity,
            epoch_state.tdcp_pu_rms,
            dt,
            rms_max=tdcp_position_config.rms_max,
            tdcp_reason=epoch_state.tdcp_pu_reason,
            dd_gate_stats=epoch_state.dd_gate_stats,
            dd_cp_input_pairs=epoch_state.dd_cp_input_pairs,
            dd_pr_gate_stats=epoch_state.dd_pr_gate_stats,
            dd_pr_input_pairs=epoch_state.dd_pr_input_pairs,
            gate_spread_m=gate_spread_m,
            gate_ess_ratio=gate_ess_ratio,
            dd_pr_raw_abs_res_median_m=epoch_state.dd_pr_raw_abs_res_median_m,
            dd_cp_raw_abs_afv_median_cycles=(
                epoch_state.dd_cp_raw_abs_afv_median_cycles
            ),
            gate_dd_carrier_min_pairs=tdcp_position_config.gate_dd_carrier_min_pairs,
            gate_dd_carrier_max_pairs=tdcp_position_config.gate_dd_carrier_max_pairs,
            gate_dd_pseudorange_max_pairs=(
                tdcp_position_config.gate_dd_pseudorange_max_pairs
            ),
            gate_min_spread_m=tdcp_position_config.gate_min_spread_m,
            gate_max_spread_m=tdcp_position_config.gate_max_spread_m,
            gate_min_ess_ratio=tdcp_position_config.gate_min_ess_ratio,
            gate_max_ess_ratio=tdcp_position_config.gate_max_ess_ratio,
            gate_dd_pr_max_raw_median_m=(
                tdcp_position_config.gate_dd_pr_max_raw_median_m
            ),
            gate_dd_cp_max_raw_afv_median_cycles=(
                tdcp_position_config.gate_dd_cp_max_raw_afv_median_cycles
            ),
            gate_logic=tdcp_position_config.gate_logic,
            gate_stop_mode=tdcp_position_config.gate_stop_mode,
            imu_stop_detected=epoch_state.imu_stop_detected,
        )
        epoch_state.tdcp_pu_gate_reason = tdcp_pu_decision.gate_reason
        if tdcp_pu_decision.apply_update:
            pf.position_update(
                tdcp_pu_decision.predicted_position,
                sigma_pos=tdcp_position_config.sigma,
            )
            stats.n_tdcp_pu_used += 1
            epoch_state.used_tdcp_pu_epoch = True
            used_tdcp_pu = True
        elif tdcp_pu_decision.gate_skipped:
            stats.n_tdcp_pu_gate_skip += 1
            stats.n_tdcp_pu_skip += 1
    else:
        stats.n_tdcp_pu_skip += 1

    return PositionEpochUpdateResult(
        used_spp_position_update=used_spp,
        used_imu_tight=used_imu_tight,
        used_doppler_position_update=used_doppler_pu,
        used_tdcp_position_update=used_tdcp_pu,
        tdcp_gate_reason=epoch_state.tdcp_pu_gate_reason,
    )
