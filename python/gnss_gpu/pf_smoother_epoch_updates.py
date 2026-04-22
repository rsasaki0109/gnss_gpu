"""Measurement update sequence for one PF smoother forward epoch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from gnss_gpu.carrier_rescue import MUPF_L1_WAVELENGTH_M
from gnss_gpu.dd_carrier_epoch_update import apply_carrier_epoch_update
from gnss_gpu.doppler_epoch_update import apply_doppler_epoch_update
from gnss_gpu.pf_smoother_epoch_measurements import ForwardEpochMeasurementInputs
from gnss_gpu.pf_smoother_epoch_state import EpochForwardState
from gnss_gpu.pf_smoother_forward_context import PfSmootherForwardPassContext
from gnss_gpu.position_epoch_update import apply_position_epoch_updates
from gnss_gpu.pseudorange_epoch_update import apply_widelane_dd_pseudorange_update


@dataclass(frozen=True)
class ForwardEpochUpdatesResult:
    spp_position_ecef: np.ndarray


def apply_forward_epoch_updates(
    context: PfSmootherForwardPassContext,
    epoch_state: EpochForwardState,
    measurement_inputs: ForwardEpochMeasurementInputs,
    sol_epoch: Any,
    measurements: list[Any],
    *,
    tow: float,
    dt: float,
) -> ForwardEpochUpdatesResult:
    run_config = context.run_config
    config_parts = context.config_parts
    observation_setup = context.observation_setup
    sat_ecef = measurement_inputs.sat_ecef
    pr = measurement_inputs.pseudoranges
    w = measurement_inputs.weights
    gate_pf_est = measurement_inputs.gate_pf_estimate
    gate_ess_ratio = measurement_inputs.gate_ess_ratio
    gate_spread_m = measurement_inputs.gate_spread_m

    apply_widelane_dd_pseudorange_update(
        context.pf,
        epoch_state,
        context.stats,
        dd_pr_computer=observation_setup.dd_pr_computer,
        wl_computer=observation_setup.wl_computer,
        tow=tow,
        measurements=measurements,
        sat_ecef=sat_ecef,
        pseudoranges=pr,
        rover_weights=w,
        pf_estimate=gate_pf_est,
        gate_pf_estimate=gate_pf_est,
        gate_spread_m=gate_spread_m,
        observations_config=config_parts.observations,
        collect_diagnostics=run_config.collect_epoch_diagnostics,
    )

    spp_position_ecef = np.array(sol_epoch.position_ecef_m[:3], dtype=np.float64)
    apply_carrier_epoch_update(
        context.pf,
        epoch_state,
        context.stats,
        dd_computer=observation_setup.dd_computer,
        carrier_bias_tracker=observation_setup.carrier_bias_tracker,
        tow=tow,
        measurements=measurements,
        sat_ecef=sat_ecef,
        pseudoranges=pr,
        spp_position_ecef=spp_position_ecef,
        gate_pf_estimate=gate_pf_est,
        gate_ess_ratio=gate_ess_ratio,
        gate_spread_m=gate_spread_m,
        prev_pf_state=context.history.prev_pf_state,
        velocity=epoch_state.velocity,
        dt=dt,
        observations_config=config_parts.observations,
        collect_diagnostics=run_config.collect_epoch_diagnostics,
    )

    apply_doppler_epoch_update(
        context.pf,
        epoch_state,
        context.stats,
        measurements=measurements,
        rover_weights=w,
        doppler_config=config_parts.doppler,
        gate_ess_ratio=gate_ess_ratio,
        gate_spread_m=gate_spread_m,
        wavelength_m=MUPF_L1_WAVELENGTH_M,
    )

    apply_position_epoch_updates(
        context.pf,
        epoch_state,
        context.stats,
        spp_position_ecef=spp_position_ecef,
        sat_ecef=sat_ecef,
        pseudoranges=pr,
        n_measurements=len(measurements),
        prev_estimate=context.history.prev_estimate,
        prev_pf_estimate=context.history.prev_pf_estimate,
        dt=dt,
        gate_ess_ratio=gate_ess_ratio,
        gate_spread_m=gate_spread_m,
        particle_filter_config=config_parts.particle_filter,
        motion_config=config_parts.motion,
        doppler_config=config_parts.doppler,
        tdcp_position_config=config_parts.tdcp_position_update,
    )

    return ForwardEpochUpdatesResult(spp_position_ecef=spp_position_ecef)
