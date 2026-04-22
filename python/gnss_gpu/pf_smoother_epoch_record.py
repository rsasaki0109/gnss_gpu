"""Store and finalize one PF smoother forward epoch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from gnss_gpu.carrier_rescue import MUPF_L1_WAVELENGTH_M
from gnss_gpu.pf_smoother_epoch_finalize import (
    ForwardEpochFinalizeResult,
    finalize_forward_epoch,
)
from gnss_gpu.pf_smoother_epoch_measurements import ForwardEpochMeasurementInputs
from gnss_gpu.pf_smoother_epoch_state import EpochForwardState
from gnss_gpu.pf_smoother_epoch_updates import ForwardEpochUpdatesResult
from gnss_gpu.pf_smoother_forward_context import PfSmootherForwardPassContext
from gnss_gpu.smoother_epoch_store import (
    SmootherEpochStoreInputs,
    append_smoother_epoch_store,
)


@dataclass(frozen=True)
class ForwardEpochRecordResult:
    store_inputs: SmootherEpochStoreInputs | None
    finalize_result: ForwardEpochFinalizeResult


def record_forward_epoch(
    context: PfSmootherForwardPassContext,
    epoch_state: EpochForwardState,
    measurement_inputs: ForwardEpochMeasurementInputs,
    updates_result: ForwardEpochUpdatesResult,
    measurements: list[Any],
    *,
    tow: float,
    dt: float,
) -> ForwardEpochRecordResult:
    run_config = context.run_config
    store_inputs = None
    if run_config.use_smoother:
        store_inputs = append_smoother_epoch_store(
            context.pf,
            context.buffers,
            sat_ecef=measurement_inputs.sat_ecef,
            pseudoranges=measurement_inputs.pseudoranges,
            weights=measurement_inputs.weights,
            spp_pos=updates_result.spp_position_ecef,
            epoch_state=epoch_state,
            dt=dt,
            position_update_sigma=run_config.position_update_sigma,
            carrier_anchor_sigma_m=run_config.carrier_anchor_sigma_m,
            carrier_afv_wavelength_m=MUPF_L1_WAVELENGTH_M,
            doppler_velocity_update_gain=run_config.doppler_velocity_update_gain,
            doppler_max_velocity_update_mps=run_config.doppler_max_velocity_update_mps,
            need_tdcp_motion=context.run_options.need_fgo_tdcp_motion,
        )

    finalize_result = finalize_forward_epoch(
        context.pf,
        context.buffers,
        context.history,
        context.stats,
        carrier_bias_tracker=context.observation_setup.carrier_bias_tracker,
        run_name=context.run_name,
        tow=tow,
        measurements=measurements,
        gt=context.dataset.gt,
        our_times=context.dataset.our_times,
        skip_valid_epochs=run_config.skip_valid_epochs,
        use_smoother=run_config.use_smoother,
        collect_epoch_diagnostics=run_config.collect_epoch_diagnostics,
        epoch_state=epoch_state,
        rbpf_velocity_kf=run_config.rbpf_velocity_kf,
        gate_ess_ratio=measurement_inputs.gate_ess_ratio,
        gate_spread_m=measurement_inputs.gate_spread_m,
        carrier_anchor_sigma_m=run_config.carrier_anchor_sigma_m,
        carrier_rescue_config=context.config_parts.observations.carrier_rescue,
    )

    return ForwardEpochRecordResult(
        store_inputs=store_inputs,
        finalize_result=finalize_result,
    )
