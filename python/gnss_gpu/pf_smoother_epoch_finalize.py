"""Forward epoch finalization for PF smoother evaluations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from gnss_gpu.carrier_bias_tracker_update import update_carrier_bias_tracker_after_epoch
from gnss_gpu.carrier_rescue import CarrierBiasState
from gnss_gpu.pf_smoother_alignment import (
    ForwardAlignmentResult,
    append_forward_alignment,
)
from gnss_gpu.pf_smoother_epoch_history import ForwardEpochHistory
from gnss_gpu.pf_smoother_epoch_state import EpochForwardState
from gnss_gpu.pf_smoother_forward_stats import ForwardRunStats
from gnss_gpu.pf_smoother_runtime import ForwardRunBuffers


@dataclass(frozen=True)
class ForwardEpochFinalizeResult:
    pf_estimate_now: np.ndarray
    pf_state_now: np.ndarray
    carrier_anchor_propagated_rows: int
    alignment: ForwardAlignmentResult


def finalize_forward_epoch(
    pf: Any,
    buffers: ForwardRunBuffers,
    history: ForwardEpochHistory,
    stats: ForwardRunStats,
    *,
    carrier_bias_tracker: dict[tuple[int, int], CarrierBiasState],
    tow: float,
    measurements: Iterable[Any],
    run_name: str,
    gt: np.ndarray,
    our_times: np.ndarray,
    skip_valid_epochs: int,
    use_smoother: bool,
    collect_epoch_diagnostics: bool,
    epoch_state: EpochForwardState,
    rbpf_velocity_kf: bool,
    gate_ess_ratio: float,
    gate_spread_m: float,
    carrier_anchor_sigma_m: float,
    carrier_rescue_config: Any,
) -> ForwardEpochFinalizeResult:
    current_measurements = list(measurements)
    pf_state_now = np.asarray(pf.estimate(), dtype=np.float64).copy()
    pf_estimate_now = pf_state_now[:3].copy()

    propagated_rows = update_carrier_bias_tracker_after_epoch(
        carrier_bias_tracker,
        epoch_state.carrier_anchor_rows,
        epoch_state.anchor_attempt,
        pf_state_now,
        tow,
        epoch_state.dd_carrier_result,
        carrier_rescue_config,
    )
    stats.n_carrier_anchor_propagated += propagated_rows

    alignment = append_forward_alignment(
        buffers,
        run_name=run_name,
        tow=tow,
        pf_estimate_now=pf_estimate_now,
        gt=gt,
        our_times=our_times,
        measurements=current_measurements,
        epoch_index=history.epochs_done,
        skip_valid_epochs=skip_valid_epochs,
        use_smoother=use_smoother,
        collect_epoch_diagnostics=collect_epoch_diagnostics,
        epoch_state=epoch_state,
        rbpf_velocity_kf=rbpf_velocity_kf,
        gate_ess_ratio=gate_ess_ratio,
        gate_spread_m=gate_spread_m,
        carrier_anchor_sigma_m=carrier_anchor_sigma_m,
    )

    history.advance(
        tow=tow,
        measurements=current_measurements,
        pf_estimate_now=pf_estimate_now,
        pf_state=pf_state_now,
    )

    return ForwardEpochFinalizeResult(
        pf_estimate_now=pf_estimate_now,
        pf_state_now=pf_state_now,
        carrier_anchor_propagated_rows=int(propagated_rows),
        alignment=alignment,
    )
