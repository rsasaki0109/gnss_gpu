"""Observation input and gate-state preparation for one forward epoch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from gnss_gpu.epoch_gate_state import EpochGateState, compute_epoch_gate_state
from gnss_gpu.epoch_observation_inputs import (
    EpochObservationInputs,
    build_epoch_observation_inputs,
)
from gnss_gpu.pf_smoother_epoch_state import EpochForwardState
from gnss_gpu.pf_smoother_forward_context import PfSmootherForwardPassContext


@dataclass(frozen=True)
class ForwardEpochMeasurementInputs:
    observation_inputs: EpochObservationInputs
    gate_state: EpochGateState
    sat_ecef: np.ndarray
    pseudoranges: np.ndarray
    weights: np.ndarray
    gate_pf_estimate: np.ndarray | None
    gate_ess_ratio: float | None
    gate_spread_m: float | None


def prepare_forward_epoch_measurements(
    context: PfSmootherForwardPassContext,
    epoch_state: EpochForwardState,
    sol_epoch: Any,
    measurements: list[Any],
) -> ForwardEpochMeasurementInputs:
    observation_inputs = build_epoch_observation_inputs(
        measurements,
        np.array(sol_epoch.position_ecef_m[:3], dtype=np.float64),
        context.pr_history,
        context.config_parts.observations,
    )
    epoch_state.carrier_anchor_rows = observation_inputs.carrier_anchor_rows
    epoch_state.dd_pr_result = None
    epoch_state.dd_carrier_result = None

    gate_state = compute_epoch_gate_state(context.pf, context.config_parts)
    epoch_state.dd_pr_gate_scale = gate_state.dd_pr_gate_scale
    epoch_state.dd_cp_gate_scale = gate_state.dd_cp_gate_scale

    return ForwardEpochMeasurementInputs(
        observation_inputs=observation_inputs,
        gate_state=gate_state,
        sat_ecef=observation_inputs.sat_ecef,
        pseudoranges=observation_inputs.pseudoranges,
        weights=observation_inputs.weights,
        gate_pf_estimate=gate_state.pf_estimate,
        gate_ess_ratio=gate_state.ess_ratio,
        gate_spread_m=gate_state.spread_m,
    )
