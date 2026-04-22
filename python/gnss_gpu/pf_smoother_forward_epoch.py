"""Single-epoch forward-pass processing for PF smoother evaluations."""

from __future__ import annotations

from typing import Any

from gnss_gpu.pf_smoother_epoch_measurements import prepare_forward_epoch_measurements
from gnss_gpu.pf_smoother_epoch_predict import apply_forward_epoch_prediction
from gnss_gpu.pf_smoother_epoch_record import record_forward_epoch
from gnss_gpu.pf_smoother_epoch_updates import apply_forward_epoch_updates
from gnss_gpu.pf_smoother_forward_context import PfSmootherForwardPassContext


def process_pf_smoother_forward_epoch(
    context: PfSmootherForwardPassContext,
    sol_epoch: Any,
    measurements: list[Any],
) -> None:
    predict_result = apply_forward_epoch_prediction(context, sol_epoch, measurements)
    epoch_state = predict_result.epoch_state
    tow = predict_result.tow
    dt = predict_result.dt

    measurement_inputs = prepare_forward_epoch_measurements(
        context,
        epoch_state,
        sol_epoch,
        measurements,
    )

    updates_result = apply_forward_epoch_updates(
        context,
        epoch_state,
        measurement_inputs,
        sol_epoch,
        measurements,
        tow=tow,
        dt=dt,
    )

    record_forward_epoch(
        context,
        epoch_state,
        measurement_inputs,
        updates_result,
        measurements=measurements,
        tow=tow,
        dt=dt,
    )
