from types import SimpleNamespace

import numpy as np

import gnss_gpu.pf_smoother_epoch_record as record_mod
from gnss_gpu.pf_smoother_epoch_record import (
    ForwardEpochRecordResult,
    record_forward_epoch,
)
from gnss_gpu.pf_smoother_epoch_updates import ForwardEpochUpdatesResult
from pf_smoother_forward_helpers import make_forward_context, make_measurement_inputs


def test_record_forward_epoch_stores_when_smoother_enabled_and_finalizes(monkeypatch):
    calls = []
    context = make_forward_context(calls)
    epoch_state = SimpleNamespace()
    measurement_inputs = make_measurement_inputs()
    updates_result = ForwardEpochUpdatesResult(
        spp_position_ecef=np.array([10.0, 20.0, 30.0], dtype=np.float64)
    )
    finalize_result = SimpleNamespace(done=True)
    store_inputs = SimpleNamespace(stored=True)

    monkeypatch.setattr(
        record_mod,
        "append_smoother_epoch_store",
        lambda *args, **kwargs: calls.append(("store", kwargs["position_update_sigma"]))
        or store_inputs,
    )
    monkeypatch.setattr(
        record_mod,
        "finalize_forward_epoch",
        lambda *args, **kwargs: calls.append(("finalize", kwargs["run_name"]))
        or finalize_result,
    )

    result = record_forward_epoch(
        context,
        epoch_state,
        measurement_inputs,
        updates_result,
        [object(), object(), object(), object()],
        tow=123.4,
        dt=0.1,
    )

    assert calls == [("store", 2.0), ("finalize", "Odaiba")]
    assert result == ForwardEpochRecordResult(
        store_inputs=store_inputs,
        finalize_result=finalize_result,
    )
