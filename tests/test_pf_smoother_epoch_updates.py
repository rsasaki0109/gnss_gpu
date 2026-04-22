from types import SimpleNamespace

import numpy as np

import gnss_gpu.pf_smoother_epoch_updates as updates_mod
from gnss_gpu.pf_smoother_epoch_updates import apply_forward_epoch_updates
from pf_smoother_forward_helpers import make_forward_context, make_measurement_inputs


def test_apply_forward_epoch_updates_runs_measurement_update_sequence(monkeypatch):
    calls = []
    context = make_forward_context(calls)
    epoch_state = SimpleNamespace(velocity=np.array([0.1, 0.2, 0.3], dtype=np.float64))
    measurement_inputs = make_measurement_inputs()

    monkeypatch.setattr(
        updates_mod,
        "apply_widelane_dd_pseudorange_update",
        lambda *args, **kwargs: calls.append(("dd_pr", kwargs["tow"])),
    )
    monkeypatch.setattr(
        updates_mod,
        "apply_carrier_epoch_update",
        lambda *args, **kwargs: calls.append(("carrier", kwargs["tow"])),
    )
    monkeypatch.setattr(
        updates_mod,
        "apply_doppler_epoch_update",
        lambda *args, **kwargs: calls.append(("doppler", kwargs["gate_ess_ratio"])),
    )
    monkeypatch.setattr(
        updates_mod,
        "apply_position_epoch_updates",
        lambda *args, **kwargs: calls.append(("position", kwargs["dt"])),
    )

    sol_epoch = SimpleNamespace(
        position_ecef_m=np.array([10.0, 20.0, 30.0], dtype=np.float64),
    )
    result = apply_forward_epoch_updates(
        context,
        epoch_state,
        measurement_inputs,
        sol_epoch,
        [object(), object(), object(), object()],
        tow=123.4,
        dt=0.1,
    )

    assert [call[0] for call in calls] == ["dd_pr", "carrier", "doppler", "position"]
    np.testing.assert_allclose(result.spp_position_ecef, [10.0, 20.0, 30.0])
