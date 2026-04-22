from types import SimpleNamespace

import numpy as np

import gnss_gpu.pf_smoother_epoch_measurements as measurement_mod
from gnss_gpu.epoch_gate_state import EpochGateState
from gnss_gpu.epoch_observation_inputs import EpochObservationInputs
from gnss_gpu.pf_smoother_epoch_measurements import prepare_forward_epoch_measurements
from pf_smoother_forward_helpers import make_forward_context


def test_prepare_forward_epoch_measurements_sets_epoch_state_and_returns_gate_inputs(
    monkeypatch,
):
    calls = []
    context = make_forward_context(calls)
    epoch_state = SimpleNamespace(
        carrier_anchor_rows=None,
        dd_pr_result="stale_pr",
        dd_carrier_result="stale_cp",
        dd_pr_gate_scale=None,
        dd_cp_gate_scale=None,
    )
    obs_inputs = EpochObservationInputs(
        sat_ecef=np.arange(12, dtype=np.float64).reshape(4, 3),
        pseudoranges=np.arange(4, dtype=np.float64),
        weights=np.ones(4, dtype=np.float64),
        carrier_anchor_rows={"anchor": {"ok": True}},
    )
    gate_state = EpochGateState(
        pf_estimate=np.array([1.0, 2.0, 3.0], dtype=np.float64),
        ess_ratio=0.5,
        spread_m=2.5,
        dd_pr_gate_scale=1.25,
        dd_cp_gate_scale=0.75,
    )

    monkeypatch.setattr(
        measurement_mod,
        "build_epoch_observation_inputs",
        lambda *args, **kwargs: obs_inputs,
    )
    monkeypatch.setattr(
        measurement_mod,
        "compute_epoch_gate_state",
        lambda *args, **kwargs: gate_state,
    )

    sol_epoch = SimpleNamespace(
        position_ecef_m=np.array([10.0, 20.0, 30.0], dtype=np.float64),
    )
    result = prepare_forward_epoch_measurements(
        context,
        epoch_state,
        sol_epoch,
        [object(), object(), object(), object()],
    )

    assert result.observation_inputs is obs_inputs
    assert result.gate_state is gate_state
    assert result.gate_ess_ratio == 0.5
    assert result.gate_spread_m == 2.5
    np.testing.assert_allclose(result.gate_pf_estimate, [1.0, 2.0, 3.0])
    assert epoch_state.carrier_anchor_rows == {"anchor": {"ok": True}}
    assert epoch_state.dd_pr_result is None
    assert epoch_state.dd_carrier_result is None
    assert epoch_state.dd_pr_gate_scale == 1.25
    assert epoch_state.dd_cp_gate_scale == 0.75
