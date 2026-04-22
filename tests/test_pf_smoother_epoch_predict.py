from types import SimpleNamespace

import numpy as np

import gnss_gpu.pf_smoother_epoch_predict as predict_mod
from pf_smoother_forward_helpers import make_forward_context


def test_apply_forward_epoch_prediction_selects_sigma_and_calls_pf_predict(monkeypatch):
    calls = []
    context = make_forward_context(calls)
    epoch_state = SimpleNamespace(
        imu_stop_detected=True,
        used_tdcp=True,
        tdcp_rms=0.2,
        velocity=np.array([0.4, 0.5, 0.6], dtype=np.float64),
    )

    monkeypatch.setattr(
        predict_mod,
        "create_epoch_forward_state",
        lambda sigma: epoch_state,
    )

    def fake_predict_motion(*args, **kwargs):
        calls.append(("predict_motion", kwargs["tow"], kwargs["tow_key"], kwargs["dt"]))
        epoch_state.velocity = np.array([0.7, 0.8, 0.9], dtype=np.float64)

    monkeypatch.setattr(predict_mod, "apply_epoch_predict_motion", fake_predict_motion)
    monkeypatch.setattr(predict_mod, "select_predict_sigma", lambda *args, **kwargs: 0.33)

    sol_epoch = SimpleNamespace(
        time=SimpleNamespace(tow=123.4),
        position_ecef_m=np.array([10.0, 20.0, 30.0], dtype=np.float64),
    )

    result = predict_mod.apply_forward_epoch_prediction(
        context,
        sol_epoch,
        [object(), object(), object(), object()],
    )

    assert result.epoch_state is epoch_state
    assert result.tow == 123.4
    assert result.tow_key == 123.4
    assert result.dt == 0.1
    assert result.sigma_predict == 0.33
    assert calls[0] == ("predict_motion", 123.4, 123.4, 0.1)
    assert calls[1][0] == "pf_predict"
    assert calls[1][1]["sigma_pos"] == 0.33
    np.testing.assert_allclose(calls[1][1]["velocity"], [0.7, 0.8, 0.9])
