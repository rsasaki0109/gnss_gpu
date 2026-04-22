from types import SimpleNamespace

import numpy as np

import gnss_gpu.epoch_observation_inputs as epoch_inputs
from gnss_gpu.epoch_observation_inputs import build_epoch_observation_inputs
from gnss_gpu.pf_smoother_config import ObservationConfig, PfSmootherConfig


def _observations(**kwargs) -> ObservationConfig:
    return PfSmootherConfig(
        n_particles=10,
        sigma_pos=1.0,
        sigma_pr=3.0,
        position_update_sigma=None,
        predict_guide="spp",
        use_smoother=False,
        **kwargs,
    ).parts().observations


def _measurement(prn, sat, pr, weight=1.0):
    return SimpleNamespace(
        prn=prn,
        satellite_ecef=np.asarray(sat, dtype=np.float64),
        corrected_pseudorange=float(pr),
        weight=float(weight),
    )


def test_build_epoch_observation_inputs_extracts_arrays_and_delegates_helpers(monkeypatch):
    calls = {}

    def fake_anchor(rows, pseudoranges, spp_position, mupf, carrier_rescue):
        calls["anchor"] = {
            "rows": rows,
            "pseudoranges": pseudoranges,
            "spp_position": spp_position,
            "anchor_enabled": carrier_rescue.anchor_enabled,
        }
        return {(0, 1): {"row": True}}

    def fake_weighting(rows, sat_ecef, pseudoranges, weights, spp_position, pr_history, **kwargs):
        calls["weighting"] = {
            "rows": rows,
            "sat_ecef": sat_ecef,
            "pseudoranges": pseudoranges,
            "weights": weights,
            "kwargs": kwargs,
        }
        return np.asarray(weights, dtype=np.float64) * 0.5

    monkeypatch.setattr(epoch_inputs, "select_carrier_anchor_rows", fake_anchor)
    monkeypatch.setattr(epoch_inputs, "apply_pseudorange_weighting", fake_weighting)

    measurements = [
        _measurement(1, [10.0, 0.0, 0.0], 100.0, 0.8),
        _measurement(2, [20.0, 0.0, 0.0], 200.0, 0.6),
    ]
    obs = _observations(
        carrier_anchor=True,
        residual_downweight=True,
        residual_threshold=9.0,
        pr_accel_downweight=True,
        pr_accel_threshold=4.0,
    )

    prepared = build_epoch_observation_inputs(
        measurements,
        np.array([1.0, 2.0, 3.0]),
        {},
        obs,
    )

    np.testing.assert_allclose(prepared.sat_ecef, [[10.0, 0.0, 0.0], [20.0, 0.0, 0.0]])
    np.testing.assert_allclose(prepared.pseudoranges, [100.0, 200.0])
    np.testing.assert_allclose(prepared.weights, [0.4, 0.3])
    assert prepared.carrier_anchor_rows == {(0, 1): {"row": True}}
    assert calls["anchor"]["anchor_enabled"] is True
    assert calls["weighting"]["kwargs"]["residual_downweight"] is True
    assert calls["weighting"]["kwargs"]["residual_threshold"] == 9.0
    assert calls["weighting"]["kwargs"]["pr_accel_downweight"] is True
    assert calls["weighting"]["kwargs"]["pr_accel_threshold"] == 4.0


def test_build_epoch_observation_inputs_updates_pr_history_with_real_weighting():
    history = {7: [100.0, 101.0]}
    prepared = build_epoch_observation_inputs(
        [_measurement(7, [20_000_000.0, 0.0, 0.0], 120.0, 1.0)],
        np.array([2_000_000.0, 0.0, 0.0]),
        history,
        _observations(pr_accel_downweight=True, pr_accel_threshold=5.0),
    )

    assert prepared.weights[0] < 1.0
    assert history[7] == [101.0, 120.0]
    assert prepared.carrier_anchor_rows == {}
