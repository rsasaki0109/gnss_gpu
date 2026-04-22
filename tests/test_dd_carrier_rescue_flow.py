from dataclasses import dataclass

import numpy as np

import gnss_gpu.dd_carrier_rescue_flow as rescue_flow
from gnss_gpu.carrier_rescue import CarrierAnchorAttempt, CarrierFallbackAttempt
from gnss_gpu.pf_smoother_config import CarrierRescueConfig, MupfConfig


@dataclass
class _DummyDDResult:
    n_dd: int


class _DummyPF:
    def __init__(self, estimate=None):
        self.estimate_value = np.asarray(
            [1.0, 2.0, 3.0, 4.0] if estimate is None else estimate,
            dtype=np.float64,
        )

    def estimate(self):
        return self.estimate_value


def test_apply_weak_dd_carrier_fallback_replacement_applies_prepared_attempt(monkeypatch):
    calls = {}

    def fake_prepare(*args, **kwargs):
        calls["prepare_kwargs"] = kwargs
        return CarrierFallbackAttempt(
            afv={"n_sat": 4},
            sigma_cycles=0.2,
            replaced_weak_dd=True,
        )

    def fake_apply(pf, attempt):
        calls["applied"] = True
        attempt.used = True
        return attempt

    monkeypatch.setattr(rescue_flow, "_prepare_dd_carrier_undiff_fallback", fake_prepare)
    monkeypatch.setattr(rescue_flow, "_apply_dd_carrier_undiff_fallback", fake_apply)

    decision = rescue_flow.apply_weak_dd_carrier_fallback_replacement(
        _DummyPF(),
        [],
        np.zeros((0, 3), dtype=np.float64),
        np.zeros(0, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        {},
        {},
        np.zeros(4, dtype=np.float64),
        10.0,
        _DummyDDResult(4),
        MupfConfig(enabled=False, snr_min=28.0, elev_min=0.2),
        CarrierRescueConfig(
            fallback_undiff=True,
            fallback_sigma_cycles=0.2,
            fallback_min_sats=4,
            fallback_prefer_tracked=True,
            anchor_max_age_s=2.0,
        ),
    )

    assert decision.dd_carrier_result is None
    assert decision.fallback_attempt.used
    assert calls["applied"] is True
    assert calls["prepare_kwargs"]["enabled"] is True
    assert calls["prepare_kwargs"]["snr_min"] == 28.0
    assert calls["prepare_kwargs"]["weak_dd_max_pairs"] == 4
    assert calls["prepare_kwargs"]["max_age_s"] == 2.0


def test_apply_weak_dd_carrier_fallback_replacement_keeps_result_when_not_ready(monkeypatch):
    dd_result = _DummyDDResult(4)

    monkeypatch.setattr(
        rescue_flow,
        "_prepare_dd_carrier_undiff_fallback",
        lambda *args, **kwargs: CarrierFallbackAttempt(),
    )

    decision = rescue_flow.apply_weak_dd_carrier_fallback_replacement(
        _DummyPF(),
        [],
        np.zeros((0, 3), dtype=np.float64),
        np.zeros(0, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        {},
        {},
        np.zeros(4, dtype=np.float64),
        10.0,
        dd_result,
        MupfConfig(enabled=False),
        CarrierRescueConfig(fallback_undiff=True),
    )

    assert decision.dd_carrier_result is dd_result
    assert not decision.fallback_attempt.used


def test_apply_post_dd_carrier_rescue_attempts_anchor_then_fallback(monkeypatch):
    calls = {}

    def fake_anchor(*args, **kwargs):
        calls["anchor_kwargs"] = kwargs
        return CarrierAnchorAttempt(
            used=True,
            state=np.asarray(args[3], dtype=np.float64),
        )

    def fake_fallback(*args, **kwargs):
        calls["fallback_kwargs"] = kwargs
        return CarrierFallbackAttempt()

    monkeypatch.setattr(rescue_flow, "_attempt_carrier_anchor_pseudorange_update", fake_anchor)
    monkeypatch.setattr(rescue_flow, "_attempt_dd_carrier_undiff_fallback", fake_fallback)

    decision = rescue_flow.apply_post_dd_carrier_rescue(
        _DummyPF(),
        [],
        np.zeros((0, 3), dtype=np.float64),
        np.zeros(0, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        {},
        {(0, 1): {}},
        np.ones(4, dtype=np.float64),
        prev_pf_state=np.zeros(4, dtype=np.float64),
        velocity=np.ones(3, dtype=np.float64),
        dt=0.1,
        tow=10.0,
        dd_carrier_result=None,
        mupf=MupfConfig(enabled=False, snr_min=27.0, elev_min=0.25),
        config=CarrierRescueConfig(
            anchor_enabled=True,
            anchor_sigma_m=0.3,
            fallback_undiff=True,
        ),
    )

    assert decision.anchor_attempt.used
    assert calls["anchor_kwargs"]["enabled"] is True
    assert calls["anchor_kwargs"]["sigma_m"] == 0.3
    assert calls["fallback_kwargs"]["used_carrier_anchor"] is True
    assert calls["fallback_kwargs"]["snr_min"] == 27.0
    assert calls["fallback_kwargs"]["elev_min"] == 0.25


def test_apply_post_dd_carrier_rescue_sets_anchor_state_after_existing_fallback(monkeypatch):
    calls = {"anchor": 0, "fallback": 0}

    monkeypatch.setattr(
        rescue_flow,
        "_attempt_carrier_anchor_pseudorange_update",
        lambda *args, **kwargs: calls.__setitem__("anchor", calls["anchor"] + 1),
    )
    monkeypatch.setattr(
        rescue_flow,
        "_attempt_dd_carrier_undiff_fallback",
        lambda *args, **kwargs: calls.__setitem__("fallback", calls["fallback"] + 1),
    )

    fallback_attempt = CarrierFallbackAttempt(used=True)
    decision = rescue_flow.apply_post_dd_carrier_rescue(
        _DummyPF(),
        [],
        np.zeros((0, 3), dtype=np.float64),
        np.zeros(0, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        {},
        {(0, 1): {}},
        np.asarray([9.0, 8.0, 7.0, 6.0], dtype=np.float64),
        prev_pf_state=None,
        velocity=None,
        dt=0.0,
        tow=10.0,
        dd_carrier_result=None,
        mupf=MupfConfig(enabled=False),
        config=CarrierRescueConfig(anchor_enabled=True),
        fallback_attempt=fallback_attempt,
    )

    assert calls == {"anchor": 0, "fallback": 0}
    assert decision.fallback_attempt is fallback_attempt
    np.testing.assert_allclose(decision.anchor_attempt.state, [9.0, 8.0, 7.0, 6.0])
