from types import SimpleNamespace

import numpy as np

import gnss_gpu.dd_carrier_epoch_update as dd_carrier_epoch_update
from gnss_gpu.carrier_rescue import CarrierAnchorAttempt, CarrierFallbackAttempt
from gnss_gpu.dd_carrier_observation import DDCarrierObservationDecision
from gnss_gpu.dd_carrier_rescue_flow import (
    PostDDCarrierRescueDecision,
    WeakDDCarrierFallbackDecision,
)
from gnss_gpu.dd_carrier_rescue_gate import DDCarrierRescueGateDecision
from gnss_gpu.pf_smoother_config import (
    CarrierRescueConfig,
    DDCarrierConfig,
    MupfConfig,
)
from gnss_gpu.pf_smoother_epoch_state import create_epoch_forward_state
from gnss_gpu.pf_smoother_forward_stats import ForwardRunStats


class _FakeParticleFilter:
    def __init__(self):
        self.calls = []
        self.estimate_value = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float64)

    def estimate(self):
        self.calls.append(("estimate",))
        return self.estimate_value

    def resample_if_needed(self):
        self.calls.append(("resample",))

    def update_carrier_afv(self, sat_ecef, carrier_phase_cycles, *, weights, sigma_cycles):
        self.calls.append(
            ("carrier_afv", sat_ecef, carrier_phase_cycles, weights, sigma_cycles)
        )

    def update_dd_carrier_afv(self, dd_result, *, sigma_cycles):
        self.calls.append(("dd_carrier", dd_result, sigma_cycles))


def _observations_config(
    *,
    mupf_enabled=False,
    dd_carrier_enabled=False,
    carrier_rescue=None,
    dd_carrier=None,
):
    return SimpleNamespace(
        mupf=MupfConfig(
            enabled=mupf_enabled,
            sigma_cycles=0.05,
            snr_min=25.0,
            elev_min=0.15,
        ),
        dd_carrier=dd_carrier
        or DDCarrierConfig(enabled=dd_carrier_enabled, sigma_cycles=0.05),
        carrier_rescue=carrier_rescue or CarrierRescueConfig(),
    )


def test_apply_carrier_epoch_update_applies_mupf_afv_sequence(monkeypatch):
    carrier_obs = SimpleNamespace(
        sat_ecef=np.zeros((4, 3), dtype=np.float64),
        carrier_phase_cycles=np.arange(4.0),
        weights=np.ones(4, dtype=np.float64),
        sigma_sequence_cycles=(2.0, 0.5, 0.05),
    )
    monkeypatch.setattr(
        dd_carrier_epoch_update,
        "build_carrier_afv_observation",
        lambda *args, **kwargs: carrier_obs,
    )

    pf = _FakeParticleFilter()
    state = create_epoch_forward_state(0.75)
    stats = ForwardRunStats()

    result = dd_carrier_epoch_update.apply_carrier_epoch_update(
        pf,
        state,
        stats,
        dd_computer=None,
        carrier_bias_tracker={},
        tow=10.0,
        measurements=["m"],
        sat_ecef=np.zeros((4, 3), dtype=np.float64),
        pseudoranges=np.arange(4.0),
        spp_position_ecef=np.zeros(3, dtype=np.float64),
        gate_pf_estimate=np.zeros(4, dtype=np.float64),
        gate_ess_ratio=None,
        gate_spread_m=None,
        prev_pf_state=None,
        velocity=None,
        dt=0.1,
        observations_config=_observations_config(mupf_enabled=True),
        collect_diagnostics=False,
    )

    assert result.used_carrier_afv is True
    assert result.used_dd_carrier is False
    assert [call[0] for call in pf.calls] == [
        "resample",
        "carrier_afv",
        "resample",
        "carrier_afv",
        "resample",
        "carrier_afv",
    ]
    assert [call[-1] for call in pf.calls if call[0] == "carrier_afv"] == [
        2.0,
        0.5,
        0.05,
    ]
    assert stats.n_dd_used == 0


def test_apply_carrier_epoch_update_updates_dd_carrier_and_rescue_stats(monkeypatch):
    dd_result = SimpleNamespace(n_dd=4)

    monkeypatch.setattr(
        dd_carrier_epoch_update,
        "compute_dd_carrier_observation",
        lambda *args, **kwargs: DDCarrierObservationDecision(
            result=dd_result,
            gate_stats="gate_stats",
            input_pairs=5,
            raw_abs_afv_median_cycles=0.2,
            raw_abs_afv_max_cycles=0.4,
            gate_pairs_rejected=2,
            gate_epoch_skipped=True,
        ),
    )
    monkeypatch.setattr(
        dd_carrier_epoch_update,
        "evaluate_dd_carrier_rescue_gate",
        lambda *args, **kwargs: DDCarrierRescueGateDecision(result=dd_result),
    )

    def fake_post_rescue(*args, **kwargs):
        return PostDDCarrierRescueDecision(
            anchor_attempt=CarrierAnchorAttempt(used=True),
            fallback_attempt=CarrierFallbackAttempt(
                used=True,
                attempted_tracked=True,
                used_tracked=True,
            ),
        )

    monkeypatch.setattr(
        dd_carrier_epoch_update,
        "apply_post_dd_carrier_rescue",
        fake_post_rescue,
    )

    pf = _FakeParticleFilter()
    state = create_epoch_forward_state(0.75)
    stats = ForwardRunStats()

    result = dd_carrier_epoch_update.apply_carrier_epoch_update(
        pf,
        state,
        stats,
        dd_computer=object(),
        carrier_bias_tracker={},
        tow=10.0,
        measurements=[],
        sat_ecef=np.zeros((4, 3), dtype=np.float64),
        pseudoranges=np.arange(4.0),
        spp_position_ecef=np.zeros(3, dtype=np.float64),
        gate_pf_estimate=np.zeros(4, dtype=np.float64),
        gate_ess_ratio=0.5,
        gate_spread_m=2.0,
        prev_pf_state=np.zeros(4, dtype=np.float64),
        velocity=np.ones(3, dtype=np.float64),
        dt=0.1,
        observations_config=_observations_config(dd_carrier_enabled=True),
        collect_diagnostics=True,
    )

    assert result.used_dd_carrier is True
    assert result.used_carrier_anchor is True
    assert result.used_carrier_fallback is True
    assert state.dd_carrier_result is dd_result
    assert state.dd_gate_stats == "gate_stats"
    assert state.dd_cp_input_pairs == 5
    assert state.dd_cp_raw_abs_afv_median_cycles == 0.2
    assert state.dd_cp_sigma_cycles == 0.05
    assert stats.n_dd_used == 1
    assert stats.n_dd_gate_pairs_rejected == 2
    assert stats.n_dd_gate_epoch_skip == 1
    assert stats.n_carrier_anchor_used == 1
    assert stats.n_dd_fallback_undiff_used == 1
    assert stats.n_dd_fallback_tracked_attempted == 1
    assert stats.n_dd_fallback_tracked_used == 1
    assert ("dd_carrier", dd_result, 0.05) in pf.calls


def test_apply_carrier_epoch_update_replaces_weak_dd_with_fallback(monkeypatch):
    weak_result = SimpleNamespace(n_dd=3)

    monkeypatch.setattr(
        dd_carrier_epoch_update,
        "compute_dd_carrier_observation",
        lambda *args, **kwargs: DDCarrierObservationDecision(
            result=weak_result,
            gate_stats=None,
            input_pairs=3,
            raw_abs_afv_median_cycles=0.5,
            raw_abs_afv_max_cycles=0.6,
            gate_pairs_rejected=0,
            gate_epoch_skipped=False,
        ),
    )
    monkeypatch.setattr(
        dd_carrier_epoch_update,
        "evaluate_dd_carrier_rescue_gate",
        lambda *args, **kwargs: DDCarrierRescueGateDecision(
            result=weak_result,
            replace_weak_with_fallback=True,
        ),
    )
    fallback_attempt = CarrierFallbackAttempt(
        used=True,
        replaced_weak_dd=True,
    )
    monkeypatch.setattr(
        dd_carrier_epoch_update,
        "apply_weak_dd_carrier_fallback_replacement",
        lambda *args, **kwargs: WeakDDCarrierFallbackDecision(
            fallback_attempt=fallback_attempt,
            dd_carrier_result=None,
        ),
    )
    monkeypatch.setattr(
        dd_carrier_epoch_update,
        "apply_post_dd_carrier_rescue",
        lambda *args, **kwargs: PostDDCarrierRescueDecision(
            anchor_attempt=CarrierAnchorAttempt(),
            fallback_attempt=kwargs["fallback_attempt"],
        ),
    )

    pf = _FakeParticleFilter()
    state = create_epoch_forward_state(0.75)
    stats = ForwardRunStats()

    result = dd_carrier_epoch_update.apply_carrier_epoch_update(
        pf,
        state,
        stats,
        dd_computer=object(),
        carrier_bias_tracker={},
        tow=10.0,
        measurements=[],
        sat_ecef=np.zeros((4, 3), dtype=np.float64),
        pseudoranges=np.arange(4.0),
        spp_position_ecef=np.zeros(3, dtype=np.float64),
        gate_pf_estimate=np.zeros(4, dtype=np.float64),
        gate_ess_ratio=0.1,
        gate_spread_m=8.0,
        prev_pf_state=None,
        velocity=None,
        dt=0.1,
        observations_config=_observations_config(dd_carrier_enabled=True),
        collect_diagnostics=False,
    )

    assert result.used_dd_carrier is False
    assert result.used_carrier_fallback is True
    assert state.dd_carrier_result is None
    assert state.fallback_attempt is fallback_attempt
    assert stats.n_dd_skip == 1
    assert stats.n_dd_fallback_undiff_used == 1
    assert stats.n_dd_fallback_weak_dd_replaced == 1
    assert all(call[0] != "dd_carrier" for call in pf.calls)
