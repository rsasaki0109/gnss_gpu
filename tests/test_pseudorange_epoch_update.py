from types import SimpleNamespace

import numpy as np

import gnss_gpu.pseudorange_epoch_update as pseudorange_epoch_update
from gnss_gpu.dd_pseudorange_observation import DDPseudorangeObservationDecision
from gnss_gpu.pf_smoother_epoch_state import create_epoch_forward_state
from gnss_gpu.pf_smoother_forward_stats import ForwardRunStats
from gnss_gpu.pseudorange_epoch_update import apply_widelane_dd_pseudorange_update
from gnss_gpu.widelane_observation import WidelaneObservationDecision


class _FakeParticleFilter:
    def __init__(self):
        self.calls = []

    def update_dd_pseudorange(self, dd_result, *, sigma_pr):
        self.calls.append(("dd_pr", dd_result, sigma_pr))

    def correct_clock_bias(self, sat_ecef, pseudoranges):
        self.calls.append(("clock", sat_ecef, pseudoranges))

    def update_gmm(self, sat_ecef, pseudoranges, *, weights, w_los, mu_nlos, sigma_nlos):
        self.calls.append(
            ("gmm", sat_ecef, pseudoranges, weights, w_los, mu_nlos, sigma_nlos)
        )

    def update(self, sat_ecef, pseudoranges, *, weights):
        self.calls.append(("pr", sat_ecef, pseudoranges, weights))


def _observations_config(*, widelane=True, dd_pr=True, use_gmm=False):
    return SimpleNamespace(
        widelane=SimpleNamespace(enabled=widelane),
        dd_pseudorange=SimpleNamespace(enabled=dd_pr),
        robust=SimpleNamespace(
            use_gmm=use_gmm,
            gmm_w_los=0.7,
            gmm_mu_nlos=15.0,
            gmm_sigma_nlos=30.0,
        ),
    )


def test_apply_widelane_dd_pseudorange_update_uses_widelane_result(monkeypatch):
    dd_result = SimpleNamespace(n_dd=3)

    def fake_widelane(*args, **kwargs):
        return WidelaneObservationDecision(
            dd_pseudorange_result=dd_result,
            stats="wl_stats",
            gate_info={"reason": "ok", "pair_rejected": 2},
            input_pairs=5,
            fixed_pairs=4,
            fix_rate=0.8,
            dd_sigma_m=0.12,
            used=True,
            gate_pair_rejected=2,
        )

    def fake_dd_pr(*args, **kwargs):
        assert kwargs["existing_result"] is dd_result
        assert kwargs["existing_input_pairs"] == 3
        return DDPseudorangeObservationDecision(
            result=dd_result,
            gate_stats="gate_stats",
            input_pairs=3,
            raw_abs_res_median_m=1.0,
            raw_abs_res_max_m=2.0,
            gate_pairs_rejected=1,
            gate_epoch_skipped=False,
        )

    monkeypatch.setattr(pseudorange_epoch_update, "compute_widelane_observation", fake_widelane)
    monkeypatch.setattr(
        pseudorange_epoch_update,
        "compute_dd_pseudorange_observation",
        fake_dd_pr,
    )
    pf = _FakeParticleFilter()
    state = create_epoch_forward_state(0.5)
    stats = ForwardRunStats()

    result = apply_widelane_dd_pseudorange_update(
        pf,
        state,
        stats,
        dd_pr_computer=object(),
        wl_computer=object(),
        tow=10.0,
        measurements=["m"],
        sat_ecef=np.zeros((4, 3)),
        pseudoranges=np.zeros(4),
        rover_weights=np.ones(4),
        pf_estimate=np.zeros(4),
        gate_pf_estimate=np.zeros(4),
        gate_spread_m=2.0,
        observations_config=_observations_config(),
        collect_diagnostics=True,
    )

    assert result.used_dd_pseudorange is True
    assert result.used_widelane is True
    assert state.used_widelane_epoch is True
    assert state.wl_stats == "wl_stats"
    assert state.wl_input_pairs == 5
    assert state.wl_fixed_pairs == 4
    assert state.dd_pr_result is dd_result
    assert state.dd_pr_sigma_epoch == 0.12
    assert state.dd_pr_gate_stats == "gate_stats"
    assert stats.n_wl_used == 1
    assert stats.n_wl_candidate_pairs == 5
    assert stats.n_wl_fixed_pairs == 4
    assert stats.n_wl_gate_pair_rejected == 2
    assert stats.n_dd_pr_used == 1
    assert stats.n_dd_pr_gate_pairs_rejected == 1
    assert pf.calls == [("dd_pr", dd_result, 0.12)]


def test_apply_widelane_dd_pseudorange_update_falls_back_to_pr_update(monkeypatch):
    monkeypatch.setattr(
        pseudorange_epoch_update,
        "compute_dd_pseudorange_observation",
        lambda *args, **kwargs: DDPseudorangeObservationDecision(
            result=None,
            gate_stats=None,
            input_pairs=0,
            raw_abs_res_median_m=None,
            raw_abs_res_max_m=None,
            gate_pairs_rejected=0,
            gate_epoch_skipped=True,
        ),
    )
    pf = _FakeParticleFilter()
    state = create_epoch_forward_state(0.5)
    stats = ForwardRunStats()
    sat_ecef = np.zeros((4, 3))
    pseudoranges = np.arange(4.0)
    weights = np.ones(4)

    result = apply_widelane_dd_pseudorange_update(
        pf,
        state,
        stats,
        dd_pr_computer=None,
        wl_computer=None,
        tow=10.0,
        measurements=[],
        sat_ecef=sat_ecef,
        pseudoranges=pseudoranges,
        rover_weights=weights,
        pf_estimate=np.zeros(4),
        gate_pf_estimate=np.zeros(4),
        gate_spread_m=2.0,
        observations_config=_observations_config(widelane=False, dd_pr=True),
        collect_diagnostics=False,
    )

    assert result.used_dd_pseudorange is False
    assert result.used_gmm_fallback is False
    assert stats.n_dd_pr_skip == 1
    assert stats.n_dd_pr_gate_epoch_skip == 1
    assert pf.calls[0] == ("clock", sat_ecef, pseudoranges)
    assert pf.calls[1] == ("pr", sat_ecef, pseudoranges, weights)


def test_apply_widelane_dd_pseudorange_update_uses_gmm_fallback(monkeypatch):
    monkeypatch.setattr(
        pseudorange_epoch_update,
        "compute_dd_pseudorange_observation",
        lambda *args, **kwargs: DDPseudorangeObservationDecision(
            result=None,
            gate_stats=None,
            input_pairs=0,
            raw_abs_res_median_m=None,
            raw_abs_res_max_m=None,
            gate_pairs_rejected=0,
            gate_epoch_skipped=False,
        ),
    )
    pf = _FakeParticleFilter()
    state = create_epoch_forward_state(0.5)
    stats = ForwardRunStats()

    result = apply_widelane_dd_pseudorange_update(
        pf,
        state,
        stats,
        dd_pr_computer=None,
        wl_computer=None,
        tow=10.0,
        measurements=[],
        sat_ecef=np.zeros((4, 3)),
        pseudoranges=np.arange(4.0),
        rover_weights=np.ones(4),
        pf_estimate=np.zeros(4),
        gate_pf_estimate=np.zeros(4),
        gate_spread_m=2.0,
        observations_config=_observations_config(widelane=False, dd_pr=False, use_gmm=True),
        collect_diagnostics=False,
    )

    assert result.used_gmm_fallback is True
    assert stats.n_dd_pr_skip == 0
    assert pf.calls[0][0] == "clock"
    assert pf.calls[1][0] == "gmm"
    assert pf.calls[1][4:] == (0.7, 15.0, 30.0)
