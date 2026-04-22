import numpy as np

import gnss_gpu.doppler_epoch_update as doppler_epoch_update
from gnss_gpu.doppler_update import DopplerUpdateDecision
from gnss_gpu.pf_smoother_config import DopplerConfig
from gnss_gpu.pf_smoother_epoch_state import create_epoch_forward_state
from gnss_gpu.pf_smoother_forward_stats import ForwardRunStats


class _FakeParticleFilter:
    def __init__(self):
        self.calls = []

    def update_doppler_kf(
        self,
        sat_ecef,
        sat_vel,
        doppler_hz,
        *,
        weights,
        wavelength,
        sigma_mps,
    ):
        self.calls.append(
            ("kf", sat_ecef, sat_vel, doppler_hz, weights, wavelength, sigma_mps)
        )

    def update_doppler(
        self,
        sat_ecef,
        sat_vel,
        doppler_hz,
        *,
        weights,
        wavelength,
        sigma_mps,
        velocity_update_gain,
        max_velocity_update_mps,
    ):
        self.calls.append(
            (
                "pp",
                sat_ecef,
                sat_vel,
                doppler_hz,
                weights,
                wavelength,
                sigma_mps,
                velocity_update_gain,
                max_velocity_update_mps,
            )
        )


def _update_payload():
    return {
        "sat_ecef": np.zeros((3, 3), dtype=np.float64),
        "sat_vel": np.ones((3, 3), dtype=np.float64),
        "doppler_hz": np.asarray([-1000.0, -1100.0, -1200.0], dtype=np.float64),
        "weights": np.asarray([0.5, 0.6, 0.7], dtype=np.float64),
        "wavelength_m": 0.19,
    }


def test_apply_doppler_epoch_update_applies_per_particle_update(monkeypatch):
    payload = _update_payload()

    monkeypatch.setattr(
        doppler_epoch_update,
        "build_doppler_update_decision",
        lambda *args, **kwargs: DopplerUpdateDecision(
            update=payload,
            sigma_mps=0.75,
            use_kf=False,
        ),
    )
    pf = _FakeParticleFilter()
    state = create_epoch_forward_state(0.75)
    stats = ForwardRunStats()

    result = doppler_epoch_update.apply_doppler_epoch_update(
        pf,
        state,
        stats,
        measurements=["m"],
        rover_weights=np.ones(3),
        doppler_config=DopplerConfig(
            per_particle=True,
            sigma_mps=0.75,
            velocity_update_gain=0.3,
            max_velocity_update_mps=9.0,
        ),
        gate_ess_ratio=None,
        gate_spread_m=None,
        wavelength_m=0.19,
    )

    assert result.used is True
    assert result.used_kf is False
    assert state.doppler_update_epoch is payload
    assert state.doppler_sigma_epoch == 0.75
    assert stats.n_doppler_pp_used == 1
    assert pf.calls[0][0] == "pp"
    assert pf.calls[0][-2:] == (0.3, 9.0)


def test_apply_doppler_epoch_update_applies_kf_update(monkeypatch):
    payload = _update_payload()

    monkeypatch.setattr(
        doppler_epoch_update,
        "build_doppler_update_decision",
        lambda *args, **kwargs: DopplerUpdateDecision(
            update=payload,
            sigma_mps=0.5,
            use_kf=True,
            gate_reason="ok",
        ),
    )
    pf = _FakeParticleFilter()
    state = create_epoch_forward_state(0.75)
    stats = ForwardRunStats()

    result = doppler_epoch_update.apply_doppler_epoch_update(
        pf,
        state,
        stats,
        measurements=["m"],
        rover_weights=np.ones(3),
        doppler_config=DopplerConfig(
            rbpf_velocity_kf=True,
            rbpf_doppler_sigma=0.5,
        ),
        gate_ess_ratio=0.2,
        gate_spread_m=2.0,
        wavelength_m=0.19,
    )

    assert result.used is True
    assert result.used_kf is True
    assert result.gate_reason == "ok"
    assert state.doppler_kf_gate_reason == "ok"
    assert stats.n_doppler_kf_used == 1
    assert pf.calls[0][0] == "kf"
    assert pf.calls[0][-2:] == (0.19, 0.5)


def test_apply_doppler_epoch_update_records_kf_gate_skip(monkeypatch):
    monkeypatch.setattr(
        doppler_epoch_update,
        "build_doppler_update_decision",
        lambda *args, **kwargs: DopplerUpdateDecision(
            update=None,
            sigma_mps=None,
            use_kf=True,
            gate_reason="min_dd_pairs",
            skipped=True,
            gate_skipped=True,
        ),
    )
    pf = _FakeParticleFilter()
    state = create_epoch_forward_state(0.75)
    stats = ForwardRunStats()

    result = doppler_epoch_update.apply_doppler_epoch_update(
        pf,
        state,
        stats,
        measurements=[],
        rover_weights=np.ones(0),
        doppler_config=DopplerConfig(rbpf_velocity_kf=True),
        gate_ess_ratio=0.1,
        gate_spread_m=8.0,
        wavelength_m=0.19,
    )

    assert result.used is False
    assert result.skipped is True
    assert result.gate_skipped is True
    assert result.gate_reason == "min_dd_pairs"
    assert state.doppler_kf_gate_reason == "min_dd_pairs"
    assert stats.n_doppler_kf_skip == 1
    assert stats.n_doppler_kf_gate_skip == 1
    assert pf.calls == []


def test_apply_doppler_epoch_update_noops_when_disabled(monkeypatch):
    calls = {"build": 0}

    def fake_build(*args, **kwargs):
        calls["build"] += 1
        return DopplerUpdateDecision(update=None, sigma_mps=None, use_kf=False)

    monkeypatch.setattr(doppler_epoch_update, "build_doppler_update_decision", fake_build)
    pf = _FakeParticleFilter()
    state = create_epoch_forward_state(0.75)
    stats = ForwardRunStats()

    result = doppler_epoch_update.apply_doppler_epoch_update(
        pf,
        state,
        stats,
        measurements=[],
        rover_weights=np.ones(0),
        doppler_config=DopplerConfig(),
        gate_ess_ratio=None,
        gate_spread_m=None,
        wavelength_m=0.19,
    )

    assert result.used is False
    assert result.skipped is False
    assert calls["build"] == 0
    assert pf.calls == []
    assert stats.n_doppler_pp_used == 0
