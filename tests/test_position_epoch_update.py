import numpy as np

import gnss_gpu.position_epoch_update as position_epoch_update
from gnss_gpu.imu_position_update import ImuTightPositionUpdateDecision
from gnss_gpu.motion_position_update import MotionPositionUpdateDecision
from gnss_gpu.pf_smoother_config import (
    DopplerConfig,
    MotionConfig,
    ParticleFilterRuntimeConfig,
    TdcpPositionUpdateConfig,
)
from gnss_gpu.pf_smoother_epoch_state import create_epoch_forward_state
from gnss_gpu.pf_smoother_forward_stats import ForwardRunStats
from gnss_gpu.tdcp_motion import TdcpPositionUpdateDecision


class _FakeParticleFilter:
    def __init__(self):
        self.calls = []

    def position_update(self, ref_ecef, *, sigma_pos):
        self.calls.append((np.asarray(ref_ecef, dtype=np.float64), sigma_pos))


def _pf_config(position_update_sigma=None):
    return ParticleFilterRuntimeConfig(
        n_particles=100,
        sigma_pos=1.0,
        sigma_pr=5.0,
        position_update_sigma=position_update_sigma,
        use_smoother=False,
    )


def test_apply_position_epoch_updates_applies_all_enabled_updates(monkeypatch):
    monkeypatch.setattr(
        position_epoch_update,
        "evaluate_imu_tight_position_update",
        lambda *args, **kwargs: ImuTightPositionUpdateDecision(
            apply_update=True,
            predicted_position=np.asarray([11.0, 12.0, 13.0], dtype=np.float64),
            sigma_pos=3.0,
            residual_rms=1.0,
            reason="ok",
        ),
    )
    monkeypatch.setattr(
        position_epoch_update,
        "evaluate_motion_position_update",
        lambda *args, **kwargs: MotionPositionUpdateDecision(
            apply_update=True,
            predicted_position=np.asarray([21.0, 22.0, 23.0], dtype=np.float64),
            reason="ok",
        ),
    )

    tdcp_kwargs = {}

    def fake_tdcp(*args, **kwargs):
        tdcp_kwargs.update(kwargs)
        return TdcpPositionUpdateDecision(
            apply_update=True,
            predicted_position=np.asarray([31.0, 32.0, 33.0], dtype=np.float64),
            gate_reason="ok",
            gate_skipped=False,
        )

    monkeypatch.setattr(position_epoch_update, "evaluate_tdcp_position_update", fake_tdcp)

    pf = _FakeParticleFilter()
    state = create_epoch_forward_state(0.75)
    state.imu_velocity = np.ones(3, dtype=np.float64)
    state.velocity = np.ones(3, dtype=np.float64)
    state.tdcp_pu_velocity = np.ones(3, dtype=np.float64)
    state.tdcp_pu_rms = 0.4
    stats = ForwardRunStats()

    result = position_epoch_update.apply_position_epoch_updates(
        pf,
        state,
        stats,
        spp_position_ecef=np.asarray([2_000_000.0, 0.0, 0.0], dtype=np.float64),
        sat_ecef=np.zeros((4, 3), dtype=np.float64),
        pseudoranges=np.zeros(4, dtype=np.float64),
        n_measurements=8,
        prev_estimate=np.zeros(3, dtype=np.float64),
        prev_pf_estimate=np.zeros(3, dtype=np.float64),
        dt=0.1,
        gate_ess_ratio=0.2,
        gate_spread_m=3.0,
        particle_filter_config=_pf_config(position_update_sigma=1.9),
        motion_config=MotionConfig(predict_guide="spp", imu_tight_coupling=True),
        doppler_config=DopplerConfig(position_update=True, pu_sigma=5.0),
        tdcp_position_config=TdcpPositionUpdateConfig(
            enabled=True,
            sigma=0.5,
            gate_logic="all",
            gate_stop_mode="moving",
        ),
    )

    assert result.used_spp_position_update is True
    assert result.used_imu_tight is True
    assert result.used_doppler_position_update is True
    assert result.used_tdcp_position_update is True
    assert result.tdcp_gate_reason == "ok"
    assert [call[1] for call in pf.calls] == [1.9, 3.0, 5.0, 0.5]
    assert stats.n_imu_tight_used == 1
    assert stats.n_tdcp_pu_used == 1
    assert state.used_imu_tight_epoch is True
    assert state.used_tdcp_pu_epoch is True
    assert tdcp_kwargs["gate_logic"] == "all"
    assert tdcp_kwargs["gate_stop_mode"] == "moving"


def test_apply_position_epoch_updates_records_imu_and_tdcp_gate_skips(monkeypatch):
    monkeypatch.setattr(
        position_epoch_update,
        "evaluate_imu_tight_position_update",
        lambda *args, **kwargs: ImuTightPositionUpdateDecision(
            apply_update=False,
            predicted_position=None,
            sigma_pos=None,
            residual_rms=None,
            reason="no_imu_velocity",
        ),
    )
    monkeypatch.setattr(
        position_epoch_update,
        "evaluate_tdcp_position_update",
        lambda *args, **kwargs: TdcpPositionUpdateDecision(
            apply_update=False,
            predicted_position=None,
            gate_reason="gate_no_match",
            gate_skipped=True,
        ),
    )

    pf = _FakeParticleFilter()
    state = create_epoch_forward_state(0.75)
    stats = ForwardRunStats()

    result = position_epoch_update.apply_position_epoch_updates(
        pf,
        state,
        stats,
        spp_position_ecef=np.zeros(3, dtype=np.float64),
        sat_ecef=np.zeros((4, 3), dtype=np.float64),
        pseudoranges=np.zeros(4, dtype=np.float64),
        n_measurements=4,
        prev_estimate=np.zeros(3, dtype=np.float64),
        prev_pf_estimate=np.zeros(3, dtype=np.float64),
        dt=0.1,
        gate_ess_ratio=None,
        gate_spread_m=None,
        particle_filter_config=_pf_config(),
        motion_config=MotionConfig(predict_guide="spp", imu_tight_coupling=True),
        doppler_config=DopplerConfig(position_update=False),
        tdcp_position_config=TdcpPositionUpdateConfig(enabled=True),
    )

    assert result.used_imu_tight is False
    assert result.used_tdcp_position_update is False
    assert result.tdcp_gate_reason == "gate_no_match"
    assert stats.n_imu_tight_skip == 1
    assert stats.n_tdcp_pu_gate_skip == 1
    assert stats.n_tdcp_pu_skip == 1
    assert pf.calls == []


def test_apply_position_epoch_updates_counts_tdcp_outer_skip_when_disabled(monkeypatch):
    calls = {"tdcp": 0}

    def fake_tdcp(*args, **kwargs):
        calls["tdcp"] += 1
        return TdcpPositionUpdateDecision(False, None, "unexpected")

    monkeypatch.setattr(position_epoch_update, "evaluate_tdcp_position_update", fake_tdcp)

    pf = _FakeParticleFilter()
    state = create_epoch_forward_state(0.75)
    stats = ForwardRunStats()

    result = position_epoch_update.apply_position_epoch_updates(
        pf,
        state,
        stats,
        spp_position_ecef=np.zeros(3, dtype=np.float64),
        sat_ecef=np.zeros((4, 3), dtype=np.float64),
        pseudoranges=np.zeros(4, dtype=np.float64),
        n_measurements=4,
        prev_estimate=np.zeros(3, dtype=np.float64),
        prev_pf_estimate=None,
        dt=0.1,
        gate_ess_ratio=None,
        gate_spread_m=None,
        particle_filter_config=_pf_config(),
        motion_config=MotionConfig(predict_guide="spp"),
        doppler_config=DopplerConfig(),
        tdcp_position_config=TdcpPositionUpdateConfig(enabled=False),
    )

    assert result.used_tdcp_position_update is False
    assert stats.n_tdcp_pu_skip == 1
    assert calls["tdcp"] == 0
    assert pf.calls == []
