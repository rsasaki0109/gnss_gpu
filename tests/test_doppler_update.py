from types import SimpleNamespace

import numpy as np

from gnss_gpu.doppler_update import build_doppler_update_decision


def _measurement(doppler, sat_vel=(0.1, 0.2, 0.3), sat_ecef=(1.0, 2.0, 3.0)):
    return SimpleNamespace(
        doppler=doppler,
        satellite_velocity=np.asarray(sat_vel, dtype=np.float64),
        satellite_ecef=np.asarray(sat_ecef, dtype=np.float64),
    )


def test_doppler_update_decision_collects_valid_rows():
    decision = build_doppler_update_decision(
        [
            _measurement(-1000.0, sat_ecef=(1.0, 0.0, 0.0)),
            _measurement(0.0, sat_ecef=(2.0, 0.0, 0.0)),
            _measurement(float("nan"), sat_ecef=(3.0, 0.0, 0.0)),
            _measurement(-1200.0, sat_ecef=(4.0, 0.0, 0.0)),
        ],
        np.array([0.5, 0.6, 0.7, 0.8]),
        min_sats=2,
        doppler_sigma_mps=0.75,
        rbpf_velocity_kf=False,
        rbpf_doppler_sigma=None,
        wavelength_m=0.19,
    )

    assert decision.skipped is False
    assert decision.use_kf is False
    assert decision.sigma_mps == 0.75
    assert decision.gate_reason is None
    assert decision.update is not None
    np.testing.assert_allclose(decision.update["doppler_hz"], [-1000.0, -1200.0])
    np.testing.assert_allclose(decision.update["weights"], [0.5, 0.8])
    assert decision.update["wavelength_m"] == 0.19


def test_doppler_update_decision_reports_min_sats_skip():
    decision = build_doppler_update_decision(
        [_measurement(-1000.0)],
        np.ones(1),
        min_sats=2,
        doppler_sigma_mps=0.75,
        rbpf_velocity_kf=True,
        rbpf_doppler_sigma=0.5,
        wavelength_m=0.19,
    )

    assert decision.update is None
    assert decision.skipped is True
    assert decision.gate_skipped is False
    assert decision.gate_reason == "min_sats"
    assert decision.use_kf is True


def test_doppler_update_decision_reports_rbpf_gate_skip():
    decision = build_doppler_update_decision(
        [_measurement(-1000.0), _measurement(-1100.0), _measurement(-1200.0)],
        np.ones(3),
        min_sats=3,
        doppler_sigma_mps=0.75,
        rbpf_velocity_kf=True,
        rbpf_doppler_sigma=0.5,
        wavelength_m=0.19,
        dd_gate_stats=SimpleNamespace(n_kept_pairs=2),
        rbpf_gate_min_dd_pairs=3,
    )

    assert decision.update is None
    assert decision.skipped is True
    assert decision.gate_skipped is True
    assert decision.gate_reason == "min_dd_pairs"


def test_doppler_update_decision_uses_rbpf_sigma_when_gated_ok():
    decision = build_doppler_update_decision(
        [_measurement(-1000.0), _measurement(-1100.0), _measurement(-1200.0)],
        np.ones(3),
        min_sats=3,
        doppler_sigma_mps=0.75,
        rbpf_velocity_kf=True,
        rbpf_doppler_sigma=0.5,
        wavelength_m=0.19,
        dd_gate_stats=SimpleNamespace(n_kept_pairs=3),
        gate_ess_ratio=0.2,
        gate_spread_m=4.0,
        rbpf_gate_min_dd_pairs=3,
        rbpf_gate_min_ess_ratio=0.1,
        rbpf_gate_max_spread_m=5.0,
    )

    assert decision.update is not None
    assert decision.skipped is False
    assert decision.use_kf is True
    assert decision.sigma_mps == 0.5
    assert decision.gate_reason == "ok"
