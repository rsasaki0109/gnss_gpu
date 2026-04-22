from types import SimpleNamespace

import numpy as np

import gnss_gpu.tdcp_motion as tdcp_motion
from gnss_gpu.tdcp_motion import (
    estimate_local_fgo_tdcp_motion,
    estimate_tdcp_position_update_motion,
    evaluate_tdcp_position_update,
)


def test_tdcp_position_update_decision_applies_valid_displacement():
    decision = evaluate_tdcp_position_update(
        np.array([10.0, 20.0, 30.0]),
        np.array([1.0, 2.0, 3.0]),
        tdcp_rms=0.5,
        dt=0.2,
        rms_max=1.0,
    )

    assert decision.apply_update is True
    assert decision.gate_reason == "ok"
    assert decision.gate_skipped is False
    np.testing.assert_allclose(decision.predicted_position, [10.2, 20.4, 30.6])


def test_tdcp_position_update_decision_respects_all_gate_logic():
    decision = evaluate_tdcp_position_update(
        np.zeros(3),
        np.ones(3),
        tdcp_rms=0.5,
        dt=1.0,
        rms_max=1.0,
        dd_gate_stats=SimpleNamespace(n_kept_pairs=5),
        dd_pr_gate_stats=SimpleNamespace(n_kept_pairs=2),
        gate_dd_carrier_min_pairs=4,
        gate_dd_pseudorange_max_pairs=1,
        gate_logic="all",
    )

    assert decision.apply_update is False
    assert decision.gate_reason == "gate_no_match"
    assert decision.gate_skipped is True


def test_tdcp_position_update_decision_respects_stop_mode():
    decision = evaluate_tdcp_position_update(
        np.zeros(3),
        np.ones(3),
        tdcp_rms=0.5,
        dt=1.0,
        rms_max=1.0,
        gate_stop_mode="stopped",
        imu_stop_detected=False,
    )

    assert decision.apply_update is False
    assert decision.gate_reason == "gate_not_stopped"
    assert decision.gate_skipped is True


def test_tdcp_position_update_decision_keeps_invalid_tdcp_out_of_gate_skip():
    decision = evaluate_tdcp_position_update(
        np.zeros(3),
        np.array([np.nan, 1.0, 2.0]),
        tdcp_rms=0.5,
        dt=1.0,
        rms_max=1.0,
        tdcp_reason="tdcp_failed",
    )

    assert decision.apply_update is False
    assert decision.gate_reason == "tdcp_failed"
    assert decision.gate_skipped is False


def test_estimate_tdcp_position_update_motion_normalizes_disabled_spp_guard(monkeypatch):
    calls = []

    def fake_estimate(*args, **kwargs):
        calls.append(kwargs)
        return np.array([1.0, 2.0, 3.0]), 0.4, "ok", None

    monkeypatch.setattr(tdcp_motion, "_estimate_tdcp_motion_velocity", fake_estimate)

    estimate = estimate_tdcp_position_update_motion(
        np.zeros(3),
        ["p"],
        ["m"],
        0.1,
        {},
        prev_tow=10.0,
        tow_key=10.1,
        elevation_weight=True,
        el_sin_floor=0.2,
        rms_max_m=3.0,
        spp_max_diff_mps=0.0,
    )

    assert estimate.ok
    assert estimate.reason == "ok"
    assert calls[0]["spp_max_diff_mps"] is None
    np.testing.assert_allclose(estimate.velocity, [1.0, 2.0, 3.0])


def test_estimate_local_fgo_tdcp_motion_preserves_spp_guard(monkeypatch):
    calls = []

    def fake_estimate(*args, **kwargs):
        calls.append(kwargs)
        return None, float("nan"), "spp_guard", 9.0

    monkeypatch.setattr(tdcp_motion, "_estimate_tdcp_motion_velocity", fake_estimate)

    estimate = estimate_local_fgo_tdcp_motion(
        np.zeros(3),
        ["p"],
        ["m"],
        0.1,
        {},
        prev_tow=10.0,
        tow_key=10.1,
        elevation_weight=False,
        el_sin_floor=0.1,
        rms_max_m=2.0,
        spp_max_diff_mps=6.0,
    )

    assert not estimate.ok
    assert estimate.reason == "spp_guard"
    assert estimate.spp_diff_mps == 9.0
    assert calls[0]["spp_max_diff_mps"] == 6.0
