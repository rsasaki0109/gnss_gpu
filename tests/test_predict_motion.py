import numpy as np

import gnss_gpu.predict_motion as predict_motion
from gnss_gpu.pf_smoother_epoch_history import ForwardEpochHistory
from gnss_gpu.pf_smoother_epoch_state import create_epoch_forward_state
from gnss_gpu.pf_smoother_forward_stats import ForwardRunStats
from gnss_gpu.predict_motion import (
    EpochPredictMotionOptions,
    ImuPredictDecision,
    TdcpPredictDecision,
    apply_epoch_predict_motion,
    evaluate_imu_predict_velocity,
    evaluate_tdcp_predict_guide,
    select_predict_sigma,
    spp_finite_difference_velocity,
)


class _FakeImuFilter:
    def __init__(self, vel_enu):
        self.vel_enu = np.asarray(vel_enu, dtype=np.float64)
        self.corrected_headings = []

    def correct_heading_spp(self, heading):
        self.corrected_headings.append(float(heading))

    def get_velocity_enu(self, prev_tow, tow):
        return self.vel_enu


def test_spp_finite_difference_velocity_rejects_invalid_or_fast_velocity():
    assert spp_finite_difference_velocity({}, prev_tow=1.0, tow_key=1.1, dt=0.1) is None

    lookup = {
        1.0: np.array([0.0, 0.0, 0.0], dtype=np.float64),
        1.1: np.array([1.0, 0.0, 0.0], dtype=np.float64),
    }
    np.testing.assert_allclose(
        spp_finite_difference_velocity(lookup, prev_tow=1.0, tow_key=1.1, dt=0.1),
        [10.0, 0.0, 0.0],
    )

    fast = {
        1.0: np.zeros(3, dtype=np.float64),
        1.1: np.array([10.0, 0.0, 0.0], dtype=np.float64),
    }
    assert spp_finite_difference_velocity(fast, prev_tow=1.0, tow_key=1.1, dt=0.1) is None


def test_select_predict_sigma_matches_stop_tdcp_and_tight_precedence():
    assert select_predict_sigma(
        10.0,
        imu_stop_detected=True,
        imu_stop_sigma_pos=0.2,
        used_tdcp=False,
        sigma_pos_tdcp=1.0,
        sigma_pos_tdcp_tight=None,
        tdcp_rms=float("nan"),
        tdcp_tight_rms_max_m=1.0,
    ) == 0.2
    assert select_predict_sigma(
        10.0,
        imu_stop_detected=False,
        imu_stop_sigma_pos=None,
        used_tdcp=True,
        sigma_pos_tdcp=2.0,
        sigma_pos_tdcp_tight=0.5,
        tdcp_rms=0.2,
        tdcp_tight_rms_max_m=1.0,
    ) == 0.5


def test_evaluate_tdcp_predict_guide_accepts_valid_velocity(monkeypatch):
    monkeypatch.setattr(
        predict_motion,
        "estimate_velocity_from_tdcp_with_metrics",
        lambda *args, **kwargs: (np.array([1.0, 2.0, 3.0]), 0.4),
    )

    decision = evaluate_tdcp_predict_guide(
        "tdcp",
        np.ones(3, dtype=np.float64),
        prev_measurements=["p"],
        measurements=["m"],
        dt=0.1,
        spp_lookup={},
        prev_tow=10.0,
        tow_key=10.1,
        elevation_weight=False,
        el_sin_floor=0.1,
        tdcp_rms_threshold=3.0,
    )

    assert decision.used_tdcp
    assert not decision.adaptive_fallback
    assert decision.tdcp_rms == 0.4
    np.testing.assert_allclose(decision.velocity, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(decision.tdcp_pu_velocity, [1.0, 2.0, 3.0])


def test_evaluate_tdcp_predict_guide_reports_adaptive_rms_fallback(monkeypatch):
    monkeypatch.setattr(
        predict_motion,
        "estimate_velocity_from_tdcp_with_metrics",
        lambda *args, **kwargs: (np.array([1.0, 0.0, 0.0]), 5.0),
    )

    decision = evaluate_tdcp_predict_guide(
        "tdcp_adaptive",
        np.ones(3, dtype=np.float64),
        prev_measurements=["p"],
        measurements=["m"],
        dt=0.1,
        spp_lookup={},
        prev_tow=10.0,
        tow_key=10.1,
        elevation_weight=False,
        el_sin_floor=0.1,
        tdcp_rms_threshold=3.0,
    )

    assert not decision.used_tdcp
    assert decision.adaptive_fallback


def test_evaluate_tdcp_predict_guide_records_spp_guard_diff(monkeypatch):
    monkeypatch.setattr(
        predict_motion,
        "estimate_velocity_from_tdcp_with_metrics",
        lambda *args, **kwargs: (np.array([20.0, 0.0, 0.0]), 0.5),
    )

    decision = evaluate_tdcp_predict_guide(
        "tdcp",
        np.ones(3, dtype=np.float64),
        prev_measurements=["p"],
        measurements=["m"],
        dt=0.1,
        spp_lookup={
            10.0: np.zeros(3, dtype=np.float64),
            10.1: np.array([0.1, 0.0, 0.0], dtype=np.float64),
        },
        prev_tow=10.0,
        tow_key=10.1,
        elevation_weight=False,
        el_sin_floor=0.1,
        tdcp_rms_threshold=3.0,
    )

    assert not decision.used_tdcp
    assert decision.tdcp_pu_spp_diff_mps == 19.0


def test_evaluate_imu_predict_velocity_detects_stop():
    decision = evaluate_imu_predict_velocity(
        _FakeImuFilter([0.0, 0.0, 0.0]),
        "imu",
        prev_tow=10.0,
        tow=10.1,
        current_position_ecef=np.array([1.0, 2.0, 3.0]),
        spp_lookup={},
        ecef_to_lla_func=lambda _x, _y, _z: (0.0, 0.0, 0.0),
        dt=0.1,
    )

    assert decision.used_imu
    assert decision.stop_detected
    np.testing.assert_allclose(decision.velocity, np.zeros(3))


def test_evaluate_imu_predict_velocity_blends_reasonable_spp_velocity():
    imu_filter = _FakeImuFilter([2.0, 0.0, 0.0])
    decision = evaluate_imu_predict_velocity(
        imu_filter,
        "imu_spp_blend",
        prev_tow=10.0,
        tow=10.1,
        current_position_ecef=np.array([1.0, 2.0, 3.0]),
        spp_lookup={
            10.0: np.zeros(3, dtype=np.float64),
            10.1: np.array([0.0, 0.2, 0.0], dtype=np.float64),
        },
        ecef_to_lla_func=lambda _x, _y, _z: (0.0, 0.0, 0.0),
        dt=0.1,
    )

    assert decision.used_imu
    assert not decision.stop_detected
    assert len(imu_filter.corrected_headings) == 1
    assert decision.spp_fd_velocity is not None
    assert decision.velocity is not None


def _predict_options(**overrides):
    values = {
        "predict_guide": "tdcp",
        "tdcp_position_update": False,
        "tdcp_elevation_weight": False,
        "tdcp_el_sin_floor": 0.1,
        "tdcp_rms_threshold": 3.0,
        "tdcp_pu_rms_max": 2.0,
        "tdcp_pu_spp_max_diff_mps": 6.0,
        "need_fgo_tdcp_motion": False,
        "fgo_local_tdcp_rms_max_m": 3.0,
        "fgo_local_tdcp_spp_max_diff_mps": 6.0,
    }
    values.update(overrides)
    return EpochPredictMotionOptions(**values)


def test_apply_epoch_predict_motion_records_imu_stop(monkeypatch):
    monkeypatch.setattr(
        predict_motion,
        "evaluate_imu_predict_velocity",
        lambda *args, **kwargs: ImuPredictDecision(
            velocity=np.zeros(3),
            imu_velocity=np.zeros(3),
            used_imu=True,
            stop_detected=True,
        ),
    )
    monkeypatch.setattr(
        predict_motion,
        "evaluate_tdcp_predict_guide",
        lambda *args, **kwargs: TdcpPredictDecision(),
    )
    state = create_epoch_forward_state(0.5)
    stats = ForwardRunStats()
    history = ForwardEpochHistory(prev_tow=10.0, prev_measurements=["prev"])

    apply_epoch_predict_motion(
        state,
        stats,
        history,
        imu_filter=object(),
        options=_predict_options(predict_guide="imu"),
        tow=10.1,
        tow_key=10.1,
        dt=0.1,
        receiver_position_ecef=np.ones(3),
        current_pf_position_ecef=np.ones(3),
        measurements=["now"],
        spp_lookup={},
        ecef_to_lla_func=lambda _x, _y, _z: (0.0, 0.0, 0.0),
    )

    assert state.used_imu is True
    assert state.imu_stop_detected is True
    np.testing.assert_allclose(state.velocity, np.zeros(3))
    assert stats.n_imu_used == 1
    assert stats.n_imu_stop_detected == 1


def test_apply_epoch_predict_motion_records_tdcp_and_fallback_metrics(monkeypatch):
    monkeypatch.setattr(
        predict_motion,
        "evaluate_imu_predict_velocity",
        lambda *args, **kwargs: ImuPredictDecision(),
    )
    monkeypatch.setattr(
        predict_motion,
        "evaluate_tdcp_predict_guide",
        lambda *args, **kwargs: TdcpPredictDecision(
            velocity=np.array([1.0, 2.0, 3.0]),
            used_tdcp=True,
            adaptive_fallback=True,
            tdcp_rms=0.4,
            tdcp_pu_velocity=np.array([1.0, 2.0, 3.0]),
            tdcp_pu_rms=0.4,
            tdcp_pu_spp_diff_mps=0.2,
        ),
    )
    state = create_epoch_forward_state(0.5)
    stats = ForwardRunStats()
    history = ForwardEpochHistory(prev_tow=10.0, prev_measurements=["prev"])

    apply_epoch_predict_motion(
        state,
        stats,
        history,
        imu_filter=None,
        options=_predict_options(predict_guide="tdcp_adaptive"),
        tow=10.1,
        tow_key=10.1,
        dt=0.1,
        receiver_position_ecef=np.ones(3),
        current_pf_position_ecef=np.ones(3),
        measurements=["now"],
        spp_lookup={},
        ecef_to_lla_func=lambda _x, _y, _z: (0.0, 0.0, 0.0),
    )

    assert state.used_tdcp is True
    assert state.tdcp_rms == 0.4
    assert state.tdcp_pu_rms == 0.4
    assert state.tdcp_pu_spp_diff_mps == 0.2
    np.testing.assert_allclose(state.velocity, [1.0, 2.0, 3.0])
    assert stats.n_tdcp_used == 1
    assert stats.n_tdcp_fallback == 1


def test_apply_epoch_predict_motion_estimates_tdcp_pu_fgo_and_spp_fallback(monkeypatch):
    class _Estimate:
        def __init__(self, velocity, rms, reason, spp_diff_mps):
            self.velocity = velocity
            self.rms = rms
            self.reason = reason
            self.spp_diff_mps = spp_diff_mps

    monkeypatch.setattr(
        predict_motion,
        "evaluate_imu_predict_velocity",
        lambda *args, **kwargs: ImuPredictDecision(),
    )
    monkeypatch.setattr(
        predict_motion,
        "evaluate_tdcp_predict_guide",
        lambda *args, **kwargs: TdcpPredictDecision(),
    )
    monkeypatch.setattr(
        predict_motion,
        "estimate_tdcp_position_update_motion",
        lambda *args, **kwargs: _Estimate(np.array([4.0, 5.0, 6.0]), 0.6, "ok", 0.3),
    )
    monkeypatch.setattr(
        predict_motion,
        "estimate_local_fgo_tdcp_motion",
        lambda *args, **kwargs: _Estimate(np.array([7.0, 8.0, 9.0]), 0.7, "ok", 0.4),
    )
    state = create_epoch_forward_state(0.5)
    stats = ForwardRunStats()
    history = ForwardEpochHistory(prev_tow=10.0, prev_measurements=["prev"])
    lookup = {
        10.0: np.array([0.0, 0.0, 0.0]),
        10.1: np.array([0.1, 0.0, 0.0]),
    }

    apply_epoch_predict_motion(
        state,
        stats,
        history,
        imu_filter=None,
        options=_predict_options(
            predict_guide="spp",
            tdcp_position_update=True,
            need_fgo_tdcp_motion=True,
        ),
        tow=10.1,
        tow_key=10.1,
        dt=0.1,
        receiver_position_ecef=np.ones(3),
        current_pf_position_ecef=np.ones(3),
        measurements=["now"],
        spp_lookup=lookup,
        ecef_to_lla_func=lambda _x, _y, _z: (0.0, 0.0, 0.0),
    )

    np.testing.assert_allclose(state.tdcp_pu_velocity, [4.0, 5.0, 6.0])
    assert state.tdcp_pu_rms == 0.6
    assert state.tdcp_pu_reason == "ok"
    assert state.tdcp_pu_spp_diff_mps == 0.3
    np.testing.assert_allclose(state.fgo_tdcp_motion_velocity, [7.0, 8.0, 9.0])
    np.testing.assert_allclose(state.velocity, [1.0, 0.0, 0.0])
    assert stats.n_fgo_tdcp_motion_used == 1


def test_apply_epoch_predict_motion_uses_heading_lookup_for_imu_heading(monkeypatch):
    seen = []
    monkeypatch.setattr(
        predict_motion,
        "evaluate_imu_predict_velocity",
        lambda *args, **kwargs: seen.append(args[5]) or ImuPredictDecision(),
    )
    monkeypatch.setattr(
        predict_motion,
        "evaluate_tdcp_predict_guide",
        lambda *args, **kwargs: TdcpPredictDecision(),
    )
    spp_lookup = {10.0: np.zeros(3)}
    fix_lookup = {10.1: np.ones(3)}

    for heading_lookup in (None, fix_lookup):
        apply_epoch_predict_motion(
            create_epoch_forward_state(0.5),
            ForwardRunStats(),
            ForwardEpochHistory(prev_tow=10.0, prev_measurements=["prev"]),
            imu_filter=object(),
            options=_predict_options(predict_guide="imu"),
            tow=10.1,
            tow_key=10.1,
            dt=0.1,
            receiver_position_ecef=np.ones(3),
            current_pf_position_ecef=np.ones(3),
            measurements=["now"],
            spp_lookup=spp_lookup,
            ecef_to_lla_func=lambda _x, _y, _z: (0.0, 0.0, 0.0),
            heading_lookup=heading_lookup,
        )

    assert seen == [spp_lookup, fix_lookup]


def test_fixed_only_heading_lookup_skips_correction_across_gaps():
    imu_filter = _FakeImuFilter([2.0, 0.0, 0.0])
    fix_only = {10.0: np.zeros(3)}  # 10.1 is not FIXED
    evaluate_imu_predict_velocity(
        imu_filter,
        "imu",
        prev_tow=10.0,
        tow=10.1,
        current_position_ecef=np.array([1.0, 2.0, 3.0]),
        spp_lookup=fix_only,
        ecef_to_lla_func=lambda _x, _y, _z: (0.0, 0.0, 0.0),
        dt=0.1,
    )
    assert imu_filter.corrected_headings == []


def _doppler_rows(rx, v_rx, *, system_id=0, clock_drift_mps=12.0):
    from types import SimpleNamespace

    from gnss_gpu.predict_motion import _L1_WAVELENGTH_M

    rng = np.random.default_rng(3)
    rows = []
    for _ in range(8):
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        if direction @ rx < 0:
            direction = -direction
        sat = rx + 2.2e7 * direction
        v_sat = rng.normal(scale=2000.0, size=3)
        los = (sat - rx) / np.linalg.norm(sat - rx)
        range_rate = los @ (v_sat - v_rx) + clock_drift_mps
        rows.append(
            SimpleNamespace(
                system_id=system_id,
                satellite_ecef=sat,
                satellite_velocity=v_sat,
                doppler=-range_rate / _L1_WAVELENGTH_M,
            )
        )
    return rows


def _lla(x, y, z):
    lon = np.arctan2(y, x)
    lat = np.arctan2(z, np.hypot(x, y))
    return lat, lon, 0.0


def test_doppler_ground_speed_recovers_horizontal_speed():
    from gnss_gpu.predict_motion import doppler_ground_speed

    rx = np.array([-3961904.9, 3348993.7, 3698211.8])
    lat, lon, _ = _lla(*rx)
    east = np.array([-np.sin(lon), np.cos(lon), 0.0])
    up = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
    v_rx = 8.0 * east + 0.5 * up

    speed = doppler_ground_speed(_doppler_rows(rx, v_rx), rx, _lla)
    assert speed is not None
    assert abs(speed - 8.0) < 1e-6
    # Standstill noise below the stop threshold reads as zero.
    assert doppler_ground_speed(_doppler_rows(rx, 0.05 * east), rx, _lla) == 0.0
    # Other constellations (different carrier) are ignored.
    assert doppler_ground_speed(_doppler_rows(rx, v_rx, system_id=3), rx, _lla) is None


def test_apply_epoch_predict_motion_holds_doppler_speed(monkeypatch):
    seen = []
    monkeypatch.setattr(
        predict_motion,
        "evaluate_imu_predict_velocity",
        lambda *args, **kwargs: seen.append(kwargs["speed_mps"]) or ImuPredictDecision(),
    )
    monkeypatch.setattr(
        predict_motion,
        "evaluate_tdcp_predict_guide",
        lambda *args, **kwargs: TdcpPredictDecision(),
    )
    speeds = iter([6.5, None])
    monkeypatch.setattr(predict_motion, "doppler_ground_speed", lambda *a, **k: next(speeds))
    history = ForwardEpochHistory(prev_tow=10.0, prev_measurements=["prev"])

    for _ in range(2):
        apply_epoch_predict_motion(
            create_epoch_forward_state(0.5),
            ForwardRunStats(),
            history,
            imu_filter=object(),
            options=_predict_options(predict_guide="imu", imu_speed_source="doppler"),
            tow=10.1,
            tow_key=10.1,
            dt=0.1,
            receiver_position_ecef=np.ones(3),
            current_pf_position_ecef=np.ones(3),
            measurements=["now"],
            spp_lookup={},
            ecef_to_lla_func=lambda _x, _y, _z: (0.0, 0.0, 0.0),
        )

    assert seen == [6.5, 6.5]


def test_stationary_interval_holds_heading_and_learns_gyro_bias():
    from gnss_gpu.imu import ComplementaryHeadingFilter

    tow = np.arange(0.0, 200.0, 0.01)
    bias = np.radians(0.15)
    imu = {
        "tow": tow,
        "accel": np.zeros((tow.size, 3)),
        "gyro": np.column_stack([np.zeros(tow.size), np.zeros(tow.size), np.full(tow.size, bias)]),
        "wheel_vel": np.zeros(tow.size),
    }
    filt = ComplementaryHeadingFilter(imu)
    t = 0.0
    while t < 199.9:
        filt.update_heading_gyro(t, t + 0.1)
        filt.apply_stationary_interval(t, t + 0.1)
        t = round(t + 0.1, 1)

    assert filt.heading == 0.0
    assert abs(filt.gyro_bias - bias) < 1e-6 * bias + 1e-12


def test_evaluate_imu_predict_velocity_applies_zupt_only_when_enabled():
    calls = []

    class _Filter(_FakeImuFilter):
        def apply_stationary_interval(self, t0, t1):
            calls.append((t0, t1))

    for enabled in (False, True):
        evaluate_imu_predict_velocity(
            _Filter([0.0, 0.0, 0.0]),
            "imu",
            prev_tow=10.0,
            tow=10.1,
            current_position_ecef=np.array([1.0, 2.0, 3.0]),
            spp_lookup={},
            ecef_to_lla_func=lambda _x, _y, _z: (0.0, 0.0, 0.0),
            dt=0.1,
            zupt_gyro_bias=enabled,
        )
    assert calls == [(10.0, 10.1)]
