from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.gsdc2023_observation_matrix import (
    ANDROID_STATE_CODE_LOCK,
    ANDROID_STATE_TOD_OK,
    ANDROID_STATE_TOW_OK,
    LIGHT_SPEED_MPS,
    RAW_GNSS_REQUIRED_COLUMNS,
    RawEpochObservation,
    android_state_tracking_ok,
    apply_matlab_signal_observation_mask,
    build_epoch_metadata_frame,
    clock_jump_from_epoch_counts,
    fill_observation_matrices,
    load_raw_gnss_frame,
    matlab_signal_observation_masks,
    receiver_clock_bias_lookup_from_epoch_meta,
    recompute_rtklib_tropo_matrix,
    repair_baseline_wls,
    select_epoch_observations,
)


def _required_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "utcTimeMillis": 1000,
        "Svid": 1,
        "ConstellationType": 1,
        "SignalType": "GPS_L1_CA",
        "RawPseudorangeMeters": 2.1e7,
        "IonosphericDelayMeters": 1.0,
        "TroposphericDelayMeters": 2.0,
        "SvClockBiasMeters": 3.0,
        "SvPositionXEcefMeters": 2.0e7,
        "SvPositionYEcefMeters": 1.0e7,
        "SvPositionZEcefMeters": 2.1e7,
        "SvElevationDegrees": 30.0,
        "Cn0DbHz": 35.0,
        "WlsPositionXEcefMeters": 6378137.0,
        "WlsPositionYEcefMeters": 0.0,
        "WlsPositionZEcefMeters": 0.0,
    }
    row.update(overrides)
    return row


def test_load_raw_gnss_frame_keeps_known_columns_and_rejects_missing_required(tmp_path) -> None:
    path = tmp_path / "device_gnss.csv"
    pd.DataFrame(
        [
            _required_row(
                State=ANDROID_STATE_CODE_LOCK | ANDROID_STATE_TOW_OK,
                PseudorangeRateMetersPerSecond=-5.0,
                UnexpectedColumn=123,
            ),
        ],
    ).to_csv(path, index=False)

    frame = load_raw_gnss_frame(path)

    assert "PseudorangeRateMetersPerSecond" in frame.columns
    assert "UnexpectedColumn" not in frame.columns

    missing_path = tmp_path / "missing.csv"
    pd.DataFrame([{col: 0 for col in RAW_GNSS_REQUIRED_COLUMNS if col != "Svid"}]).to_csv(missing_path, index=False)
    with pytest.raises(RuntimeError, match="missing columns"):
        load_raw_gnss_frame(missing_path)


def test_matlab_signal_observation_masks_separate_p_d_l_availability() -> None:
    frame = pd.DataFrame(
        [
            _required_row(
                State=ANDROID_STATE_CODE_LOCK | ANDROID_STATE_TOW_OK,
                PseudorangeRateMetersPerSecond=-1.0,
                AccumulatedDeltaRangeState=1,
                AccumulatedDeltaRangeMeters=10.0,
                MultipathIndicator=0,
            ),
            _required_row(
                Svid=2,
                Cn0DbHz=10.0,
                State=ANDROID_STATE_CODE_LOCK | ANDROID_STATE_TOW_OK,
                PseudorangeRateMetersPerSecond=-1.0,
                AccumulatedDeltaRangeState=1,
                AccumulatedDeltaRangeMeters=10.0,
                MultipathIndicator=0,
            ),
            _required_row(
                Svid=3,
                ConstellationType=3,
                SignalType="GLO_G1",
                State=ANDROID_STATE_CODE_LOCK | ANDROID_STATE_TOD_OK,
                PseudorangeRateMetersPerSecond=-1.0,
                AccumulatedDeltaRangeState=1,
                AccumulatedDeltaRangeMeters=10.0,
                MultipathIndicator=0,
            ),
            _required_row(
                Svid=4,
                State=ANDROID_STATE_CODE_LOCK | ANDROID_STATE_TOW_OK,
                PseudorangeRateMetersPerSecond=np.nan,
                AccumulatedDeltaRangeState=2,
                AccumulatedDeltaRangeMeters=10.0,
                MultipathIndicator=0,
            ),
        ],
    )

    p_ok, d_ok, l_ok = matlab_signal_observation_masks(frame, min_cn0_dbhz=20.0, min_elevation_deg=10.0)
    masked_frame, masked_count = apply_matlab_signal_observation_mask(
        frame,
        min_cn0_dbhz=20.0,
        min_elevation_deg=10.0,
    )

    np.testing.assert_array_equal(p_ok, np.array([True, False, True, True]))
    np.testing.assert_array_equal(d_ok, np.array([True, False, True, False]))
    np.testing.assert_array_equal(l_ok, np.array([True, False, True, False]))
    assert masked_count == 1
    assert list(masked_frame["Svid"]) == [1, 3, 4]
    np.testing.assert_array_equal(android_state_tracking_ok(frame), np.array([True, True, True, True]))


def test_build_epoch_metadata_prefers_low_bias_uncertainty_row() -> None:
    frame = pd.DataFrame(
        [
            _required_row(utcTimeMillis=1000, WlsPositionXEcefMeters=1.0, BiasUncertaintyNanos=2.0e4),
            _required_row(utcTimeMillis=1000, WlsPositionXEcefMeters=2.0, BiasUncertaintyNanos=10.0),
            _required_row(utcTimeMillis=2000, WlsPositionXEcefMeters=3.0, BiasUncertaintyNanos=2.0e4),
        ],
    )

    meta = build_epoch_metadata_frame(frame)

    assert list(meta["utcTimeMillis"]) == [1000, 2000]
    assert meta.loc[meta["utcTimeMillis"] == 1000, "WlsPositionXEcefMeters"].item() == 2.0
    assert meta.loc[meta["utcTimeMillis"] == 2000, "WlsPositionXEcefMeters"].item() == 3.0


def test_receiver_clock_bias_lookup_and_epoch_count_jumps() -> None:
    meta = pd.DataFrame(
        {
            "utcTimeMillis": [1000, 2000, 5000, 6000],
            "TimeNanos": [10_000_000_000, 11_000_000_000, 14_000_000_001, 15_000_000_001],
            "FullBiasNanos": [-1000, -998, -1020, -1018],
            "HardwareClockDiscontinuityCount": [1, 1, 1, 1],
        },
    )

    lookup = receiver_clock_bias_lookup_from_epoch_meta(meta)
    jumps = clock_jump_from_epoch_counts(np.array([1.0, 1.0, 2.0, np.nan, 2.0], dtype=np.float64))

    assert lookup[1000] == 0.0
    assert lookup[2000] == 2.0e-9 * LIGHT_SPEED_MPS
    assert lookup[5000] == 0.0
    assert lookup[6000] == 2.0e-9 * LIGHT_SPEED_MPS
    np.testing.assert_array_equal(jumps, np.array([False, False, True, False, False]))


def test_repair_baseline_wls_interpolates_stale_single_epoch_jump_direct() -> None:
    times_ms = np.arange(9, dtype=np.float64) * 1000.0
    base = np.array([-2711319.9, -4269168.8, 3873275.0], dtype=np.float64)
    step = np.array([3.0, -55.0, -62.0], dtype=np.float64)
    xyz = np.vstack(
        [
            base,
            base,
            base,
            base,
            base + np.array([9.0, -41.0, -57.0], dtype=np.float64),
            base + step,
            base + 2.0 * step,
            base + 3.0 * step,
            base + 4.0 * step,
        ],
    )

    repaired = repair_baseline_wls(times_ms, xyz)

    np.testing.assert_allclose(repaired[4], 0.5 * (xyz[3] + xyz[5]), atol=1e-9)


def test_select_epoch_observations_applies_start_max_and_truth_nearest() -> None:
    grouped = {
        1000.0: pd.DataFrame([_required_row(utcTimeMillis=1000, Svid=5)]),
        3000.0: pd.DataFrame(
            [
                _required_row(utcTimeMillis=3000, Svid=8),
                _required_row(utcTimeMillis=3000, Svid=2),
            ],
        ),
    }
    baseline_lookup = {2000: np.array([20.0, 21.0, 22.0], dtype=np.float64)}
    gt_times = np.array([900.0, 2100.0, 3100.0], dtype=np.float64)
    gt_ecef = np.array(
        [
            [9.0, 9.0, 9.0],
            [21.0, 21.0, 21.0],
            [31.0, 31.0, 31.0],
        ],
        dtype=np.float64,
    )

    epochs = select_epoch_observations(
        {1000.0, 2000.0, 3000.0},
        grouped,
        baseline_lookup,
        gt_times,
        gt_ecef,
        start_epoch=1,
        max_epochs=2,
        nearest_index_fn=lambda times, value: int(np.argmin(np.abs(times - value))),
    )

    assert [epoch.time_ms for epoch in epochs] == [2000.0, 3000.0]
    assert epochs[0].group.empty
    np.testing.assert_allclose(epochs[0].baseline_xyz, [20.0, 21.0, 22.0])
    np.testing.assert_allclose(epochs[0].truth_xyz, [21.0, 21.0, 21.0])
    assert list(epochs[1].group["Svid"]) == [2, 8]


def test_fill_observation_matrices_populates_signal_clock_doppler_and_adr() -> None:
    row = _required_row(
        utcTimeMillis=1000,
        Svid=7,
        RawPseudorangeMeters=21_000_000.0,
        SvClockBiasMeters=3.0,
        IonosphericDelayMeters=1.0,
        TroposphericDelayMeters=2.0,
        SvElevationDegrees=30.0,
        Cn0DbHz=35.0,
        PseudorangeRateMetersPerSecond=-4.0,
        PseudorangeRateUncertaintyMetersPerSecond=0.5,
        AccumulatedDeltaRangeState=1,
        AccumulatedDeltaRangeMeters=12.0,
        AccumulatedDeltaRangeUncertaintyMeters=0.2,
        SvVelocityXEcefMetersPerSecond=1.0,
        SvVelocityYEcefMetersPerSecond=2.0,
        SvVelocityZEcefMetersPerSecond=3.0,
        SvClockDriftMetersPerSecond=-0.01,
        bridge_p_ok=True,
        bridge_d_ok=True,
        bridge_l_ok=True,
        bridge_p_bias_ok=True,
    )
    epoch = RawEpochObservation(
        time_ms=1000.0,
        group=pd.DataFrame([row]),
        baseline_xyz=np.array([1.0, 2.0, 3.0], dtype=np.float64),
        truth_xyz=np.array([4.0, 5.0, 6.0], dtype=np.float64),
    )
    baseline_lookup = {1000: np.array([10.0, 20.0, 30.0], dtype=np.float64)}

    products = fill_observation_matrices(
        [epoch],
        source_columns=epoch.group.columns,
        baseline_lookup=baseline_lookup,
        weight_mode="sin2el",
        multi_gnss=True,
        dual_frequency=False,
        tdcp_enabled=True,
        adr_sign=-1.0,
        elapsed_ns_lookup={1000: 123.0},
        hcdc_lookup={1000: 2.0},
        clock_bias_lookup={1000: 7.0},
        clock_drift_lookup={1000: -0.3},
        gps_tgd_m_by_svid={},
        gps_matrtklib_nav_messages={},
        gps_arrival_tow_s_from_row_fn=lambda _row: 100.0,
        gps_sat_clock_bias_adjustment_m_fn=lambda _const, _svid, _signal, _tgd: 0.5,
        gps_matrtklib_sat_product_adjustment_fn=lambda **_kwargs: None,
        clock_kind_for_observation_fn=lambda const, _signal, **_kwargs: int(const) - 1,
        is_l5_signal_fn=lambda signal: "L5" in signal,
        slot_sort_key_fn=lambda key: key,
        ecef_to_lla_fn=lambda _x, _y, _z: (0.5, 0.0, 100.0),
        elevation_azimuth_fn=lambda _rx, _sat: (np.deg2rad(30.0), 0.0),
        rtklib_tropo_fn=lambda _lat, _alt, _el: 4.0,
        matlab_signal_clock_dim=7,
    )

    assert products.slot_keys == ((1, 7, "GPS_L1_CA"),)
    assert products.n_clock == 3
    assert products.visible_max == 1
    np.testing.assert_allclose(products.times_ms, [1000.0])
    np.testing.assert_allclose(products.kaggle_wls[0], [10.0, 20.0, 30.0])
    np.testing.assert_allclose(products.truth[0], [4.0, 5.0, 6.0])
    assert products.pseudorange_observable[0, 0] == 21_000_000.0
    assert products.pseudorange[0, 0] == 21_000_000.5
    assert products.sat_clock_bias_matrix[0, 0] == 3.5
    assert products.weights[0, 0] == pytest.approx(0.25)
    assert products.pseudorange_bias_weights[0, 0] == 1.0
    assert products.rtklib_tropo_m[0, 0] == 4.0
    assert products.sys_kind is not None
    assert products.sys_kind[0, 0] == 0
    np.testing.assert_allclose(products.sat_vel[0, 0], [1.0, 2.0, 3.0])
    assert products.sat_clock_drift_mps[0, 0] == -0.01
    assert products.doppler[0, 0] == 4.0
    assert products.doppler_weights[0, 0] == 4.0
    assert products.adr[0, 0] == -12.0
    assert products.adr_state[0, 0] == 1
    assert products.adr_uncertainty[0, 0] == 0.2
    assert products.elapsed_ns[0] == 123.0
    assert products.clock_counts[0] == 2.0
    assert products.clock_bias_m[0] == 7.0
    assert products.clock_drift_mps[0] == -0.3


def test_fill_observation_matrices_uses_signal_common_clock_for_gps_sat_product_adjustment() -> None:
    l5_row = _required_row(
        utcTimeMillis=1000,
        Svid=7,
        SignalType="GPS_L5_Q",
        RawPseudorangeMeters=22_000_000.0,
        SvClockBiasMeters=100.0,
        ArrivalTimeNanosSinceGpsEpoch=123_000_000_000.0,
    )
    l1_row = _required_row(
        utcTimeMillis=1000,
        Svid=7,
        SignalType="GPS_L1_CA",
        RawPseudorangeMeters=21_000_000.0,
        SvClockBiasMeters=10.0,
        ArrivalTimeNanosSinceGpsEpoch=123_000_000_000.0,
    )
    epoch = RawEpochObservation(
        time_ms=1000.0,
        group=pd.DataFrame([l5_row, l1_row]),
        baseline_xyz=np.array([1.0, 2.0, 3.0], dtype=np.float64),
        truth_xyz=np.array([4.0, 5.0, 6.0], dtype=np.float64),
    )
    calls: list[dict[str, object]] = []

    def fake_adjustment(**kwargs):
        calls.append(kwargs)
        return None

    products = fill_observation_matrices(
        [epoch],
        source_columns=epoch.group.columns,
        baseline_lookup={},
        weight_mode="sin2el",
        multi_gnss=True,
        dual_frequency=True,
        tdcp_enabled=False,
        adr_sign=1.0,
        elapsed_ns_lookup=None,
        hcdc_lookup=None,
        clock_bias_lookup={1000: -30.0},
        clock_drift_lookup=None,
        gps_tgd_m_by_svid={7: 0.5},
        gps_matrtklib_nav_messages={7: ((object(),), (object(),))},
        gps_arrival_tow_s_from_row_fn=lambda _row: 123.0,
        gps_sat_clock_bias_adjustment_m_fn=lambda _const, _svid, signal, _tgd: 5.0 if signal == "GPS_L1_CA" else 50.0,
        gps_matrtklib_sat_product_adjustment_fn=fake_adjustment,
        clock_kind_for_observation_fn=lambda const, signal, **_kwargs: 0 if signal == "GPS_L1_CA" else 1,
        is_l5_signal_fn=lambda signal: "L5" in signal,
        slot_sort_key_fn=lambda key: key,
        ecef_to_lla_fn=lambda _x, _y, _z: (0.5, 0.0, 100.0),
        elevation_azimuth_fn=lambda _rx, _sat: (np.deg2rad(30.0), 0.0),
        rtklib_tropo_fn=lambda _lat, _alt, _el: 4.0,
        matlab_signal_clock_dim=7,
    )

    assert len(calls) == 2
    assert calls[0]["l1_raw_pseudorange_m"] == 22_000_000.0
    assert calls[0]["derived_common_clock_m"] == 150.0
    assert calls[0]["receiver_clock_bias_m"] == -30.0
    assert calls[1]["l1_raw_pseudorange_m"] == 21_000_000.0
    assert calls[1]["derived_common_clock_m"] == 15.0
    l5_slot = products.slot_keys.index((1, 7, "GPS_L5_Q"))
    assert products.sat_clock_bias_matrix[0, l5_slot] == 150.0



def test_fill_observation_matrices_applies_l5_adjustment_when_clock_differs() -> None:
    l5_row = _required_row(
        utcTimeMillis=1000,
        Svid=7,
        SignalType="GPS_L5_Q",
        RawPseudorangeMeters=22_000_000.0,
        SvClockBiasMeters=100.0,
        SvPositionXEcefMeters=20_000_000.0,
        SvPositionYEcefMeters=10_000_000.0,
        SvPositionZEcefMeters=21_000_000.0,
        ArrivalTimeNanosSinceGpsEpoch=123_000_000_000.0,
    )
    l1_row = _required_row(
        utcTimeMillis=1000,
        Svid=7,
        SignalType="GPS_L1_CA",
        RawPseudorangeMeters=21_000_000.0,
        SvClockBiasMeters=10.0,
        ArrivalTimeNanosSinceGpsEpoch=123_000_000_000.0,
    )
    epoch = RawEpochObservation(
        time_ms=1000.0,
        group=pd.DataFrame([l5_row, l1_row]),
        baseline_xyz=np.array([1.0, 2.0, 3.0], dtype=np.float64),
        truth_xyz=np.array([4.0, 5.0, 6.0], dtype=np.float64),
    )
    calls: list[dict[str, object]] = []
    adjusted_xyz = np.array([30_000_000.0, 31_000_000.0, 32_000_000.0], dtype=np.float64)

    def fake_adjustment(**kwargs):
        calls.append(kwargs)
        return adjusted_xyz, np.array([1.0, 2.0, 3.0], dtype=np.float64), 42.0, -0.5

    products = fill_observation_matrices(
        [epoch],
        source_columns=epoch.group.columns,
        baseline_lookup={},
        weight_mode="sin2el",
        multi_gnss=True,
        dual_frequency=True,
        tdcp_enabled=False,
        adr_sign=1.0,
        elapsed_ns_lookup=None,
        hcdc_lookup=None,
        clock_bias_lookup={1000: -30.0},
        clock_drift_lookup=None,
        gps_tgd_m_by_svid={7: 0.5},
        gps_matrtklib_nav_messages={7: ((object(),), (object(),))},
        gps_arrival_tow_s_from_row_fn=lambda _row: 123.0,
        gps_sat_clock_bias_adjustment_m_fn=lambda _const, _svid, signal, _tgd: 5.0 if signal == "GPS_L1_CA" else 50.0,
        gps_matrtklib_sat_product_adjustment_fn=fake_adjustment,
        clock_kind_for_observation_fn=lambda const, signal, **_kwargs: 0 if signal == "GPS_L1_CA" else 1,
        is_l5_signal_fn=lambda signal: "L5" in signal,
        slot_sort_key_fn=lambda key: key,
        ecef_to_lla_fn=lambda _x, _y, _z: (0.5, 0.0, 100.0),
        elevation_azimuth_fn=lambda _rx, _sat: (np.deg2rad(30.0), 0.0),
        rtklib_tropo_fn=lambda _lat, _alt, _el: 4.0,
        matlab_signal_clock_dim=7,
    )

    assert len(calls) == 2
    assert calls[0]["l1_raw_pseudorange_m"] == 22_000_000.0
    assert calls[0]["derived_common_clock_m"] == 150.0
    assert calls[1]["l1_raw_pseudorange_m"] == 21_000_000.0
    assert calls[1]["derived_common_clock_m"] == 15.0
    l1_slot = products.slot_keys.index((1, 7, "GPS_L1_CA"))
    l5_slot = products.slot_keys.index((1, 7, "GPS_L5_Q"))
    np.testing.assert_allclose(products.sat_ecef[0, l1_slot], adjusted_xyz)
    np.testing.assert_allclose(products.sat_ecef[0, l5_slot], adjusted_xyz)
    assert products.sat_clock_bias_matrix[0, l1_slot] == 42.0
    assert products.sat_clock_bias_matrix[0, l5_slot] == 42.0


def test_fill_observation_matrices_keeps_l5_raw_product_when_clock_already_matches() -> None:
    l5_row = _required_row(
        utcTimeMillis=1000,
        Svid=7,
        SignalType="GPS_L5_Q",
        RawPseudorangeMeters=22_000_000.0,
        SvClockBiasMeters=100.0,
        SvPositionXEcefMeters=20_000_000.0,
        SvPositionYEcefMeters=10_000_000.0,
        SvPositionZEcefMeters=21_000_000.0,
        ArrivalTimeNanosSinceGpsEpoch=123_000_000_000.0,
    )
    l1_row = _required_row(
        utcTimeMillis=1000,
        Svid=7,
        SignalType="GPS_L1_CA",
        RawPseudorangeMeters=21_000_000.0,
        SvClockBiasMeters=10.0,
        ArrivalTimeNanosSinceGpsEpoch=123_000_000_000.0,
    )
    epoch = RawEpochObservation(
        time_ms=1000.0,
        group=pd.DataFrame([l5_row, l1_row]),
        baseline_xyz=np.array([1.0, 2.0, 3.0], dtype=np.float64),
        truth_xyz=np.array([4.0, 5.0, 6.0], dtype=np.float64),
    )
    calls: list[dict[str, object]] = []
    adjusted_xyz = np.array([30_000_000.0, 31_000_000.0, 32_000_000.0], dtype=np.float64)

    def fake_adjustment(**kwargs):
        calls.append(kwargs)
        return adjusted_xyz, np.array([1.0, 2.0, 3.0], dtype=np.float64), 150.003, -0.5

    products = fill_observation_matrices(
        [epoch],
        source_columns=epoch.group.columns,
        baseline_lookup={},
        weight_mode="sin2el",
        multi_gnss=True,
        dual_frequency=True,
        tdcp_enabled=False,
        adr_sign=1.0,
        elapsed_ns_lookup=None,
        hcdc_lookup=None,
        clock_bias_lookup={1000: -30.0},
        clock_drift_lookup=None,
        gps_tgd_m_by_svid={7: 0.5},
        gps_matrtklib_nav_messages={7: ((object(),), (object(),))},
        gps_arrival_tow_s_from_row_fn=lambda _row: 123.0,
        gps_sat_clock_bias_adjustment_m_fn=lambda _const, _svid, signal, _tgd: 5.0 if signal == "GPS_L1_CA" else 50.0,
        gps_matrtklib_sat_product_adjustment_fn=fake_adjustment,
        clock_kind_for_observation_fn=lambda const, signal, **_kwargs: 0 if signal == "GPS_L1_CA" else 1,
        is_l5_signal_fn=lambda signal: "L5" in signal,
        slot_sort_key_fn=lambda key: key,
        ecef_to_lla_fn=lambda _x, _y, _z: (0.5, 0.0, 100.0),
        elevation_azimuth_fn=lambda _rx, _sat: (np.deg2rad(30.0), 0.0),
        rtklib_tropo_fn=lambda _lat, _alt, _el: 4.0,
        matlab_signal_clock_dim=7,
    )

    assert len(calls) == 2
    l1_slot = products.slot_keys.index((1, 7, "GPS_L1_CA"))
    l5_slot = products.slot_keys.index((1, 7, "GPS_L5_Q"))
    np.testing.assert_allclose(products.sat_ecef[0, l1_slot], adjusted_xyz)
    np.testing.assert_allclose(products.sat_ecef[0, l5_slot], [20_000_000.0, 10_000_000.0, 21_000_000.0])
    assert products.sat_clock_bias_matrix[0, l1_slot] == 150.003
    assert products.sat_clock_bias_matrix[0, l5_slot] == 150.0


def test_fill_observation_matrices_keeps_l5_only_raw_product_when_clock_already_matches() -> None:
    row = _required_row(
        utcTimeMillis=1000,
        Svid=7,
        SignalType="GPS_L5_Q",
        RawPseudorangeMeters=22_000_000.0,
        SvClockBiasMeters=100.0,
        SvPositionXEcefMeters=20_000_000.0,
        SvPositionYEcefMeters=10_000_000.0,
        SvPositionZEcefMeters=21_000_000.0,
        ArrivalTimeNanosSinceGpsEpoch=123_000_000_000.0,
    )
    epoch = RawEpochObservation(
        time_ms=1000.0,
        group=pd.DataFrame([row]),
        baseline_xyz=np.array([1.0, 2.0, 3.0], dtype=np.float64),
        truth_xyz=np.array([4.0, 5.0, 6.0], dtype=np.float64),
    )
    calls: list[dict[str, object]] = []
    adjusted_xyz = np.array([30_000_000.0, 31_000_000.0, 32_000_000.0], dtype=np.float64)

    def fake_adjustment(**kwargs):
        calls.append(kwargs)
        return adjusted_xyz, np.array([1.0, 2.0, 3.0], dtype=np.float64), 150.003, -0.5

    products = fill_observation_matrices(
        [epoch],
        source_columns=epoch.group.columns,
        baseline_lookup={},
        weight_mode="sin2el",
        multi_gnss=True,
        dual_frequency=True,
        tdcp_enabled=False,
        adr_sign=1.0,
        elapsed_ns_lookup=None,
        hcdc_lookup=None,
        clock_bias_lookup={1000: -30.0},
        clock_drift_lookup=None,
        gps_tgd_m_by_svid={7: 0.5},
        gps_matrtklib_nav_messages={7: ((object(),), (object(),))},
        gps_arrival_tow_s_from_row_fn=lambda _row: 123.0,
        gps_sat_clock_bias_adjustment_m_fn=lambda _const, _svid, _signal, _tgd: 50.0,
        gps_matrtklib_sat_product_adjustment_fn=fake_adjustment,
        clock_kind_for_observation_fn=lambda _const, _signal, **_kwargs: 1,
        is_l5_signal_fn=lambda signal: "L5" in signal,
        slot_sort_key_fn=lambda key: key,
        ecef_to_lla_fn=lambda _x, _y, _z: (0.5, 0.0, 100.0),
        elevation_azimuth_fn=lambda _rx, _sat: (np.deg2rad(30.0), 0.0),
        rtklib_tropo_fn=lambda _lat, _alt, _el: 4.0,
        matlab_signal_clock_dim=7,
    )

    assert len(calls) == 1
    assert calls[0]["l1_raw_pseudorange_m"] == 22_000_000.0
    assert calls[0]["derived_common_clock_m"] == 150.0
    slot = products.slot_keys.index((1, 7, "GPS_L5_Q"))
    np.testing.assert_allclose(products.sat_ecef[0, slot], [20_000_000.0, 10_000_000.0, 21_000_000.0])
    assert products.sat_clock_bias_matrix[0, slot] == 150.0

def test_recompute_rtklib_tropo_matrix_overlays_valid_repaired_baseline() -> None:
    sat_ecef = np.array(
        [
            [[20_000_000.0, 0.0, 0.0]],
            [[21_000_000.0, 0.0, 0.0]],
        ],
        dtype=np.float64,
    )
    kaggle_wls = np.array(
        [
            [10.0, 0.0, 0.0],
            [np.nan, np.nan, np.nan],
        ],
        dtype=np.float64,
    )
    initial = np.array([[5.0], [6.0]], dtype=np.float64)

    tropo = recompute_rtklib_tropo_matrix(
        kaggle_wls,
        sat_ecef,
        ecef_to_lla_fn=lambda x, _y, _z: (0.0, 0.0, x),
        elevation_azimuth_fn=lambda _rx, _sat: (0.5, 0.0),
        rtklib_tropo_fn=lambda _lat, alt, _el: alt,
        initial_tropo_m=initial,
    )

    np.testing.assert_allclose(tropo, [[10.0], [6.0]])
