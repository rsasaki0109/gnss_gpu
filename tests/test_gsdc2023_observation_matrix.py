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
    build_sat_velocity_forward_difference_lookup,
    clock_jump_from_epoch_counts,
    fill_observation_matrices,
    load_raw_gnss_frame,
    load_raw_gnss_frame_epoch_window,
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


def _internal_state_row(**overrides: object) -> dict[str, object]:
    row = _required_row(
        State=ANDROID_STATE_CODE_LOCK | ANDROID_STATE_TOW_OK,
        MultipathIndicator=0,
        RawPseudorangeUncertaintyMeters=5.0,
        BiasUncertaintyNanos=10.0,
        HardwareClockDiscontinuityCount=1.0,
        ChipsetElapsedRealtimeNanos=100.0,
        FullBiasNanos=-1_000_000_000,
        BiasNanos=0.0,
        DriftNanosPerSecond=0.0,
        ArrivalTimeNanosSinceGpsEpoch=100_000_000_000.0,
        PseudorangeRateMetersPerSecond=-1.0,
        PseudorangeRateUncertaintyMetersPerSecond=1.0,
        AccumulatedDeltaRangeState=1,
        AccumulatedDeltaRangeMeters=10.0,
        AccumulatedDeltaRangeUncertaintyMeters=0.1,
        SvVelocityXEcefMetersPerSecond=0.0,
        SvVelocityYEcefMetersPerSecond=0.0,
        SvVelocityZEcefMetersPerSecond=0.0,
        SvClockDriftMetersPerSecond=0.0,
        bridge_p_ok=True,
        bridge_d_ok=True,
        bridge_l_ok=True,
        bridge_p_bias_ok=True,
    )
    row.update(overrides)
    return row


def _assert_snapshot_allclose(actual: object, expected: object) -> None:
    actual_array = np.asarray(actual)
    expected_array = np.asarray(expected)
    if expected_array.dtype.kind in "biu" and actual_array.dtype.kind in "biu":
        np.testing.assert_array_equal(actual_array, expected_array)
        return
    np.testing.assert_allclose(actual_array, expected_array, rtol=0.0, atol=1e-12, equal_nan=True)


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


def test_load_raw_gnss_frame_epoch_window_reads_complete_selected_epochs(tmp_path) -> None:
    path = tmp_path / "device_gnss.csv"
    rows = []
    for epoch_idx, utc_ms in enumerate([1000, 2000, 3000, 4000]):
        rows.append(_required_row(utcTimeMillis=utc_ms, Svid=1, UnexpectedColumn=epoch_idx))
        rows.append(_required_row(utcTimeMillis=utc_ms, Svid=2, UnexpectedColumn=epoch_idx))
    pd.DataFrame(rows).to_csv(path, index=False)

    frame = load_raw_gnss_frame_epoch_window(path, start_epoch=1, max_epochs=2, chunksize=3)

    assert frame["utcTimeMillis"].tolist() == [2000, 2000, 3000, 3000]
    assert frame["Svid"].tolist() == [1, 2, 1, 2]
    assert "UnexpectedColumn" not in frame.columns


def test_load_raw_gnss_frame_epoch_window_falls_back_for_unsorted_epochs(tmp_path) -> None:
    path = tmp_path / "device_gnss.csv"
    pd.DataFrame(
        [
            _required_row(utcTimeMillis=3000, Svid=3),
            _required_row(utcTimeMillis=1000, Svid=1),
            _required_row(utcTimeMillis=2000, Svid=2),
        ],
    ).to_csv(path, index=False)

    frame = load_raw_gnss_frame_epoch_window(path, start_epoch=0, max_epochs=2, chunksize=2)

    assert frame["utcTimeMillis"].tolist() == [1000, 2000]
    assert frame["Svid"].tolist() == [1, 2]


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


def test_sat_velocity_forward_difference_lookup_matches_rtklib_half_step() -> None:
    epochs = [
        RawEpochObservation(
            time_ms=1000.0 + 1000.0 * idx,
            group=pd.DataFrame(
                [
                    _required_row(
                        Svid=7,
                        SvVelocityXEcefMetersPerSecond=1.0 + 10.0 * idx,
                        SvVelocityYEcefMetersPerSecond=2.0 + 20.0 * idx,
                        SvVelocityZEcefMetersPerSecond=3.0 + 30.0 * idx,
                    ),
                ],
            ),
            baseline_xyz=np.zeros(3, dtype=np.float64),
            truth_xyz=np.zeros(3, dtype=np.float64),
        )
        for idx in range(3)
    ]

    lookup = build_sat_velocity_forward_difference_lookup(epochs)

    np.testing.assert_allclose(lookup[(0, 1, 7, "GPS_L1_CA")], [1.005, 2.01, 3.015])
    np.testing.assert_allclose(lookup[(1, 1, 7, "GPS_L1_CA")], [11.005, 22.01, 33.015])
    np.testing.assert_allclose(lookup[(2, 1, 7, "GPS_L1_CA")], [21.005, 42.01, 63.015])


def test_matlab_signal_observation_masks_internal_state_snapshot() -> None:
    frame = pd.DataFrame(
        [
            _internal_state_row(Svid=1),
            _internal_state_row(Svid=2, Cn0DbHz=19.0),
            _internal_state_row(
                Svid=3,
                ConstellationType=3,
                SignalType="GLO_G1",
                State=ANDROID_STATE_CODE_LOCK | ANDROID_STATE_TOD_OK,
            ),
            _internal_state_row(Svid=4, MultipathIndicator=1),
            _internal_state_row(Svid=5, RawPseudorangeMeters=5.0e6),
        ],
    )

    p_ok, d_ok, l_ok = matlab_signal_observation_masks(frame, min_cn0_dbhz=20.0, min_elevation_deg=10.0)
    masked_frame, masked_count = apply_matlab_signal_observation_mask(
        frame,
        min_cn0_dbhz=20.0,
        min_elevation_deg=10.0,
    )

    snapshot = {
        "android_state_tracking_ok": android_state_tracking_ok(frame),
        "p_ok": p_ok,
        "d_ok": d_ok,
        "l_ok": l_ok,
        "masked_svid": masked_frame["Svid"].to_numpy(dtype=np.int64),
    }

    _assert_snapshot_allclose(snapshot["android_state_tracking_ok"], [True, True, True, True, True])
    _assert_snapshot_allclose(snapshot["p_ok"], [True, False, True, False, False])
    _assert_snapshot_allclose(snapshot["d_ok"], [True, False, True, False, True])
    _assert_snapshot_allclose(snapshot["l_ok"], [True, False, True, False, True])
    _assert_snapshot_allclose(snapshot["masked_svid"], [1, 3])
    assert masked_count == 3


def test_fill_observation_matrices_internal_state_snapshot() -> None:
    epoch0_rows = [
        _internal_state_row(
            utcTimeMillis=1000,
            Svid=7,
            ConstellationType=1,
            SignalType="GPS_L1_CA",
            RawPseudorangeMeters=21_000_000.0,
            IonosphericDelayMeters=1.0,
            TroposphericDelayMeters=2.0,
            SvClockBiasMeters=3.0,
            SvPositionXEcefMeters=20_000_000.0,
            SvPositionYEcefMeters=10_000_000.0,
            SvPositionZEcefMeters=21_000_000.0,
            SvElevationDegrees=30.0,
            Cn0DbHz=35.0,
            PseudorangeRateMetersPerSecond=-4.0,
            PseudorangeRateUncertaintyMetersPerSecond=0.5,
            AccumulatedDeltaRangeMeters=12.0,
            AccumulatedDeltaRangeUncertaintyMeters=0.2,
            SvVelocityXEcefMetersPerSecond=1.0,
            SvVelocityYEcefMetersPerSecond=2.0,
            SvVelocityZEcefMetersPerSecond=3.0,
            SvClockDriftMetersPerSecond=-0.01,
        ),
        _internal_state_row(
            utcTimeMillis=1000,
            Svid=3,
            ConstellationType=6,
            SignalType="GAL_E1",
            RawPseudorangeMeters=23_000_000.0,
            IonosphericDelayMeters=0.5,
            TroposphericDelayMeters=1.5,
            SvClockBiasMeters=-2.0,
            SvPositionXEcefMeters=21_000_000.0,
            SvPositionYEcefMeters=11_000_000.0,
            SvPositionZEcefMeters=22_000_000.0,
            SvElevationDegrees=60.0,
            Cn0DbHz=40.0,
            PseudorangeRateMetersPerSecond=2.0,
            PseudorangeRateUncertaintyMetersPerSecond=1.0,
            AccumulatedDeltaRangeMeters=13.0,
            AccumulatedDeltaRangeUncertaintyMeters=0.3,
            SvVelocityXEcefMetersPerSecond=4.0,
            SvVelocityYEcefMetersPerSecond=5.0,
            SvVelocityZEcefMetersPerSecond=6.0,
            SvClockDriftMetersPerSecond=-0.02,
        ),
    ]
    epoch1_rows = [
        _internal_state_row(
            utcTimeMillis=2000,
            Svid=7,
            ConstellationType=1,
            SignalType="GPS_L5_Q",
            RawPseudorangeMeters=22_000_000.0,
            IonosphericDelayMeters=2.0,
            TroposphericDelayMeters=3.0,
            SvClockBiasMeters=100.0,
            SvPositionXEcefMeters=20_500_000.0,
            SvPositionYEcefMeters=10_500_000.0,
            SvPositionZEcefMeters=21_500_000.0,
            SvElevationDegrees=10.0,
            Cn0DbHz=45.0,
            PseudorangeRateMetersPerSecond=-6.0,
            PseudorangeRateUncertaintyMetersPerSecond=0.25,
            AccumulatedDeltaRangeMeters=14.0,
            AccumulatedDeltaRangeUncertaintyMeters=0.4,
            SvVelocityXEcefMetersPerSecond=7.0,
            SvVelocityYEcefMetersPerSecond=8.0,
            SvVelocityZEcefMetersPerSecond=9.0,
            SvClockDriftMetersPerSecond=-0.03,
            bridge_d_ok=False,
            bridge_l_ok=False,
            bridge_p_bias_ok=False,
        ),
        _internal_state_row(
            utcTimeMillis=2000,
            Svid=5,
            ConstellationType=3,
            SignalType="GLO_G1",
            RawPseudorangeMeters=24_000_000.0,
            IonosphericDelayMeters=1.0,
            TroposphericDelayMeters=1.0,
            SvClockBiasMeters=5.0,
            SvPositionXEcefMeters=22_000_000.0,
            SvPositionYEcefMeters=12_000_000.0,
            SvPositionZEcefMeters=23_000_000.0,
            SvElevationDegrees=5.0,
            Cn0DbHz=38.0,
            PseudorangeRateMetersPerSecond=-8.0,
            PseudorangeRateUncertaintyMetersPerSecond=0.25,
            AccumulatedDeltaRangeMeters=15.0,
            AccumulatedDeltaRangeUncertaintyMeters=0.5,
            SvVelocityXEcefMetersPerSecond=10.0,
            SvVelocityYEcefMetersPerSecond=11.0,
            SvVelocityZEcefMetersPerSecond=12.0,
            SvClockDriftMetersPerSecond=-0.04,
            bridge_p_ok=False,
            bridge_p_bias_ok=False,
        ),
    ]
    epoch0_group = pd.DataFrame(epoch0_rows)
    epoch1_group = pd.DataFrame(epoch1_rows)
    epochs = [
        RawEpochObservation(
            time_ms=1000.0,
            group=epoch0_group,
            baseline_xyz=np.array([1.0, 2.0, 3.0], dtype=np.float64),
            truth_xyz=np.array([4.0, 5.0, 6.0], dtype=np.float64),
        ),
        RawEpochObservation(
            time_ms=2000.0,
            group=epoch1_group,
            baseline_xyz=np.array([11.0, 22.0, 33.0], dtype=np.float64),
            truth_xyz=np.array([7.0, 8.0, 9.0], dtype=np.float64),
        ),
    ]

    def clock_kind(constellation: int, signal: str, **_kwargs: object) -> int:
        if constellation == 1:
            return 1 if "L5" in signal else 0
        if constellation == 3:
            return 2
        if constellation == 6:
            return 4
        return 6

    products = fill_observation_matrices(
        epochs,
        source_columns=epoch0_group.columns,
        baseline_lookup={1000: np.array([10.0, 20.0, 30.0], dtype=np.float64)},
        weight_mode="sin2el",
        multi_gnss=True,
        dual_frequency=True,
        tdcp_enabled=True,
        adr_sign=-1.0,
        elapsed_ns_lookup={1000: 123.0, 2000: 223.0},
        hcdc_lookup={1000: 2.0, 2000: 3.0},
        clock_bias_lookup={1000: 7.0, 2000: 8.0},
        clock_drift_lookup={1000: -0.3, 2000: -0.4},
        gps_tgd_m_by_svid={},
        gps_matrtklib_nav_messages={},
        gps_arrival_tow_s_from_row_fn=lambda _row: 100.0,
        gps_sat_clock_bias_adjustment_m_fn=lambda _const, _svid, signal, _tgd: (
            0.5 if signal == "GPS_L1_CA" else 50.0 if signal == "GPS_L5_Q" else 0.0
        ),
        gps_matrtklib_sat_product_adjustment_fn=lambda **_kwargs: None,
        clock_kind_for_observation_fn=clock_kind,
        is_l5_signal_fn=lambda signal: "L5" in signal,
        slot_sort_key_fn=lambda key: key,
        ecef_to_lla_fn=lambda _x, _y, _z: (0.0, 0.0, 100.0),
        elevation_azimuth_fn=lambda _rx, sat: (0.1 + float(sat[0]) / 1.0e9, 0.0),
        rtklib_tropo_fn=lambda _lat, _alt, el: float(el) * 100.0,
        matlab_signal_clock_dim=7,
    )

    assert products.slot_keys == (
        (1, 7, "GPS_L1_CA"),
        (1, 7, "GPS_L5_Q"),
        (3, 5, "GLO_G1"),
        (6, 3, "GAL_E1"),
    )
    assert products.n_clock == 7
    assert products.visible_max == 2
    _assert_snapshot_allclose(products.times_ms, [1000.0, 2000.0])
    _assert_snapshot_allclose(products.kaggle_wls, [[10.0, 20.0, 30.0], [11.0, 22.0, 33.0]])
    _assert_snapshot_allclose(products.truth, [[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    _assert_snapshot_allclose(
        products.pseudorange_observable,
        [[21_000_000.0, 0.0, 0.0, 23_000_000.0], [0.0, 22_000_000.0, 24_000_000.0, 0.0]],
    )
    _assert_snapshot_allclose(
        products.pseudorange,
        [[21_000_000.5, 0.0, 0.0, 22_999_996.0], [0.0, 22_000_145.0, 24_000_003.0, 0.0]],
    )
    _assert_snapshot_allclose(
        products.weights,
        [[0.25, 0.0, 0.0, 0.75], [0.0, np.sin(np.deg2rad(10.0)) ** 2, 0.0, 0.0]],
    )
    _assert_snapshot_allclose(
        products.pseudorange_bias_weights,
        [[1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0]],
    )
    _assert_snapshot_allclose(
        products.sat_clock_bias_matrix,
        [[3.5, np.nan, np.nan, -2.0], [np.nan, 150.0, 5.0, np.nan]],
    )
    _assert_snapshot_allclose(
        products.rtklib_tropo_m,
        [[12.0, np.nan, np.nan, 12.1], [np.nan, 12.05, 12.2, np.nan]],
    )
    _assert_snapshot_allclose(products.sys_kind, [[0, 0, 0, 4], [0, 1, 2, 0]])
    _assert_snapshot_allclose(
        products.sat_vel,
        [
            [[1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [4.0, 5.0, 6.0]],
            [[0.0, 0.0, 0.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0], [0.0, 0.0, 0.0]],
        ],
    )
    _assert_snapshot_allclose(
        products.sat_clock_drift_mps,
        [[-0.01, 0.0, 0.0, -0.02], [0.0, -0.03, -0.04, 0.0]],
    )
    _assert_snapshot_allclose(products.doppler, [[4.0, 0.0, 0.0, -2.0], [0.0, 6.0, 8.0, 0.0]])
    _assert_snapshot_allclose(products.doppler_weights, [[4.0, 0.0, 0.0, 1.0], [0.0, 0.0, 16.0, 0.0]])
    _assert_snapshot_allclose(
        products.adr,
        [[-12.0, np.nan, np.nan, -13.0], [np.nan, np.nan, -15.0, np.nan]],
    )
    _assert_snapshot_allclose(products.adr_state, [[1, 0, 0, 1], [0, 0, 1, 0]])
    _assert_snapshot_allclose(
        products.adr_uncertainty,
        [[0.2, np.nan, np.nan, 0.3], [np.nan, np.nan, 0.5, np.nan]],
    )
    _assert_snapshot_allclose(products.elapsed_ns, [123.0, 223.0])
    _assert_snapshot_allclose(products.clock_counts, [2.0, 3.0])
    _assert_snapshot_allclose(products.clock_bias_m, [7.0, 8.0])
    _assert_snapshot_allclose(products.clock_drift_mps, [-0.3, -0.4])


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
        ReceivedSvTimeNanos=120_000_000_000.0,
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
    assert calls[0]["l1_raw_pseudorange_m"] == 21_000_000.0
    assert calls[0]["derived_common_clock_m"] == 15.0
    assert calls[0]["receiver_clock_bias_m"] == -30.0
    assert calls[0]["received_sv_tow_s"] == pytest.approx(120.0)
    assert calls[1]["l1_raw_pseudorange_m"] == 21_000_000.0
    assert calls[1]["derived_common_clock_m"] == 15.0
    assert calls[1]["received_sv_tow_s"] == pytest.approx(120.0)
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
        SvVelocityXEcefMetersPerSecond=4.0,
        SvVelocityYEcefMetersPerSecond=5.0,
        SvVelocityZEcefMetersPerSecond=6.0,
        SvClockDriftMetersPerSecond=-0.03,
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
    assert calls[0]["l1_raw_pseudorange_m"] == 21_000_000.0
    assert calls[0]["derived_common_clock_m"] == 15.0
    assert calls[1]["l1_raw_pseudorange_m"] == 21_000_000.0
    assert calls[1]["derived_common_clock_m"] == 15.0
    l1_slot = products.slot_keys.index((1, 7, "GPS_L1_CA"))
    l5_slot = products.slot_keys.index((1, 7, "GPS_L5_Q"))
    np.testing.assert_allclose(products.sat_ecef[0, l1_slot], adjusted_xyz)
    np.testing.assert_allclose(products.sat_ecef[0, l5_slot], adjusted_xyz)
    assert products.sat_clock_bias_matrix[0, l1_slot] == 42.0
    assert products.sat_clock_bias_matrix[0, l5_slot] == 42.0


def test_fill_observation_matrices_uses_l1_product_for_l5_when_clock_already_matches() -> None:
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
        SvVelocityXEcefMetersPerSecond=4.0,
        SvVelocityYEcefMetersPerSecond=5.0,
        SvVelocityZEcefMetersPerSecond=6.0,
        SvClockDriftMetersPerSecond=-0.03,
    )
    l1_row = _required_row(
        utcTimeMillis=1000,
        Svid=7,
        SignalType="GPS_L1_CA",
        RawPseudorangeMeters=21_000_000.0,
        SvClockBiasMeters=10.0,
        SvPositionXEcefMeters=19_000_000.0,
        SvPositionYEcefMeters=9_000_000.0,
        SvPositionZEcefMeters=20_000_000.0,
        ArrivalTimeNanosSinceGpsEpoch=123_000_000_000.0,
        SvVelocityXEcefMetersPerSecond=7.0,
        SvVelocityYEcefMetersPerSecond=8.0,
        SvVelocityZEcefMetersPerSecond=9.0,
        SvClockDriftMetersPerSecond=-0.04,
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
    np.testing.assert_allclose(products.sat_ecef[0, l5_slot], [19_000_000.0, 9_000_000.0, 20_000_000.0])
    assert products.sat_clock_bias_matrix[0, l1_slot] == 150.003
    assert products.sat_clock_bias_matrix[0, l5_slot] == 150.0
    np.testing.assert_allclose(products.sat_vel[0, l5_slot], [1.0, 2.0, 3.0])
    assert products.sat_clock_drift_mps[0, l5_slot] == -0.5


def test_fill_observation_matrices_keeps_adjusted_gnss_log_only_l1_product_for_l5_clock_match() -> None:
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
        bridge_gnss_log_only=True,
    )
    l1_row = _required_row(
        utcTimeMillis=1000,
        Svid=7,
        SignalType="GPS_L1_CA",
        RawPseudorangeMeters=21_000_000.0,
        SvClockBiasMeters=10.0,
        SvPositionXEcefMeters=19_000_000.0,
        SvPositionYEcefMeters=9_000_000.0,
        SvPositionZEcefMeters=20_000_000.0,
        ArrivalTimeNanosSinceGpsEpoch=123_000_000_000.0,
        ReceivedSvTimeNanos=120_000_000_000.0,
        bridge_gnss_log_only=True,
    )
    epoch = RawEpochObservation(
        time_ms=1000.0,
        group=pd.DataFrame([l5_row, l1_row]),
        baseline_xyz=np.array([1.0, 2.0, 3.0], dtype=np.float64),
        truth_xyz=np.array([4.0, 5.0, 6.0], dtype=np.float64),
    )
    adjusted_xyz = np.array([30_000_000.0, 31_000_000.0, 32_000_000.0], dtype=np.float64)

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
        gps_matrtklib_sat_product_adjustment_fn=lambda **_kwargs: (
            adjusted_xyz,
            np.array([1.0, 2.0, 3.0], dtype=np.float64),
            150.003,
            -0.5,
        ),
        clock_kind_for_observation_fn=lambda const, signal, **_kwargs: 0 if signal == "GPS_L1_CA" else 1,
        is_l5_signal_fn=lambda signal: "L5" in signal,
        slot_sort_key_fn=lambda key: key,
        ecef_to_lla_fn=lambda _x, _y, _z: (0.5, 0.0, 100.0),
        elevation_azimuth_fn=lambda _rx, _sat: (np.deg2rad(30.0), 0.0),
        rtklib_tropo_fn=lambda _lat, _alt, _el: 4.0,
        matlab_signal_clock_dim=7,
    )

    l5_slot = products.slot_keys.index((1, 7, "GPS_L5_Q"))
    np.testing.assert_allclose(products.sat_ecef[0, l5_slot], adjusted_xyz)
    assert products.sat_clock_bias_matrix[0, l5_slot] == 150.0


def test_fill_observation_matrices_ignores_fully_masked_l1_for_l5_product_reuse() -> None:
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
        SvVelocityXEcefMetersPerSecond=4.0,
        SvVelocityYEcefMetersPerSecond=5.0,
        SvVelocityZEcefMetersPerSecond=6.0,
        SvClockDriftMetersPerSecond=-0.03,
    )
    masked_l1_row = _required_row(
        utcTimeMillis=1000,
        Svid=7,
        SignalType="GPS_L1_CA",
        RawPseudorangeMeters=21_000_000.0,
        SvClockBiasMeters=10.0,
        SvPositionXEcefMeters=19_000_000.0,
        SvPositionYEcefMeters=9_000_000.0,
        SvPositionZEcefMeters=20_000_000.0,
        ArrivalTimeNanosSinceGpsEpoch=123_000_000_000.0,
        ReceivedSvTimeNanos=120_000_000_000.0,
        bridge_p_ok=False,
        bridge_d_ok=False,
        bridge_l_ok=False,
        bridge_p_bias_ok=False,
        bridge_gnss_log_only=True,
    )
    epoch = RawEpochObservation(
        time_ms=1000.0,
        group=pd.DataFrame([l5_row, masked_l1_row]),
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

    l5_call = next(call for call in calls if call["derived_common_clock_m"] == 150.0)
    assert l5_call["l1_raw_pseudorange_m"] == 22_000_000.0
    assert l5_call["received_sv_tow_s"] != pytest.approx(120.0)
    l5_slot = products.slot_keys.index((1, 7, "GPS_L5_Q"))
    np.testing.assert_allclose(products.sat_ecef[0, l5_slot], [20_000_000.0, 10_000_000.0, 21_000_000.0])
    assert products.sat_clock_bias_matrix[0, l5_slot] == 150.0


def test_fill_observation_matrices_reuses_masked_l1_timing_for_gnss_log_only_l5_product() -> None:
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
        SvVelocityXEcefMetersPerSecond=4.0,
        SvVelocityYEcefMetersPerSecond=5.0,
        SvVelocityZEcefMetersPerSecond=6.0,
        SvClockDriftMetersPerSecond=-0.03,
        bridge_gnss_log_only=True,
    )
    masked_l1_row = _required_row(
        utcTimeMillis=1000,
        Svid=7,
        SignalType="GPS_L1_CA",
        RawPseudorangeMeters=21_000_000.0,
        SvClockBiasMeters=10.0,
        SvPositionXEcefMeters=19_000_000.0,
        SvPositionYEcefMeters=9_000_000.0,
        SvPositionZEcefMeters=20_000_000.0,
        ArrivalTimeNanosSinceGpsEpoch=123_000_000_000.0,
        ReceivedSvTimeNanos=120_000_000_000.0,
        bridge_p_ok=False,
        bridge_d_ok=False,
        bridge_l_ok=False,
        bridge_p_bias_ok=False,
        bridge_gnss_log_only=True,
    )
    epoch = RawEpochObservation(
        time_ms=1000.0,
        group=pd.DataFrame([l5_row, masked_l1_row]),
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

    l5_call = next(call for call in calls if call["l1_raw_pseudorange_m"] == 21_000_000.0)
    assert l5_call["derived_common_clock_m"] == 15.0
    assert l5_call["received_sv_tow_s"] == pytest.approx(120.0)
    l5_slot = products.slot_keys.index((1, 7, "GPS_L5_Q"))
    np.testing.assert_allclose(products.sat_ecef[0, l5_slot], adjusted_xyz)
    assert products.sat_clock_bias_matrix[0, l5_slot] == 150.0


def test_fill_observation_matrices_reuses_real_masked_l1_for_l5_product() -> None:
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
    masked_l1_row = _required_row(
        utcTimeMillis=1000,
        Svid=7,
        SignalType="GPS_L1_CA",
        RawPseudorangeMeters=21_000_000.0,
        SvClockBiasMeters=10.0,
        SvPositionXEcefMeters=19_000_000.0,
        SvPositionYEcefMeters=9_000_000.0,
        SvPositionZEcefMeters=20_000_000.0,
        ArrivalTimeNanosSinceGpsEpoch=123_000_000_000.0,
        ReceivedSvTimeNanos=120_000_000_000.0,
        bridge_p_ok=False,
        bridge_d_ok=False,
        bridge_l_ok=False,
        bridge_p_bias_ok=False,
    )
    epoch = RawEpochObservation(
        time_ms=1000.0,
        group=pd.DataFrame([l5_row, masked_l1_row]),
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

    l5_call = next(call for call in calls if call["l1_raw_pseudorange_m"] == 21_000_000.0)
    assert l5_call["l1_raw_pseudorange_m"] == 21_000_000.0
    assert l5_call["derived_common_clock_m"] == 15.0
    assert l5_call["received_sv_tow_s"] == pytest.approx(120.0)
    l5_slot = products.slot_keys.index((1, 7, "GPS_L5_Q"))
    np.testing.assert_allclose(products.sat_ecef[0, l5_slot], [19_000_000.0, 9_000_000.0, 20_000_000.0])
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
        SvVelocityXEcefMetersPerSecond=4.0,
        SvVelocityYEcefMetersPerSecond=5.0,
        SvVelocityZEcefMetersPerSecond=6.0,
        SvClockDriftMetersPerSecond=-0.03,
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
    np.testing.assert_allclose(products.sat_vel[0, slot], [1.0, 2.0, 3.0])
    assert products.sat_clock_drift_mps[0, slot] == -0.5

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
