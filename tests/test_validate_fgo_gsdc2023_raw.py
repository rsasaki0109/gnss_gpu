from __future__ import annotations

import csv
from datetime import datetime
import io
import json
import zipfile

import numpy as np
import pandas as pd
from scipy.io import savemat

import experiments.compare_gsdc2023_base_correction_series as base_compare
import experiments.export_gsdc2023_base_correction_series as base_export
import experiments.gsdc2023_raw_bridge as raw_bridge
from experiments.gsdc2023_raw_bridge import (
    BridgeResult,
    ChunkCandidateQuality,
    ChunkSelectionRecord,
    _build_trip_arrays,
    _build_tdcp_arrays,
    _clock_aid_enabled,
    _clock_drift_seed_enabled,
    _ecef_to_enu_relative,
    _effective_multi_gnss_enabled,
    _effective_position_source,
    _estimate_residual_clock_series,
    _export_bridge_outputs,
    _fit_state_with_clock_bias,
    _geometric_range_rate_with_sagnac,
    _geometric_range_with_sagnac,
    _imu_preintegration_segment,
    _enu_to_ecef_relative,
    _mask_doppler_residual_outliers,
    _segment_ranges,
    _select_auto_chunk_source,
    _select_gated_chunk_source,
    _should_refine_outlier_result,
    apply_relative_height_constraint,
    apply_phone_position_offset,
    bridge_position_columns,
    collect_matlab_parity_audit,
    estimate_speed_mps,
    estimate_rpy_from_velocity,
    load_device_imu_measurements,
    preintegrate_processed_imu,
    process_device_imu,
    project_stop_to_epochs,
    ProcessedIMU,
    run_wls,
    solver_stop_mask,
)
from experiments.evaluate import lla_to_ecef
from gnss_gpu.io.nav_rinex import NavMessage
from gnss_gpu.io.rinex import read_rinex_obs


def test_matrtklib_duplicate_nav_filter_matches_effective_eph_selection():
    messages = [
        NavMessage(prn=9, toc=datetime(2021, 12, 8, 19, 59, 44), toe=331184.0, iode=7.0),
        NavMessage(prn=9, toc=datetime(2021, 12, 8, 20, 0, 0), toe=331200.0, iode=95.0),
        NavMessage(prn=9, toc=datetime(2021, 12, 8, 21, 59, 44), toe=338384.0, iode=8.0),
    ]

    filtered = raw_bridge._filter_matrtklib_duplicate_gps_nav_messages(messages)

    assert [int(message.iode) for message in filtered] == [7, 8]
    assert int(raw_bridge._select_gps_nav_message(tuple(messages), 333151.44).iode) == 95
    assert int(raw_bridge._select_gps_nav_message(tuple(filtered), 333151.44).iode) == 7


def test_receiver_clock_bias_lookup_resets_on_time_nanos_gap():
    meta = pd.DataFrame(
        {
            "utcTimeMillis": [1000, 2000, 5000, 6000],
            "TimeNanos": [10_000_000_000, 11_000_000_000, 14_000_000_001, 15_000_000_001],
            "FullBiasNanos": [-1000, -998, -1020, -1018],
            "HardwareClockDiscontinuityCount": [1, 1, 1, 1],
        },
    )

    lookup = raw_bridge._receiver_clock_bias_lookup_from_epoch_meta(meta)

    assert lookup[1000] == 0.0
    assert lookup[2000] == 2.0e-9 * raw_bridge.LIGHT_SPEED_MPS
    assert lookup[5000] == 0.0
    assert lookup[6000] == 2.0e-9 * raw_bridge.LIGHT_SPEED_MPS


def test_receiver_clock_bias_lookup_resets_on_utc_gap_without_time_nanos():
    meta = pd.DataFrame(
        {
            "utcTimeMillis": [1000, 2000, 5000, 6000],
            "FullBiasNanos": [-1000, -998, -1020, -1018],
            "HardwareClockDiscontinuityCount": [1, 1, 1, 1],
        },
    )

    lookup = raw_bridge._receiver_clock_bias_lookup_from_epoch_meta(meta)

    assert lookup[1000] == 0.0
    assert lookup[2000] == 2.0e-9 * raw_bridge.LIGHT_SPEED_MPS
    assert lookup[5000] == 0.0
    assert lookup[6000] == 2.0e-9 * raw_bridge.LIGHT_SPEED_MPS


def test_geometric_range_with_sagnac_matches_rtklib_formula():
    sat = np.array([[20_200_000.0, 14_000_000.0, 21_700_000.0]])
    rx = np.array([1_300_000.0, -4_700_000.0, 3_900_000.0])
    expected = np.linalg.norm(sat[0] - rx) + 7.2921151467e-5 * (sat[0, 0] * rx[1] - sat[0, 1] * rx[0]) / 299792458.0

    actual = _geometric_range_with_sagnac(sat, rx)

    assert actual.shape == (1,)
    assert actual[0] == expected


def test_pseudorange_residual_mask_uses_receiver_clock_global_isb():
    rx = np.array([-3947460.0, 3431490.0, 3637870.0], dtype=np.float64)
    offsets = np.array(
        [
            [2.1e7, 0.0, 0.0],
            [0.0, 2.1e7, 0.0],
            [0.0, 0.0, 2.1e7],
            [-1.5e7, 1.5e7, 0.0],
            [1.2e7, 0.8e7, 1.4e7],
            [-1.1e7, 0.9e7, 1.3e7],
            [1.0e7, -1.2e7, 1.1e7],
            [0.9e7, 1.1e7, -1.2e7],
            [-1.3e7, -0.7e7, 1.0e7],
        ],
        dtype=np.float64,
    )
    sat_one_epoch = rx[None, :] + offsets
    sat_ecef = np.stack([sat_one_epoch, sat_one_epoch], axis=0)
    reference_xyz = np.repeat(rx[None, :], 2, axis=0)
    ranges = _geometric_range_with_sagnac(sat_one_epoch, rx)
    receiver_clock = np.array([100.0, 100.0], dtype=np.float64)
    isb = 10.0
    pseudorange = np.vstack([ranges + receiver_clock[0] + isb, ranges + receiver_clock[1] + isb])
    pseudorange[1, 4:] += 50.0
    weights = np.ones((2, offsets.shape[0]), dtype=np.float64)

    masked = raw_bridge._mask_pseudorange_residual_outliers(
        sat_ecef,
        pseudorange,
        weights,
        reference_xyz,
        threshold_m=20.0,
        receiver_clock_bias_m=receiver_clock,
        common_bias_group=np.zeros(offsets.shape[0], dtype=np.int32),
    )

    assert masked == 5
    assert np.all(weights[0] > 0.0)
    assert np.all(weights[1, :4] > 0.0)
    assert np.all(weights[1, 4:] == 0.0)


def test_pseudorange_residual_mask_uses_separate_isb_sample_weights():
    rx = np.array([-3947460.0, 3431490.0, 3637870.0], dtype=np.float64)
    offsets = np.array(
        [
            [2.1e7, 0.0, 0.0],
            [0.0, 2.1e7, 0.0],
            [0.0, 0.0, 2.1e7],
            [-1.5e7, 1.5e7, 0.0],
            [1.2e7, 0.8e7, 1.4e7],
            [-1.1e7, 0.9e7, 1.3e7],
            [1.0e7, -1.2e7, 1.1e7],
            [0.9e7, 1.1e7, -1.2e7],
            [-1.3e7, -0.7e7, 1.0e7],
        ],
        dtype=np.float64,
    )
    sat_ecef = (rx[None, :] + offsets).reshape(1, offsets.shape[0], 3)
    reference_xyz = rx.reshape(1, 3)
    ranges = _geometric_range_with_sagnac(sat_ecef[0], rx)
    receiver_clock = np.array([100.0], dtype=np.float64)
    pseudorange = ranges.reshape(1, -1) + receiver_clock[0] + 10.0
    pseudorange[0, :4] += 10.0
    weights = np.zeros((1, offsets.shape[0]), dtype=np.float64)
    weights[0, :4] = 1.0
    sample_weights = np.ones_like(weights)

    masked = raw_bridge._mask_pseudorange_residual_outliers(
        sat_ecef,
        pseudorange,
        weights,
        reference_xyz,
        threshold_m=5.0,
        receiver_clock_bias_m=receiver_clock,
        common_bias_group=np.zeros(offsets.shape[0], dtype=np.int32),
        common_bias_sample_weights=sample_weights,
    )

    assert masked == 4
    assert np.all(weights[0, :4] == 0.0)


def test_pseudorange_residual_mask_uses_receiver_clock_for_single_observation_epoch():
    rx = np.array([-3947460.0, 3431490.0, 3637870.0], dtype=np.float64)
    sat_ecef = (rx + np.array([2.1e7, 0.0, 0.0], dtype=np.float64)).reshape(1, 1, 3)
    reference_xyz = rx.reshape(1, 3)
    pseudorange = _geometric_range_with_sagnac(sat_ecef[0], rx).reshape(1, 1) + 100.0
    weights = np.ones((1, 1), dtype=np.float64)

    masked = raw_bridge._mask_pseudorange_residual_outliers(
        sat_ecef,
        pseudorange,
        weights,
        reference_xyz,
        threshold_m=20.0,
        receiver_clock_bias_m=np.array([0.0], dtype=np.float64),
        common_bias_group=np.array([0], dtype=np.int32),
        common_bias_by_group={0: 0.0},
    )

    assert masked == 1
    assert weights[0, 0] == 0.0


def test_build_trip_arrays_uses_full_epoch_span_for_pseudorange_isb(tmp_path, monkeypatch):
    trip = tmp_path / "dataset_2023" / "train" / "course" / "phone"
    trip.mkdir(parents=True)
    monkeypatch.setattr(
        raw_bridge,
        "_receiver_clock_bias_lookup_from_epoch_meta",
        lambda _epoch_meta: {1000: 0.0, 2000: 0.0, 3000: 0.0},
    )

    rx = np.array([-3947460.0, 3431490.0, 3637870.0], dtype=np.float64)
    sat_offsets = np.array(
        [
            [2.1e7, 0.0, 0.0],
            [0.0, 2.1e7, 0.0],
            [0.0, 0.0, 2.1e7],
            [-1.5e7, 1.5e7, 0.0],
            [1.2e7, 0.8e7, 1.4e7],
        ],
        dtype=np.float64,
    )
    sat_ecef = rx[None, :] + sat_offsets
    ranges = _geometric_range_with_sagnac(sat_ecef, rx)
    rows = []
    for utc_ms, isb_m in ((1000, 100.0), (2000, 0.0), (3000, -100.0)):
        for svid, (sat, range_m) in enumerate(zip(sat_ecef, ranges), start=1):
            rows.append(
                {
                    "utcTimeMillis": utc_ms,
                    "Svid": svid,
                    "ConstellationType": 1,
                    "SignalType": "GPS_L1_CA",
                    "RawPseudorangeMeters": float(range_m + isb_m),
                    "IonosphericDelayMeters": 0.0,
                    "TroposphericDelayMeters": 0.0,
                    "SvClockBiasMeters": 0.0,
                    "SvPositionXEcefMeters": sat[0],
                    "SvPositionYEcefMeters": sat[1],
                    "SvPositionZEcefMeters": sat[2],
                    "SvElevationDegrees": 35.0,
                    "Cn0DbHz": 35.0,
                    "WlsPositionXEcefMeters": rx[0],
                    "WlsPositionYEcefMeters": rx[1],
                    "WlsPositionZEcefMeters": rx[2],
                    "FullBiasNanos": -1.0e18,
                    "State": 1 | 8,
                    "MultipathIndicator": 0,
                },
            )
    _write_zipped_csv(trip / "device_gnss.csv", rows, list(rows[0].keys()))

    batch = _build_trip_arrays(
        trip,
        max_epochs=1,
        start_epoch=0,
        constellation_type=1,
        signal_type="GPS_L1_CA",
        weight_mode="sin2el",
        apply_observation_mask=True,
        pseudorange_residual_mask_m=0.0,
        doppler_residual_mask_mps=0.0,
        pseudorange_doppler_mask_m=0.0,
    )

    assert batch.times_ms.tolist() == [1000.0]
    assert batch.pseudorange_isb_by_group is not None
    assert np.isclose(batch.pseudorange_isb_by_group[0], 0.0, atol=1e-6)


def test_doppler_residual_mask_uses_receiver_drift_for_single_observation_epoch():
    rx = np.array([-3947460.0, 3431490.0, 3637870.0], dtype=np.float64)
    sat_ecef = (rx + np.array([2.1e7, 0.0, 0.0], dtype=np.float64)).reshape(1, 1, 3)
    sat_vel = np.zeros_like(sat_ecef)
    doppler = np.array([[100.0]], dtype=np.float64)
    doppler_weights = np.ones((1, 1), dtype=np.float64)

    masked = _mask_doppler_residual_outliers(
        np.array([1000.0], dtype=np.float64),
        sat_ecef,
        sat_vel,
        doppler,
        doppler_weights,
        rx.reshape(1, 3),
        threshold_mps=3.0,
        receiver_clock_drift_mps=np.array([0.0], dtype=np.float64),
    )

    assert masked == 1
    assert doppler_weights[0, 0] == 0.0


def test_doppler_residual_mask_uses_velocity_context_at_window_endpoint():
    times_ms = np.array([0.0, 1000.0, 2000.0], dtype=np.float64)
    rx0 = np.array([6378137.0, 0.0, 0.0], dtype=np.float64)
    reference_xyz = np.array(
        [
            rx0,
            rx0,
            rx0 + np.array([10.0, 0.0, 0.0], dtype=np.float64),
        ],
        dtype=np.float64,
    )
    sat_ecef = np.tile((rx0 + np.array([2.1e7, 0.0, 0.0], dtype=np.float64)).reshape(1, 1, 3), (3, 1, 1))
    sat_vel = np.zeros_like(sat_ecef)
    doppler = np.zeros((3, 1), dtype=np.float64)
    doppler_weights = np.array([[0.0], [0.0], [1.0]], dtype=np.float64)

    masked_without_context = _mask_doppler_residual_outliers(
        times_ms,
        sat_ecef,
        sat_vel,
        doppler,
        doppler_weights.copy(),
        reference_xyz,
        threshold_mps=3.0,
        receiver_clock_drift_mps=np.zeros(3, dtype=np.float64),
    )
    weights_with_context = doppler_weights.copy()
    masked_with_context = _mask_doppler_residual_outliers(
        times_ms,
        sat_ecef,
        sat_vel,
        doppler,
        weights_with_context,
        reference_xyz,
        threshold_mps=3.0,
        receiver_clock_drift_mps=np.zeros(3, dtype=np.float64),
        velocity_times_ms=np.array([0.0, 1000.0, 2000.0, 3000.0], dtype=np.float64),
        velocity_reference_xyz=np.array(
            [
                rx0,
                rx0,
                rx0 + np.array([10.0, 0.0, 0.0], dtype=np.float64),
                rx0,
            ],
            dtype=np.float64,
        ),
    )

    assert masked_without_context == 1
    assert masked_with_context == 0
    assert weights_with_context[2, 0] > 0.0


def test_doppler_residual_mask_uses_velocity_context_only_at_window_edges():
    times_ms = np.array([0.0, 1000.0, 2000.0], dtype=np.float64)
    rx0 = np.array([6378137.0, 0.0, 0.0], dtype=np.float64)
    receiver_xyz = np.array(
        [
            rx0,
            rx0 + np.array([10.0, 0.0, 0.0], dtype=np.float64),
            rx0 + np.array([20.0, 0.0, 0.0], dtype=np.float64),
        ],
        dtype=np.float64,
    )
    sat_xyz = (rx0 + np.array([2.1e7, 0.0, 0.0], dtype=np.float64)).reshape(1, 1, 3)
    sat_vel = np.zeros_like(sat_xyz)
    window_velocity = raw_bridge._receiver_velocity_from_reference(times_ms, receiver_xyz)
    window_model = _geometric_range_rate_with_sagnac(
        np.tile(sat_xyz, (3, 1, 1)),
        receiver_xyz[:, None, :],
        np.tile(sat_vel, (3, 1, 1)),
        window_velocity[:, None, :],
    )
    doppler = np.zeros((3, 1), dtype=np.float64)
    doppler[1, 0] = -float(window_model[1, 0])
    doppler_weights = np.array([[0.0], [1.0], [0.0]], dtype=np.float64)

    masked = _mask_doppler_residual_outliers(
        times_ms,
        np.tile(sat_xyz, (3, 1, 1)),
        np.tile(sat_vel, (3, 1, 1)),
        doppler,
        doppler_weights,
        receiver_xyz,
        threshold_mps=3.0,
        receiver_clock_drift_mps=np.zeros(3, dtype=np.float64),
        velocity_times_ms=times_ms,
        velocity_reference_xyz=np.array(
            [
                rx0,
                rx0 + np.array([10.0, 0.0, 0.0], dtype=np.float64),
                rx0 + np.array([200.0, 0.0, 0.0], dtype=np.float64),
            ],
            dtype=np.float64,
        ),
    )

    assert masked == 0
    assert doppler_weights[1, 0] > 0.0


def test_doppler_residual_mask_uses_clock_drift_context_at_window_endpoint():
    times_ms = np.array([0.0, 1000.0, 2000.0], dtype=np.float64)
    rx0 = np.array([6378137.0, 0.0, 0.0], dtype=np.float64)
    sat_ecef = np.tile((rx0 + np.array([2.1e7, 0.0, 0.0], dtype=np.float64)).reshape(1, 1, 3), (3, 1, 1))
    sat_vel = np.zeros_like(sat_ecef)
    doppler = np.array([[0.0], [0.0], [4.0]], dtype=np.float64)
    doppler_weights = np.array([[0.0], [0.0], [1.0]], dtype=np.float64)

    masked_without_context = _mask_doppler_residual_outliers(
        times_ms,
        sat_ecef,
        sat_vel,
        doppler,
        doppler_weights.copy(),
        np.tile(rx0, (3, 1)),
        threshold_mps=3.0,
        receiver_clock_drift_mps=np.zeros(3, dtype=np.float64),
    )
    weights_with_context = doppler_weights.copy()
    masked_with_context = _mask_doppler_residual_outliers(
        times_ms,
        sat_ecef,
        sat_vel,
        doppler,
        weights_with_context,
        np.tile(rx0, (3, 1)),
        threshold_mps=3.0,
        receiver_clock_drift_mps=np.zeros(3, dtype=np.float64),
        clock_drift_times_ms=np.array([0.0, 1000.0, 2000.0, 3000.0], dtype=np.float64),
        clock_drift_reference_mps=np.array([0.0, 0.0, 4.0, 0.0], dtype=np.float64),
    )

    assert masked_without_context == 1
    assert masked_with_context == 0
    assert weights_with_context[2, 0] > 0.0


def test_doppler_residual_mask_uses_clock_drift_context_only_at_window_edges():
    times_ms = np.array([0.0, 1000.0, 2000.0], dtype=np.float64)
    rx0 = np.array([6378137.0, 0.0, 0.0], dtype=np.float64)
    sat_ecef = np.tile((rx0 + np.array([2.1e7, 0.0, 0.0], dtype=np.float64)).reshape(1, 1, 3), (3, 1, 1))
    sat_vel = np.zeros_like(sat_ecef)
    doppler = np.array([[0.0], [4.0], [0.0]], dtype=np.float64)
    doppler_weights = np.array([[0.0], [1.0], [0.0]], dtype=np.float64)

    masked = _mask_doppler_residual_outliers(
        times_ms,
        sat_ecef,
        sat_vel,
        doppler,
        doppler_weights,
        np.tile(rx0, (3, 1)),
        threshold_mps=3.0,
        receiver_clock_drift_mps=np.zeros(3, dtype=np.float64),
        clock_drift_times_ms=times_ms,
        clock_drift_reference_mps=np.array([0.0, 4.0, 0.0], dtype=np.float64),
    )

    assert masked == 1
    assert doppler_weights[1, 0] == 0.0


def test_repair_baseline_wls_interpolates_stale_single_epoch_jump():
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

    repaired = raw_bridge._repair_baseline_wls(times_ms, xyz)

    np.testing.assert_allclose(repaired[4], 0.5 * (xyz[3] + xyz[5]), atol=1e-9)


def _write_zipped_csv(path, rows, fieldnames):
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(path.name, buf.getvalue())


def _rinex_header_line(body, label):
    return f"{body:<60}{label}\n"


def _rinex_obs_field(value):
    return f"{value:14.3f}  "


def test_read_rinex2_observation_multiline_codes(tmp_path):
    rinex_path = tmp_path / "base.obs"
    rinex_path.write_text(
        "".join(
            [
                _rinex_header_line("     2.11           OBSERVATION DATA    M (MIXED)", "RINEX VERSION / TYPE"),
                _rinex_header_line("     6    L1    L2    C1    P2    P1    C5", "# / TYPES OF OBSERV"),
                _rinex_header_line("    30.0000", "INTERVAL"),
                _rinex_header_line("", "END OF HEADER"),
                " 20 12 10  0  0  0.0000000  0  1G05\n",
                "".join(_rinex_obs_field(v) for v in [1.0, 2.0, 21000003.0, 4.0, 21000005.0]) + "\n",
                _rinex_obs_field(21000006.0) + "\n",
            ],
        ),
        encoding="utf-8",
    )

    obs = read_rinex_obs(rinex_path)

    assert len(obs.epochs) == 1
    assert obs.header.obs_types[""] == ["L1", "L2", "C1", "P2", "P1", "C5"]
    assert obs.epochs[0].satellites == ["G05"]
    sat_obs = obs.epochs[0].observations["G05"]
    assert sat_obs["C1"] == 21000003.0
    assert sat_obs["P1"] == 21000005.0
    assert sat_obs["C5"] == 21000006.0


def test_build_trip_arrays_from_raw_zip(tmp_path):
    trip = tmp_path / "dataset_2023" / "train" / "course" / "phone"
    trip.mkdir(parents=True)

    gt = pd.DataFrame(
        [
            {
                "MessageType": "Fix",
                "Provider": "GT",
                "LatitudeDegrees": 35.0,
                "LongitudeDegrees": 139.0,
                "AltitudeMeters": 10.0,
                "UnixTimeMillis": 1000,
            },
            {
                "MessageType": "Fix",
                "Provider": "GT",
                "LatitudeDegrees": 35.00001,
                "LongitudeDegrees": 139.00001,
                "AltitudeMeters": 10.0,
                "UnixTimeMillis": 2000,
            },
        ],
    )
    gt.to_csv(trip / "ground_truth.csv", index=False)

    rows = []
    for utc_ms in (1000, 2000):
        for svid in range(1, 5):
            rows.append(
                {
                    "utcTimeMillis": utc_ms,
                    "Svid": svid,
                    "ConstellationType": 1,
                    "SignalType": "GPS_L1_CA",
                    "RawPseudorangeMeters": 2.1e7 + 1000 * svid,
                    "IonosphericDelayMeters": 2.0,
                    "TroposphericDelayMeters": 3.0,
                    "SvClockBiasMeters": 10.0,
                    "SvPositionXEcefMeters": 2.6e7 - 1e5 * svid,
                    "SvPositionYEcefMeters": 1.3e7 + 2e5 * svid,
                    "SvPositionZEcefMeters": 2.1e7 - 3e5 * svid,
                    "SvElevationDegrees": 30.0 + svid,
                    "Cn0DbHz": 35.0 + svid,
                    "WlsPositionXEcefMeters": -3947460.0 + utc_ms * 0.001,
                    "WlsPositionYEcefMeters": 3431490.0 + utc_ms * 0.001,
                    "WlsPositionZEcefMeters": 3637870.0 + utc_ms * 0.001,
                },
            )
            # lower-C/N0 duplicate should be dropped
            rows.append(
                {
                    "utcTimeMillis": utc_ms,
                    "Svid": svid,
                    "ConstellationType": 1,
                    "SignalType": "GPS_L1_CA",
                    "RawPseudorangeMeters": 2.1e7 + 1000 * svid + 500.0,
                    "IonosphericDelayMeters": 2.0,
                    "TroposphericDelayMeters": 3.0,
                    "SvClockBiasMeters": 10.0,
                    "SvPositionXEcefMeters": 2.6e7 - 1e5 * svid,
                    "SvPositionYEcefMeters": 1.3e7 + 2e5 * svid,
                    "SvPositionZEcefMeters": 2.1e7 - 3e5 * svid,
                    "SvElevationDegrees": 30.0 + svid,
                    "Cn0DbHz": 10.0,
                    "WlsPositionXEcefMeters": -3947460.0 + utc_ms * 0.001,
                    "WlsPositionYEcefMeters": 3431490.0 + utc_ms * 0.001,
                    "WlsPositionZEcefMeters": 3637870.0 + utc_ms * 0.001,
                },
            )

    _write_zipped_csv(trip / "device_gnss.csv", rows, list(rows[0].keys()))

    batch = _build_trip_arrays(
        trip,
        max_epochs=10,
        start_epoch=0,
        constellation_type=1,
        signal_type="GPS_L1_CA",
        weight_mode="sin2el",
    )

    assert batch.sat_ecef.shape == (2, 4, 3)
    assert batch.pseudorange.shape == (2, 4)
    assert batch.weights.shape == (2, 4)
    assert batch.kaggle_wls.shape == (2, 3)
    assert batch.truth.shape == (2, 3)
    expected_pr = 2.1e7 + 1000 * 1 + 10.0 - 2.0 - 3.0
    assert np.isclose(batch.pseudorange[0, 0], expected_pr)
    assert batch.weights[0, 0] > 0.0
    assert batch.has_truth is True


def test_build_trip_arrays_applies_base_correction_when_enabled(tmp_path, monkeypatch):
    data_root = tmp_path / "dataset_2023"
    trip = data_root / "train" / "course" / "phone"
    trip.mkdir(parents=True)

    rows = []
    for svid in range(1, 5):
        rows.append(
            {
                "utcTimeMillis": 1000,
                "Svid": svid,
                "ConstellationType": 1,
                "SignalType": "GPS_L1_CA",
                "RawPseudorangeMeters": 2.1e7 + 1000 * svid,
                "IonosphericDelayMeters": 2.0,
                "TroposphericDelayMeters": 3.0,
                "SvClockBiasMeters": 10.0,
                "SvPositionXEcefMeters": 2.6e7 - 1e5 * svid,
                "SvPositionYEcefMeters": 1.3e7 + 2e5 * svid,
                "SvPositionZEcefMeters": 2.1e7 - 3e5 * svid,
                "SvElevationDegrees": 30.0 + svid,
                "Cn0DbHz": 35.0 + svid,
                "WlsPositionXEcefMeters": -3947460.0,
                "WlsPositionYEcefMeters": 3431490.0,
                "WlsPositionZEcefMeters": 3637870.0,
            },
        )
    _write_zipped_csv(trip / "device_gnss.csv", rows, list(rows[0].keys()))

    def fake_base_correction(data_root_arg, trip_arg, times_ms, slot_keys, signal_type):
        assert data_root_arg == data_root
        assert trip_arg == "train/course/phone"
        assert signal_type == "GPS_L1_CA"
        assert times_ms.tolist() == [1000.0]
        assert slot_keys == [
            (1, 1, "GPS_L1_CA"),
            (1, 2, "GPS_L1_CA"),
            (1, 3, "GPS_L1_CA"),
            (1, 4, "GPS_L1_CA"),
        ]
        return np.array([[4.5, np.nan, 1.25, np.nan]], dtype=np.float64)

    monkeypatch.setattr(raw_bridge, "compute_base_pseudorange_correction_matrix", fake_base_correction)
    batch = _build_trip_arrays(
        trip,
        max_epochs=10,
        start_epoch=0,
        constellation_type=1,
        signal_type="GPS_L1_CA",
        weight_mode="sin2el",
        apply_base_correction=True,
        data_root=data_root,
        trip="train/course/phone",
    )

    expected_pr1 = 2.1e7 + 1000 * 1 + 10.0 - 2.0 - 3.0
    expected_pr2 = 2.1e7 + 1000 * 2 + 10.0 - 2.0 - 3.0
    assert np.isclose(batch.pseudorange[0, 0], expected_pr1 - 4.5)
    assert np.isclose(batch.pseudorange[0, 1], expected_pr2)
    assert np.isclose(batch.pseudorange[0, 2], 2.1e7 + 3000 + 10.0 - 2.0 - 3.0 - 1.25)
    assert batch.base_correction_count == 2


def test_base_pseudorange_observation_selector_handles_rinex3_signal_codes():
    obs = {
        "C1C": 21_000_001.0,
        "C1X": 21_000_002.0,
        "C5Q": 22_000_005.0,
    }

    assert raw_bridge._select_base_pseudorange_observation(obs, "GPS_L1_CA") == ("C1C", 21_000_001.0)
    assert raw_bridge._select_base_pseudorange_observation(obs, "GAL_E1_C_P") == ("C1C", 21_000_001.0)
    assert raw_bridge._select_base_pseudorange_observation(obs, "GPS_L5_Q") == ("C5Q", 22_000_005.0)
    assert raw_bridge._select_base_pseudorange_observation({"C1X": 21_000_010.0}, "GAL_E1_C_P") == (
        "C1X",
        21_000_010.0,
    )


def test_base_iono_scale_matches_matrtklib_frequency_compensation():
    assert raw_bridge._signal_type_iono_scale("GPS_L1_CA") == 1.0
    assert raw_bridge._signal_type_iono_scale("GAL_E1_C_P") == 1.0
    assert np.isclose(raw_bridge._signal_type_iono_scale("GPS_L5_Q"), raw_bridge.GPS_L5_TGD_SCALE)
    assert np.isclose(raw_bridge._signal_type_iono_scale("GAL_E5A_Q"), raw_bridge.GPS_L5_TGD_SCALE)


def test_base_position_can_match_matlab_pre_offset_residual_coordinate(tmp_path):
    data_root = tmp_path / "dataset_2023"
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    pd.DataFrame(
        [
            {
                "Base": "slac",
                "Year": 2020,
                "X": -2700112.7,
                "Y": -4292747.3,
                "Z": 3855195.5,
            },
        ],
    ).to_csv(base_dir / "base_position.csv", index=False)
    pd.DataFrame([{"Base": "slac", "E": 1.0, "N": 2.0, "U": 3.0}]).to_csv(
        base_dir / "base_offset.csv",
        index=False,
    )

    no_offset = raw_bridge._read_base_station_xyz(data_root, "2020-08-04-course", "slac", apply_offset=False)
    with_offset = raw_bridge._read_base_station_xyz(data_root, "2020-08-04-course", "slac")

    np.testing.assert_allclose(no_offset, np.array([-2700112.7, -4292747.3, 3855195.5]))
    assert np.linalg.norm(with_offset - no_offset) > 1.0


def test_base_time_span_mask_rounds_like_matrtklib_select_time_span():
    base_times = np.arange(0.0, 181.0, 30.0)
    mask = raw_bridge._matlab_base_time_span_mask(
        base_times,
        phone_start_gps_s=209.0,
        phone_end_gps_s=239.0,
        base_dt_s=30.0,
    )

    np.testing.assert_array_equal(mask, np.array([False, True, True, True, True, True, True]))


def test_compute_base_pseudorange_correction_matrix_covers_multi_gnss(tmp_path, monkeypatch):
    data_root = tmp_path / "dataset_2023"
    times_ms = np.array([1000.0, 2000.0], dtype=np.float64)
    phone_times = raw_bridge._unix_ms_to_gps_abs_seconds(times_ms)
    calls: list[tuple[str, tuple[str, ...]]] = []

    monkeypatch.setattr(raw_bridge, "_base_setting", lambda *args: ("BASE", "V3"))

    def fake_load_base_residuals(
        data_root_str,
        split,
        course,
        base_name,
        rinex_type,
        signal_type,
        phone_start_gps_s,
        phone_end_gps_s,
        sat_ids_key,
    ):
        assert data_root_str == str(data_root)
        assert (split, course, base_name, rinex_type) == ("train", "course", "BASE", "V3")
        assert np.isclose(phone_start_gps_s, phone_times[0])
        assert np.isclose(phone_end_gps_s, phone_times[-1])
        calls.append((signal_type, sat_ids_key))
        residuals = np.zeros((phone_times.size, len(sat_ids_key)), dtype=np.float64)
        for col, sat_id in enumerate(sat_ids_key):
            residuals[:, col] = 10.0 * (col + 1) + len(str(sat_id))
        return phone_times, sat_ids_key, residuals

    monkeypatch.setattr(raw_bridge, "_load_base_residual_series_cached", fake_load_base_residuals)

    correction = raw_bridge.compute_base_pseudorange_correction_matrix(
        data_root,
        "train/course/phone",
        times_ms,
        [
            (1, 3, "GPS_L1_CA"),
            (6, 11, "GAL_E1_C_P"),
            (4, 193, "QZS_L1_CA"),
            (3, 1, "BDS_B1I"),
        ],
        "GPS_L1_CA",
    )

    assert {call[1] for call in calls} == {("G03",), ("E11",), ("J01",)}
    np.testing.assert_allclose(correction[:, 0], np.array([13.0, 13.0]))
    np.testing.assert_allclose(correction[:, 1], np.array([13.0, 13.0]))
    np.testing.assert_allclose(correction[:, 2], np.array([13.0, 13.0]))
    assert np.isnan(correction[:, 3]).all()


def test_export_base_correction_series_writes_slot_matrix(tmp_path, monkeypatch):
    data_root = tmp_path / "dataset_2023"
    trip = data_root / "train" / "course" / "phone"
    trip.mkdir(parents=True)

    rows = []
    for svid in range(1, 5):
        rows.append(
            {
                "utcTimeMillis": 1000,
                "Svid": svid,
                "ConstellationType": 1,
                "SignalType": "GPS_L1_CA",
                "RawPseudorangeMeters": 2.1e7 + 1000 * svid,
                "IonosphericDelayMeters": 2.0,
                "TroposphericDelayMeters": 3.0,
                "SvClockBiasMeters": 10.0,
                "SvPositionXEcefMeters": 2.6e7 - 1e5 * svid,
                "SvPositionYEcefMeters": 1.3e7 + 2e5 * svid,
                "SvPositionZEcefMeters": 2.1e7 - 3e5 * svid,
                "SvElevationDegrees": 30.0 + svid,
                "Cn0DbHz": 35.0 + svid,
                "WlsPositionXEcefMeters": -3947460.0,
                "WlsPositionYEcefMeters": 3431490.0,
                "WlsPositionZEcefMeters": 3637870.0,
            },
        )
    _write_zipped_csv(trip / "device_gnss.csv", rows, list(rows[0].keys()))

    def fake_base_correction(data_root_arg, trip_arg, times_ms, slot_keys, signal_type):
        assert data_root_arg == data_root
        assert trip_arg == "train/course/phone"
        assert times_ms.tolist() == [1000.0]
        assert slot_keys == [
            (1, 1, "GPS_L1_CA"),
            (1, 2, "GPS_L1_CA"),
            (1, 3, "GPS_L1_CA"),
            (1, 4, "GPS_L1_CA"),
        ]
        assert signal_type == "GPS_L1_CA"
        return np.array([[1.0, np.nan, -2.5, 0.5]], dtype=np.float64)

    monkeypatch.setattr(base_export, "compute_base_pseudorange_correction_matrix", fake_base_correction)
    monkeypatch.setattr(
        base_export,
        "collect_matlab_parity_audit",
        lambda data_root_arg, trip_arg: {
            "base_correction_status": "base_correction_ready",
            "base_correction_ready": True,
        },
    )

    out_dir = tmp_path / "export"
    summary = base_export.export_base_correction_series(
        data_root,
        "train/course/phone",
        out_dir,
        max_epochs=10,
        start_epoch=0,
        constellation_type=1,
        signal_type="GPS_L1_CA",
        weight_mode="sin2el",
        multi_gnss=False,
        dual_frequency=False,
    )

    wide = pd.read_csv(out_dir / "base_correction_wide.csv")
    long_df = pd.read_csv(out_dir / "base_correction_long.csv")
    meta = json.loads((out_dir / "base_correction_summary.json").read_text(encoding="utf-8"))

    assert summary["finite_correction_count"] == 3
    assert meta["applied_correction_count"] == 3
    assert list(wide.columns) == [
        "UnixTimeMillis",
        "c1_s01_GPS_L1_CA",
        "c1_s02_GPS_L1_CA",
        "c1_s03_GPS_L1_CA",
        "c1_s04_GPS_L1_CA",
    ]
    assert np.isclose(wide.loc[0, "c1_s03_GPS_L1_CA"], -2.5)
    assert len(long_df) == 4
    assert long_df["CorrectionMeters"].notna().sum() == 3


def test_compare_base_correction_series_joins_matlab_and_bridge_csv(tmp_path):
    matlab_csv = tmp_path / "matlab.csv"
    bridge_csv = tmp_path / "bridge_long.csv"
    out_dir = tmp_path / "comparison"
    pd.DataFrame(
        [
            {
                "freq": "L1",
                "epoch_index": 1,
                "utcTimeMillis": 1000,
                "sys": 1,
                "svid": 3,
                "correction_m": 10.0,
            },
            {
                "freq": "L1",
                "epoch_index": 1,
                "utcTimeMillis": 1000,
                "sys": 8,
                "svid": 11,
                "correction_m": -2.0,
            },
            {
                "freq": "L5",
                "epoch_index": 2,
                "utcTimeMillis": 2000,
                "sys": 1,
                "svid": 5,
                "correction_m": 5.0,
            },
        ],
    ).to_csv(matlab_csv, index=False)
    pd.DataFrame(
        [
            {
                "UnixTimeMillis": 1000,
                "EpochIndex": 0,
                "ConstellationType": 1,
                "Svid": 3,
                "SignalType": "GPS_L1_CA",
                "CorrectionMeters": 10.5,
                "ObservationWeightPositive": True,
            },
            {
                "UnixTimeMillis": 1000,
                "EpochIndex": 0,
                "ConstellationType": 6,
                "Svid": 11,
                "SignalType": "GAL_E1_C_P",
                "CorrectionMeters": -1.0,
                "ObservationWeightPositive": True,
            },
            {
                "UnixTimeMillis": 3000,
                "EpochIndex": 2,
                "ConstellationType": 4,
                "Svid": 1,
                "SignalType": "QZS_L1_CA",
                "CorrectionMeters": 7.0,
                "ObservationWeightPositive": True,
            },
        ],
    ).to_csv(bridge_csv, index=False)

    merged, by_freq_sys, summary = base_compare.compare_base_correction_series(
        matlab_csv,
        bridge_csv,
        out_dir,
    )

    assert summary["matched_count"] == 2
    assert summary["matlab_only_count"] == 1
    assert summary["bridge_only_count"] == 1
    assert np.isclose(summary["median_abs_delta_m"], 0.75)
    assert (out_dir / "base_correction_series_comparison.csv").is_file()
    assert set(merged["_merge"].astype(str)) == {"both", "left_only", "right_only"}
    gal = by_freq_sys[(by_freq_sys["freq"] == "L1") & (by_freq_sys["sys"] == 8)].iloc[0]
    assert gal["matched_count"] == 1


def test_build_trip_arrays_keeps_l1_l5_as_dual_frequency_slots(tmp_path):
    trip = tmp_path / "dataset_2023" / "train" / "course" / "phone"
    trip.mkdir(parents=True)

    rows = []
    for svid in range(1, 5):
        for signal_type, extra_pr, cn0 in (("GPS_L1_CA", 0.0, 35.0), ("GPS_L5_Q", 25.0, 32.0)):
            rows.append(
                {
                    "utcTimeMillis": 1000,
                    "Svid": svid,
                    "ConstellationType": 1,
                    "SignalType": signal_type,
                    "RawPseudorangeMeters": 2.1e7 + 1000 * svid + extra_pr,
                    "IonosphericDelayMeters": 2.0,
                    "TroposphericDelayMeters": 3.0,
                    "SvClockBiasMeters": 10.0,
                    "SvPositionXEcefMeters": 2.6e7 - 1e5 * svid,
                    "SvPositionYEcefMeters": 1.3e7 + 2e5 * svid,
                    "SvPositionZEcefMeters": 2.1e7 - 3e5 * svid,
                    "SvElevationDegrees": 30.0 + svid,
                    "Cn0DbHz": cn0 + svid,
                    "WlsPositionXEcefMeters": -3947460.0,
                    "WlsPositionYEcefMeters": 3431490.0,
                    "WlsPositionZEcefMeters": 3637870.0,
                },
            )
    _write_zipped_csv(trip / "device_gnss.csv", rows, list(rows[0].keys()))

    single = _build_trip_arrays(
        trip,
        max_epochs=10,
        start_epoch=0,
        constellation_type=1,
        signal_type="GPS_L1_CA",
        weight_mode="sin2el",
    )
    dual = _build_trip_arrays(
        trip,
        max_epochs=10,
        start_epoch=0,
        constellation_type=1,
        signal_type="GPS_L1_CA",
        weight_mode="sin2el",
        dual_frequency=True,
    )

    assert single.n_sat_slots == 4
    assert dual.n_sat_slots == 8
    assert dual.slot_keys == (
        (1, 1, "GPS_L1_CA"),
        (1, 1, "GPS_L5_Q"),
        (1, 2, "GPS_L1_CA"),
        (1, 2, "GPS_L5_Q"),
        (1, 3, "GPS_L1_CA"),
        (1, 3, "GPS_L5_Q"),
        (1, 4, "GPS_L1_CA"),
        (1, 4, "GPS_L5_Q"),
    )
    assert dual.n_clock == raw_bridge.MATLAB_SIGNAL_CLOCK_DIM
    assert dual.dual_frequency is True
    np.testing.assert_array_equal(dual.sys_kind[0], np.array([0, 4, 0, 4, 0, 4, 0, 4], dtype=np.int32))
    assert np.isclose(dual.pseudorange[0, 0], single.pseudorange[0, 0])
    assert np.isclose(dual.pseudorange[0, 1], single.pseudorange[0, 0] + 25.0)


def test_build_trip_arrays_multi_gnss_dual_frequency_uses_matlab_signal_clock_kinds(tmp_path):
    trip = tmp_path / "dataset_2023" / "train" / "course" / "phone"
    trip.mkdir(parents=True)

    rows = []
    signals = [
        (1, 1, "GPS_L1_CA", 0.0),
        (1, 1, "GPS_L5_Q", 25.0),
        (6, 11, "GAL_E1_C_P", 40.0),
        (6, 11, "GAL_E5A_Q", 65.0),
        (4, 193, "QZS_L1_CA", 80.0),
        (4, 193, "QZS_L5_Q", 105.0),
    ]
    for idx, (constellation, svid, signal_type, extra_pr) in enumerate(signals, start=1):
        rows.append(
            {
                "utcTimeMillis": 1000,
                "Svid": svid,
                "ConstellationType": constellation,
                "SignalType": signal_type,
                "RawPseudorangeMeters": 2.1e7 + 1000 * idx + extra_pr,
                "IonosphericDelayMeters": 2.0,
                "TroposphericDelayMeters": 3.0,
                "SvClockBiasMeters": 10.0,
                "SvPositionXEcefMeters": 2.6e7 - 1e5 * idx,
                "SvPositionYEcefMeters": 1.3e7 + 2e5 * idx,
                "SvPositionZEcefMeters": 2.1e7 - 3e5 * idx,
                "SvElevationDegrees": 30.0 + idx,
                "Cn0DbHz": 35.0 + idx,
                "WlsPositionXEcefMeters": -3947460.0,
                "WlsPositionYEcefMeters": 3431490.0,
                "WlsPositionZEcefMeters": 3637870.0,
            },
        )
    _write_zipped_csv(trip / "device_gnss.csv", rows, list(rows[0].keys()))

    batch = _build_trip_arrays(
        trip,
        max_epochs=10,
        start_epoch=0,
        constellation_type=1,
        signal_type="GPS_L1_CA",
        weight_mode="sin2el",
        multi_gnss=True,
        dual_frequency=True,
    )

    assert batch.n_clock == raw_bridge.MATLAB_SIGNAL_CLOCK_DIM
    assert batch.slot_keys == (
        (1, 1, "GPS_L1_CA"),
        (1, 1, "GPS_L5_Q"),
        (6, 11, "GAL_E1_C_P"),
        (6, 11, "GAL_E5A_Q"),
        (4, 193, "QZS_L1_CA"),
        (4, 193, "QZS_L5_Q"),
    )
    np.testing.assert_array_equal(batch.sys_kind[0], np.array([0, 4, 2, 5, 0, 4], dtype=np.int32))


def test_multi_system_for_clock_kind_preserves_legacy_and_matlab_signal_mappings():
    assert raw_bridge._multi_system_for_clock_kind(4, raw_bridge.MATLAB_SIGNAL_CLOCK_DIM) == raw_bridge.SYSTEM_GPS
    assert raw_bridge._multi_system_for_clock_kind(5, raw_bridge.MATLAB_SIGNAL_CLOCK_DIM) == raw_bridge.SYSTEM_GALILEO
    assert raw_bridge._multi_system_for_clock_kind(2, raw_bridge.MATLAB_SIGNAL_CLOCK_DIM) == raw_bridge.SYSTEM_GALILEO
    assert raw_bridge._multi_system_for_clock_kind(2, 3) == raw_bridge.SYSTEM_QZSS


def test_run_wls_uses_signal_clock_kinds_as_independent_bias_labels(monkeypatch):
    calls = []

    class FakeSignalClockSolver:
        def __init__(self, systems, max_iter, tol):
            self.systems = tuple(systems)
            self.max_iter = max_iter
            self.tol = tol

        def solve(self, sat_ecef, pseudoranges, system_ids, weights=None):
            del sat_ecef, pseudoranges, weights
            calls.append((self.systems, tuple(int(value) for value in system_ids)))
            return np.array([1000.0, 2000.0, 3000.0], dtype=np.float64), {0: 100.0, 4: 115.0}, 3

    monkeypatch.setattr(raw_bridge, "MultiGNSSSolver", FakeSignalClockSolver)
    sat_ecef = np.ones((1, 5, 3), dtype=np.float64)
    pseudorange = np.ones((1, 5), dtype=np.float64)
    weights = np.ones((1, 5), dtype=np.float64)
    sys_kind = np.array([[0, 4, 0, 4, 0]], dtype=np.int32)

    state = run_wls(
        sat_ecef,
        pseudorange,
        weights,
        sys_kind=sys_kind,
        n_clock=raw_bridge.MATLAB_SIGNAL_CLOCK_DIM,
    )

    assert calls == [((0, 4), (0, 4, 0, 4, 0))]
    np.testing.assert_allclose(state[0, :3], np.array([1000.0, 2000.0, 3000.0]))
    assert state[0, 3] == 100.0
    assert state[0, 3 + 4] == 15.0


def test_build_trip_arrays_applies_matlab_style_observation_mask(tmp_path):
    trip = tmp_path / "dataset_2023" / "train" / "course" / "phone"
    trip.mkdir(parents=True)

    base = {
        "utcTimeMillis": 1000,
        "ConstellationType": 1,
        "SignalType": "GPS_L1_CA",
        "RawPseudorangeMeters": 2.1e7,
        "IonosphericDelayMeters": 2.0,
        "TroposphericDelayMeters": 3.0,
        "SvClockBiasMeters": 10.0,
        "SvPositionXEcefMeters": 2.6e7,
        "SvPositionYEcefMeters": 1.3e7,
        "SvPositionZEcefMeters": 2.1e7,
        "SvElevationDegrees": 30.0,
        "Cn0DbHz": 35.0,
        "WlsPositionXEcefMeters": -3947460.0,
        "WlsPositionYEcefMeters": 3431490.0,
        "WlsPositionZEcefMeters": 3637870.0,
        "State": 1 | 8,
        "MultipathIndicator": 0,
    }
    rows = []
    for svid in range(1, 9):
        row = dict(base, Svid=svid, RawPseudorangeMeters=2.1e7 + 1000 * svid)
        if svid == 5:
            row["Cn0DbHz"] = 10.0
        elif svid == 6:
            row["MultipathIndicator"] = 1
        elif svid == 7:
            row["State"] = 1
        elif svid == 8:
            row["RawPseudorangeMeters"] = 4.5e7
        rows.append(row)
    _write_zipped_csv(trip / "device_gnss.csv", rows, list(rows[0].keys()))

    batch = _build_trip_arrays(
        trip,
        max_epochs=10,
        start_epoch=0,
        constellation_type=1,
        signal_type="GPS_L1_CA",
        weight_mode="sin2el",
        apply_observation_mask=True,
        pseudorange_residual_mask_m=0.0,
        doppler_residual_mask_mps=0.0,
        pseudorange_doppler_mask_m=0.0,
    )

    assert batch.sat_ecef.shape == (1, 4, 3)
    assert np.count_nonzero(batch.weights > 0.0) == 4
    assert batch.observation_mask_count == 4
    assert batch.residual_mask_count == 0


def test_run_wls_keeps_fallback_for_underconstrained_epochs():
    sat_ecef = np.zeros((2, 3, 3), dtype=np.float64)
    pseudorange = np.zeros((2, 3), dtype=np.float64)
    weights = np.ones((2, 3), dtype=np.float64)
    fallback = np.array(
        [
            [-3947460.0, 3431490.0, 3637870.0],
            [-3947461.0, 3431491.0, 3637871.0],
        ],
        dtype=np.float64,
    )

    state = run_wls(sat_ecef, pseudorange, weights, fallback_xyz=fallback)

    np.testing.assert_allclose(state[:, :3], fallback)


def test_build_trip_arrays_masks_pseudorange_residual_outlier(tmp_path):
    trip = tmp_path / "dataset_2023" / "train" / "course" / "phone"
    trip.mkdir(parents=True)

    rx = np.array([-3947460.0, 3431490.0, 3637870.0], dtype=np.float64)
    sat_offsets = np.array(
        [
            [2.1e7, 0.0, 0.0],
            [0.0, 2.1e7, 0.0],
            [0.0, 0.0, 2.1e7],
            [-1.5e7, 1.5e7, 0.0],
            [1.2e7, 0.8e7, 1.4e7],
        ],
        dtype=np.float64,
    )
    rows = []
    receiver_clock_m = 120.0
    iono = 2.0
    tropo = 3.0
    sv_clock = 10.0
    for svid, offset in enumerate(sat_offsets, start=1):
        sat = rx + offset
        corrected_pr = float(np.linalg.norm(offset) + receiver_clock_m)
        if svid == 5:
            corrected_pr += 1000.0
        raw_pr = corrected_pr - sv_clock + iono + tropo
        rows.append(
            {
                "utcTimeMillis": 1000,
                "Svid": svid,
                "ConstellationType": 1,
                "SignalType": "GPS_L1_CA",
                "RawPseudorangeMeters": raw_pr,
                "IonosphericDelayMeters": iono,
                "TroposphericDelayMeters": tropo,
                "SvClockBiasMeters": sv_clock,
                "SvPositionXEcefMeters": sat[0],
                "SvPositionYEcefMeters": sat[1],
                "SvPositionZEcefMeters": sat[2],
                "SvElevationDegrees": 35.0,
                "Cn0DbHz": 35.0,
                "WlsPositionXEcefMeters": rx[0],
                "WlsPositionYEcefMeters": rx[1],
                "WlsPositionZEcefMeters": rx[2],
                "State": 1 | 8,
                "MultipathIndicator": 0,
            },
        )
    _write_zipped_csv(trip / "device_gnss.csv", rows, list(rows[0].keys()))

    batch = _build_trip_arrays(
        trip,
        max_epochs=10,
        start_epoch=0,
        constellation_type=1,
        signal_type="GPS_L1_CA",
        weight_mode="sin2el",
        apply_observation_mask=True,
        pseudorange_residual_mask_m=300.0,
    )

    assert batch.observation_mask_count == 0
    assert batch.residual_mask_count == 1
    assert np.count_nonzero(batch.weights[0] > 0.0) == 4
    assert batch.weights[0, 4] == 0.0


def test_build_trip_arrays_masks_doppler_residual_outlier(tmp_path):
    trip = tmp_path / "dataset_2023" / "train" / "course" / "phone"
    trip.mkdir(parents=True)

    rx = np.array([-3947460.0, 3431490.0, 3637870.0], dtype=np.float64)
    sat_offsets = np.array(
        [
            [2.1e7, 0.0, 0.0],
            [0.0, 2.1e7, 0.0],
            [0.0, 0.0, 2.1e7],
            [-1.5e7, 1.5e7, 0.0],
            [1.2e7, 0.8e7, 1.4e7],
        ],
        dtype=np.float64,
    )
    unit = sat_offsets / np.linalg.norm(sat_offsets, axis=1)[:, None]
    sat_vel = unit * 100.0
    clock_drift_mps = 7.0
    iono = 2.0
    tropo = 3.0
    sv_clock = 10.0
    rows = []
    for svid, offset in enumerate(sat_offsets, start=1):
        sat = rx + offset
        corrected_pr = float(np.linalg.norm(offset) + 120.0)
        raw_pr = corrected_pr - sv_clock + iono + tropo
        geom_rate = float(np.dot(unit[svid - 1], sat_vel[svid - 1]))
        doppler = geom_rate + clock_drift_mps
        if svid == 5:
            doppler += 5.5
        rows.append(
            {
                "utcTimeMillis": 1000,
                "Svid": svid,
                "ConstellationType": 1,
                "SignalType": "GPS_L1_CA",
                "RawPseudorangeMeters": raw_pr,
                "IonosphericDelayMeters": iono,
                "TroposphericDelayMeters": tropo,
                "SvClockBiasMeters": sv_clock,
                "SvPositionXEcefMeters": sat[0],
                "SvPositionYEcefMeters": sat[1],
                "SvPositionZEcefMeters": sat[2],
                "SvVelocityXEcefMetersPerSecond": sat_vel[svid - 1, 0],
                "SvVelocityYEcefMetersPerSecond": sat_vel[svid - 1, 1],
                "SvVelocityZEcefMetersPerSecond": sat_vel[svid - 1, 2],
                "SvElevationDegrees": 35.0,
                "Cn0DbHz": 35.0,
                "WlsPositionXEcefMeters": rx[0],
                "WlsPositionYEcefMeters": rx[1],
                "WlsPositionZEcefMeters": rx[2],
                "State": 1 | 8,
                "MultipathIndicator": 0,
                "PseudorangeRateMetersPerSecond": -doppler,
                "PseudorangeRateUncertaintyMetersPerSecond": 0.1,
            },
        )
    _write_zipped_csv(trip / "device_gnss.csv", rows, list(rows[0].keys()))

    batch = _build_trip_arrays(
        trip,
        max_epochs=10,
        start_epoch=0,
        constellation_type=1,
        signal_type="GPS_L1_CA",
        weight_mode="sin2el",
        apply_observation_mask=True,
        pseudorange_residual_mask_m=0.0,
        doppler_residual_mask_mps=1.0,
    )

    assert batch.observation_mask_count == 0
    assert batch.residual_mask_count == 0
    assert batch.doppler_residual_mask_count == 1
    assert np.count_nonzero(batch.weights[0] > 0.0) == 5
    assert np.count_nonzero(batch.doppler_weights[0] > 0.0) == 4
    assert batch.weights[0, 4] > 0.0
    assert batch.doppler_weights[0, 4] == 0.0


def test_doppler_residual_mask_uses_satellite_clock_drift():
    times_ms = np.array([1000.0], dtype=np.float64)
    rx = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    sat_ecef = np.array(
        [
            [
                [2.1e7, 0.0, 0.0],
                [0.0, 2.1e7, 0.0],
                [0.0, 0.0, 2.1e7],
                [-2.1e7, 0.0, 0.0],
                [0.0, -2.1e7, 0.0],
            ],
        ],
        dtype=np.float64,
    )
    los = sat_ecef / np.linalg.norm(sat_ecef, axis=2)[:, :, None]
    sat_vel = los * 100.0
    geom_rate = np.sum(los * sat_vel, axis=2)
    sat_clock_drift = np.array([[0.0, 2.0, 4.0, 6.0, 8.0]], dtype=np.float64)
    model = geom_rate - sat_clock_drift
    receiver_common = 7.0
    doppler = -model + receiver_common
    weights = np.ones_like(doppler)

    masked = _mask_doppler_residual_outliers(
        times_ms,
        sat_ecef,
        sat_vel,
        doppler,
        weights,
        rx,
        threshold_mps=3.0,
        sat_clock_drift_mps=sat_clock_drift,
    )

    assert masked == 0
    assert np.count_nonzero(weights > 0.0) == 5


def test_observation_mask_keeps_doppler_and_tdcp_when_only_pseudorange_state_fails(tmp_path):
    trip = tmp_path / "dataset_2023" / "train" / "course" / "phone"
    trip.mkdir(parents=True)

    rows = []
    for epoch_idx, utc_ms in enumerate((1000, 2000)):
        for svid in range(1, 6):
            state = 1 | 8
            if svid == 5:
                state = 1
            rows.append(
                {
                    "utcTimeMillis": utc_ms,
                    "Svid": svid,
                    "ConstellationType": 1,
                    "SignalType": "GPS_L1_CA",
                    "RawPseudorangeMeters": 2.1e7 + 1000 * svid + epoch_idx,
                    "IonosphericDelayMeters": 2.0,
                    "TroposphericDelayMeters": 3.0,
                    "SvClockBiasMeters": 10.0,
                    "SvPositionXEcefMeters": 2.6e7 - 1e5 * svid,
                    "SvPositionYEcefMeters": 1.3e7 + 2e5 * svid,
                    "SvPositionZEcefMeters": 2.1e7 - 3e5 * svid,
                    "SvElevationDegrees": 35.0,
                    "Cn0DbHz": 35.0,
                    "WlsPositionXEcefMeters": -3947460.0,
                    "WlsPositionYEcefMeters": 3431490.0,
                    "WlsPositionZEcefMeters": 3637870.0,
                    "State": state,
                    "MultipathIndicator": 0,
                    "PseudorangeRateMetersPerSecond": 1.0,
                    "PseudorangeRateUncertaintyMetersPerSecond": 0.1,
                    "AccumulatedDeltaRangeState": 1,
                    "AccumulatedDeltaRangeMeters": 10.0 * svid + float(epoch_idx),
                    "AccumulatedDeltaRangeUncertaintyMeters": 0.02,
                },
            )
    _write_zipped_csv(trip / "device_gnss.csv", rows, list(rows[0].keys()))

    batch = _build_trip_arrays(
        trip,
        max_epochs=10,
        start_epoch=0,
        constellation_type=1,
        signal_type="GPS_L1_CA",
        weight_mode="sin2el",
        use_tdcp=True,
        apply_observation_mask=True,
        pseudorange_residual_mask_m=0.0,
        doppler_residual_mask_mps=0.0,
        pseudorange_doppler_mask_m=0.0,
        tdcp_consistency_threshold_m=0.2,
    )

    assert batch.observation_mask_count == 2
    assert np.count_nonzero(batch.weights[:, 4] > 0.0) == 0
    assert np.count_nonzero(batch.doppler_weights[:, 4] > 0.0) == 2
    assert batch.tdcp_weights is not None
    assert batch.tdcp_weights[0, 4] > 0.0


def test_build_trip_arrays_masks_pseudorange_doppler_inconsistency(tmp_path):
    trip = tmp_path / "dataset_2023" / "train" / "course" / "phone"
    trip.mkdir(parents=True)

    rows = []
    iono = 2.0
    tropo = 3.0
    sv_clock = 10.0
    for epoch_idx, utc_ms in enumerate((1000, 2000)):
        for svid in range(1, 6):
            corrected_pr = 2.1e7 + 1000.0 * svid + 10.0 * epoch_idx
            if svid == 5 and epoch_idx == 1:
                corrected_pr += 120.0
            rows.append(
                {
                    "utcTimeMillis": utc_ms,
                    "Svid": svid,
                    "ConstellationType": 1,
                    "SignalType": "GPS_L1_CA",
                    "RawPseudorangeMeters": corrected_pr - sv_clock + iono + tropo,
                    "IonosphericDelayMeters": iono,
                    "TroposphericDelayMeters": tropo,
                    "SvClockBiasMeters": sv_clock,
                    "SvPositionXEcefMeters": 2.6e7 - 1e5 * svid,
                    "SvPositionYEcefMeters": 1.3e7 + 2e5 * svid,
                    "SvPositionZEcefMeters": 2.1e7 - 3e5 * svid,
                    "SvElevationDegrees": 35.0,
                    "Cn0DbHz": 35.0,
                    "WlsPositionXEcefMeters": -3947460.0,
                    "WlsPositionYEcefMeters": 3431490.0,
                    "WlsPositionZEcefMeters": 3637870.0,
                    "State": 1 | 8,
                    "MultipathIndicator": 0,
                    "PseudorangeRateMetersPerSecond": -10.0,
                    "PseudorangeRateUncertaintyMetersPerSecond": 0.1,
                },
            )
    _write_zipped_csv(trip / "device_gnss.csv", rows, list(rows[0].keys()))

    batch = _build_trip_arrays(
        trip,
        max_epochs=10,
        start_epoch=0,
        constellation_type=1,
        signal_type="GPS_L1_CA",
        weight_mode="sin2el",
        apply_observation_mask=True,
        pseudorange_residual_mask_m=0.0,
        doppler_residual_mask_mps=0.0,
        pseudorange_doppler_mask_m=20.0,
    )

    assert batch.observation_mask_count == 0
    assert batch.pseudorange_doppler_mask_count == 2
    assert np.count_nonzero(batch.weights[0] > 0.0) == 4
    assert np.count_nonzero(batch.weights[1] > 0.0) == 4
    assert batch.weights[0, 4] == 0.0
    assert batch.weights[1, 4] == 0.0
    assert batch.doppler_weights[0, 4] > 0.0
    assert batch.doppler_weights[1, 4] > 0.0


def test_dual_frequency_pseudorange_doppler_mask_uses_l5_threshold(tmp_path):
    trip = tmp_path / "dataset_2023" / "train" / "course" / "phone"
    trip.mkdir(parents=True)

    rows = []
    iono = 2.0
    tropo = 3.0
    sv_clock = 10.0
    for epoch_idx, utc_ms in enumerate((1000, 2000)):
        for svid in range(1, 5):
            for signal_type, extra in (("GPS_L1_CA", 0.0), ("GPS_L5_Q", 20.0)):
                corrected_pr = 2.1e7 + 1000.0 * svid + extra + 10.0 * epoch_idx
                if svid == 4 and signal_type == "GPS_L5_Q" and epoch_idx == 1:
                    corrected_pr += 35.0
                rows.append(
                    {
                        "utcTimeMillis": utc_ms,
                        "Svid": svid,
                        "ConstellationType": 1,
                        "SignalType": signal_type,
                        "RawPseudorangeMeters": corrected_pr - sv_clock + iono + tropo,
                        "IonosphericDelayMeters": iono,
                        "TroposphericDelayMeters": tropo,
                        "SvClockBiasMeters": sv_clock,
                        "SvPositionXEcefMeters": 2.6e7 - 1e5 * svid,
                        "SvPositionYEcefMeters": 1.3e7 + 2e5 * svid,
                        "SvPositionZEcefMeters": 2.1e7 - 3e5 * svid,
                        "SvElevationDegrees": 35.0,
                        "Cn0DbHz": 35.0,
                        "WlsPositionXEcefMeters": -3947460.0,
                        "WlsPositionYEcefMeters": 3431490.0,
                        "WlsPositionZEcefMeters": 3637870.0,
                        "State": 1 | 8,
                        "MultipathIndicator": 0,
                        "PseudorangeRateMetersPerSecond": -10.0,
                        "PseudorangeRateUncertaintyMetersPerSecond": 0.1,
                    },
                )
    _write_zipped_csv(trip / "device_gnss.csv", rows, list(rows[0].keys()))

    batch = _build_trip_arrays(
        trip,
        max_epochs=10,
        start_epoch=0,
        constellation_type=1,
        signal_type="GPS_L1_CA",
        weight_mode="sin2el",
        apply_observation_mask=True,
        dual_frequency=True,
        pseudorange_residual_mask_m=0.0,
        doppler_residual_mask_mps=0.0,
    )

    assert batch.pseudorange_doppler_mask_count == 2
    assert batch.slot_keys[6] == (1, 4, "GPS_L1_CA")
    assert batch.slot_keys[7] == (1, 4, "GPS_L5_Q")
    assert np.count_nonzero(batch.weights[:, 6] > 0.0) == 2
    assert np.count_nonzero(batch.weights[:, 7] > 0.0) == 0


def test_pseudorange_doppler_consistency_uses_matlab_scalar_interval():
    times_ms = np.array([0.0, 1000.0, 2000.0, 5000.0], dtype=np.float64)
    pseudorange = np.array([[0.0], [10.0], [20.0], [50.0]], dtype=np.float64)
    weights = np.ones_like(pseudorange)
    doppler = np.full_like(pseudorange, -10.0)
    doppler_weights = np.ones_like(pseudorange)

    masked = raw_bridge._mask_pseudorange_doppler_consistency(
        times_ms,
        pseudorange,
        weights,
        doppler,
        doppler_weights,
        phone="pixel5",
        threshold_m=15.0,
    )

    assert masked == 2
    assert weights[0, 0] > 0.0
    assert weights[1, 0] > 0.0
    assert weights[2, 0] == 0.0
    assert weights[3, 0] == 0.0


def test_build_trip_arrays_without_ground_truth(tmp_path):
    trip = tmp_path / "dataset_2023" / "test" / "course" / "phone"
    trip.mkdir(parents=True)

    rows = []
    for utc_ms in (1000, 2000, 3000):
        for svid in range(1, 5):
            rows.append(
                {
                    "utcTimeMillis": utc_ms,
                    "Svid": svid,
                    "ConstellationType": 1,
                    "SignalType": "GPS_L1_CA",
                    "RawPseudorangeMeters": 2.1e7 + 1000 * svid,
                    "IonosphericDelayMeters": 2.0,
                    "TroposphericDelayMeters": 3.0,
                    "SvClockBiasMeters": 10.0,
                    "SvPositionXEcefMeters": 2.6e7 - 1e5 * svid,
                    "SvPositionYEcefMeters": 1.3e7 + 2e5 * svid,
                    "SvPositionZEcefMeters": 2.1e7 - 3e5 * svid,
                    "SvElevationDegrees": 30.0 + svid,
                    "Cn0DbHz": 35.0 + svid,
                    "WlsPositionXEcefMeters": -3947460.0 + utc_ms * 0.001,
                    "WlsPositionYEcefMeters": 3431490.0 + utc_ms * 0.001,
                    "WlsPositionZEcefMeters": 3637870.0 + utc_ms * 0.001,
                },
            )

    _write_zipped_csv(trip / "device_gnss.csv", rows, list(rows[0].keys()))

    batch = _build_trip_arrays(
        trip,
        max_epochs=10,
        start_epoch=0,
        constellation_type=1,
        signal_type="GPS_L1_CA",
        weight_mode="sin2el",
    )

    assert batch.has_truth is False
    assert np.isnan(batch.truth).all()


def test_build_trip_arrays_repairs_baseline_and_prefers_low_bias_uncertainty(tmp_path):
    trip = tmp_path / "dataset_2023" / "test" / "course" / "phone"
    trip.mkdir(parents=True)

    base_xyz = np.array([-3947460.0, 3431490.0, 3637870.0], dtype=np.float64)
    baseline_x = {
        1000: 0.0,
        2000: 1.0,
        3000: np.nan,
        4000: 3.0,
        5000: 1000.0,
        6000: 5.0,
    }
    rows = []
    for utc_ms in sorted(baseline_x):
        for svid in range(1, 5):
            rows.append(
                {
                    "utcTimeMillis": utc_ms,
                    "Svid": svid,
                    "ConstellationType": 1,
                    "SignalType": "GPS_L1_CA",
                    "RawPseudorangeMeters": 2.1e7 + 1000 * svid,
                    "IonosphericDelayMeters": 2.0,
                    "TroposphericDelayMeters": 3.0,
                    "SvClockBiasMeters": 10.0,
                    "SvPositionXEcefMeters": 2.6e7 - 1e5 * svid,
                    "SvPositionYEcefMeters": 1.3e7 + 2e5 * svid,
                    "SvPositionZEcefMeters": 2.1e7 - 3e5 * svid,
                    "SvElevationDegrees": 35.0 + svid,
                    "Cn0DbHz": 35.0 + svid,
                    "WlsPositionXEcefMeters": base_xyz[0] + baseline_x[utc_ms],
                    "WlsPositionYEcefMeters": base_xyz[1],
                    "WlsPositionZEcefMeters": base_xyz[2],
                    "BiasUncertaintyNanos": 10.0,
                },
            )
            if utc_ms == 1000:
                rows.append(
                    {
                        "utcTimeMillis": utc_ms,
                        "Svid": svid,
                        "ConstellationType": 1,
                        "SignalType": "GPS_L1_CA",
                        "RawPseudorangeMeters": 2.1e7 + 1000 * svid + 500.0,
                        "IonosphericDelayMeters": 2.0,
                        "TroposphericDelayMeters": 3.0,
                        "SvClockBiasMeters": 10.0,
                        "SvPositionXEcefMeters": 2.6e7 - 1e5 * svid,
                        "SvPositionYEcefMeters": 1.3e7 + 2e5 * svid,
                        "SvPositionZEcefMeters": 2.1e7 - 3e5 * svid,
                        "SvElevationDegrees": 35.0 + svid,
                        "Cn0DbHz": 60.0 + svid,
                        "WlsPositionXEcefMeters": 9999.0,
                        "WlsPositionYEcefMeters": base_xyz[1],
                        "WlsPositionZEcefMeters": base_xyz[2],
                        "BiasUncertaintyNanos": 2.0e4,
                    },
                )

    _write_zipped_csv(trip / "device_gnss.csv", rows, list(rows[0].keys()))

    batch = _build_trip_arrays(
        trip,
        max_epochs=10,
        start_epoch=0,
        constellation_type=1,
        signal_type="GPS_L1_CA",
        weight_mode="sin2el",
    )

    assert np.allclose(
        batch.kaggle_wls[:, 0],
        base_xyz[0] + np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
        atol=1e-6,
    )
    assert np.allclose(batch.kaggle_wls[:, 1], base_xyz[1], atol=1e-6)
    assert np.allclose(batch.kaggle_wls[:, 2], base_xyz[2], atol=1e-6)


def test_build_trip_arrays_disables_tdcp_across_hardware_clock_discontinuity(tmp_path):
    trip = tmp_path / "dataset_2023" / "test" / "course" / "phone"
    trip.mkdir(parents=True)

    base_xyz = np.array([-3947460.0, 3431490.0, 3637870.0], dtype=np.float64)
    rows = []
    hcdc = {1000: 0.0, 2000: 1.0, 3000: 1.0}
    for epoch_idx, utc_ms in enumerate((1000, 2000, 3000)):
        for svid in range(1, 5):
            rows.append(
                {
                    "utcTimeMillis": utc_ms,
                    "Svid": svid,
                    "ConstellationType": 1,
                    "SignalType": "GPS_L1_CA",
                    "RawPseudorangeMeters": 2.1e7 + 1000 * svid,
                    "IonosphericDelayMeters": 2.0,
                    "TroposphericDelayMeters": 3.0,
                    "SvClockBiasMeters": 10.0,
                    "SvPositionXEcefMeters": 2.6e7 - 1e5 * svid,
                    "SvPositionYEcefMeters": 1.3e7 + 2e5 * svid,
                    "SvPositionZEcefMeters": 2.1e7 - 3e5 * svid,
                    "SvElevationDegrees": 40.0 + svid,
                    "Cn0DbHz": 35.0 + svid,
                    "WlsPositionXEcefMeters": base_xyz[0] + float(epoch_idx),
                    "WlsPositionYEcefMeters": base_xyz[1],
                    "WlsPositionZEcefMeters": base_xyz[2],
                    "BiasUncertaintyNanos": 10.0,
                    "HardwareClockDiscontinuityCount": hcdc[utc_ms],
                    "PseudorangeRateMetersPerSecond": 1.0,
                    "PseudorangeRateUncertaintyMetersPerSecond": 0.1,
                    "AccumulatedDeltaRangeState": 1,
                    "AccumulatedDeltaRangeMeters": 10.0 * svid + epoch_idx,
                    "AccumulatedDeltaRangeUncertaintyMeters": 0.02,
                },
            )

    _write_zipped_csv(trip / "device_gnss.csv", rows, list(rows[0].keys()))

    batch = _build_trip_arrays(
        trip,
        max_epochs=10,
        start_epoch=0,
        constellation_type=1,
        signal_type="GPS_L1_CA",
        weight_mode="sin2el",
        use_tdcp=True,
    )

    assert batch.clock_jump is not None
    assert batch.clock_jump.tolist() == [False, True, False]
    assert batch.tdcp_weights is not None
    assert np.allclose(batch.tdcp_weights[0], 0.0)
    assert np.all(batch.tdcp_weights[1] > 0.0)


def test_build_trip_arrays_disables_tdcp_for_blocklisted_phone(tmp_path):
    trip = tmp_path / "dataset_2023" / "test" / "course" / "samsunga32"
    trip.mkdir(parents=True)

    base_xyz = np.array([-3947460.0, 3431490.0, 3637870.0], dtype=np.float64)
    rows = []
    for epoch_idx, utc_ms in enumerate((1000, 2000, 3000)):
        for svid in range(1, 5):
            rows.append(
                {
                    "utcTimeMillis": utc_ms,
                    "Svid": svid,
                    "ConstellationType": 1,
                    "SignalType": "GPS_L1_CA",
                    "RawPseudorangeMeters": 2.1e7 + 1000 * svid,
                    "IonosphericDelayMeters": 2.0,
                    "TroposphericDelayMeters": 3.0,
                    "SvClockBiasMeters": 10.0,
                    "SvPositionXEcefMeters": 2.6e7 - 1e5 * svid,
                    "SvPositionYEcefMeters": 1.3e7 + 2e5 * svid,
                    "SvPositionZEcefMeters": 2.1e7 - 3e5 * svid,
                    "SvElevationDegrees": 40.0 + svid,
                    "Cn0DbHz": 35.0 + svid,
                    "WlsPositionXEcefMeters": base_xyz[0] + float(epoch_idx),
                    "WlsPositionYEcefMeters": base_xyz[1],
                    "WlsPositionZEcefMeters": base_xyz[2],
                    "PseudorangeRateMetersPerSecond": -1.0,
                    "PseudorangeRateUncertaintyMetersPerSecond": 0.1,
                    "AccumulatedDeltaRangeState": 1,
                    "AccumulatedDeltaRangeMeters": 10.0 * svid + epoch_idx,
                    "AccumulatedDeltaRangeUncertaintyMeters": 0.02,
                },
            )

    _write_zipped_csv(trip / "device_gnss.csv", rows, list(rows[0].keys()))

    batch = _build_trip_arrays(
        trip,
        max_epochs=10,
        start_epoch=0,
        constellation_type=1,
        signal_type="GPS_L1_CA",
        weight_mode="sin2el",
        use_tdcp=True,
    )

    assert batch.tdcp_meas is None
    assert batch.tdcp_weights is None


def test_build_trip_arrays_applies_tdcp_loffset_for_samsung_a_family(tmp_path):
    trip = tmp_path / "dataset_2023" / "test" / "course" / "sm-a505u"
    trip.mkdir(parents=True)

    base_xyz = np.array([-3947460.0, 3431490.0, 3637870.0], dtype=np.float64)
    rows = []
    for epoch_idx, utc_ms in enumerate((1000, 2000)):
        for svid in range(1, 5):
            rows.append(
                {
                    "utcTimeMillis": utc_ms,
                    "Svid": svid,
                    "ConstellationType": 1,
                    "SignalType": "GPS_L1_CA",
                    "RawPseudorangeMeters": 2.1e7 + 1000 * svid,
                    "IonosphericDelayMeters": 2.0,
                    "TroposphericDelayMeters": 3.0,
                    "SvClockBiasMeters": 10.0,
                    "SvPositionXEcefMeters": 2.6e7 - 1e5 * svid,
                    "SvPositionYEcefMeters": 1.3e7 + 2e5 * svid,
                    "SvPositionZEcefMeters": 2.1e7 - 3e5 * svid,
                    "SvElevationDegrees": 40.0 + svid,
                    "Cn0DbHz": 35.0 + svid,
                    "WlsPositionXEcefMeters": base_xyz[0] + float(epoch_idx),
                    "WlsPositionYEcefMeters": base_xyz[1],
                    "WlsPositionZEcefMeters": base_xyz[2],
                    "PseudorangeRateMetersPerSecond": 1.617,
                    "PseudorangeRateUncertaintyMetersPerSecond": 0.1,
                    "AccumulatedDeltaRangeState": 1,
                    "AccumulatedDeltaRangeMeters": 10.0 * svid - 0.5 * epoch_idx,
                    "AccumulatedDeltaRangeUncertaintyMeters": 0.02,
                },
            )

    _write_zipped_csv(trip / "device_gnss.csv", rows, list(rows[0].keys()))

    batch = _build_trip_arrays(
        trip,
        max_epochs=10,
        start_epoch=0,
        constellation_type=1,
        signal_type="GPS_L1_CA",
        weight_mode="sin2el",
        use_tdcp=True,
        tdcp_consistency_threshold_m=0.2,
        tdcp_geometry_correction=False,
    )

    assert batch.tdcp_meas is not None
    assert batch.tdcp_weights is not None
    np.testing.assert_allclose(batch.tdcp_meas[0, :4], np.full(4, 1.617), atol=1e-9)
    assert np.all(batch.tdcp_weights[0, :4] > 0.0)
    assert batch.tdcp_consistency_mask_count == 0


def test_build_tdcp_arrays_counts_consistency_rejects_and_ignores_masked_doppler():
    adr = np.array(
        [
            [0.0, 0.0],
            [10.0, 10.0],
        ],
        dtype=np.float64,
    )
    adr_state = np.ones_like(adr, dtype=np.int32)
    adr_uncertainty = np.full_like(adr, 0.02)
    doppler = np.array(
        [
            [-20.0, -20.0],
            [-20.0, -20.0],
        ],
        dtype=np.float64,
    )
    doppler_weights = np.ones_like(adr)
    doppler_weights[0, 1] = 0.0

    tdcp_meas, tdcp_weights, mask_count = _build_tdcp_arrays(
        adr,
        adr_state,
        adr_uncertainty,
        doppler,
        np.array([1.0, 0.0], dtype=np.float64),
        consistency_threshold_m=1.5,
        doppler_weights=doppler_weights,
    )

    assert mask_count == 1
    assert tdcp_meas is not None
    assert tdcp_weights is not None
    assert tdcp_weights[0, 0] == 0.0
    assert tdcp_weights[0, 1] > 0.0


def test_build_tdcp_arrays_propagates_consistency_rejects_to_adjacent_pairs():
    adr = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [100.0, 2.0],
            [101.0, 3.0],
        ],
        dtype=np.float64,
    )
    adr_state = np.ones_like(adr, dtype=np.int32)
    adr_uncertainty = np.full_like(adr, 0.02)
    doppler = np.full_like(adr, -1.0)

    tdcp_meas, tdcp_weights, mask_count = _build_tdcp_arrays(
        adr,
        adr_state,
        adr_uncertainty,
        doppler,
        np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float64),
        consistency_threshold_m=1.5,
    )

    assert mask_count == 1
    assert tdcp_meas is not None
    assert tdcp_weights is not None
    assert np.all(tdcp_weights[:, 0] == 0.0)
    assert np.all(tdcp_weights[:, 1] > 0.0)


def test_build_tdcp_arrays_uses_matlab_scalar_interval_for_consistency():
    adr = np.array([[0.0], [1.0], [2.0]], dtype=np.float64)
    adr_state = np.ones_like(adr, dtype=np.int32)
    adr_uncertainty = np.full_like(adr, 0.02)
    doppler = np.full_like(adr, -1.0)

    tdcp_meas, tdcp_weights, mask_count = _build_tdcp_arrays(
        adr,
        adr_state,
        adr_uncertainty,
        doppler,
        np.array([1.0, 3.0, 0.0], dtype=np.float64),
        consistency_threshold_m=1.5,
    )

    assert mask_count == 0
    assert tdcp_meas is not None
    assert tdcp_weights is not None
    assert np.all(tdcp_weights[:, 0] > 0.0)


def test_build_trip_arrays_applies_tdcp_weight_scale(tmp_path):
    trip = tmp_path / "dataset_2023" / "train" / "course" / "phone"
    trip.mkdir(parents=True)

    rows = []
    for epoch_idx, utc_ms in enumerate((1000, 2000)):
        for svid in range(1, 5):
            rows.append(
                {
                    "utcTimeMillis": utc_ms,
                    "Svid": svid,
                    "ConstellationType": 1,
                    "SignalType": "GPS_L1_CA",
                    "RawPseudorangeMeters": 2.1e7 + 1000 * svid,
                    "IonosphericDelayMeters": 0.0,
                    "TroposphericDelayMeters": 0.0,
                    "SvClockBiasMeters": 0.0,
                    "SvPositionXEcefMeters": 2.2e7 + 1000 * svid,
                    "SvPositionYEcefMeters": 1.4e7 + 1000 * epoch_idx,
                    "SvPositionZEcefMeters": 2.1e7,
                    "SvElevationDegrees": 35.0,
                    "Cn0DbHz": 35.0,
                    "WlsPositionXEcefMeters": -3947460.0,
                    "WlsPositionYEcefMeters": 3431490.0,
                    "WlsPositionZEcefMeters": 3637870.0,
                    "PseudorangeRateMetersPerSecond": 1.0,
                    "PseudorangeRateUncertaintyMetersPerSecond": 0.1,
                    "AccumulatedDeltaRangeState": 1,
                    "AccumulatedDeltaRangeMeters": 100.0 * svid + float(epoch_idx),
                    "AccumulatedDeltaRangeUncertaintyMeters": 0.02,
                },
            )
    _write_zipped_csv(trip / "device_gnss.csv", rows, list(rows[0].keys()))

    batch = _build_trip_arrays(
        trip,
        max_epochs=10,
        start_epoch=0,
        constellation_type=1,
        signal_type="GPS_L1_CA",
        weight_mode="sin2el",
        use_tdcp=True,
        tdcp_consistency_threshold_m=1.5,
        tdcp_weight_scale=0.001,
    )

    assert batch.tdcp_weights is not None
    expected_unscaled = 1.0 / ((0.02**2 + 0.02**2))
    np.testing.assert_allclose(batch.tdcp_weights[0, :4], expected_unscaled * 0.001, rtol=1e-12)


def test_matlab_residual_diagnostics_mask_preserves_tdcp_signal_weights(tmp_path):
    trip = tmp_path / "dataset_2023" / "train" / "course" / "phone"
    trip.mkdir(parents=True)

    rows = []
    for epoch_idx, utc_ms in enumerate((1000, 2000)):
        for svid in range(1, 5):
            adr_uncertainty = 0.05 if svid == 1 else 0.02
            rows.append(
                {
                    "utcTimeMillis": utc_ms,
                    "Svid": svid,
                    "ConstellationType": 1,
                    "SignalType": "GPS_L1_CA",
                    "RawPseudorangeMeters": 2.1e7 + 1000 * svid,
                    "IonosphericDelayMeters": 0.0,
                    "TroposphericDelayMeters": 0.0,
                    "SvClockBiasMeters": 0.0,
                    "SvPositionXEcefMeters": 2.2e7 + 1000 * svid,
                    "SvPositionYEcefMeters": 1.4e7 + 1000 * epoch_idx,
                    "SvPositionZEcefMeters": 2.1e7,
                    "SvElevationDegrees": 35.0,
                    "Cn0DbHz": 35.0,
                    "WlsPositionXEcefMeters": -3947460.0,
                    "WlsPositionYEcefMeters": 3431490.0,
                    "WlsPositionZEcefMeters": 3637870.0,
                    "PseudorangeRateMetersPerSecond": 1.0,
                    "PseudorangeRateUncertaintyMetersPerSecond": 0.1,
                    "AccumulatedDeltaRangeState": 1,
                    "AccumulatedDeltaRangeMeters": 100.0 * svid + float(epoch_idx),
                    "AccumulatedDeltaRangeUncertaintyMeters": adr_uncertainty,
                },
            )
    _write_zipped_csv(trip / "device_gnss.csv", rows, list(rows[0].keys()))

    diagnostics_path = trip / "phone_data_residual_diagnostics.csv"
    pd.DataFrame(
        [
            {
                "freq": "L1",
                "utcTimeMillis": utc_ms,
                "sys": 1,
                "svid": 1,
                "p_factor_finite": "1",
                "d_factor_finite": "1",
                "l_factor_finite": "1",
            }
            for utc_ms in (1000, 2000)
        ],
    ).to_csv(diagnostics_path, index=False)

    batch = _build_trip_arrays(
        trip,
        max_epochs=10,
        start_epoch=0,
        constellation_type=1,
        signal_type="GPS_L1_CA",
        weight_mode="sin2el",
        use_tdcp=True,
        tdcp_consistency_threshold_m=1.5,
        tdcp_weight_scale=0.001,
        matlab_residual_diagnostics_mask_path=diagnostics_path,
    )

    assert batch.tdcp_weights is not None
    expected_unscaled = 1.0 / ((0.05**2 + 0.05**2))
    np.testing.assert_allclose(batch.tdcp_weights[0, 0], expected_unscaled * 0.001, rtol=1e-12)
    np.testing.assert_array_equal(batch.tdcp_weights[0, 1:], np.zeros(3, dtype=np.float64))


def test_build_trip_arrays_applies_tdcp_geometry_correction(tmp_path):
    trip = tmp_path / "dataset_2023" / "train" / "course" / "phone"
    trip.mkdir(parents=True)

    baseline = np.array([-3947460.0, 3431490.0, 3637870.0], dtype=np.float64)
    rows = []
    for epoch_idx, utc_ms in enumerate((1000, 2000)):
        for svid in range(1, 5):
            rows.append(
                {
                    "utcTimeMillis": utc_ms,
                    "Svid": svid,
                    "ConstellationType": 1,
                    "SignalType": "GPS_L1_CA",
                    "RawPseudorangeMeters": 2.1e7 + 1000 * svid,
                    "IonosphericDelayMeters": 0.0,
                    "TroposphericDelayMeters": 0.0,
                    "SvClockBiasMeters": 0.0,
                    "SvPositionXEcefMeters": 2.2e7 + 1000 * svid + 50.0 * epoch_idx,
                    "SvPositionYEcefMeters": 1.4e7,
                    "SvPositionZEcefMeters": 2.1e7,
                    "SvElevationDegrees": 35.0,
                    "Cn0DbHz": 35.0,
                    "WlsPositionXEcefMeters": baseline[0],
                    "WlsPositionYEcefMeters": baseline[1],
                    "WlsPositionZEcefMeters": baseline[2],
                    "PseudorangeRateMetersPerSecond": 1.0,
                    "PseudorangeRateUncertaintyMetersPerSecond": 0.1,
                    "AccumulatedDeltaRangeState": 1,
                    "AccumulatedDeltaRangeMeters": 100.0 * svid + 10.0 * epoch_idx,
                    "AccumulatedDeltaRangeUncertaintyMeters": 0.02,
                },
            )
    _write_zipped_csv(trip / "device_gnss.csv", rows, list(rows[0].keys()))

    kwargs = dict(
        max_epochs=10,
        start_epoch=0,
        constellation_type=1,
        signal_type="GPS_L1_CA",
        weight_mode="sin2el",
        use_tdcp=True,
        tdcp_consistency_threshold_m=1e9,
    )
    raw = _build_trip_arrays(trip, tdcp_geometry_correction=False, **kwargs)
    corrected = _build_trip_arrays(trip, tdcp_geometry_correction=True, **kwargs)

    assert raw.tdcp_meas is not None
    assert corrected.tdcp_meas is not None
    assert corrected.tdcp_geometry_correction_count == 4
    rho0 = _geometric_range_with_sagnac(raw.sat_ecef[0, 0], raw.kaggle_wls[0])
    rho1 = _geometric_range_with_sagnac(raw.sat_ecef[1, 0], raw.kaggle_wls[1])
    np.testing.assert_allclose(corrected.tdcp_meas[0, 0], raw.tdcp_meas[0, 0] - (rho1 - rho0), rtol=1e-12)


def test_build_trip_arrays_estimates_clock_drift_for_mi8_family(tmp_path):
    trip = tmp_path / "dataset_2023" / "test" / "course" / "mi8"
    trip.mkdir(parents=True)

    base_xyz = np.array([-3947460.0, 3431490.0, 3637870.0], dtype=np.float64)
    rows = []
    for epoch_idx, utc_ms in enumerate((1000, 2000, 3000, 4000)):
        full_bias = -1.0e18 - (100.0 * epoch_idx / 299792458.0) * 1.0e9
        for svid in range(1, 5):
            rows.append(
                {
                    "utcTimeMillis": utc_ms,
                    "Svid": svid,
                    "ConstellationType": 1,
                    "SignalType": "GPS_L1_CA",
                    "RawPseudorangeMeters": 2.1e7 + 1000 * svid,
                    "IonosphericDelayMeters": 2.0,
                    "TroposphericDelayMeters": 3.0,
                    "SvClockBiasMeters": 10.0,
                    "SvPositionXEcefMeters": 2.6e7 - 1e5 * svid,
                    "SvPositionYEcefMeters": 1.3e7 + 2e5 * svid,
                    "SvPositionZEcefMeters": 2.1e7 - 3e5 * svid,
                    "SvElevationDegrees": 40.0 + svid,
                    "Cn0DbHz": 35.0 + svid,
                    "WlsPositionXEcefMeters": base_xyz[0] + float(epoch_idx),
                    "WlsPositionYEcefMeters": base_xyz[1],
                    "WlsPositionZEcefMeters": base_xyz[2],
                    "FullBiasNanos": full_bias,
                    "BiasNanos": 0.0,
                },
            )

    _write_zipped_csv(trip / "device_gnss.csv", rows, list(rows[0].keys()))

    batch = _build_trip_arrays(
        trip,
        max_epochs=10,
        start_epoch=0,
        constellation_type=1,
        signal_type="GPS_L1_CA",
        weight_mode="sin2el",
    )

    assert batch.clock_drift_mps is not None
    assert np.all(np.isfinite(batch.clock_drift_mps))
    assert np.all(batch.clock_drift_mps > 50.0)
    assert np.all(batch.clock_drift_mps < 140.0)


def test_estimate_residual_clock_series_recovers_bias_and_drift():
    times_ms = np.array([1000.0, 2000.0, 3000.0], dtype=np.float64)
    baseline_xyz = np.array(
        [
            [-3947460.0, 3431490.0, 3637870.0],
            [-3947459.0, 3431490.5, 3637870.25],
            [-3947458.0, 3431491.0, 3637870.5],
        ],
        dtype=np.float64,
    )
    sat_ecef = np.array(
        [
            [
                [2.60e7, 1.31e7, 2.11e7],
                [2.58e7, 1.33e7, 2.09e7],
                [2.57e7, 1.28e7, 2.15e7],
                [2.62e7, 1.26e7, 2.08e7],
            ],
        ]
        * 3,
        dtype=np.float64,
    )
    sat_vel = np.array(
        [
            [
                [10.0, -5.0, 2.0],
                [6.0, 3.0, -4.0],
                [-4.0, 7.0, 1.0],
                [3.0, -6.0, 5.0],
            ],
        ]
        * 3,
        dtype=np.float64,
    )
    clock_bias = np.array([120.0, 132.0, 144.0], dtype=np.float64)
    clock_drift = np.array([12.0, 12.0, 12.0], dtype=np.float64)
    ranges = _geometric_range_with_sagnac(sat_ecef, baseline_xyz[:, None, :])
    rx_vel = np.gradient(baseline_xyz, times_ms * 1e-3, axis=0, edge_order=1)
    geom_rate = _geometric_range_rate_with_sagnac(sat_ecef, baseline_xyz[:, None, :], sat_vel, rx_vel[:, None, :])
    pseudorange = ranges + clock_bias[:, None]
    doppler = clock_drift[:, None] - geom_rate
    sys_kind = np.array(
        [
            [0, 0, 1, 2],
            [0, 0, 1, 2],
            [0, 0, 1, 2],
        ],
        dtype=np.int32,
    )

    est_bias, est_drift = _estimate_residual_clock_series(
        times_ms,
        baseline_xyz,
        sat_ecef,
        pseudorange,
        sat_vel,
        doppler,
        sys_kind=sys_kind,
    )

    assert est_bias is not None
    assert est_drift is not None
    assert np.allclose(est_bias, clock_bias, atol=1e-6)
    assert np.allclose(est_drift, clock_drift, atol=1e-6)


def test_receiver_velocity_matches_matlab_scalar_interval_gradient():
    times_ms = np.array([0.0, 2000.0, 3000.0, 4000.0], dtype=np.float64)
    xyz = np.column_stack(
        [
            np.array([0.0, 2.0, 3.0, 4.0], dtype=np.float64),
            np.zeros(4, dtype=np.float64),
            np.zeros(4, dtype=np.float64),
        ],
    )

    velocity = raw_bridge._receiver_velocity_from_reference(times_ms, xyz)

    np.testing.assert_allclose(velocity[:, 0], np.array([2.0, 1.5, 1.0, 1.0]), atol=1e-12)
    np.testing.assert_allclose(velocity[:, 1:], 0.0, atol=1e-12)


def test_build_trip_arrays_prefers_residual_clock_series_for_blocklist_phone(tmp_path):
    trip = tmp_path / "dataset_2023" / "test" / "course" / "sm-a205u"
    trip.mkdir(parents=True)

    base_xyz = np.array(
        [
            [-3947460.0, 3431490.0, 3637870.0],
            [-3947459.0, 3431490.5, 3637870.25],
            [-3947458.0, 3431491.0, 3637870.5],
            [-3947457.0, 3431491.5, 3637870.75],
        ],
        dtype=np.float64,
    )
    sat_xyz = np.array(
        [
            [2.60e7, 1.31e7, 2.11e7],
            [2.58e7, 1.33e7, 2.09e7],
            [2.57e7, 1.28e7, 2.15e7],
            [2.62e7, 1.26e7, 2.08e7],
        ],
        dtype=np.float64,
    )
    sat_vel = np.array(
        [
            [10.0, -5.0, 2.0],
            [6.0, 3.0, -4.0],
            [-4.0, 7.0, 1.0],
            [3.0, -6.0, 5.0],
        ],
        dtype=np.float64,
    )
    times_ms = np.array([1000, 2000, 3000, 4000], dtype=np.int64)
    clock_bias = np.array([100.0, 112.0, 812.0, 824.0], dtype=np.float64)
    clock_drift = np.array([12.0, 12.0, 12.0, 12.0], dtype=np.float64)
    rx_vel = np.gradient(base_xyz, times_ms.astype(np.float64) * 1e-3, axis=0, edge_order=1)

    rows = []
    for epoch_idx, utc_ms in enumerate(times_ms):
        ranges = _geometric_range_with_sagnac(sat_xyz, base_xyz[epoch_idx])
        geom_rate = _geometric_range_rate_with_sagnac(
            sat_xyz,
            base_xyz[epoch_idx],
            sat_vel,
            rx_vel[epoch_idx],
        )
        for svid in range(1, 5):
            rows.append(
                {
                    "utcTimeMillis": int(utc_ms),
                    "Svid": svid,
                    "ConstellationType": 1,
                    "SignalType": "GPS_L1_CA",
                    "RawPseudorangeMeters": ranges[svid - 1] + clock_bias[epoch_idx],
                    "IonosphericDelayMeters": 0.0,
                    "TroposphericDelayMeters": 0.0,
                    "SvClockBiasMeters": 0.0,
                    "SvPositionXEcefMeters": sat_xyz[svid - 1, 0],
                    "SvPositionYEcefMeters": sat_xyz[svid - 1, 1],
                    "SvPositionZEcefMeters": sat_xyz[svid - 1, 2],
                    "SvVelocityXEcefMetersPerSecond": sat_vel[svid - 1, 0],
                    "SvVelocityYEcefMetersPerSecond": sat_vel[svid - 1, 1],
                    "SvVelocityZEcefMetersPerSecond": sat_vel[svid - 1, 2],
                    "SvElevationDegrees": 40.0 + svid,
                    "Cn0DbHz": 35.0 + svid,
                    "WlsPositionXEcefMeters": base_xyz[epoch_idx, 0],
                    "WlsPositionYEcefMeters": base_xyz[epoch_idx, 1],
                    "WlsPositionZEcefMeters": base_xyz[epoch_idx, 2],
                    "FullBiasNanos": -1.0e18,
                    "BiasNanos": 0.0,
                    "DriftNanosPerSecond": -1000.0 / 299792458.0 * 1.0e9,
                    "State": 1 | 8,
                    "MultipathIndicator": 0,
                    "PseudorangeRateMetersPerSecond": geom_rate[svid - 1] - clock_drift[epoch_idx],
                    "PseudorangeRateUncertaintyMetersPerSecond": 0.1,
                },
            )

    _write_zipped_csv(trip / "device_gnss.csv", rows, list(rows[0].keys()))

    batch = _build_trip_arrays(
        trip,
        max_epochs=10,
        start_epoch=0,
        constellation_type=1,
        signal_type="GPS_L1_CA",
        weight_mode="sin2el",
    )

    assert batch.clock_bias_m is not None
    assert batch.clock_drift_mps is not None
    assert batch.clock_jump is not None
    assert np.allclose(batch.clock_bias_m, clock_bias, atol=1e-6)
    assert np.allclose(batch.clock_drift_mps, clock_drift, atol=1e-6)
    assert batch.clock_jump.tolist() == [False, False, True, False]

    masked_batch = _build_trip_arrays(
        trip,
        max_epochs=10,
        start_epoch=0,
        constellation_type=1,
        signal_type="GPS_L1_CA",
        weight_mode="sin2el",
        apply_observation_mask=True,
        pseudorange_residual_mask_m=20.0,
        doppler_residual_mask_mps=0.0,
        pseudorange_doppler_mask_m=0.0,
    )

    assert masked_batch.residual_mask_count == 0
    assert np.all(masked_batch.weights > 0.0)


def test_segment_ranges_split_at_clock_jumps():
    clock_jump = np.array([False, False, True, False, True, False], dtype=bool)
    assert _segment_ranges(0, 6, clock_jump) == [(0, 2), (2, 4), (4, 6)]
    assert _segment_ranges(1, 5, clock_jump) == [(1, 2), (2, 4), (4, 5)]


def test_clock_aid_enabled_for_pixel4_and_clock_blocklist():
    assert _clock_aid_enabled("pixel4") is True
    assert _clock_aid_enabled("sm-a205u") is True
    assert _clock_aid_enabled("pixel5") is False
    assert _clock_aid_enabled("pixel4xl") is False


def test_clock_drift_seed_enabled_skips_sm_a505u_only():
    assert _clock_drift_seed_enabled("pixel4") is True
    assert _clock_drift_seed_enabled("sm-a205u") is True
    assert _clock_drift_seed_enabled("sm-a505u") is False
    assert _clock_drift_seed_enabled("pixel5") is False


def test_collect_matlab_parity_audit_reports_missing_base1(tmp_path):
    data_root = tmp_path / "dataset_2023"
    trip = data_root / "test" / "course" / "phone"
    trip.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "Course": "course",
                "Phone": "phone",
                "Type": "Street",
                "L5": 0,
                "BDS": 0,
                "RINEX": "V3",
                "Base1": np.nan,
                "IdxStart": 1,
                "IdxEnd": 10,
                "RPYReset": 0,
            },
        ],
    ).to_csv(data_root / "settings_test.csv", index=False)

    audit = collect_matlab_parity_audit(data_root, "test/course/phone")

    assert audit["settings_csv_present"] is True
    assert audit["setting_row_present"] is True
    assert audit["base_correction_status"] == "base1_missing"
    assert audit["base_correction_ready"] is False


def test_collect_matlab_parity_audit_quick_skips_raw_imu_parsing(tmp_path):
    data_root = tmp_path / "dataset_2023"
    trip = data_root / "train" / "course" / "phone"
    trip.mkdir(parents=True)
    (trip / "device_imu.csv").write_text("not,a,valid,imu\n", encoding="utf-8")
    (trip / "device_gnss.csv").write_text("not,a,valid,gnss\n", encoding="utf-8")
    pd.DataFrame(
        [
            {
                "Course": "course",
                "Phone": "phone",
                "Type": "Street",
                "L5": 0,
                "BDS": 0,
                "RINEX": "V3",
                "Base1": np.nan,
                "IdxStart": 1,
                "IdxEnd": 5,
                "RPYReset": 0,
            },
        ],
    ).to_csv(data_root / "settings_train.csv", index=False)

    audit = collect_matlab_parity_audit(data_root, "train/course/phone", include_imu_sync=False)

    assert audit["device_imu_present"] is True
    assert audit["imu_rows_acc"] == 0
    assert audit["gnss_elapsed_present"] is False
    assert audit["imu_sync_ready"] is False


def test_collect_matlab_parity_audit_detects_ready_base_correction_inputs(tmp_path):
    data_root = tmp_path / "dataset_2023"
    course_dir = data_root / "train" / "course"
    trip = course_dir / "phone"
    trip.mkdir(parents=True)
    (trip / "device_imu.csv").write_text("x\n", encoding="utf-8")
    (trip / "ground_truth.csv").write_text("x\n", encoding="utf-8")
    (course_dir / "brdc.23n").write_text("nav\n", encoding="utf-8")
    (course_dir / "BASE_rnx3.obs").write_text("obs\n", encoding="utf-8")
    base_dir = tmp_path / "base"
    base_dir.mkdir(parents=True)
    (base_dir / "base_position.csv").write_text("Base,Year,X,Y,Z\nBASE,2020,1,2,3\n", encoding="utf-8")
    (base_dir / "base_offset.csv").write_text("Base,E,N,U\nBASE,0,0,0\n", encoding="utf-8")
    pd.DataFrame(
        [
            {
                "Course": "course",
                "Phone": "phone",
                "Type": "Street",
                "L5": 0,
                "BDS": 0,
                "RINEX": "V3",
                "Base1": "BASE",
                "IdxStart": 1,
                "IdxEnd": 10,
                "RPYReset": 0,
            },
        ],
    ).to_csv(data_root / "settings_train.csv", index=False)

    audit = collect_matlab_parity_audit(data_root, "train/course/phone")

    assert audit["base_correction_status"] == "base_correction_ready"
    assert audit["base_correction_ready"] is True
    assert audit["base_obs_file_present"] is True
    assert audit["broadcast_nav_present"] is True
    assert audit["device_imu_present"] is True
    assert audit["ground_truth_present"] is True


def test_load_device_imu_measurements_and_process_stop_detection(tmp_path):
    trip = tmp_path / "dataset_2023" / "train" / "course" / "phone"
    trip.mkdir(parents=True)
    imu_rows = []
    for idx, utc_ms in enumerate((1000, 2000, 3000, 4000, 5000)):
        elapsed_ns = float(utc_ms) * 1e6
        acc_x = 0.0 if idx < 3 else 1.0
        gyro_x = 0.01 if idx < 3 else 0.2
        imu_rows.extend(
            [
                {
                    "MessageType": "UncalAccel",
                    "utcTimeMillis": utc_ms,
                    "elapsedRealtimeNanos": elapsed_ns,
                    "MeasurementX": acc_x,
                    "MeasurementY": 0.0,
                    "MeasurementZ": 9.81,
                    "BiasX": 0.1 * (idx + 1),
                    "BiasY": -0.2 * (idx + 1),
                    "BiasZ": 0.3,
                },
                {
                    "MessageType": "UncalGyro",
                    "utcTimeMillis": utc_ms,
                    "elapsedRealtimeNanos": elapsed_ns,
                    "MeasurementX": gyro_x,
                    "MeasurementY": 0.0,
                    "MeasurementZ": 0.0,
                    "BiasX": 0.01 * (idx + 1),
                    "BiasY": -0.02 * (idx + 1),
                    "BiasZ": 0.03,
                },
                {
                    "MessageType": "UncalMag",
                    "utcTimeMillis": utc_ms,
                    "elapsedRealtimeNanos": elapsed_ns,
                    "MeasurementX": 30.0,
                    "MeasurementY": -20.0,
                    "MeasurementZ": 5.0,
                    "BiasX": 1.0,
                    "BiasY": 2.0,
                    "BiasZ": 3.0,
                },
            ],
        )
    _write_zipped_csv(trip / "device_imu.csv", imu_rows, list(imu_rows[0].keys()))

    acc, gyro, mag = load_device_imu_measurements(trip)

    assert acc is not None and gyro is not None and mag is not None
    assert acc.xyz.shape == (5, 3)
    assert gyro.xyz.shape == (5, 3)
    assert mag.bias.shape == (5, 3)
    np.testing.assert_allclose(acc.bias[:, 0], np.array([0.1, 0.2, 0.3, 0.4, 0.5]))
    np.testing.assert_allclose(gyro.bias[:, 0], np.array([0.01, 0.02, 0.03, 0.04, 0.05]))

    gnss_times_ms = np.array([1000, 2000, 3000, 4000, 5000], dtype=np.float64)
    gnss_elapsed_ns = gnss_times_ms * 1e6
    acc_proc, gyro_proc, idx_stop = process_device_imu(acc, gyro, gnss_times_ms, gnss_elapsed_ns)
    stop_epochs = project_stop_to_epochs(acc_proc.times_ms, idx_stop, gnss_times_ms)

    assert acc_proc.sync_coefficient == 0.5
    assert gyro_proc.sync_coefficient == 0.5
    assert acc_proc.bias is not None
    assert gyro_proc.bias is not None
    np.testing.assert_allclose(acc_proc.bias, acc.bias)
    np.testing.assert_allclose(gyro_proc.bias, gyro.bias)
    assert stop_epochs.tolist()[:3] == [True, True, True]
    assert stop_epochs.tolist()[-2:] == [False, False]


def test_preintegrate_processed_imu_between_gnss_epochs():
    times_ms = np.array([0.0, 500.0, 1000.0, 1500.0])
    dt_s = np.full(times_ms.size, 0.5)
    acc = ProcessedIMU(
        times_ms=times_ms,
        xyz=np.tile(np.array([2.0, 0.0, 0.0]), (times_ms.size, 1)),
        dt_s=dt_s,
        norm_3d=np.full(times_ms.size, 2.0),
        norm_std=np.zeros(times_ms.size),
        sync_coefficient=1.0,
    )
    gyro = ProcessedIMU(
        times_ms=times_ms,
        xyz=np.tile(np.array([0.0, 0.0, 0.1]), (times_ms.size, 1)),
        dt_s=dt_s,
        norm_3d=np.full(times_ms.size, 0.1),
        norm_std=np.zeros(times_ms.size),
        sync_coefficient=1.0,
    )

    preint = preintegrate_processed_imu(acc, gyro, np.array([0.0, 1000.0, 1500.0]))

    np.testing.assert_allclose(preint.delta_t_s, np.array([1.0, 0.5]))
    np.testing.assert_array_equal(preint.sample_count, np.array([3, 2], dtype=np.int32))
    np.testing.assert_allclose(preint.delta_v_body[:, 0], np.array([2.0, 1.0]))
    np.testing.assert_allclose(preint.delta_p_body[:, 0], np.array([1.0, 0.25]))
    np.testing.assert_allclose(preint.delta_angle_rad[:, 2], np.array([0.1, 0.05]))


def test_preintegrate_processed_imu_tracks_interval_bias_means():
    times_ms = np.array([0.0, 500.0, 1000.0, 1500.0])
    dt_s = np.full(times_ms.size, 0.5)
    acc = ProcessedIMU(
        times_ms=times_ms,
        xyz=np.tile(np.array([2.0, 0.0, 0.0]), (times_ms.size, 1)),
        dt_s=dt_s,
        norm_3d=np.full(times_ms.size, 2.0),
        norm_std=np.zeros(times_ms.size),
        sync_coefficient=1.0,
        bias=np.column_stack(
            [
                np.array([1.0, 3.0, 5.0, 7.0]),
                np.zeros(times_ms.size),
                np.zeros(times_ms.size),
            ],
        ),
    )
    gyro = ProcessedIMU(
        times_ms=times_ms,
        xyz=np.tile(np.array([0.0, 0.0, 0.1]), (times_ms.size, 1)),
        dt_s=dt_s,
        norm_3d=np.full(times_ms.size, 0.1),
        norm_std=np.zeros(times_ms.size),
        sync_coefficient=1.0,
        bias=np.column_stack(
            [
                np.array([0.1, 0.3, 0.5, 0.7]),
                np.zeros(times_ms.size),
                np.zeros(times_ms.size),
            ],
        ),
    )

    preint = preintegrate_processed_imu(acc, gyro, np.array([0.0, 1000.0, 1500.0]))

    assert preint.acc_bias_mean_sensor is not None
    assert preint.gyro_bias_mean_sensor is not None
    np.testing.assert_allclose(preint.acc_bias_mean_sensor[:, 0], np.array([3.0, 6.0]))
    np.testing.assert_allclose(preint.gyro_bias_mean_sensor[:, 0], np.array([0.3, 0.6]))
    np.testing.assert_allclose(preint.acc_bias_mean_sensor[:, 1:], 0.0)
    np.testing.assert_allclose(preint.gyro_bias_mean_sensor[:, 1:], 0.0)


def test_preintegrate_processed_imu_ecef_frame_removes_stationary_gravity():
    times_ms = np.array([0.0, 500.0, 1000.0])
    dt_s = np.full(times_ms.size, 0.5)
    rot_body_sensor = raw_bridge._eul_xyz_to_rotm(raw_bridge.IMU_MOUNTING_ANGLE_RAD.reshape(1, 3))[0]
    stationary_acc_sensor = rot_body_sensor.T @ np.array([0.0, 0.0, raw_bridge.IMU_GRAVITY_MPS2], dtype=np.float64)
    acc = ProcessedIMU(
        times_ms=times_ms,
        xyz=np.tile(stationary_acc_sensor, (times_ms.size, 1)),
        dt_s=dt_s,
        norm_3d=np.full(times_ms.size, raw_bridge.IMU_GRAVITY_MPS2),
        norm_std=np.zeros(times_ms.size),
        sync_coefficient=1.0,
    )
    gyro = ProcessedIMU(
        times_ms=times_ms,
        xyz=np.zeros((times_ms.size, 3), dtype=np.float64),
        dt_s=dt_s,
        norm_3d=np.zeros(times_ms.size),
        norm_std=np.zeros(times_ms.size),
        sync_coefficient=1.0,
    )
    origin_xyz = np.asarray(lla_to_ecef(np.deg2rad(35.0), np.deg2rad(139.0), 10.0), dtype=np.float64)
    reference_xyz = np.tile(origin_xyz.reshape(1, 3), (times_ms.size, 1))

    preint = preintegrate_processed_imu(
        acc,
        gyro,
        times_ms,
        delta_frame="ecef",
        reference_xyz_ecef=reference_xyz,
    )

    assert preint.delta_frame == "ecef"
    np.testing.assert_allclose(preint.delta_v_body, 0.0, atol=1e-9)
    np.testing.assert_allclose(preint.delta_p_body, 0.0, atol=1e-9)


def test_imu_preintegration_segment_masks_invalid_intervals():
    preint = raw_bridge.IMUPreintegration(
        epoch_times_ms=np.array([0.0, 1000.0, 2000.0, 3000.0]),
        delta_t_s=np.array([1.0, 0.0, 1.0]),
        delta_v_body=np.array(
            [
                [0.1, 0.2, 0.3],
                [9.0, 9.0, 9.0],
                [0.4, 0.5, 0.6],
            ],
            dtype=np.float64,
        ),
        delta_p_body=np.array(
            [
                [1.0, 2.0, 3.0],
                [8.0, 8.0, 8.0],
                [4.0, 5.0, 6.0],
            ],
            dtype=np.float64,
        ),
        delta_angle_rad=np.zeros((3, 3), dtype=np.float64),
        sample_count=np.array([5, 0, 7], dtype=np.int32),
    )

    delta_p, delta_v, count = _imu_preintegration_segment(preint, 0, 4)

    assert count == 2
    assert delta_p is not None and delta_v is not None
    np.testing.assert_allclose(delta_p[0], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(delta_v[2], [0.4, 0.5, 0.6])
    assert np.isnan(delta_p[1]).all()
    assert np.isnan(delta_v[1]).all()


def test_run_fgo_chunked_forwards_opt_in_imu_prior(monkeypatch):
    true_pos = np.array([1.0e6, 2.0e6, 3.0e6], dtype=np.float64)
    sat = np.array(
        [
            [2.1e7, 0.0, 0.0],
            [0.0, 2.2e7, 0.0],
            [0.0, 0.0, 2.3e7],
            [1.8e7, 1.8e7, 1.8e7],
        ],
        dtype=np.float64,
    )
    n_epoch, n_sat = 3, 4
    sat_ecef = np.tile(sat.reshape(1, n_sat, 3), (n_epoch, 1, 1))
    pseudorange = np.zeros((n_epoch, n_sat), dtype=np.float64)
    for t in range(n_epoch):
        for s in range(n_sat):
            pseudorange[t, s] = _geometric_range_with_sagnac(true_pos, sat[s])
    weights = np.ones((n_epoch, n_sat), dtype=np.float64)
    raw_wls = np.zeros((n_epoch, 4), dtype=np.float64)
    raw_wls[:, :3] = true_pos + np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.4, 0.0, 0.0]])
    preint = raw_bridge.IMUPreintegration(
        epoch_times_ms=np.array([0.0, 1000.0, 2000.0]),
        delta_t_s=np.array([1.0, 1.0]),
        delta_v_body=np.array([[0.1, 0.0, 0.0], [0.2, 0.0, 0.0]], dtype=np.float64),
        delta_p_body=np.array([[0.5, 0.0, 0.0], [0.6, 0.0, 0.0]], dtype=np.float64),
        delta_angle_rad=np.zeros((2, 3), dtype=np.float64),
        sample_count=np.array([10, 10], dtype=np.int32),
    )
    batch = raw_bridge.TripArrays(
        times_ms=np.array([0.0, 1000.0, 2000.0], dtype=np.float64),
        sat_ecef=sat_ecef,
        pseudorange=pseudorange,
        weights=weights,
        kaggle_wls=raw_wls[:, :3],
        truth=np.full((n_epoch, 3), np.nan, dtype=np.float64),
        max_sats=n_sat,
        has_truth=False,
        n_clock=1,
        dt=np.array([1.0, 1.0, 0.0], dtype=np.float64),
        imu_preintegration=preint,
        absolute_height_ref_ecef=raw_wls[:, :3] + np.array([0.0, 0.0, 1.0], dtype=np.float64),
        absolute_height_ref_count=3,
    )
    captured: dict[str, np.ndarray | float | None] = {}

    def fake_fgo_gnss_lm_vd(*args, **kwargs):
        captured["state_shape"] = args[3].shape
        captured["imu_delta_p"] = kwargs.get("imu_delta_p")
        captured["imu_delta_v"] = kwargs.get("imu_delta_v")
        captured["imu_position_sigma_m"] = kwargs.get("imu_position_sigma_m")
        captured["imu_velocity_sigma_mps"] = kwargs.get("imu_velocity_sigma_mps")
        captured["imu_accel_bias_prior_sigma_mps2"] = kwargs.get("imu_accel_bias_prior_sigma_mps2")
        captured["imu_accel_bias_between_sigma_mps2"] = kwargs.get("imu_accel_bias_between_sigma_mps2")
        captured["absolute_height_ref_ecef"] = kwargs.get("absolute_height_ref_ecef")
        captured["absolute_height_sigma_m"] = kwargs.get("absolute_height_sigma_m")
        captured["enu_up_ecef"] = kwargs.get("enu_up_ecef")
        return 1, 0.0

    monkeypatch.setattr(raw_bridge, "fgo_gnss_lm_vd", fake_fgo_gnss_lm_vd)

    raw_bridge.run_fgo_chunked(
        batch,
        raw_wls,
        clock_jump=None,
        clock_drift_seed_mps=None,
        clock_use_average_drift=False,
        tdcp_use_drift=False,
        stop_mask=None,
        motion_sigma_m=0.0,
        clock_drift_sigma_m=0.0,
        stop_velocity_sigma_mps=0.0,
        stop_position_sigma_m=0.0,
        apply_imu_prior=True,
        imu_position_sigma_m=12.5,
        imu_velocity_sigma_mps=2.5,
        imu_accel_bias_state=True,
        imu_accel_bias_prior_sigma_mps2=9.5,
        imu_accel_bias_between_sigma_mps2=0.75,
        fgo_iters=1,
        tol=1e-7,
        chunk_epochs=0,
        use_vd=True,
        apply_absolute_height=True,
        absolute_height_sigma_m=0.75,
    )

    assert captured["state_shape"] == (3, 11)
    np.testing.assert_allclose(captured["imu_delta_p"], preint.delta_p_body)
    np.testing.assert_allclose(captured["imu_delta_v"], preint.delta_v_body)
    assert captured["imu_position_sigma_m"] == 12.5
    assert captured["imu_velocity_sigma_mps"] == 2.5
    assert captured["imu_accel_bias_prior_sigma_mps2"] == 9.5
    assert captured["imu_accel_bias_between_sigma_mps2"] == 0.75
    np.testing.assert_allclose(captured["absolute_height_ref_ecef"], batch.absolute_height_ref_ecef)
    assert captured["absolute_height_sigma_m"] == 0.75
    assert captured["enu_up_ecef"] is not None


def test_run_fgo_chunked_masks_imu_prior_across_factor_dt_gap(monkeypatch):
    true_pos = np.array([1.0e6, 2.0e6, 3.0e6], dtype=np.float64)
    sat = np.array(
        [
            [2.1e7, 0.0, 0.0],
            [0.0, 2.2e7, 0.0],
            [0.0, 0.0, 2.3e7],
            [1.8e7, 1.8e7, 1.8e7],
        ],
        dtype=np.float64,
    )
    n_epoch, n_sat = 3, 4
    sat_ecef = np.tile(sat.reshape(1, n_sat, 3), (n_epoch, 1, 1))
    pseudorange = np.zeros((n_epoch, n_sat), dtype=np.float64)
    for t in range(n_epoch):
        for s in range(n_sat):
            pseudorange[t, s] = _geometric_range_with_sagnac(true_pos, sat[s])
    weights = np.ones((n_epoch, n_sat), dtype=np.float64)
    raw_wls = np.zeros((n_epoch, 4), dtype=np.float64)
    raw_wls[:, :3] = true_pos
    preint = raw_bridge.IMUPreintegration(
        epoch_times_ms=np.array([0.0, 1000.0, 2500.0]),
        delta_t_s=np.array([1.0, 1.5]),
        delta_v_body=np.array([[0.1, 0.0, 0.0], [9.0, 9.0, 9.0]], dtype=np.float64),
        delta_p_body=np.array([[0.5, 0.0, 0.0], [8.0, 8.0, 8.0]], dtype=np.float64),
        delta_angle_rad=np.zeros((2, 3), dtype=np.float64),
        sample_count=np.array([10, 10], dtype=np.int32),
    )
    batch = raw_bridge.TripArrays(
        times_ms=np.array([0.0, 1000.0, 2500.0], dtype=np.float64),
        sat_ecef=sat_ecef,
        pseudorange=pseudorange,
        weights=weights,
        kaggle_wls=raw_wls[:, :3],
        truth=np.full((n_epoch, 3), np.nan, dtype=np.float64),
        max_sats=n_sat,
        has_truth=False,
        n_clock=1,
        dt=np.array([1.0, 0.0, 0.0], dtype=np.float64),
        imu_preintegration=preint,
        factor_dt_gap_count=1,
    )
    captured: dict[str, np.ndarray | None] = {}

    def fake_fgo_gnss_lm_vd(*args, **kwargs):
        captured["imu_delta_p"] = kwargs.get("imu_delta_p")
        captured["imu_delta_v"] = kwargs.get("imu_delta_v")
        return 1, 0.0

    monkeypatch.setattr(raw_bridge, "fgo_gnss_lm_vd", fake_fgo_gnss_lm_vd)

    raw_bridge.run_fgo_chunked(
        batch,
        raw_wls,
        clock_jump=None,
        clock_drift_seed_mps=None,
        clock_use_average_drift=False,
        tdcp_use_drift=False,
        stop_mask=None,
        motion_sigma_m=0.0,
        clock_drift_sigma_m=0.0,
        stop_velocity_sigma_mps=0.0,
        stop_position_sigma_m=0.0,
        apply_imu_prior=True,
        imu_position_sigma_m=12.5,
        imu_velocity_sigma_mps=2.5,
        fgo_iters=1,
        tol=1e-7,
        chunk_epochs=0,
        use_vd=True,
    )

    assert captured["imu_delta_p"] is not None and captured["imu_delta_v"] is not None
    assert captured["imu_delta_p"].shape == (1, 3)
    assert captured["imu_delta_v"].shape == (1, 3)
    np.testing.assert_allclose(captured["imu_delta_p"][0], preint.delta_p_body[0])
    np.testing.assert_allclose(captured["imu_delta_v"][0], preint.delta_v_body[0])


def test_solver_stop_mask_filters_fast_epochs():
    origin_xyz = np.asarray(lla_to_ecef(np.deg2rad(35.0), np.deg2rad(139.0), 10.0), dtype=np.float64)
    enu = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    xyz = _enu_to_ecef_relative(enu, origin_xyz)
    times_ms = np.array([0, 1000, 2000, 3000, 4000], dtype=np.float64)

    speed_mps = estimate_speed_mps(xyz, times_ms)
    assert np.all(speed_mps[:3] < 0.1)
    assert np.all(speed_mps[3:] >= 0.5)

    mask = solver_stop_mask(np.ones(5, dtype=bool), xyz, times_ms)
    assert mask is not None
    assert mask.tolist() == [True, True, True, False, False]


def test_collect_matlab_parity_audit_reports_imu_sync_ready(tmp_path):
    data_root = tmp_path / "dataset_2023"
    trip = data_root / "train" / "course" / "phone"
    trip.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "Course": "course",
                "Phone": "phone",
                "Type": "Street",
                "L5": 0,
                "BDS": 0,
                "RINEX": "V3",
                "Base1": np.nan,
                "IdxStart": 1,
                "IdxEnd": 5,
                "RPYReset": 0,
            },
        ],
    ).to_csv(data_root / "settings_train.csv", index=False)

    gnss_rows = []
    for utc_ms in (1000, 2000, 3000, 4000, 5000):
        for svid in range(1, 5):
            gnss_rows.append(
                {
                    "utcTimeMillis": utc_ms,
                    "Svid": svid,
                    "ConstellationType": 1,
                    "SignalType": "GPS_L1_CA",
                    "RawPseudorangeMeters": 2.1e7 + 1000 * svid,
                    "IonosphericDelayMeters": 2.0,
                    "TroposphericDelayMeters": 3.0,
                    "SvClockBiasMeters": 10.0,
                    "SvPositionXEcefMeters": 2.6e7 - 1e5 * svid,
                    "SvPositionYEcefMeters": 1.3e7 + 2e5 * svid,
                    "SvPositionZEcefMeters": 2.1e7 - 3e5 * svid,
                    "SvElevationDegrees": 30.0 + svid,
                    "Cn0DbHz": 35.0 + svid,
                    "WlsPositionXEcefMeters": -3947460.0,
                    "WlsPositionYEcefMeters": 3431490.0,
                    "WlsPositionZEcefMeters": 3637870.0,
                    "ChipsetElapsedRealtimeNanos": float(utc_ms) * 1e6,
                },
            )
    _write_zipped_csv(trip / "device_gnss.csv", gnss_rows, list(gnss_rows[0].keys()))

    imu_rows = []
    for idx, utc_ms in enumerate((1000, 2000, 3000, 4000, 5000)):
        elapsed_ns = float(utc_ms) * 1e6
        acc_x = 0.0 if idx < 3 else 1.0
        gyro_x = 0.01 if idx < 3 else 0.2
        imu_rows.extend(
            [
                {
                    "MessageType": "UncalAccel",
                    "utcTimeMillis": utc_ms,
                    "elapsedRealtimeNanos": elapsed_ns,
                    "MeasurementX": acc_x,
                    "MeasurementY": 0.0,
                    "MeasurementZ": 9.81,
                    "BiasX": 0.0,
                    "BiasY": 0.0,
                    "BiasZ": 0.0,
                },
                {
                    "MessageType": "UncalGyro",
                    "utcTimeMillis": utc_ms,
                    "elapsedRealtimeNanos": elapsed_ns,
                    "MeasurementX": gyro_x,
                    "MeasurementY": 0.0,
                    "MeasurementZ": 0.0,
                    "BiasX": 0.0,
                    "BiasY": 0.0,
                    "BiasZ": 0.0,
                },
            ],
        )
    _write_zipped_csv(trip / "device_imu.csv", imu_rows, list(imu_rows[0].keys()))

    audit = collect_matlab_parity_audit(data_root, "train/course/phone")

    assert audit["device_imu_present"] is True
    assert audit["gnss_elapsed_present"] is True
    assert audit["imu_sync_ready"] is True
    assert audit["stop_epoch_count"] >= 1


def test_estimate_rpy_from_velocity_keeps_east_heading():
    vel_enu = np.tile(np.array([[2.0, 0.0, 0.0]], dtype=np.float64), (40, 1))
    rpy = estimate_rpy_from_velocity(vel_enu)
    assert rpy.shape == (40, 3)
    assert np.allclose(rpy[:, :2], 0.0)
    assert np.allclose(np.rad2deg(rpy[:, 2]), -180.0)


def test_apply_phone_position_offset_shifts_along_heading():
    origin_xyz = np.array([6378137.0, 0.0, 0.0], dtype=np.float64)
    enu = np.column_stack(
        [
            np.linspace(0.0, 49.0, 50, dtype=np.float64),
            np.zeros(50, dtype=np.float64),
            np.zeros(50, dtype=np.float64),
        ],
    )
    xyz = _enu_to_ecef_relative(enu, origin_xyz)

    offset_xyz = apply_phone_position_offset(xyz, "pixel4")
    offset_enu = _ecef_to_enu_relative(offset_xyz, origin_xyz)

    delta = offset_enu - enu
    assert np.allclose(delta[:, 0], -0.15, atol=1e-3)
    assert np.allclose(delta[:, 1], 0.0, atol=1e-3)
    assert np.allclose(delta[:, 2], 0.0, atol=1e-3)


def test_apply_relative_height_constraint_equalizes_revisit_up():
    origin_xyz = np.array([6378137.0, 0.0, 0.0], dtype=np.float64)
    enu = np.array(
        [
            [0.0, 0.0, 0.0],
            [60.0, 0.0, 5.0],
            [120.0, 0.0, 12.0],
            [60.0, 0.0, 25.0],
            [2.0, 2.0, 30.0],
        ],
        dtype=np.float64,
    )
    xyz = _enu_to_ecef_relative(enu, origin_xyz)

    corrected_xyz = apply_relative_height_constraint(xyz, xyz)
    corrected_enu = _ecef_to_enu_relative(corrected_xyz, origin_xyz)

    assert np.allclose(corrected_enu[[0, 4], 2], 15.0, atol=1e-3)
    assert np.allclose(corrected_enu[[1, 3], 2], 15.0, atol=1e-3)
    assert np.isclose(corrected_enu[2, 2], enu[2, 2], atol=1e-3)


def test_apply_relative_height_constraint_skips_stop_epochs():
    origin_xyz = np.array([6378137.0, 0.0, 0.0], dtype=np.float64)
    enu = np.array(
        [
            [0.0, 0.0, 0.0],
            [60.0, 0.0, 5.0],
            [120.0, 0.0, 12.0],
            [60.0, 0.0, 25.0],
            [2.0, 2.0, 30.0],
        ],
        dtype=np.float64,
    )
    xyz = _enu_to_ecef_relative(enu, origin_xyz)
    stop_mask = np.array([False, False, False, False, True], dtype=bool)

    corrected_xyz = apply_relative_height_constraint(xyz, xyz, stop_mask)
    corrected_enu = _ecef_to_enu_relative(corrected_xyz, origin_xyz)

    assert np.isclose(corrected_enu[0, 2], enu[0, 2], atol=1e-3)
    assert np.isclose(corrected_enu[4, 2], enu[4, 2], atol=1e-3)


def test_load_absolute_height_reference_ecef_maps_nearby_ref_hight(tmp_path):
    course_dir = tmp_path / "train" / "course"
    course_dir.mkdir(parents=True)
    origin_xyz = np.asarray(lla_to_ecef(np.deg2rad(35.0), np.deg2rad(139.0), 10.0), dtype=np.float64)
    query_enu = np.array(
        [
            [0.0, 0.0, 0.0],
            [20.0, 0.0, 9.0],
            [0.0, 5.0, 3.0],
        ],
        dtype=np.float64,
    )
    query_xyz = _enu_to_ecef_relative(query_enu, origin_xyz)
    savemat(
        course_dir / "ref_hight.mat",
        {
            "posgt": {
                "enu": np.array([[1.0, 1.0, 100.0], [100.0, 100.0, 50.0]], dtype=np.float64),
                "up": np.array([100.0, 50.0], dtype=np.float64),
            },
        },
    )

    ref_xyz, count = raw_bridge.load_absolute_height_reference_ecef(course_dir, query_xyz, dist_m=15.0)

    assert count == 2
    assert ref_xyz is not None
    ref_enu = _ecef_to_enu_relative(ref_xyz, origin_xyz)
    np.testing.assert_allclose(ref_enu[[0, 2], 2], np.array([100.0, 100.0]), atol=1e-3)
    assert not np.isfinite(ref_enu[1]).any()


def test_effective_multi_gnss_enabled_disables_mi8_family():
    assert _effective_multi_gnss_enabled("test/course/mi8", True) is False
    assert _effective_multi_gnss_enabled("test/course/xiaomimi8", True) is False
    assert _effective_multi_gnss_enabled("test/course/pixel5", True) is True
    assert _effective_multi_gnss_enabled("test/course/mi8", False) is False


def test_effective_position_source_uses_raw_wls_for_mi8_family():
    assert _effective_position_source("test/course/mi8", "gated") == "gated"
    assert _effective_position_source("test/course/xiaomimi8", "auto") == "raw_wls"
    assert _effective_position_source("test/course/mi8", "baseline") == "baseline"
    assert _effective_position_source("test/course/pixel5", "gated") == "gated"


def test_should_refine_outlier_result_only_for_large_gated_auto_errors():
    assert _should_refine_outlier_result("gated", 200, 1200.0) is True
    assert _should_refine_outlier_result("auto", 200, 1000.1) is True
    assert _should_refine_outlier_result("raw_wls", 200, 5000.0) is False
    assert _should_refine_outlier_result("gated", 30, 5000.0) is False
    assert _should_refine_outlier_result("gated", 200, 999.0) is False


def test_export_bridge_outputs(tmp_path):
    export_dir = tmp_path / "bridge"
    times_ms = np.array([1000, 2000], dtype=np.float64)
    ecef = np.array(
        [
            [-3947460.0, 3431490.0, 3637870.0],
            [-3947459.5, 3431490.5, 3637870.5],
        ],
        dtype=np.float64,
    )
    metrics = {
        "rms_2d": 1.0,
        "rms_3d": 1.2,
        "mean_2d": 0.9,
        "mean_3d": 1.0,
        "std_2d": 0.2,
        "p50": 0.8,
        "p67": 0.9,
        "p95": 1.4,
        "max_2d": 1.5,
        "n_epochs": 2,
    }
    result = BridgeResult(
        trip="train/course/phone",
        signal_type="GPS_L1_CA",
        weight_mode="sin2el",
        selected_source_mode="auto",
        times_ms=times_ms,
        kaggle_wls=ecef,
        raw_wls=np.column_stack([ecef, np.zeros(2)]),
        fgo_state=np.column_stack([ecef + 1.0, np.zeros(2)]),
        selected_state=np.column_stack([ecef, np.zeros(2)]),
        selected_sources=np.array(["baseline", "fgo"], dtype=object),
        truth=ecef - 1.0,
        max_sats=7,
        fgo_iters=3,
        failed_chunks=0,
        selected_mse_pr=12.5,
        baseline_mse_pr=10.0,
        raw_wls_mse_pr=11.0,
        fgo_mse_pr=12.5,
        selected_source_counts={"baseline": 1, "raw_wls": 0, "fgo": 1},
        metrics_selected=metrics,
        metrics_kaggle=metrics,
        metrics_raw_wls=metrics,
        metrics_fgo=metrics,
    )

    _export_bridge_outputs(export_dir, result)

    pos = pd.read_csv(export_dir / "bridge_positions.csv")
    meta = json.loads((export_dir / "bridge_metrics.json").read_text(encoding="utf-8"))

    assert list(pos["UnixTimeMillis"]) == [1000, 2000]
    assert list(pos["SelectedSource"]) == ["baseline", "fgo"]
    assert "FgoLatitudeDegrees" in pos.columns
    assert "LatitudeDegrees" in pos.columns
    assert "GroundTruthLongitudeDegrees" in pos.columns
    assert meta["trip"] == "train/course/phone"
    assert meta["fgo_iters"] == 3
    assert meta["fgo_score_m"] == 1.1
    assert meta["selected_score_m"] == 1.1
    assert meta["baseline_mse_pr"] == 10.0


def test_export_bridge_outputs_without_ground_truth(tmp_path):
    export_dir = tmp_path / "bridge"
    times_ms = np.array([1000], dtype=np.float64)
    ecef = np.array([[-3947460.0, 3431490.0, 3637870.0]], dtype=np.float64)

    result = BridgeResult(
        trip="test/course/phone",
        signal_type="GPS_L1_CA",
        weight_mode="sin2el",
        selected_source_mode="baseline",
        times_ms=times_ms,
        kaggle_wls=ecef,
        raw_wls=np.column_stack([ecef, np.zeros(1)]),
        fgo_state=np.column_stack([ecef + 1.0, np.zeros(1)]),
        selected_state=np.column_stack([ecef, np.zeros(1)]),
        selected_sources=np.array(["raw_wls"], dtype=object),
        truth=None,
        max_sats=7,
        fgo_iters=2,
        failed_chunks=1,
        selected_mse_pr=5.0,
        baseline_mse_pr=4.0,
        raw_wls_mse_pr=5.0,
        fgo_mse_pr=6.0,
        selected_source_counts={"baseline": 0, "raw_wls": 1, "fgo": 0},
        metrics_selected=None,
        metrics_kaggle=None,
        metrics_raw_wls=None,
        metrics_fgo=None,
    )

    _export_bridge_outputs(export_dir, result)

    pos = pd.read_csv(export_dir / "bridge_positions.csv")
    meta = json.loads((export_dir / "bridge_metrics.json").read_text(encoding="utf-8"))

    assert np.isnan(pos.loc[0, "GroundTruthLatitudeDegrees"])
    assert meta["fgo_score_m"] is None
    assert meta["selected_source_counts"]["raw_wls"] == 1


def test_bridge_position_columns_uses_selected_output_for_gated_and_auto():
    columns = {
        "LatitudeDegrees",
        "LongitudeDegrees",
        "BaselineLatitudeDegrees",
        "BaselineLongitudeDegrees",
        "RawWlsLatitudeDegrees",
        "RawWlsLongitudeDegrees",
        "FgoLatitudeDegrees",
        "FgoLongitudeDegrees",
    }

    assert bridge_position_columns("gated", columns) == ("LatitudeDegrees", "LongitudeDegrees")
    assert bridge_position_columns("auto", columns) == ("LatitudeDegrees", "LongitudeDegrees")


def test_bridge_position_columns_uses_named_candidate_columns():
    columns = {
        "LatitudeDegrees",
        "LongitudeDegrees",
        "BaselineLatitudeDegrees",
        "BaselineLongitudeDegrees",
        "RawWlsLatitudeDegrees",
        "RawWlsLongitudeDegrees",
        "FgoLatitudeDegrees",
        "FgoLongitudeDegrees",
    }

    assert bridge_position_columns("baseline", columns) == (
        "BaselineLatitudeDegrees",
        "BaselineLongitudeDegrees",
    )
    assert bridge_position_columns("raw_wls", columns) == (
        "RawWlsLatitudeDegrees",
        "RawWlsLongitudeDegrees",
    )
    assert bridge_position_columns("fgo", columns) == ("FgoLatitudeDegrees", "FgoLongitudeDegrees")


def test_fit_state_with_clock_bias_estimates_bias_and_residual():
    sat_ecef = np.array(
        [
            [
                [15600000.0, 0.0, 20100000.0],
                [0.0, 17600000.0, 21300000.0],
                [-16600000.0, 0.0, 20800000.0],
                [0.0, -18600000.0, 21700000.0],
            ],
        ],
        dtype=np.float64,
    )
    xyz = np.array([[1113194.9, -4841695.5, 3985355.2]], dtype=np.float64)
    rho = _geometric_range_with_sagnac(sat_ecef[0], xyz[0])
    bias = 73.0
    pseudorange = (rho + bias).reshape(1, -1)
    weights = np.ones_like(pseudorange)

    state, weighted_sse, weight_sum, per_epoch_wmse = _fit_state_with_clock_bias(
        sat_ecef,
        pseudorange,
        weights,
        xyz,
    )

    assert np.isclose(state[0, 3], bias)
    assert np.isclose(weighted_sse, 0.0, atol=1e-9)
    assert np.isclose(weight_sum, 4.0)
    assert np.isclose(per_epoch_wmse[0], 0.0, atol=1e-9)


def test_build_trip_arrays_multi_gnss_with_tdcp(tmp_path):
    trip = tmp_path / "dataset_2023" / "train" / "course" / "phone"
    trip.mkdir(parents=True)

    rows = []
    epoch_rows = [
        (1000, [
            (1, 1, "GPS_L1_CA", -0.60, 100.0, 25, 2500.0),
            (1, 2, "GPS_L1_CA", -0.50, 200.0, 25, 1500.0),
            (6, 11, "GAL_E1_C_P", -0.40, 300.0, 25, 500.0),
            (4, 193, "QZS_L1_CA", -0.30, 400.0, 25, -500.0),
        ]),
        (2000, [
            (4, 193, "QZS_L1_CA", -0.30, 400.3, 29, -500.0),
            (6, 11, "GAL_E1_C_P", -0.40, 300.4, 25, 500.0),
            (1, 2, "GPS_L1_CA", -0.50, 200.5, 25, 1500.0),
            (1, 1, "GPS_L1_CA", -0.60, 100.6, 25, 2500.0),
        ]),
    ]

    for utc_ms, sats in epoch_rows:
        for idx, (constellation, svid, signal_type, pr_rate, adr, adr_state, svx) in enumerate(sats, start=1):
            rows.append(
                {
                    "utcTimeMillis": utc_ms,
                    "Svid": svid,
                    "ConstellationType": constellation,
                    "SignalType": signal_type,
                    "RawPseudorangeMeters": 2.1e7 + 1000 * idx,
                    "IonosphericDelayMeters": 2.0,
                    "TroposphericDelayMeters": 3.0,
                    "SvClockBiasMeters": 10.0,
                    "SvPositionXEcefMeters": 2.6e7 - 1e5 * idx,
                    "SvPositionYEcefMeters": 1.3e7 + 2e5 * idx,
                    "SvPositionZEcefMeters": 2.1e7 - 3e5 * idx,
                    "SvElevationDegrees": 30.0 + idx,
                    "Cn0DbHz": 35.0 + idx,
                    "WlsPositionXEcefMeters": -3947460.0 + utc_ms * 0.001,
                    "WlsPositionYEcefMeters": 3431490.0 + utc_ms * 0.001,
                    "WlsPositionZEcefMeters": 3637870.0 + utc_ms * 0.001,
                    "PseudorangeRateMetersPerSecond": pr_rate,
                    "PseudorangeRateUncertaintyMetersPerSecond": 0.2,
                    "AccumulatedDeltaRangeState": adr_state,
                    "AccumulatedDeltaRangeMeters": adr,
                    "AccumulatedDeltaRangeUncertaintyMeters": 0.02,
                    "SvVelocityXEcefMetersPerSecond": svx,
                    "SvVelocityYEcefMetersPerSecond": -20.0 * idx,
                    "SvVelocityZEcefMetersPerSecond": 30.0 * idx,
                },
            )

    _write_zipped_csv(trip / "device_gnss.csv", rows, list(rows[0].keys()))

    batch = _build_trip_arrays(
        trip,
        max_epochs=10,
        start_epoch=0,
        constellation_type=1,
        signal_type="GPS_L1_CA",
        weight_mode="sin2el",
        multi_gnss=True,
        use_tdcp=True,
        tdcp_consistency_threshold_m=1e9,
        tdcp_geometry_correction=False,
    )

    assert batch.n_clock == 3
    assert batch.n_sat_slots == 4
    np.testing.assert_array_equal(batch.sys_kind[0], np.array([0, 0, 1, 2], dtype=np.int32))
    np.testing.assert_allclose(batch.dt, np.array([1.0, 0.0]))
    assert batch.doppler is not None
    assert batch.doppler_weights is not None
    assert batch.sat_vel is not None
    assert batch.tdcp_meas is not None
    assert batch.tdcp_weights is not None
    np.testing.assert_allclose(batch.doppler[0, :4], np.array([0.60, 0.50, 0.40, 0.30]))
    np.testing.assert_allclose(batch.sat_vel[0, 2], np.array([500.0, -60.0, 90.0]))
    np.testing.assert_allclose(batch.tdcp_meas[0, :3], np.array([0.6, 0.5, 0.4]), atol=1e-9)
    assert batch.tdcp_meas[0, 3] == 0.0
    assert np.all(batch.tdcp_weights[0, :3] > 0.0)
    assert batch.tdcp_weights[0, 3] == 0.0


def test_fit_state_with_clock_bias_multi_clock():
    sat_ecef = np.array(
        [
            [
                [15600000.0, 0.0, 20100000.0],
                [0.0, 17600000.0, 21300000.0],
                [-16600000.0, 0.0, 20800000.0],
                [0.0, -18600000.0, 21700000.0],
                [20300000.0, 4000000.0, 11000000.0],
                [-20400000.0, -5000000.0, 12000000.0],
            ],
        ],
        dtype=np.float64,
    )
    xyz = np.array([[1113194.9, -4841695.5, 3985355.2]], dtype=np.float64)
    rho = _geometric_range_with_sagnac(sat_ecef[0], xyz[0])
    sys_kind = np.array([[0, 0, 1, 1, 2, 2]], dtype=np.int32)
    clock_bias = np.array([73.0, 12.0, -8.0], dtype=np.float64)
    total_bias = np.array([73.0, 73.0, 85.0, 85.0, 65.0, 65.0], dtype=np.float64)
    pseudorange = (rho + total_bias).reshape(1, -1)
    weights = np.ones_like(pseudorange)

    state, weighted_sse, weight_sum, per_epoch_wmse = _fit_state_with_clock_bias(
        sat_ecef,
        pseudorange,
        weights,
        xyz,
        sys_kind=sys_kind,
        n_clock=3,
    )

    np.testing.assert_allclose(state[0, 3:], clock_bias, atol=1e-9)
    assert np.isclose(weighted_sse, 0.0, atol=1e-9)
    assert np.isclose(weight_sum, 6.0)
    assert np.isclose(per_epoch_wmse[0], 0.0, atol=1e-9)


def test_select_auto_chunk_source_prefers_smoother_candidate_when_mse_is_close():
    candidates = {
        "baseline": ChunkCandidateQuality(
            mse_pr=100.0,
            step_mean_m=2.0,
            step_p95_m=5.0,
            accel_mean_m=1.0,
            accel_p95_m=3.0,
            bridge_jump_m=2.0,
            baseline_gap_mean_m=0.0,
            baseline_gap_p95_m=0.0,
            baseline_gap_max_m=0.0,
            quality_score=1.0,
        ),
        "fgo": ChunkCandidateQuality(
            mse_pr=106.0,
            step_mean_m=1.0,
            step_p95_m=2.0,
            accel_mean_m=0.4,
            accel_p95_m=1.2,
            bridge_jump_m=0.5,
            baseline_gap_mean_m=1.0,
            baseline_gap_p95_m=2.0,
            baseline_gap_max_m=3.0,
            quality_score=0.78,
        ),
    }

    assert _select_auto_chunk_source(candidates) == "fgo"


def test_select_gated_chunk_source_stays_baseline_without_clear_quality_gain():
    record = ChunkSelectionRecord(
        start_epoch=0,
        end_epoch=200,
        auto_source="fgo",
        candidates={
            "baseline": ChunkCandidateQuality(
                mse_pr=120.0,
                step_mean_m=1.8,
                step_p95_m=4.0,
                accel_mean_m=0.9,
                accel_p95_m=2.0,
                bridge_jump_m=0.4,
                baseline_gap_mean_m=0.0,
                baseline_gap_p95_m=0.0,
                baseline_gap_max_m=0.0,
                quality_score=1.0,
            ),
            "fgo": ChunkCandidateQuality(
                mse_pr=128.0,
                step_mean_m=1.6,
                step_p95_m=3.7,
                accel_mean_m=0.8,
                accel_p95_m=1.9,
                bridge_jump_m=0.4,
                baseline_gap_mean_m=0.8,
                baseline_gap_p95_m=1.2,
                baseline_gap_max_m=1.5,
                quality_score=0.97,
            ),
        },
    )

    assert _select_gated_chunk_source(record, baseline_threshold=500.0) == "baseline"


def test_select_gated_chunk_source_switches_when_quality_gain_is_clear():
    record = ChunkSelectionRecord(
        start_epoch=0,
        end_epoch=200,
        auto_source="fgo",
        candidates={
            "baseline": ChunkCandidateQuality(
                mse_pr=170.0,
                step_mean_m=2.5,
                step_p95_m=6.0,
                accel_mean_m=1.2,
                accel_p95_m=3.0,
                bridge_jump_m=1.0,
                baseline_gap_mean_m=0.0,
                baseline_gap_p95_m=0.0,
                baseline_gap_max_m=0.0,
                quality_score=1.0,
            ),
            "fgo": ChunkCandidateQuality(
                mse_pr=178.0,
                step_mean_m=1.1,
                step_p95_m=2.2,
                accel_mean_m=0.5,
                accel_p95_m=1.1,
                bridge_jump_m=0.3,
                baseline_gap_mean_m=0.9,
                baseline_gap_p95_m=1.5,
                baseline_gap_max_m=2.0,
                quality_score=0.78,
            ),
        },
    )

    assert _select_gated_chunk_source(record, baseline_threshold=500.0) == "fgo"


def test_select_gated_chunk_source_rejects_fgo_with_large_baseline_gap():
    record = ChunkSelectionRecord(
        start_epoch=0,
        end_epoch=200,
        auto_source="fgo",
        candidates={
            "baseline": ChunkCandidateQuality(
                mse_pr=20.854,
                step_mean_m=3.8,
                step_p95_m=12.538,
                accel_mean_m=1.6,
                accel_p95_m=6.0,
                bridge_jump_m=0.5,
                baseline_gap_mean_m=0.0,
                baseline_gap_p95_m=0.0,
                baseline_gap_max_m=0.0,
                quality_score=1.0,
            ),
            "fgo": ChunkCandidateQuality(
                mse_pr=13.240,
                step_mean_m=3.4,
                step_p95_m=13.409,
                accel_mean_m=1.3,
                accel_p95_m=5.0,
                bridge_jump_m=0.4,
                baseline_gap_mean_m=7.0,
                baseline_gap_p95_m=15.975,
                baseline_gap_max_m=21.748,
                quality_score=0.911,
            ),
        },
    )

    assert _select_gated_chunk_source(record, baseline_threshold=500.0) == "baseline"


def test_select_gated_chunk_source_rejects_raw_wls_when_baseline_is_not_catastrophic():
    record = ChunkSelectionRecord(
        start_epoch=0,
        end_epoch=200,
        auto_source="raw_wls",
        candidates={
            "baseline": ChunkCandidateQuality(
                mse_pr=8.622,
                step_mean_m=6.0,
                step_p95_m=20.099,
                accel_mean_m=2.0,
                accel_p95_m=6.543,
                bridge_jump_m=0.5,
                baseline_gap_mean_m=0.0,
                baseline_gap_p95_m=0.0,
                baseline_gap_max_m=0.0,
                quality_score=1.0,
            ),
            "raw_wls": ChunkCandidateQuality(
                mse_pr=5.709,
                step_mean_m=7.0,
                step_p95_m=21.185,
                accel_mean_m=5.0,
                accel_p95_m=17.126,
                bridge_jump_m=0.5,
                baseline_gap_mean_m=5.0,
                baseline_gap_p95_m=10.862,
                baseline_gap_max_m=21.082,
                quality_score=0.827,
            ),
        },
    )

    assert _select_gated_chunk_source(record, baseline_threshold=500.0) == "baseline"


def test_select_gated_chunk_source_allows_raw_wls_for_mi8_baseline_jump():
    record = ChunkSelectionRecord(
        start_epoch=0,
        end_epoch=200,
        auto_source="baseline",
        candidates={
            "baseline": ChunkCandidateQuality(
                mse_pr=15.848,
                step_mean_m=1086.0,
                step_p95_m=8151.3,
                accel_mean_m=1758.0,
                accel_p95_m=11423.0,
                bridge_jump_m=0.0,
                baseline_gap_mean_m=0.0,
                baseline_gap_p95_m=0.0,
                baseline_gap_max_m=0.0,
                quality_score=1.0,
            ),
            "raw_wls": ChunkCandidateQuality(
                mse_pr=9.791,
                step_mean_m=1096.0,
                step_p95_m=8151.3,
                accel_mean_m=1778.0,
                accel_p95_m=11423.0,
                bridge_jump_m=0.0,
                baseline_gap_mean_m=11.6,
                baseline_gap_p95_m=35.7,
                baseline_gap_max_m=118.4,
                quality_score=0.549,
            ),
        },
    )

    assert _select_gated_chunk_source(record, baseline_threshold=500.0) == "baseline"
    assert (
        _select_gated_chunk_source(
            record,
            baseline_threshold=500.0,
            allow_raw_wls_on_mi8_baseline_jump=True,
        )
        == "raw_wls"
    )


def test_select_gated_chunk_source_rejects_raw_wls_for_mi8_jump_when_gap_is_large():
    record = ChunkSelectionRecord(
        start_epoch=0,
        end_epoch=200,
        auto_source="baseline",
        candidates={
            "baseline": ChunkCandidateQuality(
                mse_pr=15.0,
                step_mean_m=1000.0,
                step_p95_m=5000.0,
                accel_mean_m=1000.0,
                accel_p95_m=8000.0,
                bridge_jump_m=0.0,
                baseline_gap_mean_m=0.0,
                baseline_gap_p95_m=0.0,
                baseline_gap_max_m=0.0,
                quality_score=1.0,
            ),
            "raw_wls": ChunkCandidateQuality(
                mse_pr=9.0,
                step_mean_m=10.0,
                step_p95_m=20.0,
                accel_mean_m=4.0,
                accel_p95_m=8.0,
                bridge_jump_m=0.0,
                baseline_gap_mean_m=300.0,
                baseline_gap_p95_m=450.0,
                baseline_gap_max_m=600.0,
                quality_score=0.5,
            ),
        },
    )

    assert (
        _select_gated_chunk_source(
            record,
            baseline_threshold=500.0,
            allow_raw_wls_on_mi8_baseline_jump=True,
        )
        == "baseline"
    )


def test_select_gated_chunk_source_uses_safe_fgo_when_raw_wls_is_unsafe():
    record = ChunkSelectionRecord(
        start_epoch=0,
        end_epoch=200,
        auto_source="raw_wls",
        candidates={
            "baseline": ChunkCandidateQuality(
                mse_pr=25.500,
                step_mean_m=8.0,
                step_p95_m=25.926,
                accel_mean_m=10.0,
                accel_p95_m=26.685,
                bridge_jump_m=0.5,
                baseline_gap_mean_m=0.0,
                baseline_gap_p95_m=0.0,
                baseline_gap_max_m=0.0,
                quality_score=1.0,
            ),
            "raw_wls": ChunkCandidateQuality(
                mse_pr=9.500,
                step_mean_m=9.0,
                step_p95_m=23.996,
                accel_mean_m=12.0,
                accel_p95_m=30.063,
                bridge_jump_m=0.5,
                baseline_gap_mean_m=8.0,
                baseline_gap_p95_m=15.054,
                baseline_gap_max_m=21.382,
                quality_score=0.701,
            ),
            "fgo": ChunkCandidateQuality(
                mse_pr=9.082,
                step_mean_m=9.0,
                step_p95_m=23.973,
                accel_mean_m=12.0,
                accel_p95_m=30.397,
                bridge_jump_m=0.5,
                baseline_gap_mean_m=8.0,
                baseline_gap_p95_m=16.357,
                baseline_gap_max_m=46.062,
                quality_score=0.725,
            ),
        },
    )

    assert _select_gated_chunk_source(record, baseline_threshold=500.0) == "fgo"


def test_select_gated_chunk_source_rejects_fgo_when_baseline_pr_is_already_low():
    record = ChunkSelectionRecord(
        start_epoch=0,
        end_epoch=200,
        auto_source="raw_wls",
        candidates={
            "baseline": ChunkCandidateQuality(
                mse_pr=11.335,
                step_mean_m=7.753,
                step_p95_m=19.473,
                accel_mean_m=6.361,
                accel_p95_m=11.806,
                bridge_jump_m=0.0,
                baseline_gap_mean_m=0.0,
                baseline_gap_p95_m=0.0,
                baseline_gap_max_m=0.0,
                quality_score=1.0,
            ),
            "fgo": ChunkCandidateQuality(
                mse_pr=4.792,
                step_mean_m=7.749,
                step_p95_m=19.067,
                accel_mean_m=6.008,
                accel_p95_m=13.447,
                bridge_jump_m=0.0,
                baseline_gap_mean_m=7.688,
                baseline_gap_p95_m=13.960,
                baseline_gap_max_m=21.145,
                quality_score=0.624,
            ),
        },
    )

    assert _select_gated_chunk_source(record, baseline_threshold=500.0) == "baseline"


def test_select_gated_chunk_source_rejects_fgo_when_raw_wls_has_better_pr_mse():
    record = ChunkSelectionRecord(
        start_epoch=0,
        end_epoch=200,
        auto_source="fgo",
        candidates={
            "baseline": ChunkCandidateQuality(
                mse_pr=22.535,
                step_mean_m=7.525,
                step_p95_m=26.406,
                accel_mean_m=3.0,
                accel_p95_m=6.312,
                bridge_jump_m=0.0,
                baseline_gap_mean_m=0.0,
                baseline_gap_p95_m=0.0,
                baseline_gap_max_m=0.0,
                quality_score=1.0,
            ),
            "fgo": ChunkCandidateQuality(
                mse_pr=14.275,
                step_mean_m=9.318,
                step_p95_m=26.849,
                accel_mean_m=6.0,
                accel_p95_m=12.916,
                bridge_jump_m=0.0,
                baseline_gap_mean_m=8.0,
                baseline_gap_p95_m=12.835,
                baseline_gap_max_m=23.113,
                quality_score=0.781,
            ),
            "raw_wls": ChunkCandidateQuality(
                mse_pr=13.446,
                step_mean_m=13.278,
                step_p95_m=28.591,
                accel_mean_m=14.0,
                accel_p95_m=32.206,
                bridge_jump_m=0.0,
                baseline_gap_mean_m=12.0,
                baseline_gap_p95_m=17.138,
                baseline_gap_max_m=28.190,
                quality_score=1.163,
            ),
        },
    )

    assert _select_gated_chunk_source(record, baseline_threshold=500.0) == "baseline"


def test_select_gated_chunk_source_uses_raw_wls_on_high_baseline_pr_mse_rescue():
    record = ChunkSelectionRecord(
        start_epoch=1400,
        end_epoch=1512,
        auto_source="baseline",
        candidates={
            "baseline": ChunkCandidateQuality(
                mse_pr=56.713,
                step_mean_m=8.0,
                step_p95_m=22.106,
                accel_mean_m=10.0,
                accel_p95_m=32.112,
                bridge_jump_m=0.0,
                baseline_gap_mean_m=0.0,
                baseline_gap_p95_m=0.0,
                baseline_gap_max_m=0.0,
                quality_score=1.0,
            ),
            "raw_wls": ChunkCandidateQuality(
                mse_pr=15.667,
                step_mean_m=14.0,
                step_p95_m=60.398,
                accel_mean_m=28.0,
                accel_p95_m=116.126,
                bridge_jump_m=0.0,
                baseline_gap_mean_m=35.0,
                baseline_gap_p95_m=94.627,
                baseline_gap_max_m=123.432,
                quality_score=1.813,
            ),
        },
    )

    assert _select_gated_chunk_source(record, baseline_threshold=500.0) == "raw_wls"


def test_select_gated_chunk_source_rejects_relaxed_high_pr_mse_rescue():
    record = ChunkSelectionRecord(
        start_epoch=1400,
        end_epoch=1512,
        auto_source="baseline",
        candidates={
            "baseline": ChunkCandidateQuality(
                mse_pr=119.0495,
                step_mean_m=8.0,
                step_p95_m=22.106,
                accel_mean_m=10.0,
                accel_p95_m=32.112,
                bridge_jump_m=0.0,
                baseline_gap_mean_m=0.0,
                baseline_gap_p95_m=0.0,
                baseline_gap_max_m=0.0,
                quality_score=1.0,
            ),
            "raw_wls": ChunkCandidateQuality(
                mse_pr=44.3967,
                step_mean_m=14.0,
                step_p95_m=60.398,
                accel_mean_m=28.0,
                accel_p95_m=116.126,
                bridge_jump_m=0.0,
                baseline_gap_mean_m=35.0,
                baseline_gap_p95_m=94.627,
                baseline_gap_max_m=123.432,
                quality_score=1.813,
            ),
        },
    )

    assert _select_gated_chunk_source(record, baseline_threshold=500.0) == "baseline"


def test_select_gated_chunk_source_rejects_raw_wls_rescue_below_baseline_pr_floor():
    record = ChunkSelectionRecord(
        start_epoch=1000,
        end_epoch=1200,
        auto_source="baseline",
        candidates={
            "baseline": ChunkCandidateQuality(
                mse_pr=47.458,
                step_mean_m=25.0,
                step_p95_m=74.166,
                accel_mean_m=35.0,
                accel_p95_m=124.0,
                bridge_jump_m=0.0,
                baseline_gap_mean_m=0.0,
                baseline_gap_p95_m=0.0,
                baseline_gap_max_m=0.0,
                quality_score=1.0,
            ),
            "raw_wls": ChunkCandidateQuality(
                mse_pr=16.568,
                step_mean_m=30.0,
                step_p95_m=105.511,
                accel_mean_m=50.0,
                accel_p95_m=200.0,
                bridge_jump_m=0.0,
                baseline_gap_mean_m=40.0,
                baseline_gap_p95_m=56.054,
                baseline_gap_max_m=282.253,
                quality_score=0.732,
            ),
        },
    )

    assert _select_gated_chunk_source(record, baseline_threshold=500.0) == "baseline"


def test_select_gated_chunk_source_rejects_raw_wls_rescue_on_nonfinite_pr_mse():
    record = ChunkSelectionRecord(
        start_epoch=800,
        end_epoch=1000,
        auto_source="raw_wls",
        candidates={
            "baseline": ChunkCandidateQuality(
                mse_pr=56.713,
                step_mean_m=8.0,
                step_p95_m=22.106,
                accel_mean_m=10.0,
                accel_p95_m=32.112,
                bridge_jump_m=0.0,
                baseline_gap_mean_m=0.0,
                baseline_gap_p95_m=0.0,
                baseline_gap_max_m=0.0,
                quality_score=1.0,
            ),
            "raw_wls": ChunkCandidateQuality(
                mse_pr=float("inf"),
                step_mean_m=100000.0,
                step_p95_m=1142004.98,
                accel_mean_m=100000.0,
                accel_p95_m=1000000.0,
                bridge_jump_m=0.0,
                baseline_gap_mean_m=0.0,
                baseline_gap_p95_m=0.0,
                baseline_gap_max_m=0.0,
                quality_score=1.0,
            ),
        },
    )

    assert _select_gated_chunk_source(record, baseline_threshold=500.0) == "baseline"


def test_select_gated_chunk_source_rejects_fgo_without_enough_quality_gain():
    record = ChunkSelectionRecord(
        start_epoch=0,
        end_epoch=200,
        auto_source="raw_wls",
        candidates={
            "baseline": ChunkCandidateQuality(
                mse_pr=12.115,
                step_mean_m=8.0,
                step_p95_m=19.772,
                accel_mean_m=5.0,
                accel_p95_m=13.474,
                bridge_jump_m=0.5,
                baseline_gap_mean_m=0.0,
                baseline_gap_p95_m=0.0,
                baseline_gap_max_m=0.0,
                quality_score=1.0,
            ),
            "raw_wls": ChunkCandidateQuality(
                mse_pr=7.981,
                step_mean_m=10.0,
                step_p95_m=24.198,
                accel_mean_m=12.0,
                accel_p95_m=27.679,
                bridge_jump_m=0.5,
                baseline_gap_mean_m=6.0,
                baseline_gap_p95_m=11.909,
                baseline_gap_max_m=37.866,
                quality_score=0.824,
            ),
            "fgo": ChunkCandidateQuality(
                mse_pr=8.828,
                step_mean_m=10.0,
                step_p95_m=24.848,
                accel_mean_m=12.0,
                accel_p95_m=32.129,
                bridge_jump_m=0.5,
                baseline_gap_mean_m=8.0,
                baseline_gap_p95_m=15.378,
                baseline_gap_max_m=47.870,
                quality_score=0.930,
            ),
        },
    )

    assert _select_gated_chunk_source(record, baseline_threshold=500.0) == "baseline"


def test_select_gated_chunk_source_can_use_tdcp_off_fgo_candidate():
    record = ChunkSelectionRecord(
        start_epoch=0,
        end_epoch=200,
        auto_source="fgo",
        candidates={
            "baseline": ChunkCandidateQuality(
                mse_pr=20.0,
                step_mean_m=3.0,
                step_p95_m=12.0,
                accel_mean_m=1.0,
                accel_p95_m=4.0,
                bridge_jump_m=0.5,
                baseline_gap_mean_m=0.0,
                baseline_gap_p95_m=0.0,
                baseline_gap_max_m=0.0,
                quality_score=1.0,
            ),
            "fgo": ChunkCandidateQuality(
                mse_pr=18.0,
                step_mean_m=2.0,
                step_p95_m=8.0,
                accel_mean_m=0.8,
                accel_p95_m=3.0,
                bridge_jump_m=0.4,
                baseline_gap_mean_m=9.0,
                baseline_gap_p95_m=18.0,
                baseline_gap_max_m=24.0,
                quality_score=0.70,
            ),
            "fgo_no_tdcp": ChunkCandidateQuality(
                mse_pr=18.5,
                step_mean_m=2.1,
                step_p95_m=8.3,
                accel_mean_m=0.8,
                accel_p95_m=3.1,
                bridge_jump_m=0.4,
                baseline_gap_mean_m=4.0,
                baseline_gap_p95_m=9.0,
                baseline_gap_max_m=12.0,
                quality_score=0.76,
            ),
        },
    )

    assert _select_gated_chunk_source(record, baseline_threshold=500.0) == "fgo_no_tdcp"


def test_select_gated_chunk_source_keeps_safe_tdcp_fgo_when_tdcp_off_is_tied():
    record = ChunkSelectionRecord(
        start_epoch=0,
        end_epoch=200,
        auto_source="fgo_no_tdcp",
        candidates={
            "baseline": ChunkCandidateQuality(
                mse_pr=20.0,
                step_mean_m=3.0,
                step_p95_m=12.0,
                accel_mean_m=1.0,
                accel_p95_m=4.0,
                bridge_jump_m=0.5,
                baseline_gap_mean_m=0.0,
                baseline_gap_p95_m=0.0,
                baseline_gap_max_m=0.0,
                quality_score=1.0,
            ),
            "fgo": ChunkCandidateQuality(
                mse_pr=18.6,
                step_mean_m=2.0,
                step_p95_m=8.0,
                accel_mean_m=0.8,
                accel_p95_m=3.0,
                bridge_jump_m=0.4,
                baseline_gap_mean_m=4.2,
                baseline_gap_p95_m=9.1,
                baseline_gap_max_m=12.0,
                quality_score=0.864,
            ),
            "fgo_no_tdcp": ChunkCandidateQuality(
                mse_pr=18.5,
                step_mean_m=2.1,
                step_p95_m=8.3,
                accel_mean_m=0.8,
                accel_p95_m=3.1,
                bridge_jump_m=0.4,
                baseline_gap_mean_m=4.0,
                baseline_gap_p95_m=9.0,
                baseline_gap_max_m=12.0,
                quality_score=0.861,
            ),
        },
    )

    assert _select_gated_chunk_source(record, baseline_threshold=500.0) == "fgo"


def test_select_gated_chunk_source_uses_tdcp_off_when_tdcp_increases_baseline_gap():
    record = ChunkSelectionRecord(
        start_epoch=0,
        end_epoch=200,
        auto_source="fgo",
        candidates={
            "baseline": ChunkCandidateQuality(
                mse_pr=20.0,
                step_mean_m=20.0,
                step_p95_m=27.5,
                accel_mean_m=4.0,
                accel_p95_m=9.0,
                bridge_jump_m=0.5,
                baseline_gap_mean_m=0.0,
                baseline_gap_p95_m=0.0,
                baseline_gap_max_m=0.0,
                quality_score=1.0,
            ),
            "fgo": ChunkCandidateQuality(
                mse_pr=7.07,
                step_mean_m=20.0,
                step_p95_m=27.7,
                accel_mean_m=3.0,
                accel_p95_m=5.8,
                bridge_jump_m=0.4,
                baseline_gap_mean_m=8.0,
                baseline_gap_p95_m=13.17,
                baseline_gap_max_m=32.8,
                quality_score=0.642,
            ),
            "fgo_no_tdcp": ChunkCandidateQuality(
                mse_pr=6.95,
                step_mean_m=20.0,
                step_p95_m=28.0,
                accel_mean_m=3.2,
                accel_p95_m=6.5,
                bridge_jump_m=0.4,
                baseline_gap_mean_m=7.8,
                baseline_gap_p95_m=12.94,
                baseline_gap_max_m=34.9,
                quality_score=0.645,
            ),
        },
    )

    assert _select_gated_chunk_source(record, baseline_threshold=500.0) == "fgo_no_tdcp"


def test_select_gated_chunk_source_rejects_tdcp_off_fgo_with_large_baseline_gap():
    record = ChunkSelectionRecord(
        start_epoch=0,
        end_epoch=200,
        auto_source="fgo_no_tdcp",
        candidates={
            "baseline": ChunkCandidateQuality(
                mse_pr=20.0,
                step_mean_m=3.0,
                step_p95_m=12.0,
                accel_mean_m=1.0,
                accel_p95_m=4.0,
                bridge_jump_m=0.5,
                baseline_gap_mean_m=0.0,
                baseline_gap_p95_m=0.0,
                baseline_gap_max_m=0.0,
                quality_score=1.0,
            ),
            "fgo_no_tdcp": ChunkCandidateQuality(
                mse_pr=18.5,
                step_mean_m=2.1,
                step_p95_m=8.3,
                accel_mean_m=0.8,
                accel_p95_m=3.1,
                bridge_jump_m=0.4,
                baseline_gap_mean_m=9.0,
                baseline_gap_p95_m=18.0,
                baseline_gap_max_m=24.0,
                quality_score=0.76,
            ),
        },
    )

    assert _select_gated_chunk_source(record, baseline_threshold=500.0) == "baseline"


def test_select_gated_chunk_source_overrides_baseline_on_catastrophic_gap():
    record = ChunkSelectionRecord(
        start_epoch=0,
        end_epoch=200,
        auto_source="baseline",
        candidates={
            "baseline": ChunkCandidateQuality(
                mse_pr=40000.0,
                step_mean_m=2.0,
                step_p95_m=8.0,
                accel_mean_m=1.0,
                accel_p95_m=3.0,
                bridge_jump_m=0.5,
                baseline_gap_mean_m=0.0,
                baseline_gap_p95_m=0.0,
                baseline_gap_max_m=0.0,
                quality_score=1.0,
            ),
            "raw_wls": ChunkCandidateQuality(
                mse_pr=30000.0,
                step_mean_m=2.1,
                step_p95_m=8.2,
                accel_mean_m=1.1,
                accel_p95_m=3.1,
                bridge_jump_m=0.5,
                baseline_gap_mean_m=15.0,
                baseline_gap_p95_m=20.0,
                baseline_gap_max_m=2500.0,
                quality_score=1.1,
            ),
        },
    )

    assert _select_gated_chunk_source(record, baseline_threshold=500.0) == "raw_wls"


def test_select_gated_chunk_source_keeps_high_pr_baseline_when_candidates_are_worse():
    record = ChunkSelectionRecord(
        start_epoch=0,
        end_epoch=200,
        auto_source="fgo",
        candidates={
            "baseline": ChunkCandidateQuality(
                mse_pr=700.0,
                step_mean_m=4.0,
                step_p95_m=14.0,
                accel_mean_m=1.2,
                accel_p95_m=3.0,
                bridge_jump_m=0.5,
                baseline_gap_mean_m=0.0,
                baseline_gap_p95_m=0.0,
                baseline_gap_max_m=0.0,
                quality_score=1.0,
            ),
            "fgo": ChunkCandidateQuality(
                mse_pr=900.0,
                step_mean_m=1.5,
                step_p95_m=4.0,
                accel_mean_m=0.5,
                accel_p95_m=1.0,
                bridge_jump_m=0.2,
                baseline_gap_mean_m=3.0,
                baseline_gap_p95_m=6.0,
                baseline_gap_max_m=8.0,
                quality_score=0.2,
            ),
            "raw_wls": ChunkCandidateQuality(
                mse_pr=850.0,
                step_mean_m=1.8,
                step_p95_m=5.0,
                accel_mean_m=0.6,
                accel_p95_m=1.2,
                bridge_jump_m=0.3,
                baseline_gap_mean_m=4.0,
                baseline_gap_p95_m=7.0,
                baseline_gap_max_m=9.0,
                quality_score=0.3,
            ),
        },
    )

    assert _select_gated_chunk_source(record, baseline_threshold=500.0) == "baseline"


def test_select_gated_chunk_source_ignores_catastrophic_candidate_when_baseline_pr_is_low():
    record = ChunkSelectionRecord(
        start_epoch=0,
        end_epoch=200,
        auto_source="raw_wls",
        candidates={
            "baseline": ChunkCandidateQuality(
                mse_pr=17.159,
                step_mean_m=6.0,
                step_p95_m=16.748,
                accel_mean_m=7.0,
                accel_p95_m=15.028,
                bridge_jump_m=0.0,
                baseline_gap_mean_m=0.0,
                baseline_gap_p95_m=0.0,
                baseline_gap_max_m=0.0,
                quality_score=1.0,
            ),
            "raw_wls": ChunkCandidateQuality(
                mse_pr=9.022,
                step_mean_m=8.0,
                step_p95_m=26.051,
                accel_mean_m=18.0,
                accel_p95_m=41.526,
                bridge_jump_m=0.0,
                baseline_gap_mean_m=20.0,
                baseline_gap_p95_m=27.439,
                baseline_gap_max_m=1675.787,
                quality_score=1.374,
            ),
        },
    )

    assert _select_gated_chunk_source(record, baseline_threshold=500.0) == "baseline"


def test_select_gated_chunk_source_rejects_catastrophic_raw_wls_with_implausible_motion():
    record = ChunkSelectionRecord(
        start_epoch=0,
        end_epoch=200,
        auto_source="raw_wls",
        candidates={
            "baseline": ChunkCandidateQuality(
                mse_pr=3519056.1531199794,
                step_mean_m=4.0,
                step_p95_m=21.220942594632188,
                accel_mean_m=1.2,
                accel_p95_m=3.0,
                bridge_jump_m=0.5,
                baseline_gap_mean_m=0.0,
                baseline_gap_p95_m=0.0,
                baseline_gap_max_m=0.0,
                quality_score=1.0,
            ),
            "raw_wls": ChunkCandidateQuality(
                mse_pr=2358467.694703256,
                step_mean_m=1600.0,
                step_p95_m=15863.785548140342,
                accel_mean_m=1000.0,
                accel_p95_m=3000.0,
                bridge_jump_m=0.5,
                baseline_gap_mean_m=4000.0,
                baseline_gap_p95_m=10828.702581415682,
                baseline_gap_max_m=62901.588575134316,
                quality_score=376.33264170804114,
            ),
        },
    )

    assert _select_gated_chunk_source(record, baseline_threshold=500.0) == "baseline"


def test_select_gated_chunk_source_prefers_raw_wls_when_high_baseline_fgo_has_worse_pr_mse():
    record = ChunkSelectionRecord(
        start_epoch=0,
        end_epoch=200,
        auto_source="baseline",
        candidates={
            "baseline": ChunkCandidateQuality(
                mse_pr=593.2585739364114,
                step_mean_m=5.0,
                step_p95_m=54.716833837440134,
                accel_mean_m=1.2,
                accel_p95_m=3.0,
                bridge_jump_m=0.5,
                baseline_gap_mean_m=0.0,
                baseline_gap_p95_m=0.0,
                baseline_gap_max_m=0.0,
                quality_score=1.0,
            ),
            "raw_wls": ChunkCandidateQuality(
                mse_pr=340.02240275341495,
                step_mean_m=10.0,
                step_p95_m=81.11465847694961,
                accel_mean_m=1.3,
                accel_p95_m=3.2,
                bridge_jump_m=0.5,
                baseline_gap_mean_m=60.0,
                baseline_gap_p95_m=141.04417807735396,
                baseline_gap_max_m=180.0,
                quality_score=0.8,
            ),
            "fgo_no_tdcp": ChunkCandidateQuality(
                mse_pr=404.78081585689944,
                step_mean_m=4.0,
                step_p95_m=19.94341427298643,
                accel_mean_m=1.0,
                accel_p95_m=2.8,
                bridge_jump_m=0.5,
                baseline_gap_mean_m=50.0,
                baseline_gap_p95_m=109.77561573112344,
                baseline_gap_max_m=140.0,
                quality_score=0.2,
            ),
        },
    )

    assert _select_gated_chunk_source(record, baseline_threshold=500.0) == "raw_wls"


def test_select_gated_chunk_source_keeps_plausible_catastrophic_raw_wls_after_motion_guard():
    record = ChunkSelectionRecord(
        start_epoch=1000,
        end_epoch=1190,
        auto_source="raw_wls",
        candidates={
            "baseline": ChunkCandidateQuality(
                mse_pr=7138.391759882097,
                step_mean_m=4.0,
                step_p95_m=14.031165674318725,
                accel_mean_m=1.2,
                accel_p95_m=3.0,
                bridge_jump_m=0.5,
                baseline_gap_mean_m=0.0,
                baseline_gap_p95_m=0.0,
                baseline_gap_max_m=0.0,
                quality_score=1.0,
            ),
            "raw_wls": ChunkCandidateQuality(
                mse_pr=5897.7679641067,
                step_mean_m=4.2,
                step_p95_m=19.4061330586764,
                accel_mean_m=1.3,
                accel_p95_m=3.2,
                bridge_jump_m=0.5,
                baseline_gap_mean_m=8.0,
                baseline_gap_p95_m=15.6892564306083,
                baseline_gap_max_m=1787.3164492887113,
                quality_score=1.6560789641992357,
            ),
        },
    )

    assert _select_gated_chunk_source(record, baseline_threshold=500.0) == "raw_wls"
