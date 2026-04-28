from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import experiments.compare_gsdc2023_residual_values as residual_values
import experiments.gsdc2023_raw_bridge as raw_bridge
from experiments.compare_gsdc2023_gnss_log_observation_counts import compare_gnss_log_observation_counts
from experiments.compare_gsdc2023_gnss_log_residual_prekeys import compare_gnss_log_residual_prekeys
from experiments.compare_gsdc2023_residual_diagnostics_factor_mask import (
    build_factor_mask_from_residual_diagnostics,
    compare_residual_diagnostics_factor_mask,
)
from experiments.gsdc2023_gnss_log_reader import C_LIGHT, GPS_WEEK_NANOS, gnss_log_observation_counts, load_gnss_log_observations


def _raw_line(
    *,
    utc_ms: int,
    time_nanos: int,
    svid: int,
    carrier_hz: float,
    received_sv_time_nanos: int,
    full_bias_nanos: int = 0,
    bias_nanos: float = 0.0,
    cn0: float = 35.0,
    state: int = (1 << 0) | (1 << 3),
    adr_state: int = 1,
    adr_m: float = 1000.0,
    multipath: int = 0,
) -> str:
    values = [
        utc_ms,
        time_nanos,
        18,
        "",
        full_bias_nanos,
        bias_nanos,
        1.0,
        0.0,
        1.0,
        0,
        svid,
        0.0,
        state,
        received_sv_time_nanos,
        10,
        cn0,
        -100.0,
        0.1,
        adr_state,
        adr_m,
        0.1,
        carrier_hz,
        "",
        "",
        "",
        multipath,
        "",
        1,
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        123456789,
    ]
    return "Raw," + ",".join(str(value) for value in values) + "\n"


def _write_log(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# Raw,utcTimeMillis,...\n"
        + _raw_line(
            utc_ms=1000,
            time_nanos=20_000_000_000,
            svid=1,
            carrier_hz=1_575_420_000.0,
            received_sv_time_nanos=19_930_000_000,
        )
        + _raw_line(
            utc_ms=1000,
            time_nanos=20_000_000_000,
            svid=1,
            carrier_hz=1_176_450_000.0,
            received_sv_time_nanos=19_930_000_000,
            adr_m=0.0,
        )
        + _raw_line(
            utc_ms=2000,
            time_nanos=21_000_000_000,
            svid=1,
            carrier_hz=1_575_420_000.0,
            received_sv_time_nanos=20_930_000_000,
        )
        + _raw_line(
            utc_ms=2000,
            time_nanos=21_000_000_000,
            svid=1,
            carrier_hz=1_176_450_000.0,
            received_sv_time_nanos=20_930_000_000,
        ),
        encoding="utf-8",
    )


def test_gnss_log_observation_counts_match_gobsphone_raw_availability(tmp_path):
    log_path = tmp_path / "gnss_log.txt"
    _write_log(log_path)

    counts = gnss_log_observation_counts(log_path)

    assert counts["L1"] == {"P": 2, "D": 2, "L": 2}
    assert counts["L5"] == {"P": 2, "D": 2, "L": 1}


def test_gnss_log_reader_skips_missing_carrier_frequency(tmp_path):
    log_path = tmp_path / "gnss_log.txt"
    log_path.write_text(
        _raw_line(
            utc_ms=1000,
            time_nanos=20_000_000_000,
            svid=1,
            carrier_hz=1_575_420_000.0,
            received_sv_time_nanos=19_930_000_000,
        )
        + _raw_line(
            utc_ms=1000,
            time_nanos=20_000_000_000,
            svid=2,
            carrier_hz="",
            received_sv_time_nanos=19_930_000_000,
        ),
        encoding="utf-8",
    )

    obs = load_gnss_log_observations(log_path)

    missing = obs.loc[obs["Svid"] == 2].iloc[0]
    assert missing["freq"] == ""
    assert np.isnan(missing["freq_hz"])
    counts = gnss_log_observation_counts(log_path)
    assert counts["L1"] == {"P": 1, "D": 1, "L": 1}
    assert counts["L5"] == {"P": 0, "D": 0, "L": 0}


def test_gnss_log_corrected_pseudorange_uses_adjusted_sat_clock(tmp_path):
    trip = tmp_path / "train" / "course" / "phone"
    log_path = trip / "supplemental" / "gnss_log.txt"
    _write_log(log_path)

    raw_frame = pd.DataFrame(
        [
            {
                "utcTimeMillis": 1000,
                "ConstellationType": 1,
                "Svid": 1,
                "SignalType": "GPS_L1_CA",
                "SvClockBiasMeters": 5.0,
                "IonosphericDelayMeters": 2.0,
                "TroposphericDelayMeters": 3.0,
            },
        ],
    )
    sat_clock_bias_m = np.array([[8.0]], dtype=np.float64)
    rtklib_tropo_m = np.array([[4.0]], dtype=np.float64)

    pseudorange, weights, observable = raw_bridge._gnss_log_corrected_pseudorange_matrix(
        trip,
        raw_frame,
        np.array([1000.0], dtype=np.float64),
        ((1, 1, "GPS_L1_CA"),),
        {},
        rtklib_tropo_m=rtklib_tropo_m,
        sat_clock_bias_m=sat_clock_bias_m,
        phone_name="phone",
    )

    raw_pseudorange = load_gnss_log_observations(log_path).query("freq == 'L1'")["PseudorangeMeters"].iloc[0]
    assert weights[0, 0] == 1.0
    assert observable[0, 0] == pytest.approx(raw_pseudorange, abs=1.0e-6)
    assert pseudorange[0, 0] == pytest.approx(raw_pseudorange + 8.0 - 2.0 - 4.0, abs=1.0e-6)


def test_gnss_log_synthetic_rows_include_arrival_time_for_nav_selection(tmp_path):
    trip = tmp_path / "train" / "course" / "phone"
    log_path = trip / "supplemental" / "gnss_log.txt"
    _write_log(log_path)

    columns = [
        "utcTimeMillis",
        "Svid",
        "ConstellationType",
        "SignalType",
        "RawPseudorangeMeters",
        "Cn0DbHz",
        "State",
        "MultipathIndicator",
        "PseudorangeRateMetersPerSecond",
        "PseudorangeRateUncertaintyMetersPerSecond",
        "AccumulatedDeltaRangeState",
        "AccumulatedDeltaRangeMeters",
        "AccumulatedDeltaRangeUncertaintyMeters",
        "CarrierFrequencyHz",
        "ArrivalTimeNanosSinceGpsEpoch",
        "ReceivedSvTimeNanos",
        "ReceivedSvTimeUncertaintyNanos",
        "TimeOffsetNanos",
        "SvPositionXEcefMeters",
        "SvPositionYEcefMeters",
        "SvPositionZEcefMeters",
        "SvVelocityXEcefMetersPerSecond",
        "SvVelocityYEcefMetersPerSecond",
        "SvVelocityZEcefMetersPerSecond",
        "SvClockBiasMeters",
        "SvClockDriftMetersPerSecond",
        "IonosphericDelayMeters",
        "TroposphericDelayMeters",
        "SvElevationDegrees",
        "SvAzimuthDegrees",
    ]
    raw_frame = pd.DataFrame(
        [
            {
                "utcTimeMillis": 1000,
                "Svid": 1,
                "ConstellationType": 1,
                "SignalType": "GPS_L1_CA",
                "SvPositionXEcefMeters": 2.1e7,
                "SvPositionYEcefMeters": 0.0,
                "SvPositionZEcefMeters": 0.0,
                "SvVelocityXEcefMetersPerSecond": 0.0,
                "SvVelocityYEcefMetersPerSecond": 0.0,
                "SvVelocityZEcefMetersPerSecond": 0.0,
                "SvClockBiasMeters": 5.0,
                "SvClockDriftMetersPerSecond": 0.0,
                "IonosphericDelayMeters": 2.0,
                "TroposphericDelayMeters": 3.0,
                "SvElevationDegrees": 45.0,
                "SvAzimuthDegrees": 90.0,
            },
        ],
    )
    appended = raw_bridge._append_gnss_log_only_gps_rows(
        pd.DataFrame(columns=columns),
        raw_frame,
        pd.DataFrame([{"utcTimeMillis": 1000}]),
        trip,
        phone_name="phone",
        dual_frequency=True,
    )

    l1_row = appended[appended["SignalType"] == "GPS_L1_CA"].iloc[0]
    tow_rx_s = load_gnss_log_observations(log_path).query("freq == 'L1'")["tow_rx_s"].iloc[0]
    assert l1_row["ArrivalTimeNanosSinceGpsEpoch"] == pytest.approx(tow_rx_s * 1.0e9)


def test_gnss_log_pseudorange_preserves_large_nanosecond_integer_precision(tmp_path):
    log_path = tmp_path / "gnss_log.txt"
    time_nanos = 1_047_929_361_000_000
    full_bias_nanos = -1_276_032_541_079_953_176
    bias_nanos = 0.0380287170410156
    received_sv_time_nanos = 347_670_368_542_152
    log_path.write_text(
        _raw_line(
            utc_ms=1_593_045_252_440,
            time_nanos=time_nanos,
            full_bias_nanos=full_bias_nanos,
            bias_nanos=bias_nanos,
            svid=10,
            carrier_hz=1_575_420_000.0,
            received_sv_time_nanos=received_sv_time_nanos,
        ),
        encoding="utf-8",
    )

    obs = load_gnss_log_observations(log_path)

    elapsed_nanos = time_nanos - full_bias_nanos
    week = int(np.floor(elapsed_nanos / GPS_WEEK_NANOS))
    tow_rx_s = (elapsed_nanos - week * GPS_WEEK_NANOS) / 1.0e9 - bias_nanos / 1.0e9
    expected = (tow_rx_s - received_sv_time_nanos / 1.0e9) * C_LIGHT
    assert obs.loc[obs["Svid"] == 10, "PseudorangeMeters"].iloc[0] == pytest.approx(expected, abs=1.0e-6)


def test_gnss_log_signal_mask_applies_matlab_exobs_rules(tmp_path):
    log_path = tmp_path / "gnss_log.txt"
    log_path.write_text(
        _raw_line(
            utc_ms=1000,
            time_nanos=20_000_000_000,
            svid=1,
            carrier_hz=1_575_420_000.0,
            received_sv_time_nanos=19_930_000_000,
        )
        + _raw_line(
            utc_ms=2000,
            time_nanos=21_000_000_000,
            svid=2,
            carrier_hz=1_575_420_000.0,
            received_sv_time_nanos=20_930_000_000,
            cn0=19.0,
        )
        + _raw_line(
            utc_ms=3000,
            time_nanos=22_000_000_000,
            svid=3,
            carrier_hz=1_575_420_000.0,
            received_sv_time_nanos=21_930_000_000,
            state=1 << 0,
        )
        + _raw_line(
            utc_ms=4000,
            time_nanos=23_000_000_000,
            svid=4,
            carrier_hz=1_575_420_000.0,
            received_sv_time_nanos=22_930_000_000,
            adr_state=(1 << 0) | (1 << 1),
        ),
        encoding="utf-8",
    )

    counts = gnss_log_observation_counts(log_path, apply_signal_mask=True)

    assert counts["L1"]["P"] == 2
    assert counts["L1"]["D"] == 3
    assert counts["L1"]["L"] == 2


def test_compare_gnss_log_observation_counts_reports_parity(tmp_path):
    trip_dir = tmp_path / "train" / "course" / "phone"
    _write_log(trip_dir / "supplemental" / "gnss_log.txt")
    pd.DataFrame(
        [
            {"freq": "L1", "field": "P", "count": 2},
            {"freq": "L1", "field": "D", "count": 2},
            {"freq": "L1", "field": "L", "count": 2},
            {"freq": "L5", "field": "P", "count": 2},
            {"freq": "L5", "field": "D", "count": 2},
            {"freq": "L5", "field": "L", "count": 1},
        ],
    ).to_csv(trip_dir / "phone_data_observation_counts.csv", index=False)

    comparison, summary = compare_gnss_log_observation_counts(trip_dir)

    assert comparison["count_delta"].eq(0).all()
    assert summary["count_parity_ratio"] == 1.0
    assert summary["matched_abs_delta_total"] == 0


def test_compare_gnss_log_residual_prekeys_reports_navigation_gap(tmp_path):
    trip_dir = tmp_path / "train" / "course" / "phone"
    _write_log(trip_dir / "supplemental" / "gnss_log.txt")
    pd.DataFrame(
        [
            {
                "freq": "L1",
                "epoch_index": 1,
                "utcTimeMillis": 1000,
                "sys": 1,
                "svid": 1,
                "p_residual_m": 0.0,
                "d_residual_mps": 0.0,
                "p_pre_finite": 1,
                "d_pre_finite": 1,
                "p_factor_finite": 1,
                "d_factor_finite": 1,
            }
        ],
    ).to_csv(trip_dir / "phone_data_residual_diagnostics.csv", index=False)

    merged, summary_by_field, summary = compare_gnss_log_residual_prekeys(trip_dir)

    assert summary["total_matched_count"] == 2
    assert summary["total_matlab_only"] == 0
    assert summary["total_gnss_log_only"] == 6
    assert set(merged["side"]) == {"both", "gnss_log_only"}
    assert summary_by_field.set_index(["field", "freq"]).loc[("P", "L1"), "gnss_log_only"] == 1


def test_residual_diagnostics_rebuilds_factor_mask(tmp_path):
    trip_dir = tmp_path / "train" / "course" / "phone"
    trip_dir.mkdir(parents=True)
    diagnostics = pd.DataFrame(
        [
            {
                "freq": "L1",
                "epoch_index": 1,
                "utcTimeMillis": 1000,
                "sys": 1,
                "svid": 3,
                "sat_col": 1,
                "p_residual_m": 0.0,
                "d_residual_mps": 0.0,
                "p_pre_finite": 1,
                "d_pre_finite": 1,
                "l_pre_finite": 1,
                "p_factor_finite": 1,
                "d_factor_finite": 1,
                "l_factor_finite": 1,
            },
            {
                "freq": "L1",
                "epoch_index": 2,
                "utcTimeMillis": 2000,
                "sys": 1,
                "svid": 3,
                "sat_col": 1,
                "p_residual_m": 0.0,
                "d_residual_mps": 0.0,
                "p_pre_finite": 1,
                "d_pre_finite": 1,
                "l_pre_finite": 1,
                "p_factor_finite": 0,
                "d_factor_finite": 1,
                "l_factor_finite": 1,
            },
        ],
    )
    diagnostics.to_csv(trip_dir / "phone_data_residual_diagnostics.csv", index=False)

    expected = build_factor_mask_from_residual_diagnostics(trip_dir / "phone_data_residual_diagnostics.csv")
    expected.to_csv(trip_dir / "phone_data_factor_mask.csv", index=False)

    merged, summary_by_field, summary = compare_residual_diagnostics_factor_mask(trip_dir)

    assert summary["symmetric_parity"] == 1.0
    assert summary["total_factor_mask_only"] == 0
    assert summary["total_diagnostics_only"] == 0
    assert merged["side"].eq("both").all()
    assert summary_by_field.set_index(["field", "freq"]).loc[("L", "L1"), "matched_count"] == 1


def test_compare_residual_values_joins_and_filters_epochs(tmp_path, monkeypatch):
    trip_dir = tmp_path / "train" / "course" / "phone"
    trip_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "freq": "L1",
                "epoch_index": 1,
                "utcTimeMillis": 1000,
                "sys": 1,
                "svid": 3,
                "p_residual_m": 10.0,
                "d_residual_mps": 1.0,
                "p_pre_respc_m": 110.0,
                "d_pre_resd_m": 11.0,
                "p_corrected_m": 210.0,
                "p_range_m": 200.0,
                "d_obs_mps": 31.0,
                "d_model_mps": 20.0,
                "sat_x_m": 100.0,
                "sat_y_m": 200.0,
                "sat_z_m": 300.0,
                "sat_vx_mps": 1.0,
                "sat_vy_mps": 2.0,
                "sat_vz_mps": 3.0,
                "sat_clock_bias_m": 5.0,
                "sat_clock_drift_mps": 0.5,
                "sat_iono_m": 2.0,
                "sat_trop_m": 3.0,
                "sat_elevation_deg": 30.0,
                "rcv_x_m": 0.0,
                "rcv_y_m": 0.0,
                "rcv_z_m": 0.0,
                "rcv_vx_mps": 0.0,
                "rcv_vy_mps": 0.0,
                "rcv_vz_mps": 0.0,
                "obs_clk_m": 90.0,
                "obs_dclk_m": 12.0,
                "p_isb_m": 10.0,
                "p_clock_bias_m": 100.0,
                "d_clock_bias_mps": 10.0,
                "p_pre_finite": 1,
                "d_pre_finite": 1,
            },
            {
                "freq": "L1",
                "epoch_index": 2,
                "utcTimeMillis": 2000,
                "sys": 1,
                "svid": 4,
                "p_residual_m": 99.0,
                "d_residual_mps": 9.0,
                "p_pre_respc_m": 199.0,
                "d_pre_resd_m": 19.0,
                "p_corrected_m": 299.0,
                "p_range_m": 200.0,
                "d_obs_mps": 39.0,
                "d_model_mps": 20.0,
                "sat_x_m": 100.0,
                "sat_y_m": 200.0,
                "sat_z_m": 300.0,
                "sat_vx_mps": 1.0,
                "sat_vy_mps": 2.0,
                "sat_vz_mps": 3.0,
                "sat_clock_bias_m": 5.0,
                "sat_clock_drift_mps": 0.5,
                "sat_iono_m": 2.0,
                "sat_trop_m": 3.0,
                "sat_elevation_deg": 30.0,
                "rcv_x_m": 0.0,
                "rcv_y_m": 0.0,
                "rcv_z_m": 0.0,
                "rcv_vx_mps": 0.0,
                "rcv_vy_mps": 0.0,
                "rcv_vz_mps": 0.0,
                "obs_clk_m": 190.0,
                "obs_dclk_m": 18.0,
                "p_isb_m": 9.0,
                "p_clock_bias_m": 199.0,
                "d_clock_bias_mps": 18.0,
                "p_pre_finite": 1,
                "d_pre_finite": 1,
            },
        ],
    ).to_csv(trip_dir / "phone_data_residual_diagnostics.csv", index=False)

    def fake_bridge_frame(path, *, max_epochs=0, multi_gnss=False):
        assert path == trip_dir
        assert max_epochs == 1
        assert multi_gnss is True
        return pd.DataFrame(
            [
                {
                    "field": "P",
                    "freq": "L1",
                    "epoch_index": 1,
                    "utcTimeMillis": 1000,
                    "sys": 1,
                    "svid": 3,
                    "bridge_residual": 12.0,
                    "bridge_pre_residual": 112.0,
                    "bridge_common_bias": 100.0,
                    "bridge_observation": 212.0,
                    "bridge_model": 100.0,
                    "bridge_sat_x": 101.0,
                    "bridge_sat_y": 200.0,
                    "bridge_sat_z": 300.0,
                    "bridge_sat_vx": 1.0,
                    "bridge_sat_vy": 3.0,
                    "bridge_sat_vz": 3.0,
                    "bridge_sat_clock_bias": 6.0,
                    "bridge_sat_clock_drift": 0.6,
                    "bridge_sat_iono": 2.5,
                    "bridge_sat_trop": 2.0,
                    "bridge_sat_elevation": 31.0,
                    "bridge_rcv_x": 3.0,
                    "bridge_rcv_y": 4.0,
                    "bridge_rcv_z": 0.0,
                    "bridge_rcv_vx": 0.0,
                    "bridge_rcv_vy": 0.0,
                    "bridge_rcv_vz": 2.0,
                },
                {
                    "field": "D",
                    "freq": "L1",
                    "epoch_index": 1,
                    "utcTimeMillis": 1000,
                    "sys": 1,
                    "svid": 3,
                    "bridge_residual": -2.0,
                    "bridge_pre_residual": 8.0,
                    "bridge_common_bias": 10.0,
                    "bridge_observation": 28.0,
                    "bridge_model": 20.0,
                    "bridge_sat_x": 101.0,
                    "bridge_sat_y": 200.0,
                    "bridge_sat_z": 300.0,
                    "bridge_sat_vx": 1.0,
                    "bridge_sat_vy": 3.0,
                    "bridge_sat_vz": 3.0,
                    "bridge_sat_clock_bias": 6.0,
                    "bridge_sat_clock_drift": 0.6,
                    "bridge_sat_iono": 2.5,
                    "bridge_sat_trop": 2.0,
                    "bridge_sat_elevation": 31.0,
                    "bridge_rcv_x": 3.0,
                    "bridge_rcv_y": 4.0,
                    "bridge_rcv_z": 0.0,
                    "bridge_rcv_vx": 0.0,
                    "bridge_rcv_vy": 0.0,
                    "bridge_rcv_vz": 2.0,
                },
            ],
        )

    monkeypatch.setattr(residual_values, "build_bridge_residual_frame", fake_bridge_frame)

    merged, summary_by_field, summary = residual_values.compare_residual_values(
        trip_dir,
        max_epochs=1,
        multi_gnss=True,
    )

    assert set(merged["epoch_index"]) == {1}
    assert summary["total_matlab_count"] == 2
    assert summary["total_bridge_count"] == 2
    assert summary["total_matched_count"] == 2
    assert summary["total_matlab_only"] == 0
    assert summary["total_bridge_only"] == 0
    assert summary["median_abs_delta"] == 2.5
    assert summary["p95_abs_delta"] == pytest.approx(2.95)
    assert summary["median_abs_pre_residual_delta"] == 2.5
    assert summary["median_abs_common_bias_delta"] == 0.0
    assert summary["median_abs_observation_delta"] == 2.5
    assert summary["median_abs_model_delta"] == 50.0
    assert summary["median_abs_sat_position_delta_norm"] == 1.0
    assert summary["median_abs_sat_velocity_delta_norm"] == 1.0
    assert summary["median_abs_sat_clock_bias_delta"] == 1.0
    assert summary["median_abs_sat_iono_delta"] == 0.5
    assert summary["median_abs_sat_trop_delta"] == 1.0
    assert summary["median_abs_rcv_position_delta_norm"] == 5.0
    assert summary["median_abs_rcv_velocity_delta_norm"] == 2.0
    by_field = summary_by_field.set_index(["field", "freq"])
    assert by_field.loc[("P", "L1"), "mean_delta"] == 2.0
    assert by_field.loc[("D", "L1"), "mean_delta"] == -3.0
    assert by_field.loc[("P", "L1"), "median_abs_pre_residual_delta"] == 2.0
    assert by_field.loc[("D", "L1"), "median_abs_pre_residual_delta"] == 3.0
    merged_by_field = merged.set_index(["field", "freq"])
    assert merged_by_field.loc[("P", "L1"), "matlab_pre_residual"] == 110.0
    assert merged_by_field.loc[("P", "L1"), "matlab_common_bias"] == 100.0
    assert merged_by_field.loc[("P", "L1"), "pre_residual_delta"] == 2.0
    assert merged_by_field.loc[("P", "L1"), "common_bias_delta"] == 0.0
    assert merged_by_field.loc[("P", "L1"), "observation_delta"] == 2.0
    assert merged_by_field.loc[("P", "L1"), "model_delta"] == -100.0
    assert merged_by_field.loc[("P", "L1"), "sat_position_delta_norm"] == 1.0
    assert merged_by_field.loc[("P", "L1"), "sat_clock_bias_delta"] == 1.0
    assert merged_by_field.loc[("P", "L1"), "rcv_position_delta_norm"] == 5.0
    assert merged_by_field.loc[("D", "L1"), "matlab_pre_residual"] == 11.0
    assert merged_by_field.loc[("D", "L1"), "matlab_common_bias"] == 10.0
    assert merged_by_field.loc[("D", "L1"), "pre_residual_delta"] == -3.0
    assert merged_by_field.loc[("D", "L1"), "common_bias_delta"] == 0.0
    assert merged_by_field.loc[("D", "L1"), "observation_delta"] == -3.0
    assert merged_by_field.loc[("D", "L1"), "model_delta"] == 0.0


def test_compare_residual_values_respects_settings_epoch_window(tmp_path, monkeypatch):
    trip_dir = tmp_path / "train" / "course" / "phone"
    trip_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "Course": "course",
                "Phone": "phone",
                "IdxStart": 2,
                "IdxEnd": 2,
            },
        ],
    ).to_csv(tmp_path / "settings_train.csv", index=False)
    pd.DataFrame(
        [
            {
                "freq": "L1",
                "epoch_index": epoch_index,
                "utcTimeMillis": epoch_index * 1000,
                "sys": 1,
                "svid": 3,
                "p_residual_m": float(epoch_index),
                "d_residual_mps": float(epoch_index),
                "p_pre_finite": 1,
                "d_pre_finite": 1,
            }
            for epoch_index in (1, 2, 3)
        ],
    ).to_csv(trip_dir / "phone_data_residual_diagnostics.csv", index=False)

    def fake_bridge_frame(path, *, max_epochs=0, multi_gnss=False):
        assert path == trip_dir
        assert max_epochs == 0
        assert multi_gnss is False
        return pd.DataFrame(
            [
                {
                    "field": field,
                    "freq": "L1",
                    "epoch_index": 2,
                    "utcTimeMillis": 2000,
                    "sys": 1,
                    "svid": 3,
                    "bridge_residual": 2.0,
                }
                for field in ("P", "D")
            ],
        )

    monkeypatch.setattr(residual_values, "build_bridge_residual_frame", fake_bridge_frame)

    merged, _summary_by_field, summary = residual_values.compare_residual_values(trip_dir)

    assert set(merged["epoch_index"]) == {2}
    assert summary["total_matlab_count"] == 2
    assert summary["total_bridge_count"] == 2
    assert summary["total_matched_count"] == 2
    assert summary["total_matlab_only"] == 0
    assert summary["total_bridge_only"] == 0


def test_residual_value_bridge_respects_settings_epoch_window(tmp_path, monkeypatch):
    trip_dir = tmp_path / "train" / "course" / "phone"
    trip_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "Course": "course",
                "Phone": "phone",
                "IdxStart": 2,
                "IdxEnd": 2,
            },
        ],
    ).to_csv(tmp_path / "settings_train.csv", index=False)
    sat_ecef = np.array(
        [
            [
                [2.1e7, 0.0, 0.0],
                [0.0, 2.1e7, 0.0],
                [0.0, 0.0, 2.1e7],
                [1.2e7, 1.2e7, 1.2e7],
            ],
        ],
        dtype=np.float64,
    )
    sat_vel = np.array(
        [
            [
                [100.0, 0.0, 0.0],
                [0.0, -80.0, 0.0],
                [0.0, 0.0, 50.0],
                [30.0, 60.0, -90.0],
            ],
        ],
        dtype=np.float64,
    )
    rx = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    los = sat_ecef / np.linalg.norm(sat_ecef, axis=2)[:, :, None]
    geom_rate = np.sum(los * sat_vel, axis=2)
    common = -3.0
    residual = np.array([[0.5, -0.75, 1.25, -1.5]], dtype=np.float64)
    raw_pseudorange_rate = geom_rate + common + residual
    batch = SimpleNamespace(
        times_ms=np.array([2000.0], dtype=np.float64),
        weights=np.zeros((1, 4), dtype=np.float64),
        pseudorange=np.zeros((1, 4), dtype=np.float64),
        sat_ecef=sat_ecef,
        sat_vel=sat_vel,
        doppler=-raw_pseudorange_rate,
        doppler_weights=np.ones((1, 4), dtype=np.float64),
        kaggle_wls=rx,
        slot_keys=tuple((1, svid, "GPS_L1_CA") for svid in range(1, 5)),
        sys_kind=None,
        n_clock=1,
        clock_drift_mps=np.array([-common], dtype=np.float64),
    )

    build_calls = []

    def fake_build_trip_arrays(path, **kwargs):
        assert path == trip_dir
        build_calls.append((kwargs["start_epoch"], kwargs["max_epochs"]))
        if kwargs["start_epoch"] == 1:
            assert kwargs["max_epochs"] == 1
        else:
            assert kwargs["start_epoch"] == 0
            assert kwargs["max_epochs"] == 1_000_000_000
        return batch

    def fake_component_frame(path, times_ms, **kwargs):
        assert path == trip_dir
        assert list(times_ms) == [2000.0]
        assert kwargs["epoch_offset"] == 1
        return pd.DataFrame()

    monkeypatch.setattr(residual_values, "_build_trip_arrays", fake_build_trip_arrays)
    monkeypatch.setattr(residual_values, "_receiver_velocity_from_reference", lambda *args, **kwargs: np.zeros_like(rx))
    monkeypatch.setattr(residual_values, "_bridge_component_frame", fake_component_frame)

    frame = residual_values.build_bridge_residual_frame(trip_dir, max_epochs=0)

    assert set(frame["field"]) == {"D"}
    assert set(frame["epoch_index"]) == {2}
    np.testing.assert_allclose(frame.sort_values("svid")["bridge_residual"], residual[0])
    assert build_calls == [(1, 1), (0, 1_000_000_000)]


def test_residual_value_bridge_doppler_uses_matlab_resd_convention(tmp_path, monkeypatch):
    sat_ecef = np.array(
        [
            [
                [2.1e7, 0.0, 0.0],
                [0.0, 2.1e7, 0.0],
                [0.0, 0.0, 2.1e7],
                [1.2e7, 1.2e7, 1.2e7],
            ],
        ],
        dtype=np.float64,
    )
    sat_vel = np.array(
        [
            [
                [100.0, 0.0, 0.0],
                [0.0, -80.0, 0.0],
                [0.0, 0.0, 50.0],
                [30.0, 60.0, -90.0],
            ],
        ],
        dtype=np.float64,
    )
    rx = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    los = sat_ecef / np.linalg.norm(sat_ecef, axis=2)[:, :, None]
    geom_rate = np.sum(los * sat_vel, axis=2)
    sat_clock_drift = np.array([[0.0, 0.4, -0.2, 0.7]], dtype=np.float64)
    model = geom_rate - sat_clock_drift
    matlab_common = -7.0
    matlab_residual = np.array([[0.25, -0.5, 1.0, -1.25]], dtype=np.float64)
    raw_pseudorange_rate = model + matlab_common + matlab_residual
    batch = SimpleNamespace(
        times_ms=np.array([1000.0], dtype=np.float64),
        weights=np.zeros((1, 4), dtype=np.float64),
        pseudorange=np.zeros((1, 4), dtype=np.float64),
        sat_ecef=sat_ecef,
        sat_vel=sat_vel,
        sat_clock_drift_mps=sat_clock_drift,
        doppler=-raw_pseudorange_rate,
        doppler_weights=np.ones((1, 4), dtype=np.float64),
        kaggle_wls=rx,
        slot_keys=tuple((1, svid, "GPS_L1_CA") for svid in range(1, 5)),
        sys_kind=None,
        n_clock=1,
        clock_drift_mps=np.array([-matlab_common], dtype=np.float64),
    )

    monkeypatch.setattr(residual_values, "_build_trip_arrays", lambda *args, **kwargs: batch)
    monkeypatch.setattr(residual_values, "_receiver_velocity_from_reference", lambda *args, **kwargs: np.zeros_like(rx))
    monkeypatch.setattr(residual_values, "_bridge_component_frame", lambda *args, **kwargs: pd.DataFrame())

    frame = residual_values.build_bridge_residual_frame(tmp_path, max_epochs=1)
    frame = frame.sort_values("svid").reset_index(drop=True)

    np.testing.assert_allclose(frame["bridge_observation"], raw_pseudorange_rate[0])
    np.testing.assert_allclose(frame["bridge_model"], model[0])
    np.testing.assert_allclose(frame["bridge_pre_residual"], matlab_common + matlab_residual[0])
    np.testing.assert_allclose(frame["bridge_common_bias"], np.full(4, matlab_common))
    np.testing.assert_allclose(frame["bridge_residual"], matlab_residual[0])
