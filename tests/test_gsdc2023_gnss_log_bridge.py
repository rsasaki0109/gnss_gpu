from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.gsdc2023_gnss_log_bridge import (
    append_gnss_log_only_gps_rows,
    gnss_log_corrected_pseudorange_products,
    gnss_log_matlab_epoch_times_ms,
)
from experiments.gsdc2023_gnss_log_reader import load_gnss_log_observations


def _raw_line(
    *,
    utc_ms: int,
    time_nanos: int,
    svid: int,
    carrier_hz: float,
    received_sv_time_nanos: int,
    adr_m: float = 1000.0,
) -> str:
    values = [
        utc_ms,
        time_nanos,
        18,
        "",
        0,
        0.0,
        1.0,
        0.0,
        1.0,
        0,
        svid,
        0.0,
        (1 << 0) | (1 << 3),
        received_sv_time_nanos,
        10,
        35.0,
        -100.0,
        0.1,
        1,
        adr_m,
        0.1,
        carrier_hz,
        "",
        "",
        "",
        0,
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
        ),
        encoding="utf-8",
    )


def test_corrected_pseudorange_products_injects_sat_clock_adjustment(tmp_path):
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
                "TroposphericDelayMeters": 99.0,
            },
        ],
    )
    rtklib_tropo_m = np.array([[4.0]], dtype=np.float64)

    products = gnss_log_corrected_pseudorange_products(
        trip,
        raw_frame,
        np.array([1000.0], dtype=np.float64),
        ((1, 1, "GPS_L1_CA"),),
        {},
        phone_name="phone",
        rtklib_tropo_m=rtklib_tropo_m,
        sat_clock_adjustment_m=lambda *_args: 3.0,
    )

    raw_pseudorange = load_gnss_log_observations(log_path).query("freq == 'L1'")["PseudorangeMeters"].iloc[0]
    assert products is not None
    assert products.weights[0, 0] == 1.0
    assert products.observable_pseudorange[0, 0] == pytest.approx(raw_pseudorange, abs=1.0e-6)
    assert products.pseudorange[0, 0] == pytest.approx(raw_pseudorange + 8.0 - 2.0 - 4.0, abs=1.0e-6)


def test_corrected_pseudorange_products_overlays_gps_slots_in_mixed_constellation_batch(tmp_path):
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
            {
                "utcTimeMillis": 1000,
                "ConstellationType": 6,
                "Svid": 7,
                "SignalType": "GAL_E1_C_P",
                "SvClockBiasMeters": 11.0,
                "IonosphericDelayMeters": 13.0,
                "TroposphericDelayMeters": 17.0,
            },
        ],
    )

    products = gnss_log_corrected_pseudorange_products(
        trip,
        raw_frame,
        np.array([1000.0], dtype=np.float64),
        ((1, 1, "GPS_L1_CA"), (6, 7, "GAL_E1_C_P")),
        {},
        phone_name="phone",
        sat_clock_adjustment_m=lambda *_args: 3.0,
    )

    raw_pseudorange = load_gnss_log_observations(log_path).query("freq == 'L1'")["PseudorangeMeters"].iloc[0]
    assert products is not None
    assert products.weights.tolist() == [[1.0, 0.0]]
    assert products.observable_pseudorange[0, 0] == pytest.approx(raw_pseudorange, abs=1.0e-6)
    assert products.pseudorange[0, 0] == pytest.approx(raw_pseudorange + 8.0 - 2.0 - 3.0, abs=1.0e-6)
    assert products.pseudorange[0, 1] == 0.0


def test_append_gnss_log_rows_sets_arrival_time_and_preserves_epoch_fields(tmp_path):
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
        "SvClockBiasMeters",
        "IonosphericDelayMeters",
        "TroposphericDelayMeters",
        "SvElevationDegrees",
        "WlsPositionXEcefMeters",
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
                "SvClockBiasMeters": 5.0,
                "IonosphericDelayMeters": 2.0,
                "TroposphericDelayMeters": 3.0,
                "SvElevationDegrees": 45.0,
            },
        ],
    )

    appended = append_gnss_log_only_gps_rows(
        pd.DataFrame(columns=columns),
        raw_frame,
        pd.DataFrame([{"utcTimeMillis": 1000, "WlsPositionXEcefMeters": 12.0}]),
        trip,
        phone_name="phone",
        dual_frequency=True,
    )

    l1_row = appended[appended["SignalType"] == "GPS_L1_CA"].iloc[0]
    tow_rx_s = load_gnss_log_observations(log_path).query("freq == 'L1'")["tow_rx_s"].iloc[0]
    assert l1_row["ArrivalTimeNanosSinceGpsEpoch"] == pytest.approx(tow_rx_s * 1.0e9)
    assert l1_row["WlsPositionXEcefMeters"] == 12.0


def test_gnss_log_epoch_times_are_unique_sorted_trip_times(tmp_path):
    trip = tmp_path / "train" / "course" / "phone"
    _write_log(trip / "supplemental" / "gnss_log.txt")

    assert gnss_log_matlab_epoch_times_ms(trip) == (1000,)
