import math

import pytest

from gnss_gpu.gnss_time_scales import (
    GPS_WEEK_SEC,
    GPST_MINUS_BDT_S,
    GPST_MINUS_UTC_S,
    broadcast_rx_seconds_of_week,
    unwrap_week_seconds,
)


# --- unwrap_week_seconds --------------------------------------------------


def test_unwrap_keeps_in_range_values_unchanged():
    assert unwrap_week_seconds(123456.0) == 123456.0


def test_unwrap_folds_negative_back_into_week():
    assert unwrap_week_seconds(-1.0) == pytest.approx(GPS_WEEK_SEC - 1.0)


def test_unwrap_folds_above_week_back_into_week():
    assert unwrap_week_seconds(GPS_WEEK_SEC + 5.0) == pytest.approx(5.0)


def test_unwrap_preserves_nan_so_caller_can_short_circuit():
    assert math.isnan(unwrap_week_seconds(float("nan")))


def test_unwrap_preserves_infinity():
    assert math.isinf(unwrap_week_seconds(float("inf")))


# --- broadcast_rx_seconds_of_week ----------------------------------------


def test_broadcast_rx_gps_passes_through():
    assert broadcast_rx_seconds_of_week("G", 100.0) == 100.0
    assert broadcast_rx_seconds_of_week("E", 200.0) == 200.0
    assert broadcast_rx_seconds_of_week("J", 300.0) == 300.0


def test_broadcast_rx_beidou_subtracts_gpst_bdt_offset():
    assert broadcast_rx_seconds_of_week("C", 100.0) == pytest.approx(100.0 - GPST_MINUS_BDT_S)


def test_broadcast_rx_glonass_subtracts_gpst_utc_offset():
    assert broadcast_rx_seconds_of_week("R", 1000.0) == pytest.approx(1000.0 - GPST_MINUS_UTC_S)


def test_broadcast_rx_beidou_wraps_across_week_boundary():
    # 5 s into the week — BDT alignment must wrap to the prior week tail.
    out = broadcast_rx_seconds_of_week("C", 5.0)
    assert out == pytest.approx(GPS_WEEK_SEC + 5.0 - GPST_MINUS_BDT_S)
    assert 0.0 <= out < GPS_WEEK_SEC


def test_broadcast_rx_offsets_are_round_trip_consistent_for_known_records():
    # If a record's broadcast sow is X, then for a receiver at X + offset we
    # must recover X exactly — this guards against accidental sign flips in
    # ``broadcast_rx_seconds_of_week``.
    record_bdt = 12345.0
    recv_gpst = record_bdt + GPST_MINUS_BDT_S
    assert broadcast_rx_seconds_of_week("C", recv_gpst) == pytest.approx(record_bdt)

    record_utc = 67890.0
    recv_gpst = record_utc + GPST_MINUS_UTC_S
    assert broadcast_rx_seconds_of_week("R", recv_gpst) == pytest.approx(record_utc)


def test_offsets_remain_at_documented_values():
    # Bumping these constants requires re-validating against the canonical
    # leap-second / BDT-alignment tables — keep the assertion as a tripwire.
    assert GPST_MINUS_BDT_S == 14.0
    assert GPST_MINUS_UTC_S == 18.0
