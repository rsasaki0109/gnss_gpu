"""Unit tests for the GICI rtk_imu_tc PPC2024 batch evaluator.

Covers the deterministic parsing / matching / scoring core (no GICI workspace
or PPC2024 dataset required).
"""

from __future__ import annotations

import math

from experiments.eval_gici_tc_ppc2024_batch import (
    dmm_to_deg,
    evaluate_file,
    horizontal_error_m,
    interp_ref,
    parse_gga,
    run_key_from_filename,
)


def test_run_key_from_filename():
    assert run_key_from_filename("test_nagoya1.txt") == "nagoya/run1"
    assert run_key_from_filename("test_nagoya2_combo4.txt") == "nagoya/run2"
    assert run_key_from_filename("test_tokyo3_loosepr.txt") == "tokyo/run3"
    assert run_key_from_filename("readme.txt") is None
    assert run_key_from_filename("test_osaka1.txt") is None


def test_dmm_to_deg():
    # 3512.3456 N -> 35 deg + 12.3456 min
    assert dmm_to_deg("3512.3456", "N") == 35 + 12.3456 / 60.0
    # West / South negate
    assert dmm_to_deg("13945.6789", "W") == -(139 + 45.6789 / 60.0)
    assert math.isnan(dmm_to_deg("", "N"))


def test_parse_gga_time_is_gps_tod_plus_day_base():
    day_base = 518400.0  # GPS week day 6 base
    line = "$GPGGA,085300.00,3500.0000,N,13500.0000,E,4,12,0.8,40.0,M,,,,*00"
    g = parse_gga(line, day_base)
    assert g is not None
    # 08:53:00 -> 8*3600+53*60 = 31980 s of day
    assert g["tow"] == day_base + 31980.0
    assert g["fix"] == 4
    assert abs(g["lat"] - 35.0) < 1e-9
    assert abs(g["lon"] - 135.0) < 1e-9


def test_parse_gga_rejects_non_gga_and_short():
    assert parse_gga("$GPRMC,085300,A,3500.0,N,13500.0,E,*00", 0.0) is None
    assert parse_gga("$GPGGA,085300.00,,,,,4*00", 0.0) is None


def test_interp_ref_linear_and_out_of_range():
    ref = [
        {"tow": 100.0, "lat": 0.0, "lon": 0.0},
        {"tow": 200.0, "lat": 1.0, "lon": 2.0},
    ]
    mid = interp_ref(ref, 150.0)
    assert mid is not None
    assert abs(mid["lat"] - 0.5) < 1e-12
    assert abs(mid["lon"] - 1.0) < 1e-12
    assert interp_ref(ref, 99.0) is None
    assert interp_ref(ref, 201.0) is None


def test_horizontal_error_zero_and_east_offset():
    assert horizontal_error_m(35.0, 135.0, 35.0, 135.0) < 1e-6
    # ~1 arcsec of longitude at 35N is a known small distance > 0
    d = horizontal_error_m(35.0, 135.001, 35.0, 135.0)
    assert 80.0 < d < 95.0  # ~91 m per 0.001 deg lon at 35N


def test_evaluate_file_with_leap_offset(tmp_path):
    # Reference: vehicle moving east; a +18 s GICI lead must be corrected to
    # land on the matching reference position (error ~0), whereas leap=0 lands
    # 18 s early (large error).
    ref = [{"tow": 1000.0 + t, "lat": 35.0, "lon": 135.0 + 1e-4 * t} for t in range(0, 60)]
    day_base = math.floor(ref[0]["tow"] / 86400.0) * 86400.0
    tod = ref[0]["tow"] - day_base  # GICI emits GPS-time-of-day
    # GICI reports the position it actually had at tow=1018, but timestamps it
    # 18 s early (its clock leads the reference TOW by the leap seconds).
    lon = 135.0 + 1e-4 * 18  # position at +18 s
    hms = f"{int(tod // 3600):02d}{int((tod % 3600) // 60):02d}{tod % 60:05.2f}"
    nmea = tmp_path / "test_nagoya1_synthetic.txt"
    nmea.write_text(
        f"$GPGGA,{hms},3500.0000,N,{135 * 100 + (lon - 135) * 60 * 100:.4f},E,4,10,0.7,40,M,,,,*00\n"
    )

    res0 = evaluate_file(nmea, ref, leap_seconds=0.0)
    res18 = evaluate_file(nmea, ref, leap_seconds=18.0)
    assert res18["n_eval"] == 1
    # +18 s correction brings the timestamp to tow=1018 where the reference
    # position matches the reported one: near-zero error.
    assert res18["p50_m"] < res0["p50_m"]
    assert res18["coverage_pct"] > 0.0
