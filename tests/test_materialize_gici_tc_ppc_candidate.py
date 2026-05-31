"""Unit tests for the GICI rtk_imu_tc -> PPC candidate materialiser core."""

from __future__ import annotations

import csv
import math

from experiments.materialize_gici_tc_ppc_candidate import (
    _DEFAULT_VARIANT,
    _dmm_to_deg,
    lla_to_ecef,
    materialize_run,
    parse_gga_full,
)


def test_lla_to_ecef_equator_prime_meridian():
    x, y, z = lla_to_ecef(0.0, 0.0, 0.0)
    assert abs(x - 6378137.0) < 1e-3  # WGS84 semi-major on the equator
    assert abs(y) < 1e-6
    assert abs(z) < 1e-6


def test_lla_to_ecef_north_pole_height():
    x, y, z = lla_to_ecef(90.0, 0.0, 100.0)
    b = 6378137.0 * (1.0 - 1.0 / 298.257223563)
    assert abs(x) < 1e-3 and abs(y) < 1e-3
    assert abs(z - (b + 100.0)) < 1e-3


def test_dmm_to_deg():
    assert abs(_dmm_to_deg("3509.9222496", "N") - (35 + 9.9222496 / 60.0)) < 1e-12
    assert abs(_dmm_to_deg("13652.8688154", "E") - (136 + 52.8688154 / 60.0)) < 1e-12
    assert _dmm_to_deg("13652.0", "W") < 0


def test_parse_gga_full_time_leap_and_ellipsoidal_height():
    # 08:54:01.2 GPS-tod, +18 s leap, day_base 518400 -> tow 518400+32041.2+18
    line = "$GPGGA,085401.200,3509.9222496,N,13652.8688154,E,4,30,1.0,3.920,M,37.720,M,22096.0,0000*7A"
    g = parse_gga_full(line, day_base=518400.0, leap_seconds=18.0)
    assert g is not None
    assert abs(g["tow"] - round(518400.0 + 32041.2 + 18.0, 1)) < 1e-6
    assert g["fix"] == 4
    assert g["nsat"] == 30
    # ellipsoidal height = MSL alt (3.920) + geoid separation (37.720)
    assert abs(g["h"] - (3.920 + 37.720)) < 1e-9


def test_parse_gga_full_rejects_short_and_non_gga():
    assert parse_gga_full("$GPRMC,085401.2,A,3509.9,N,13652.8,E*00", 0.0, 18.0) is None
    assert parse_gga_full("$GPGGA,085401.2,3509.9,N,13652.8,E,4*00", 0.0, 18.0) is None


def test_default_variant_covers_all_six_runs():
    assert set(_DEFAULT_VARIANT) == {
        "nagoya/run1", "nagoya/run2", "nagoya/run3",
        "tokyo/run1", "tokyo/run2", "tokyo/run3",
    }


def test_materialize_run_writes_pos_and_diag(tmp_path):
    # Minimal reference (gives day_base) + one GGA line.
    ref = tmp_path / "reference.csv"
    ref.write_text("GPS TOW (s),Latitude (deg),Longitude (deg)\n550380.0,35.0,136.0\n")
    nmea = tmp_path / "test_nagoya1_synthetic.txt"
    nmea.write_text(
        "$GPGGA,085401.200,3509.9222496,N,13652.8688154,E,4,30,1.0,3.920,M,37.720,M,,*00\n"
        "$GPGGA,085401.400,3509.9228110,N,13652.8689827,E,5,28,1.0,4.265,M,37.720,M,,*00\n"
    )
    out_dir = tmp_path / "gici_tc"
    s = materialize_run(
        nmea_path=nmea, reference_csv=ref, out_dir=out_dir,
        city="nagoya", run="run1", leap_seconds=18.0, fix4_only=False,
        synth_ratio=99.0, synth_rms_fix=0.01, synth_rms_float=0.05,
    )
    assert s["epochs_out"] == 2 and s["fix4"] == 1 and s["fix5"] == 1
    pos_lines = [ln for ln in (out_dir / "nagoya_run1_full.pos").read_text().splitlines() if not ln.startswith("%")]
    assert len(pos_lines) == 2
    parts = pos_lines[0].split()
    assert int(parts[8]) == 4  # status column
    assert all(math.isfinite(float(p)) for p in parts[2:8])  # ECEF + lla finite

    diag = list(csv.DictReader((out_dir / "nagoya_run1_full.csv").open()))
    assert len(diag) == 2
    assert diag[0]["output_added"] == "1"
    assert diag[0]["final_status"] == "4"
    assert float(diag[0]["final_ratio"]) == 99.0
    # fix=5 row uses the float residual rms
    assert abs(float(diag[1]["final_residual_rms"]) - 0.05) < 1e-9


def test_fix4_only_drops_float(tmp_path):
    ref = tmp_path / "reference.csv"
    ref.write_text("GPS TOW (s),Latitude (deg),Longitude (deg)\n550380.0,35.0,136.0\n")
    nmea = tmp_path / "test_tokyo1_synthetic.txt"
    nmea.write_text(
        "$GPGGA,085401.200,3509.9222496,N,13652.8688154,E,4,30,1.0,3.9,M,37.7,M,,*00\n"
        "$GPGGA,085401.400,3509.9228110,N,13652.8689827,E,5,28,1.0,4.2,M,37.7,M,,*00\n"
    )
    out_dir = tmp_path / "gici_tc"
    s = materialize_run(
        nmea_path=nmea, reference_csv=ref, out_dir=out_dir,
        city="tokyo", run="run1", leap_seconds=18.0, fix4_only=True,
        synth_ratio=99.0, synth_rms_fix=0.01, synth_rms_float=0.05,
    )
    assert s["epochs_out"] == 1 and s["fix4"] == 1 and s["fix5"] == 0
