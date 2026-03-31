"""Tests for NMEA writer."""

from __future__ import annotations

import math
import tempfile
from datetime import datetime, date
from pathlib import Path

import numpy as np
import pytest

from gnss_gpu.io.nmea_writer import NMEAWriter, positions_to_nmea, ecef_to_nmea
from gnss_gpu.io.nmea import parse_nmea, _verify_checksum


# ---------------------------------------------------------------------------
# Checksum
# ---------------------------------------------------------------------------

class TestChecksum:
    def test_known_value(self):
        # Body between $ and * for a simple sentence
        body = "GPGGA,123456.00,3546.1234,N,13945.6789,E,1,08,1.0,100.000,M,0.000,M,,"
        chk = NMEAWriter._checksum(body)
        # Verify it's a 2-char hex string
        assert len(chk) == 2
        assert int(chk, 16) >= 0

    def test_checksum_consistency(self):
        writer = NMEAWriter()
        sentence = writer.gga(35.0, 139.0, 100.0,
                              time_utc=datetime(2024, 1, 15, 12, 0, 0))
        assert _verify_checksum(sentence)

    def test_checksum_all_sentence_types(self):
        writer = NMEAWriter()
        t = datetime(2024, 6, 1, 10, 30, 0)
        sentences = [
            writer.gga(35.0, 139.0, 50.0, time_utc=t),
            writer.rmc(35.0, 139.0, time_utc=t),
            writer.gsa([1, 3, 7, 14], pdop=2.0, hdop=1.2, vdop=1.5),
            writer.vtg(course_deg=45.0, speed_knots=5.0, speed_kmh=9.26),
        ]
        for s in sentences:
            assert _verify_checksum(s), f"Checksum failed for: {s}"


# ---------------------------------------------------------------------------
# deg_to_nmea conversion
# ---------------------------------------------------------------------------

class TestDegToNmea:
    def test_positive_latitude(self):
        val, d = NMEAWriter._deg_to_nmea(35.769167, is_lat=True)
        assert d == "N"
        # 35 degrees 46.1500 minutes
        deg_part = int(val[:2])
        min_part = float(val[2:])
        assert deg_part == 35
        assert abs(min_part - 46.1500) < 0.001

    def test_negative_latitude(self):
        val, d = NMEAWriter._deg_to_nmea(-33.8688, is_lat=True)
        assert d == "S"
        deg_part = int(val[:2])
        min_part = float(val[2:])
        assert deg_part == 33
        assert abs(min_part - 52.128) < 0.01

    def test_positive_longitude(self):
        val, d = NMEAWriter._deg_to_nmea(139.6917, is_lat=False)
        assert d == "E"
        deg_part = int(val[:3])
        min_part = float(val[3:])
        assert deg_part == 139
        assert abs(min_part - 41.502) < 0.01

    def test_negative_longitude(self):
        val, d = NMEAWriter._deg_to_nmea(-122.4194, is_lat=False)
        assert d == "W"
        deg_part = int(val[:3])
        min_part = float(val[3:])
        assert deg_part == 122
        assert abs(min_part - 25.164) < 0.01

    def test_zero_latitude(self):
        val, d = NMEAWriter._deg_to_nmea(0.0, is_lat=True)
        assert d == "N"
        assert float(val) == pytest.approx(0.0)

    def test_zero_longitude(self):
        val, d = NMEAWriter._deg_to_nmea(0.0, is_lat=False)
        assert d == "E"
        assert float(val) == pytest.approx(0.0)

    def test_lat_format_width(self):
        # Latitude should be DDMM.MMMM (2 digit degrees)
        val, _ = NMEAWriter._deg_to_nmea(5.5, is_lat=True)
        assert val[0:2] == "05"

    def test_lon_format_width(self):
        # Longitude should be DDDMM.MMMM (3 digit degrees)
        val, _ = NMEAWriter._deg_to_nmea(5.5, is_lat=False)
        assert val[0:3] == "005"


# ---------------------------------------------------------------------------
# GGA
# ---------------------------------------------------------------------------

class TestGGA:
    def test_format(self):
        writer = NMEAWriter()
        t = datetime(2024, 3, 15, 12, 34, 56)
        s = writer.gga(35.6812, 139.7671, 40.0, time_utc=t,
                       fix_quality=1, n_sats=8, hdop=1.2)
        assert s.startswith("$GPGGA,")
        assert "*" in s
        fields = s.split("*")[0].lstrip("$").split(",")
        assert fields[0] == "GPGGA"
        assert fields[1].startswith("123456")
        assert fields[3] in ("N", "S")
        assert fields[5] in ("E", "W")
        assert fields[6] == "1"      # fix quality
        assert fields[7] == "08"     # n_sats
        assert fields[10] == "M"

    def test_checksum_valid(self):
        writer = NMEAWriter()
        s = writer.gga(35.0, 139.0, 100.0, time_utc=datetime(2024, 1, 1, 0, 0, 0))
        assert _verify_checksum(s)

    def test_southern_hemisphere(self):
        writer = NMEAWriter()
        s = writer.gga(-33.8688, 151.2093, 10.0, time_utc=datetime(2024, 1, 1))
        fields = s.split("*")[0].lstrip("$").split(",")
        assert fields[3] == "S"
        assert fields[5] == "E"

    def test_custom_talker(self):
        writer = NMEAWriter(talker_id="GN")
        s = writer.gga(35.0, 139.0, 0.0, time_utc=datetime(2024, 1, 1))
        assert s.startswith("$GNGGA,")


# ---------------------------------------------------------------------------
# RMC
# ---------------------------------------------------------------------------

class TestRMC:
    def test_format(self):
        writer = NMEAWriter()
        t = datetime(2024, 3, 15, 12, 34, 56)
        s = writer.rmc(35.6812, 139.7671, time_utc=t,
                       speed_knots=5.2, course_deg=123.4)
        assert s.startswith("$GPRMC,")
        fields = s.split("*")[0].lstrip("$").split(",")
        assert fields[0] == "GPRMC"
        assert fields[2] == "A"  # status
        assert fields[9] == "150324"  # DDMMYY

    def test_date_field(self):
        writer = NMEAWriter()
        t = datetime(2025, 12, 25, 0, 0, 0)
        s = writer.rmc(0.0, 0.0, time_utc=t)
        fields = s.split("*")[0].lstrip("$").split(",")
        assert fields[9] == "251225"

    def test_void_status(self):
        writer = NMEAWriter()
        s = writer.rmc(0.0, 0.0, time_utc=datetime(2024, 1, 1), status="V")
        fields = s.split("*")[0].lstrip("$").split(",")
        assert fields[2] == "V"

    def test_checksum_valid(self):
        writer = NMEAWriter()
        s = writer.rmc(35.0, 139.0, time_utc=datetime(2024, 1, 1))
        assert _verify_checksum(s)


# ---------------------------------------------------------------------------
# GSA
# ---------------------------------------------------------------------------

class TestGSA:
    def test_format(self):
        writer = NMEAWriter()
        s = writer.gsa([1, 3, 7, 14], pdop=2.5, hdop=1.3, vdop=2.1)
        assert s.startswith("$GPGSA,")
        fields = s.split("*")[0].lstrip("$").split(",")
        assert fields[1] == "A"   # mode
        assert fields[2] == "3"   # fix type
        # PRNs in fields 3..14
        assert fields[3] == "01"
        assert fields[4] == "03"
        assert fields[5] == "07"
        assert fields[6] == "14"
        # Empty slots
        for i in range(7, 15):
            assert fields[i] == ""
        # DOP values
        assert fields[15] == "2.5"
        assert fields[16] == "1.3"
        assert fields[17] == "2.1"

    def test_checksum_valid(self):
        writer = NMEAWriter()
        s = writer.gsa([1], pdop=1.0, hdop=0.8, vdop=0.6)
        assert _verify_checksum(s)

    def test_max_12_prns(self):
        writer = NMEAWriter()
        prns = list(range(1, 16))  # 15 PRNs, should be truncated to 12
        s = writer.gsa(prns)
        fields = s.split("*")[0].lstrip("$").split(",")
        non_empty = [f for f in fields[3:15] if f]
        assert len(non_empty) == 12


# ---------------------------------------------------------------------------
# GSV
# ---------------------------------------------------------------------------

class TestGSV:
    def test_single_message(self):
        writer = NMEAWriter()
        sats = [(1, 45, 120, 35), (3, 30, 200, 28)]
        sentences = writer.gsv(sats)
        assert len(sentences) == 1
        assert sentences[0].startswith("$GPGSV,")
        assert _verify_checksum(sentences[0])

    def test_multiple_messages(self):
        writer = NMEAWriter()
        sats = [(i, 45, 120, 35) for i in range(1, 10)]  # 9 sats -> 3 msgs
        sentences = writer.gsv(sats)
        assert len(sentences) == 3
        for s in sentences:
            assert _verify_checksum(s)


# ---------------------------------------------------------------------------
# VTG
# ---------------------------------------------------------------------------

class TestVTG:
    def test_format(self):
        writer = NMEAWriter()
        s = writer.vtg(course_deg=90.5, speed_knots=10.0, speed_kmh=18.52)
        assert s.startswith("$GPVTG,")
        fields = s.split("*")[0].lstrip("$").split(",")
        # Standard VTG: GPVTG,cogt,T,cogm,M,sogn,N,sogk,K
        assert fields[2] == "T"
        assert fields[4] == "M"
        assert fields[6] == "N"
        assert fields[8] == "K"

    def test_checksum_valid(self):
        writer = NMEAWriter()
        s = writer.vtg()
        assert _verify_checksum(s)


# ---------------------------------------------------------------------------
# write_epoch
# ---------------------------------------------------------------------------

class TestWriteEpoch:
    def test_returns_sentences(self):
        writer = NMEAWriter()
        t = datetime(2024, 6, 15, 8, 30, 0)
        sentences = writer.write_epoch(
            35.6812, 139.7671, 40.0, time_utc=t,
            n_sats=8, hdop=1.0, prn_list=[1, 3, 7],
        )
        # GGA + RMC + GSA + VTG = 4
        assert len(sentences) == 4
        assert "GGA" in sentences[0]
        assert "RMC" in sentences[1]
        assert "GSA" in sentences[2]
        assert "VTG" in sentences[3]

    def test_no_prn_list(self):
        writer = NMEAWriter()
        sentences = writer.write_epoch(35.0, 139.0, 50.0,
                                       time_utc=datetime(2024, 1, 1))
        # GGA + RMC + VTG = 3  (no GSA without prn_list)
        assert len(sentences) == 3


# ---------------------------------------------------------------------------
# positions_to_nmea
# ---------------------------------------------------------------------------

class TestPositionsToNmea:
    def test_returns_string(self):
        lat = np.array([35.0, 35.001])
        lon = np.array([139.0, 139.001])
        alt = np.array([50.0, 51.0])
        result = positions_to_nmea(lat, lon, alt)
        assert isinstance(result, str)
        assert "$GPGGA" in result
        assert "$GPRMC" in result

    def test_file_output(self):
        lat = np.array([35.0])
        lon = np.array([139.0])
        alt = np.array([50.0])

        with tempfile.NamedTemporaryFile(suffix=".nmea", delete=False) as f:
            path = Path(f.name)

        try:
            result = positions_to_nmea(lat, lon, alt, filepath=path)
            assert result is None
            content = path.read_text()
            assert "$GPGGA" in content
        finally:
            path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# ecef_to_nmea roundtrip
# ---------------------------------------------------------------------------

class TestEcefToNmea:
    def test_roundtrip(self):
        """ECEF -> NMEA -> parse -> verify coordinates match."""
        # Tokyo Station approximate ECEF
        # lat=35.6812, lon=139.7671, alt~40m
        lat_expected = 35.6812
        lon_expected = 139.7671
        alt_expected = 40.0

        # Convert LLA to ECEF (WGS84)
        a = 6378137.0
        f = 1.0 / 298.257223563
        e2 = 2.0 * f - f * f
        lat_r = math.radians(lat_expected)
        lon_r = math.radians(lon_expected)
        sin_lat = math.sin(lat_r)
        cos_lat = math.cos(lat_r)
        N = a / math.sqrt(1.0 - e2 * sin_lat * sin_lat)
        x = (N + alt_expected) * cos_lat * math.cos(lon_r)
        y = (N + alt_expected) * cos_lat * math.sin(lon_r)
        z = (N * (1.0 - e2) + alt_expected) * sin_lat

        ecef = np.array([[x, y, z]])
        t = [datetime(2024, 6, 15, 12, 0, 0)]

        # Write to NMEA file
        with tempfile.NamedTemporaryFile(suffix=".nmea", delete=False) as f:
            path = Path(f.name)

        try:
            ecef_to_nmea(ecef, times=t, filepath=path)
            messages = parse_nmea(path)
        finally:
            path.unlink(missing_ok=True)

        # Find GGA message
        gga_msgs = [m for m in messages if hasattr(m, "altitude")]
        assert len(gga_msgs) == 1
        gga = gga_msgs[0]

        assert abs(gga.latitude - lat_expected) < 0.001
        assert abs(gga.longitude - lon_expected) < 0.001
        assert abs(gga.altitude - alt_expected) < 1.0

    def test_multiple_positions(self):
        """Verify multiple ECEF positions produce correct number of sentences."""
        a = 6378137.0
        f = 1.0 / 298.257223563
        e2 = 2.0 * f - f * f

        positions = []
        for lat_d, lon_d in [(35.0, 139.0), (36.0, 140.0)]:
            lat_r = math.radians(lat_d)
            lon_r = math.radians(lon_d)
            sin_lat = math.sin(lat_r)
            cos_lat = math.cos(lat_r)
            N = a / math.sqrt(1.0 - e2 * sin_lat * sin_lat)
            alt = 0.0
            x = (N + alt) * cos_lat * math.cos(lon_r)
            y = (N + alt) * cos_lat * math.sin(lon_r)
            z = (N * (1.0 - e2) + alt) * sin_lat
            positions.append([x, y, z])

        ecef = np.array(positions)
        result = ecef_to_nmea(ecef)
        assert isinstance(result, str)
        # Each epoch produces GGA + RMC + VTG = 3 lines, 2 epochs = 6
        assert result.count("$GPGGA") == 2
        assert result.count("$GPRMC") == 2

    def test_returns_string_without_filepath(self):
        a = 6378137.0
        f = 1.0 / 298.257223563
        e2 = 2.0 * f - f * f
        lat_r = math.radians(35.0)
        lon_r = math.radians(139.0)
        sin_lat = math.sin(lat_r)
        cos_lat = math.cos(lat_r)
        N = a / math.sqrt(1.0 - e2 * sin_lat * sin_lat)
        x = (N) * cos_lat * math.cos(lon_r)
        y = (N) * cos_lat * math.sin(lon_r)
        z = (N * (1.0 - e2)) * sin_lat

        ecef = np.array([[x, y, z]])
        result = ecef_to_nmea(ecef)
        assert isinstance(result, str)
        assert "$GPGGA" in result
