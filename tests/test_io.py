"""Tests for RINEX and NMEA parsers."""

import tempfile
from pathlib import Path

from gnss_gpu.io.nmea import parse_nmea, GGAMessage, RMCMessage
from gnss_gpu.io.rinex import read_rinex_obs


def test_nmea_gga():
    content = "$GPGGA,092750.000,3544.7041,N,13939.6735,E,1,08,0.9,545.4,M,46.9,M,,*5D\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".nmea", delete=False) as f:
        f.write(content)
        f.flush()
        msgs = parse_nmea(f.name)

    assert len(msgs) == 1
    msg = msgs[0]
    assert isinstance(msg, GGAMessage)
    assert abs(msg.latitude - 35.745068333) < 0.001
    assert abs(msg.longitude - 139.661225) < 0.001
    assert msg.fix_quality == 1
    assert msg.n_satellites == 8
    assert abs(msg.altitude - 545.4) < 0.01


def test_nmea_rmc():
    content = "$GPRMC,092750.000,A,3544.7041,N,13939.6735,E,0.02,31.66,310326,,,A*50\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".nmea", delete=False) as f:
        f.write(content)
        f.flush()
        msgs = parse_nmea(f.name)

    assert len(msgs) == 1
    msg = msgs[0]
    assert isinstance(msg, RMCMessage)
    assert msg.status == "A"
    assert msg.time.year == 2026
    assert msg.time.month == 3
    assert msg.time.day == 31


def test_nmea_checksum_fail():
    content = "$GPGGA,092750.000,3544.7041,N,13939.6735,E,1,08,0.9,545.4,M,46.9,M,,*FF\n"  # wrong checksum
    with tempfile.NamedTemporaryFile(mode="w", suffix=".nmea", delete=False) as f:
        f.write(content)
        f.flush()
        msgs = parse_nmea(f.name)

    assert len(msgs) == 0


def test_rinex_header():
    header_lines = [
        "     3.04           OBSERVATION DATA    G                   RINEX VERSION / TYPE\n",
        "test                                                        MARKER NAME         \n",
        " -3957196.1328  3310204.8922  3737910.8017                  APPROX POSITION XYZ \n",
        "G    4 C1C L1C D1C S1C                                      SYS / # / OBS TYPES \n",
        "     1.000                                                  INTERVAL            \n",
        "                                                            END OF HEADER       \n",
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".obs", delete=False) as f:
        f.writelines(header_lines)
        f.flush()
        obs = read_rinex_obs(f.name)

    assert obs.header.version == 3.04
    assert obs.header.marker_name == "test"
    assert "G" in obs.header.obs_types
    assert obs.header.obs_types["G"] == ["C1C", "L1C", "D1C", "S1C"]
