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


def test_rinex2_observation_epochs(tmp_path):
    path = tmp_path / "base.23o"
    lines = [
        "     2.11           OBSERVATION DATA    G                   RINEX VERSION / TYPE\n",
        "NOAA CORS                                                   MARKER NAME         \n",
        " -2695156.4450 -4299130.6400  3851527.4380                  APPROX POSITION XYZ \n",
        "     3    C1    L1    S1                                      # / TYPES OF OBSERV\n",
        "    30.000                                                  INTERVAL            \n",
        "                                                            END OF HEADER       \n",
        " 23  5 23  0  0  0.0000000  0  3G01G02G11\n",
        "  20200001.000      110.000       45.000  \n",
        "  20200002.000      120.000       46.000  \n",
        "  20200011.000      130.000       47.000  \n",
    ]
    path.write_text("".join(lines))

    obs = read_rinex_obs(path)

    assert obs.header.version == 2.11
    assert obs.header.approx_position.tolist() == [-2695156.445, -4299130.64, 3851527.438]
    assert obs.header.interval == 30.0
    assert obs.header.obs_types["G"] == ["C1", "L1", "S1"]
    assert len(obs.epochs) == 1
    assert obs.epochs[0].time.year == 2023
    assert obs.epochs[0].satellites == ["G01", "G02", "G11"]
    assert obs.epochs[0].observations["G02"]["C1"] == 20200002.0
