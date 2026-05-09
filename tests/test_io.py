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


def test_rinex2_blank_trailing_observation_field_keeps_width(tmp_path):
    def _header(body, label):
        return f"{body:<60}{label}\n"

    def _field(value):
        if value is None:
            return " " * 16
        return f"{value:14.3f}  "

    path = tmp_path / "base.20o"
    path.write_text(
        "".join(
            [
                _header("     2.11           OBSERVATION DATA    M (MIXED)", "RINEX VERSION / TYPE"),
                _header("     6    L1    L2    C1    P2    P1    C5", "# / TYPES OF OBSERV"),
                _header("    30.000", "INTERVAL"),
                _header("", "END OF HEADER"),
                " 20  8  4  0  0  0.0000000  0  1G08\n",
                "".join(_field(value) for value in [117.0, 91.0, 22_372_270.0, 22_372_280.0, None]) + "\n",
                _field(22_372_279.0) + "\n",
            ],
        ),
        encoding="utf-8",
    )

    obs = read_rinex_obs(path)
    sat_obs = obs.epochs[0].observations["G08"]

    assert sat_obs["C1"] == 22_372_270.0
    assert sat_obs["P1"] == 0.0
    assert sat_obs["C5"] == 22_372_279.0


def test_rinex3_long_observation_rows_do_not_consume_next_satellite(tmp_path):
    def _header(body, label):
        return f"{body:<60}{label}\n"

    def _field(value):
        return f"{value:14.3f}  "

    obs_types = [
        "C1C", "L1C", "D1C", "S1C",
        "C2W", "L2W", "D2W", "S2W",
        "C5Q", "L5Q", "D5Q", "S5Q",
    ]
    rinex_path = tmp_path / "long_rows.obs"
    rinex_path.write_text(
        "".join(
            [
                _header("     3.04           OBSERVATION DATA    G", "RINEX VERSION / TYPE"),
                _header(f"G   {len(obs_types):2d} " + " ".join(obs_types), "SYS / # / OBS TYPES"),
                _header("", "END OF HEADER"),
                "> 2024 07 20 09 52 30.0000000  0  2\n",
                "G10" + "".join(_field(v) for v in range(1, 13)) + "\n",
                "G15" + "".join(_field(v) for v in range(101, 113)) + "\n",
            ]
        ),
        encoding="utf-8",
    )

    obs = read_rinex_obs(rinex_path)

    assert len(obs.epochs) == 1
    assert obs.epochs[0].satellites == ["G10", "G15"]
    assert obs.epochs[0].observations["G10"]["C1C"] == 1.0
    assert obs.epochs[0].observations["G10"]["S5Q"] == 12.0
    assert obs.epochs[0].observations["G15"]["C1C"] == 101.0
    assert obs.epochs[0].observations["G15"]["S5Q"] == 112.0
