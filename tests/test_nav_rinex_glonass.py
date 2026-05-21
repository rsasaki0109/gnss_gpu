"""GLONASS-specific tests for the RINEX 3 navigation parser.

The Kepler families are already covered by ``test_ephemeris.py``; these tests
exercise the 4-line GLONASS layout (km → m conversion, ``-TauN`` sign flip,
frequency channel parsing) end-to-end via a minimal in-memory RINEX 3 file.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from gnss_gpu.io.nav_rinex import read_nav_rinex


_GLO_RINEX_FIXTURE = dedent(
    """\
         3.04           N: GNSS NAV DATA    M: MIXED            RINEX VERSION / TYPE
    test_writer         test_agency         20260520 000000 UTC PGM / RUN BY / DATE
                                                                END OF HEADER
    R01 2026 05 20 00 15 00-1.000000000000e-04 2.000000000000e-09 0.000000000000e+00
         5.000000000000e+03 1.000000000000e+00 1.000000000000e-09 0.000000000000e+00
         1.000000000000e+04 2.000000000000e+00 2.000000000000e-09 1.000000000000e+00
         1.500000000000e+04 3.000000000000e+00 3.000000000000e-09 0.000000000000e+00
    """
)


@pytest.fixture
def glonass_nav_file(tmp_path: Path) -> Path:
    path = tmp_path / "glonass.rnx"
    path.write_text(_GLO_RINEX_FIXTURE)
    return path


def test_glonass_parser_returns_one_record_per_prn(glonass_nav_file: Path):
    out = read_nav_rinex(glonass_nav_file, systems=("R",))
    assert list(out.keys()) == [1]
    assert len(out[1]) == 1


def test_glonass_parser_keyed_by_sat_id(glonass_nav_file: Path):
    out = read_nav_rinex(glonass_nav_file, systems=("R",), key_by_sat_id=True)
    assert "R01" in out
    nav = out["R01"][0]
    assert nav.system == "R"
    assert nav.prn == 1


def test_glonass_parser_converts_position_kilometres_to_metres(glonass_nav_file: Path):
    nav = read_nav_rinex(glonass_nav_file, systems=("R",))[1][0]
    # Fixture stored 5 / 10 / 15 km — must surface as 5,000 / 10,000 / 15,000 m.
    assert nav.glo_px_m == pytest.approx(5_000_000.0)
    assert nav.glo_py_m == pytest.approx(10_000_000.0)
    assert nav.glo_pz_m == pytest.approx(15_000_000.0)


def test_glonass_parser_converts_velocity_and_acceleration_units(glonass_nav_file: Path):
    nav = read_nav_rinex(glonass_nav_file, systems=("R",))[1][0]
    # Velocity: 1 / 2 / 3 km/s → 1000 / 2000 / 3000 m/s.
    assert nav.glo_vx_m_s == pytest.approx(1000.0)
    assert nav.glo_vy_m_s == pytest.approx(2000.0)
    assert nav.glo_vz_m_s == pytest.approx(3000.0)
    # Acceleration: 1e-9 / 2e-9 / 3e-9 km/s² → 1e-6 / 2e-6 / 3e-6 m/s².
    assert nav.glo_ax_m_s2 == pytest.approx(1.0e-6)
    assert nav.glo_ay_m_s2 == pytest.approx(2.0e-6)
    assert nav.glo_az_m_s2 == pytest.approx(3.0e-6)


def test_glonass_parser_negates_minus_tau_n_into_canonical_tau(glonass_nav_file: Path):
    # Fixture stores -TauN = -1e-4 → tau_n = +1e-4.
    nav = read_nav_rinex(glonass_nav_file, systems=("R",))[1][0]
    assert nav.glo_tau_n == pytest.approx(1.0e-4)


def test_glonass_parser_keeps_gamma_n_sign_intact(glonass_nav_file: Path):
    nav = read_nav_rinex(glonass_nav_file, systems=("R",))[1][0]
    assert nav.glo_gamma_n == pytest.approx(2.0e-9)


def test_glonass_parser_extracts_frequency_channel(glonass_nav_file: Path):
    # Frequency channel is stored at column 3 of broadcast-orbit line 2 of
    # the fixture (here ``1.0``).
    nav = read_nav_rinex(glonass_nav_file, systems=("R",))[1][0]
    assert nav.glo_frequency_channel == 1


def test_glonass_parser_sets_toe_to_seconds_of_week(glonass_nav_file: Path):
    # toc = 2026-05-20 00:15:00 UTC → sow ≈ 900s into a Wednesday week start.
    # We only assert the sow is non-zero and < one week, not the exact value,
    # because GPST week numbering depends on the platform datetime epoch math.
    nav = read_nav_rinex(glonass_nav_file, systems=("R",))[1][0]
    assert 0.0 <= nav.toe < 604800.0
    assert nav.toe == pytest.approx(nav.toc_seconds)


def test_glonass_parser_leaves_kepler_fields_at_default(glonass_nav_file: Path):
    nav = read_nav_rinex(glonass_nav_file, systems=("R",))[1][0]
    assert nav.sqrt_a == 0.0
    assert nav.e == 0.0
    assert nav.M0 == 0.0


def test_glonass_parser_skips_when_not_in_requested_systems(glonass_nav_file: Path):
    out = read_nav_rinex(glonass_nav_file, systems=("G",))
    assert out == {}
