"""Tests for UrbanNavLoader using synthetic inline CSV data.

No actual UrbanNav download is required; all test data is created in
temporary directories.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from gnss_gpu.io.urbannav import (
    UrbanNavLoader,
    GnssObs,
    _llh_to_ecef,
    _pick_observation_value,
    _SYSTEM_PR_FALLBACKS,
    _SYSTEM_SNR_FALLBACKS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write(directory: Path, filename: str, content: str) -> Path:
    p = directory / filename
    p.write_text(content)
    return p


# ---------------------------------------------------------------------------
# _llh_to_ecef unit test
# ---------------------------------------------------------------------------

def test_llh_to_ecef_equator():
    """A point on the equator at lon=0 should map to (a, 0, 0) approximately."""
    xyz = _llh_to_ecef(0.0, 0.0, 0.0)
    assert abs(xyz[0] - 6_378_137.0) < 1.0
    assert abs(xyz[1]) < 1.0
    assert abs(xyz[2]) < 1.0


def test_llh_to_ecef_north_pole():
    """North pole at (90, 0, 0) should give x≈0, y≈0, z≈b."""
    xyz = _llh_to_ecef(90.0, 0.0, 0.0)
    b = 6_356_752.3142  # WGS-84 semi-minor axis
    assert abs(xyz[0]) < 1.0
    assert abs(xyz[1]) < 1.0
    assert abs(xyz[2] - b) < 1.0


# ---------------------------------------------------------------------------
# UrbanNavLoader construction
# ---------------------------------------------------------------------------

def test_loader_init_valid():
    with tempfile.TemporaryDirectory() as td:
        loader = UrbanNavLoader(td)
        assert loader.data_dir == Path(td)


def test_loader_init_missing_dir():
    with pytest.raises(FileNotFoundError):
        UrbanNavLoader("/nonexistent/path/abc123")


def test_is_run_directory_true_for_tokyo_style_layout():
    with tempfile.TemporaryDirectory() as td:
        run_dir = Path(td)
        for name in ("reference.csv", "base.nav", "base_trimble.obs", "rover_ublox.obs"):
            _write(run_dir, name, "")

        assert UrbanNavLoader.is_run_directory(run_dir)


def test_is_run_directory_false_when_required_files_missing():
    with tempfile.TemporaryDirectory() as td:
        run_dir = Path(td)
        _write(run_dir, "reference.csv", "")
        _write(run_dir, "base.nav", "")

        assert not UrbanNavLoader.is_run_directory(run_dir)


# ---------------------------------------------------------------------------
# load_gnss_csv
# ---------------------------------------------------------------------------

GNSS_CSV = """\
GPS_time,PRN,pseudorange,carrier,CN0,doppler
1.0,G01,20000000.0,105000000.0,42.5,-1500.0
1.0,G05,21000000.0,110000000.0,38.0,-2000.0
2.0,G01,20000100.0,105000010.0,41.0,-1490.0
2.0,G05,21000100.0,110000010.0,37.5,-1990.0
2.0,G12,22000000.0,115000000.0,35.0,-1200.0
"""


def test_load_gnss_csv_basic():
    with tempfile.TemporaryDirectory() as td:
        _write(Path(td), "run1_GNSS.csv", GNSS_CSV)
        loader = UrbanNavLoader(td)
        data = loader.load_gnss_csv()

    assert set(data.keys()) >= {"time", "prn", "pseudorange", "carrier", "cn0", "doppler"}
    assert len(data["time"]) == 5
    assert list(data["prn"]) == ["G01", "G05", "G01", "G05", "G12"]
    assert data["pseudorange"][0] == pytest.approx(20_000_000.0)
    assert data["cn0"][1] == pytest.approx(38.0)
    assert data["doppler"][2] == pytest.approx(-1490.0)


def test_load_gnss_csv_explicit_path():
    with tempfile.TemporaryDirectory() as td:
        csv_path = _write(Path(td), "custom.csv", GNSS_CSV)
        loader = UrbanNavLoader(td)
        data = loader.load_gnss_csv(filepath=csv_path)

    assert len(data["time"]) == 5


def test_load_gnss_csv_no_file():
    with tempfile.TemporaryDirectory() as td:
        loader = UrbanNavLoader(td)
        with pytest.raises(FileNotFoundError):
            loader.load_gnss_csv()


def test_load_gnss_csv_empty():
    with tempfile.TemporaryDirectory() as td:
        _write(Path(td), "run_GNSS.csv", "GPS_time,PRN,pseudorange,carrier,CN0,doppler\n")
        loader = UrbanNavLoader(td)
        data = loader.load_gnss_csv()

    assert len(data["time"]) == 0


def test_load_gnss_csv_missing_optional_cols():
    """CSV with only time, PRN and pseudorange; carrier/CN0/doppler absent."""
    content = "GPS_time,PRN,pseudorange\n1.0,G01,20000000.0\n2.0,G03,21000000.0\n"
    with tempfile.TemporaryDirectory() as td:
        _write(Path(td), "run_GNSS.csv", content)
        loader = UrbanNavLoader(td)
        data = loader.load_gnss_csv()

    assert len(data["time"]) == 2
    assert np.all(np.isnan(data["carrier"]))
    assert np.all(np.isnan(data["cn0"]))
    assert np.all(np.isnan(data["doppler"]))


def test_load_gnss_csv_extra_columns():
    """Extra numeric columns should be included in the returned dict."""
    content = "GPS_time,PRN,pseudorange,elevation\n1.0,G01,20000000.0,25.3\n"
    with tempfile.TemporaryDirectory() as td:
        _write(Path(td), "run_GNSS.csv", content)
        loader = UrbanNavLoader(td)
        data = loader.load_gnss_csv()

    assert "elevation" in data
    assert data["elevation"][0] == pytest.approx(25.3)


# ---------------------------------------------------------------------------
# load_ground_truth – ECEF variant
# ---------------------------------------------------------------------------

GT_ECEF_CSV = """\
GPS_time,x,y,z
1.0,-2414266.9197,5386768.9868,2407460.0288
2.0,-2414267.0000,5386769.0000,2407460.1000
3.0,-2414267.1000,5386769.1000,2407460.2000
"""

GT_REFERENCE_STYLE_CSV = """\
GPS TOW (s), GPS Week, ECEF X (m), ECEF Y (m), ECEF Z (m), Ellipsoid Height (m)
273375.10,2032,-3963426.7981,3350882.1576,3694865.5458,44.6995
273375.20,2032,-3963426.7981,3350882.1576,3694865.5458,44.6995
"""


def test_load_ground_truth_ecef():
    with tempfile.TemporaryDirectory() as td:
        _write(Path(td), "run_groundtruth.csv", GT_ECEF_CSV)
        loader = UrbanNavLoader(td)
        times, ecef = loader.load_ground_truth()

    assert times.shape == (3,)
    assert ecef.shape == (3, 3)
    assert times[0] == pytest.approx(1.0)
    assert ecef[0, 0] == pytest.approx(-2_414_266.9197)


def test_load_ground_truth_reference_csv_aliases():
    with tempfile.TemporaryDirectory() as td:
        _write(Path(td), "reference.csv", GT_REFERENCE_STYLE_CSV)
        loader = UrbanNavLoader(td)
        times, ecef = loader.load_ground_truth()

    assert times.shape == (2,)
    assert times[0] == pytest.approx(273375.10)
    assert ecef[0, 0] == pytest.approx(-3_963_426.7981)


def test_pick_observation_value_prefers_requested_code():
    code, value = _pick_observation_value(
        "G",
        {"C1C": 20_000_000.0, "C1W": 21_000_000.0},
        "C1C",
        _SYSTEM_PR_FALLBACKS,
    )
    assert code == "C1C"
    assert value == pytest.approx(20_000_000.0)


def test_pick_observation_value_falls_back_by_constellation():
    pr_code, pr_value = _pick_observation_value(
        "E",
        {"C1X": 24_000_000.0, "S1X": 39.0},
        "C1C",
        _SYSTEM_PR_FALLBACKS,
    )
    snr_code, snr_value = _pick_observation_value(
        "E",
        {"C1X": 24_000_000.0, "S1X": 39.0},
        "S1C",
        _SYSTEM_SNR_FALLBACKS,
    )
    assert pr_code == "C1X"
    assert pr_value == pytest.approx(24_000_000.0)
    assert snr_code == "S1X"
    assert snr_value == pytest.approx(39.0)


# ---------------------------------------------------------------------------
# load_ground_truth – geodetic (lat/lon/alt) variant
# ---------------------------------------------------------------------------

GT_LLH_CSV = """\
GPS_time,latitude,longitude,altitude
1.0,22.3198,114.1694,50.0
2.0,22.3199,114.1695,50.5
"""

GT_HK_REFERENCE_CSV = """\
ROS Time (s), GPS Week, GPS TOW (s), Latitude (deg), Longitude (deg), Altitude (m)
1556456283.924171925,2051,46699.000000000,22.301155545,114.179000313,6.613695497
1556456284.924304008,2051,46700.000000000,22.301155461,114.179000322,6.605289413
"""


def test_load_ground_truth_llh():
    with tempfile.TemporaryDirectory() as td:
        _write(Path(td), "run_groundtruth.csv", GT_LLH_CSV)
        loader = UrbanNavLoader(td)
        times, ecef = loader.load_ground_truth()

    assert times.shape == (2,)
    assert ecef.shape == (2, 3)
    # ECEF magnitude for a point near sea level should be ~6.37e6 m
    assert 6.3e6 < np.linalg.norm(ecef[0]) < 6.4e6


def test_load_ground_truth_hk_reference_aliases():
    with tempfile.TemporaryDirectory() as td:
        _write(Path(td), "reference.csv", GT_HK_REFERENCE_CSV)
        loader = UrbanNavLoader(td)
        times, ecef = loader.load_ground_truth()

    assert times.shape == (2,)
    assert times[0] == pytest.approx(46699.0)
    assert ecef.shape == (2, 3)
    assert 6.3e6 < np.linalg.norm(ecef[0]) < 6.4e6


def test_load_ground_truth_llh_no_alt():
    """Altitude column missing; loader should default to 0."""
    content = "GPS_time,latitude,longitude\n1.0,22.3198,114.1694\n"
    with tempfile.TemporaryDirectory() as td:
        _write(Path(td), "run_groundtruth.csv", content)
        loader = UrbanNavLoader(td)
        times, ecef = loader.load_ground_truth()

    assert ecef.shape == (1, 3)
    assert 6.3e6 < np.linalg.norm(ecef[0]) < 6.4e6


def test_load_ground_truth_no_usable_cols():
    content = "GPS_time,speed\n1.0,0.5\n"
    with tempfile.TemporaryDirectory() as td:
        _write(Path(td), "run_groundtruth.csv", content)
        loader = UrbanNavLoader(td)
        with pytest.raises(ValueError, match="neither ECEF"):
            loader.load_ground_truth()


def test_load_ground_truth_no_file():
    with tempfile.TemporaryDirectory() as td:
        loader = UrbanNavLoader(td)
        with pytest.raises(FileNotFoundError):
            loader.load_ground_truth()


def test_load_ground_truth_explicit_path():
    with tempfile.TemporaryDirectory() as td:
        p = _write(Path(td), "custom_gt.csv", GT_ECEF_CSV)
        loader = UrbanNavLoader(td)
        times, ecef = loader.load_ground_truth(filepath=p)
    assert len(times) == 3


# ---------------------------------------------------------------------------
# load_rinex_obs / load_rinex_nav – absent files return None
# ---------------------------------------------------------------------------

def test_load_rinex_obs_none_when_missing():
    with tempfile.TemporaryDirectory() as td:
        loader = UrbanNavLoader(td)
        assert loader.load_rinex_obs() is None


def test_load_rinex_nav_none_when_missing():
    with tempfile.TemporaryDirectory() as td:
        loader = UrbanNavLoader(td)
        assert loader.load_rinex_nav() is None


# ---------------------------------------------------------------------------
# epochs()
# ---------------------------------------------------------------------------

def test_epochs_basic():
    with tempfile.TemporaryDirectory() as td:
        _write(Path(td), "run_GNSS.csv", GNSS_CSV)
        _write(Path(td), "run_groundtruth.csv", GT_ECEF_CSV)
        loader = UrbanNavLoader(td)
        result = list(loader.epochs())

    # GNSS epochs: t=1.0 and t=2.0; GT times: 1.0, 2.0, 3.0 – both match
    assert len(result) == 2
    obs0, pos0 = result[0]
    assert isinstance(obs0, GnssObs)
    assert obs0.time == pytest.approx(1.0)
    assert obs0.prn == ["G01", "G05"]
    assert obs0.pseudorange.shape == (2,)
    assert pos0.shape == (3,)

    obs1, pos1 = result[1]
    assert obs1.time == pytest.approx(2.0)
    assert obs1.prn == ["G01", "G05", "G12"]


def test_epochs_no_match_outside_tolerance():
    """Ground truth times far from GNSS epochs should yield no results."""
    gt_far = "GPS_time,x,y,z\n100.0,-2414266.0,5386769.0,2407460.0\n"
    with tempfile.TemporaryDirectory() as td:
        _write(Path(td), "run_GNSS.csv", GNSS_CSV)
        _write(Path(td), "run_groundtruth.csv", gt_far)
        loader = UrbanNavLoader(td)
        result = list(loader.epochs(time_tolerance=0.5))

    assert len(result) == 0


def test_epochs_tolerance_respected():
    """With a large enough tolerance, all GNSS epochs should match."""
    gt_far = "GPS_time,x,y,z\n100.0,-2414266.0,5386769.0,2407460.0\n"
    with tempfile.TemporaryDirectory() as td:
        _write(Path(td), "run_GNSS.csv", GNSS_CSV)
        _write(Path(td), "run_groundtruth.csv", gt_far)
        loader = UrbanNavLoader(td)
        result = list(loader.epochs(time_tolerance=200.0))

    assert len(result) == 2


def test_epochs_extra_columns_preserved():
    content = "GPS_time,PRN,pseudorange,elevation\n1.0,G01,20000000.0,25.3\n"
    gt = "GPS_time,x,y,z\n1.0,-2414266.0,5386769.0,2407460.0\n"
    with tempfile.TemporaryDirectory() as td:
        _write(Path(td), "run_GNSS.csv", content)
        _write(Path(td), "run_groundtruth.csv", gt)
        loader = UrbanNavLoader(td)
        result = list(loader.epochs())

    assert len(result) == 1
    obs, _ = result[0]
    assert "elevation" in obs.extra
    assert obs.extra["elevation"][0] == pytest.approx(25.3)


# ---------------------------------------------------------------------------
# Column-name alias robustness
# ---------------------------------------------------------------------------

def test_gnss_csv_alternate_column_names():
    """Alternate column name aliases should be resolved correctly."""
    content = "timestamp,SatID,C1,L1,SNR,D1\n0.5,G02,19500000.0,102000000.0,40.0,-1800.0\n"
    with tempfile.TemporaryDirectory() as td:
        _write(Path(td), "run_GNSS.csv", content)
        loader = UrbanNavLoader(td)
        data = loader.load_gnss_csv()

    assert data["time"][0] == pytest.approx(0.5)
    assert data["prn"][0] == "G02"
    assert data["pseudorange"][0] == pytest.approx(19_500_000.0)
    assert data["carrier"][0] == pytest.approx(102_000_000.0)
    assert data["cn0"][0] == pytest.approx(40.0)
    assert data["doppler"][0] == pytest.approx(-1800.0)


def test_ground_truth_alternate_ecef_names():
    content = "time,ECEF_X,ECEF_Y,ECEF_Z\n1.0,-2414266.9,5386768.9,2407460.0\n"
    with tempfile.TemporaryDirectory() as td:
        _write(Path(td), "run_groundtruth.csv", content)
        loader = UrbanNavLoader(td)
        times, ecef = loader.load_ground_truth()

    assert ecef[0, 0] == pytest.approx(-2_414_266.9)


# ---------------------------------------------------------------------------
# GnssObs dataclass
# ---------------------------------------------------------------------------

def test_gnss_obs_dataclass():
    obs = GnssObs(
        time=1.0,
        prn=["G01"],
        pseudorange=np.array([20_000_000.0]),
        carrier=np.array([105_000_000.0]),
        cn0=np.array([42.0]),
        doppler=np.array([-1500.0]),
    )
    assert obs.time == pytest.approx(1.0)
    assert obs.prn == ["G01"]
    assert obs.extra == {}
