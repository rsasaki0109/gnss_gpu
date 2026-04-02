"""Tests for PPCDatasetLoader using small inline CSV fixtures."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from gnss_gpu.io.ppc import PPCDatasetLoader


def _write(directory: Path, filename: str, content: str) -> Path:
    path = directory / filename
    path.write_text(content)
    return path


REFERENCE_CSV = """\
GPS TOW (s),GPS Week,Latitude (deg),Longitude (deg),Ellipsoid Height (m),ECEF X (m),ECEF Y (m),ECEF Z (m)
187470.0,2324,35.66755248,139.79109598,39.557,-3961767.630,3349008.712,3698309.780
187470.2,2324,35.66755248,139.79109598,39.557,-3961767.630,3349008.712,3698309.780
"""

IMU_CSV = """\
GPS TOW (s), GPS Week, Acc X (m/s^2), Acc Y (m/s^2), Acc Z (m/s^2), Ang Rate X (deg/s),  Ang Rate Y (deg/s),  Ang Rate Z (deg/s)
187470.00,2324,0.1,-0.2,9.8,0.01,0.02,0.03
187470.01,2324,0.2,-0.1,9.7,0.04,0.05,0.06
"""


def test_is_run_directory():
    with tempfile.TemporaryDirectory() as td:
        directory = Path(td)
        for filename in ("rover.obs", "base.obs", "base.nav", "reference.csv"):
            _write(directory, filename, "")
        assert PPCDatasetLoader.is_run_directory(directory)


def test_loader_init_missing_dir():
    with pytest.raises(FileNotFoundError):
        PPCDatasetLoader("/nonexistent/path/ppc-run")


def test_load_ground_truth_ecef_columns():
    with tempfile.TemporaryDirectory() as td:
        directory = Path(td)
        _write(directory, "reference.csv", REFERENCE_CSV)
        loader = PPCDatasetLoader(directory)
        times, ecef = loader.load_ground_truth()

    assert times.shape == (2,)
    assert ecef.shape == (2, 3)
    assert times[0] == pytest.approx(187470.0)
    assert ecef[0, 0] == pytest.approx(-3961767.630)


def test_load_imu_columns():
    with tempfile.TemporaryDirectory() as td:
        directory = Path(td)
        _write(directory, "imu.csv", IMU_CSV)
        loader = PPCDatasetLoader(directory)
        imu = loader.load_imu()

    assert set(imu) == {"time", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"}
    assert imu["time"].shape == (2,)
    assert imu["acc_z"][0] == pytest.approx(9.8)
    assert imu["gyro_z"][1] == pytest.approx(0.06)


def test_load_ground_truth_latlon_fallback():
    content = """\
GPS TOW (s),Latitude (deg),Longitude (deg),Ellipsoid Height (m)
187470.0,35.66755248,139.79109598,39.557
"""
    with tempfile.TemporaryDirectory() as td:
        directory = Path(td)
        _write(directory, "reference.csv", content)
        loader = PPCDatasetLoader(directory)
        _, ecef = loader.load_ground_truth()

    assert ecef.shape == (1, 3)
    assert 6.3e6 < np.linalg.norm(ecef[0]) < 6.5e6
