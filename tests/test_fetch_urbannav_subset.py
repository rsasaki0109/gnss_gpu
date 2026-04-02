"""Tests for UrbanNav subset-selection helpers."""

from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

from fetch_urbannav_subset import select_run_entries


class _FakeZip:
    def __init__(self, names):
        self._names = list(names)

    def namelist(self):
        return list(self._names)


def test_select_run_entries_picks_core_files_for_one_run():
    zf = _FakeZip(
        [
            "Tokyo_Data/Odaiba/base.nav",
            "Tokyo_Data/Odaiba/base_trimble.obs",
            "Tokyo_Data/Odaiba/reference.csv",
            "Tokyo_Data/Odaiba/rover_ublox.obs",
            "Tokyo_Data/Odaiba/rover_trimble.obs",
            "Tokyo_Data/Odaiba/imu.csv",
            "Tokyo_Data/Odaiba/lidar.bag",
            "Tokyo_Data/Shinjuku/base.nav",
        ]
    )

    selected = select_run_entries(zf, "Odaiba")

    assert selected == [
        "Tokyo_Data/Odaiba/base.nav",
        "Tokyo_Data/Odaiba/base_trimble.obs",
        "Tokyo_Data/Odaiba/imu.csv",
        "Tokyo_Data/Odaiba/reference.csv",
        "Tokyo_Data/Odaiba/rover_trimble.obs",
        "Tokyo_Data/Odaiba/rover_ublox.obs",
    ]


def test_select_run_entries_can_skip_imu_and_include_lidar():
    zf = _FakeZip(
        [
            "Tokyo_Data/Shinjuku/base.nav",
            "Tokyo_Data/Shinjuku/base_trimble.obs",
            "Tokyo_Data/Shinjuku/reference.csv",
            "Tokyo_Data/Shinjuku/rover_ublox.obs",
            "Tokyo_Data/Shinjuku/lidar.bag",
            "Tokyo_Data/Shinjuku/imu.csv",
        ]
    )

    selected = select_run_entries(zf, "Shinjuku", include_imu=False, include_lidar=True)

    assert selected == [
        "Tokyo_Data/Shinjuku/base.nav",
        "Tokyo_Data/Shinjuku/base_trimble.obs",
        "Tokyo_Data/Shinjuku/lidar.bag",
        "Tokyo_Data/Shinjuku/reference.csv",
        "Tokyo_Data/Shinjuku/rover_ublox.obs",
    ]
