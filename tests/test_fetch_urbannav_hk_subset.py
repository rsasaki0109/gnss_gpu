"""Tests for UrbanNav Hong Kong subset-selection helpers."""

from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

from fetch_urbannav_hk_subset import (
    select_hk20190428_gnss_members,
    select_hk20190428_support_members,
)


def test_select_hk20190428_gnss_members_normalizes_expected_files():
    names = [
        "RINEX/COM3_190428_124409.obs",
        "RINEX/hksc1180.19o",
        "RINEX/hksc1180.19n",
    ]

    selected = select_hk20190428_gnss_members(names)

    assert selected == {
        "rover_ublox.obs": "RINEX/COM3_190428_124409.obs",
        "base_hksc.obs": "RINEX/hksc1180.19o",
        "base.nav": "RINEX/hksc1180.19n",
    }


def test_select_hk20190428_support_members_picks_csvs():
    names = ["imu.csv", "reference.csv", "notes.txt"]

    selected = select_hk20190428_support_members(names)

    assert selected == {
        "imu.csv": "imu.csv",
        "reference.csv": "reference.csv",
    }
