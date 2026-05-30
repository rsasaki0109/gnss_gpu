from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from experiments.audit_gsdc2023_dd_carrier_support import (
    carrier_wavelength_m,
    course_base_obs_path_for_template,
    rover_measurements_for_epoch,
    snap_tow_to_base_epoch,
)
from gnss_gpu.dd_carrier import GPS_L1_WAVELENGTH, GPS_L5_WAVELENGTH


def test_carrier_wavelength_uses_signal_band() -> None:
    assert carrier_wavelength_m(1, "GPS_L1_CA") == GPS_L1_WAVELENGTH
    assert carrier_wavelength_m(1, "GPS_L5_Q") == GPS_L5_WAVELENGTH


def test_rover_measurements_for_epoch_converts_adr_meters_to_cycles() -> None:
    batch = SimpleNamespace(
        adr=np.array([[GPS_L1_WAVELENGTH * 1000.0, GPS_L5_WAVELENGTH * 2000.0, 0.0]], dtype=np.float64),
        adr_state=np.array([[1, 1, 1]], dtype=np.int32),
        sat_ecef=np.array(
            [
                [
                    [20_200_000.0, 1000.0, 1_000_000.0],
                    [20_300_000.0, 2000.0, 1_000_000.0],
                    [0.0, 0.0, 0.0],
                ],
            ],
            dtype=np.float64,
        ),
        kaggle_wls=np.array([[6_378_137.0, 0.0, 0.0]], dtype=np.float64),
        slot_keys=((1, 3, "GPS_L1_CA"), (6, 11, "GAL_E5_Q"), (1, 5, "GPS_L1_CA")),
    )

    measurements = rover_measurements_for_epoch(batch, 0)

    assert len(measurements) == 2
    assert measurements[0].prn == 3
    assert measurements[0].carrier_phase == 1000.0
    assert measurements[1].prn == 11
    assert measurements[1].carrier_phase == 2000.0
    assert measurements[1].system_id == 2


def test_snap_tow_to_base_epoch_uses_nearest_within_tolerance() -> None:
    computer = SimpleNamespace(_base_tow_keys=np.array([90.0, 120.0, 150.0], dtype=np.float64))

    assert snap_tow_to_base_epoch(computer, 120.44, 0.6) == 120.0
    assert snap_tow_to_base_epoch(computer, 120.7, 0.6) is None


def test_course_base_obs_path_template_prefers_existing_course_file(tmp_path) -> None:
    course_dir = tmp_path / "train" / "course"
    course_dir.mkdir(parents=True)
    highrate = course_dir / "SLAC_1hz.obs"
    highrate.write_text("rinex\n", encoding="ascii")

    assert course_base_obs_path_for_template(
        tmp_path,
        "train",
        "course",
        "SLAC",
        "rnx3",
        base_obs_template="{base}_1hz.obs",
    ) == highrate


def test_course_base_obs_path_template_falls_back_when_missing(tmp_path) -> None:
    standard = tmp_path / "train" / "course" / "SLAC_rnx3.obs"
    standard.parent.mkdir(parents=True)
    standard.write_text("rinex\n", encoding="ascii")

    assert course_base_obs_path_for_template(
        tmp_path,
        "train",
        "course",
        "SLAC",
        "rnx3",
        base_obs_template="{base}_1hz.obs",
    ) == standard
