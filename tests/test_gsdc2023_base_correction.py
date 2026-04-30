from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd

import experiments.gsdc2023_base_correction as base_correction
from experiments.gsdc2023_base_correction import (
    GPS_L5_TGD_SCALE,
    compute_base_pseudorange_correction_matrix,
    filter_matrtklib_duplicate_gps_nav_messages,
    matlab_base_time_span_mask,
    read_base_station_xyz,
    rinex_header_base_xyz_or_metadata,
    select_base_pseudorange_observation,
    select_gps_nav_message,
    signal_type_iono_scale,
    unix_ms_to_gps_abs_seconds,
)
from gnss_gpu.io.nav_rinex import NavMessage


def test_matrtklib_nav_filter_and_selection_direct():
    messages = [
        NavMessage(prn=9, toc=datetime(2021, 12, 8, 19, 59, 44), toe=331184.0, iode=7.0),
        NavMessage(prn=9, toc=datetime(2021, 12, 8, 20, 0, 0), toe=331200.0, iode=95.0),
        NavMessage(prn=9, toc=datetime(2021, 12, 8, 21, 59, 44), toe=338384.0, iode=8.0),
    ]

    filtered = filter_matrtklib_duplicate_gps_nav_messages(messages)

    assert [int(message.iode) for message in filtered] == [7, 8]
    assert int(select_gps_nav_message(tuple(messages), 333151.44).iode) == 95
    assert int(select_gps_nav_message(tuple(filtered), 333151.44).iode) == 7


def test_matrtklib_sat_product_adjusts_same_selected_message_when_raw_product_is_stale(monkeypatch):
    selected = SimpleNamespace(toe=100.0, iode=7.0, tgd=0.0, clock_m=10.0, x0=1000.0)

    def fake_compute_single_cpu(message, tow_s, _code):
        return np.array([message.x0 + tow_s, 2.0, 3.0], dtype=np.float64), message.clock_m / base_correction.LIGHT_SPEED_MPS

    monkeypatch.setattr(
        base_correction.Ephemeris,
        "_compute_single_cpu",
        staticmethod(fake_compute_single_cpu),
    )

    close = base_correction.gps_matrtklib_sat_product_adjustment(
        svid=13,
        arrival_tow_s=100.0,
        l1_raw_pseudorange_m=base_correction.LIGHT_SPEED_MPS * 10.0,
        derived_common_clock_m=10.001,
        nav_messages_by_svid={13: ((selected,), (selected,))},
    )
    assert close is None

    adjusted = base_correction.gps_matrtklib_sat_product_adjustment(
        svid=13,
        arrival_tow_s=100.0,
        l1_raw_pseudorange_m=base_correction.LIGHT_SPEED_MPS * 10.0,
        derived_common_clock_m=10.1,
        nav_messages_by_svid={13: ((selected,), (selected,))},
    )

    assert adjusted is not None
    sat_pos, sat_vel, sat_clock_bias_m, sat_clock_drift_mps = adjusted
    np.testing.assert_allclose(sat_pos, np.array([1090.0, 2.0, 3.0]), atol=1.0e-6)
    np.testing.assert_allclose(sat_vel, np.array([1.0, 0.0, 0.0]), atol=1.0e-9)
    assert np.isclose(sat_clock_bias_m, 10.0)
    assert np.isclose(sat_clock_drift_mps, 0.0)


def test_matrtklib_sat_product_uses_broadcast_clock_for_transmit_time(monkeypatch):
    selected = SimpleNamespace(
        toe=0.0,
        toc_seconds=0.0,
        iode=7.0,
        tgd=20.0 / base_correction.LIGHT_SPEED_MPS,
        af0=1.0e-8,
        af1=0.0,
        af2=0.0,
        clock_m=80.0,
    )

    def fake_compute_single_cpu(message, tow_s, _code):
        return (
            np.array([tow_s * base_correction.LIGHT_SPEED_MPS, 0.0, 0.0], dtype=np.float64),
            message.clock_m / base_correction.LIGHT_SPEED_MPS,
        )

    monkeypatch.setattr(
        base_correction.Ephemeris,
        "_compute_single_cpu",
        staticmethod(fake_compute_single_cpu),
    )

    transmit_tow_s = 900.0
    adjusted = base_correction.gps_matrtklib_sat_product_adjustment(
        svid=13,
        arrival_tow_s=1000.0,
        l1_raw_pseudorange_m=base_correction.LIGHT_SPEED_MPS * 100.0,
        derived_common_clock_m=80.1,
        nav_messages_by_svid={13: ((selected,), (selected,))},
    )

    assert adjusted is not None
    sat_pos, _sat_vel, sat_clock_bias_m, _sat_clock_drift_mps = adjusted
    expected_tow_s = transmit_tow_s - selected.af0
    np.testing.assert_allclose(sat_pos[0], expected_tow_s * base_correction.LIGHT_SPEED_MPS, atol=1.0e-5)
    assert np.isclose(sat_clock_bias_m, 100.0)


def test_matrtklib_sat_product_applies_receiver_clock_to_transmit_time(monkeypatch):
    selected = SimpleNamespace(
        toe=0.0,
        toc_seconds=0.0,
        iode=7.0,
        tgd=0.0,
        af0=1.0e-8,
        af1=0.0,
        af2=0.0,
        clock_m=10.0,
    )

    def fake_compute_single_cpu(message, tow_s, _code):
        return np.array([tow_s * base_correction.LIGHT_SPEED_MPS, 0.0, 0.0], dtype=np.float64), message.clock_m / base_correction.LIGHT_SPEED_MPS

    monkeypatch.setattr(
        base_correction.Ephemeris,
        "_compute_single_cpu",
        staticmethod(fake_compute_single_cpu),
    )

    adjusted = base_correction.gps_matrtklib_sat_product_adjustment(
        svid=13,
        arrival_tow_s=1000.0,
        l1_raw_pseudorange_m=base_correction.LIGHT_SPEED_MPS * 100.0,
        derived_common_clock_m=10.1,
        nav_messages_by_svid={13: ((selected,), (selected,))},
        receiver_clock_bias_m=-30.0,
    )

    assert adjusted is not None
    sat_pos, _sat_vel, _sat_clock_bias_m, _sat_clock_drift_mps = adjusted
    expected_tow_s = 900.0 - 30.0 / base_correction.LIGHT_SPEED_MPS - selected.af0
    np.testing.assert_allclose(sat_pos[0], expected_tow_s * base_correction.LIGHT_SPEED_MPS, atol=1.0e-5)


def test_base_observation_selector_and_iono_scale_direct():
    obs = {
        "C1C": 21_000_001.0,
        "C1X": 21_000_002.0,
        "C5Q": 22_000_005.0,
    }

    assert select_base_pseudorange_observation(obs, "GPS_L1_CA") == ("C1C", 21_000_001.0)
    assert select_base_pseudorange_observation(obs, "GAL_E1_C_P") == ("C1C", 21_000_001.0)
    assert select_base_pseudorange_observation(obs, "GPS_L5_Q") == ("C5Q", 22_000_005.0)
    assert select_base_pseudorange_observation({"C1X": 21_000_010.0}, "GAL_E1_C_P") == (
        "C1X",
        21_000_010.0,
    )
    assert signal_type_iono_scale("GPS_L1_CA") == 1.0
    assert signal_type_iono_scale("GAL_E1_C_P") == 1.0
    assert np.isclose(signal_type_iono_scale("GPS_L5_Q"), GPS_L5_TGD_SCALE)
    assert np.isclose(signal_type_iono_scale("GAL_E5A_Q"), GPS_L5_TGD_SCALE)


def test_base_position_offset_direct(tmp_path):
    data_root = tmp_path / "dataset_2023"
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    pd.DataFrame(
        [
            {
                "Base": "slac",
                "Year": 2020,
                "X": -2700112.7,
                "Y": -4292747.3,
                "Z": 3855195.5,
            },
        ],
    ).to_csv(base_dir / "base_position.csv", index=False)
    pd.DataFrame([{"Base": "slac", "E": 1.0, "N": 2.0, "U": 3.0}]).to_csv(
        base_dir / "base_offset.csv",
        index=False,
    )

    no_offset = read_base_station_xyz(data_root, "2020-08-04-course", "slac", apply_offset=False)
    with_offset = read_base_station_xyz(data_root, "2020-08-04-course", "slac")

    np.testing.assert_allclose(no_offset, np.array([-2700112.7, -4292747.3, 3855195.5]))
    assert np.linalg.norm(with_offset - no_offset) > 1.0


def test_rinex_header_base_xyz_preferred_when_valid():
    metadata_xyz = np.array([1.0, 2.0, 3.0])
    header_xyz = np.array([-2703115.266, -4291768.344, 3854247.955])
    base_obs = SimpleNamespace(header=SimpleNamespace(approx_position=header_xyz))

    selected_xyz = rinex_header_base_xyz_or_metadata(base_obs, metadata_xyz)

    np.testing.assert_allclose(selected_xyz, header_xyz)


def test_rinex_header_base_xyz_falls_back_to_metadata_when_missing_or_invalid():
    metadata_xyz = np.array([-2700729.3481, -4293104.2572, 3854473.9477])

    missing_header = SimpleNamespace()
    small_header = SimpleNamespace(header=SimpleNamespace(approx_position=np.array([1.0, 2.0, 3.0])))
    nan_header = SimpleNamespace(
        header=SimpleNamespace(approx_position=np.array([np.nan, -4291768.344, 3854247.955])),
    )

    for base_obs in [missing_header, small_header, nan_header]:
        selected_xyz = rinex_header_base_xyz_or_metadata(base_obs, metadata_xyz)
        np.testing.assert_allclose(selected_xyz, metadata_xyz)


def test_base_time_span_mask_direct():
    base_times = np.arange(0.0, 181.0, 30.0)

    mask = matlab_base_time_span_mask(
        base_times,
        phone_start_gps_s=209.0,
        phone_end_gps_s=239.0,
        base_dt_s=30.0,
    )

    np.testing.assert_array_equal(mask, np.array([False, True, True, True, True, True, True]))


def test_compute_base_correction_matrix_uses_injected_dependencies(tmp_path):
    data_root = tmp_path / "dataset_2023"
    times_ms = np.array([1000.0, 2000.0], dtype=np.float64)
    phone_times = unix_ms_to_gps_abs_seconds(times_ms)
    calls: list[tuple[str, tuple[str, ...]]] = []

    def fake_setting(root, split, course, phone):
        assert root == data_root
        assert (split, course, phone) == ("train", "course", "phone")
        return "BASE", "V3"

    def fake_span(root, split, course, phone, selected_phone_times_gps_s):
        assert root == data_root
        assert (split, course, phone) == ("train", "course", "phone")
        np.testing.assert_allclose(selected_phone_times_gps_s, phone_times)
        return float(phone_times[0]), float(phone_times[-1])

    def fake_load_base_residuals(
        data_root_str,
        split,
        course,
        base_name,
        rinex_type,
        signal_type,
        phone_start_gps_s,
        phone_end_gps_s,
        sat_ids_key,
    ):
        assert data_root_str == str(data_root)
        assert (split, course, base_name, rinex_type) == ("train", "course", "BASE", "V3")
        assert np.isclose(phone_start_gps_s, phone_times[0])
        assert np.isclose(phone_end_gps_s, phone_times[-1])
        calls.append((signal_type, sat_ids_key))
        residuals = np.zeros((phone_times.size, len(sat_ids_key)), dtype=np.float64)
        for col, sat_id in enumerate(sat_ids_key):
            residuals[:, col] = 10.0 * (col + 1) + len(str(sat_id))
        return phone_times, sat_ids_key, residuals

    correction = compute_base_pseudorange_correction_matrix(
        data_root,
        "train/course/phone",
        times_ms,
        [
            (1, 3, "GPS_L1_CA"),
            (6, 11, "GAL_E1_C_P"),
            (4, 193, "QZS_L1_CA"),
            (3, 1, "BDS_B1I"),
        ],
        "GPS_L1_CA",
        base_setting_fn=fake_setting,
        base_residual_loader=fake_load_base_residuals,
        phone_span_fn=fake_span,
    )

    assert {call[1] for call in calls} == {("G03",), ("E11",), ("J01",)}
    np.testing.assert_allclose(correction[:, 0], np.array([13.0, 13.0]))
    np.testing.assert_allclose(correction[:, 1], np.array([13.0, 13.0]))
    np.testing.assert_allclose(correction[:, 2], np.array([13.0, 13.0]))
    assert np.isnan(correction[:, 3]).all()
