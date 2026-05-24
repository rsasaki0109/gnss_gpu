"""Tests for ``experiments.gsdc2023_per_type_kernel``.

Pins the per-Type / per-phone Huber + motion-sigma table to taroz
``parameters.m`` values and exercises the settings.csv lookup path.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from experiments.gsdc2023_per_type_kernel import (
    CARRIER_HUBER_K_BY_TYPE,
    DOPPLER_HUBER_K_BY_TYPE,
    DOPPLER_HUBER_K_PHONE_OVERRIDES,
    MOTION_SIGMA_M_BY_TYPE,
    MOTION_SIGMA_PHONE_OVERRIDES,
    PR_HUBER_K_BY_TYPE,
    TRIP_TYPE_HIGHWAY,
    TRIP_TYPE_MIX,
    TRIP_TYPE_STREET,
    load_settings_lookup,
    per_type_kernel_for,
    trip_type_from_data_root,
)


def test_pr_huber_k_table_matches_taroz_parameters_m():
    assert PR_HUBER_K_BY_TYPE[TRIP_TYPE_STREET] == pytest.approx(0.1)
    assert PR_HUBER_K_BY_TYPE[TRIP_TYPE_HIGHWAY] == pytest.approx(0.2)
    assert PR_HUBER_K_BY_TYPE[TRIP_TYPE_MIX] == pytest.approx(0.1)


def test_doppler_huber_k_table_matches_taroz():
    assert DOPPLER_HUBER_K_BY_TYPE[TRIP_TYPE_STREET] == pytest.approx(0.4)
    assert DOPPLER_HUBER_K_BY_TYPE[TRIP_TYPE_HIGHWAY] == pytest.approx(0.8)
    assert DOPPLER_HUBER_K_BY_TYPE[TRIP_TYPE_MIX] == pytest.approx(0.4)


def test_carrier_huber_k_table_matches_taroz():
    assert CARRIER_HUBER_K_BY_TYPE[TRIP_TYPE_STREET] == pytest.approx(0.2)
    assert CARRIER_HUBER_K_BY_TYPE[TRIP_TYPE_HIGHWAY] == pytest.approx(0.5)


def test_motion_sigma_table_matches_taroz():
    assert MOTION_SIGMA_M_BY_TYPE[TRIP_TYPE_STREET] == pytest.approx(0.05)
    assert MOTION_SIGMA_M_BY_TYPE[TRIP_TYPE_HIGHWAY] == pytest.approx(0.01)
    assert MOTION_SIGMA_M_BY_TYPE[TRIP_TYPE_MIX] == pytest.approx(0.01)


def test_phone_overrides_table_matches_taroz():
    assert DOPPLER_HUBER_K_PHONE_OVERRIDES["pixel4"] == pytest.approx(0.2)
    assert MOTION_SIGMA_PHONE_OVERRIDES["mi8"] == pytest.approx(0.1)
    assert MOTION_SIGMA_PHONE_OVERRIDES["xiaomimi8"] == pytest.approx(0.1)


def test_per_type_kernel_for_highway_default():
    k = per_type_kernel_for(TRIP_TYPE_HIGHWAY, phone="pixel5")
    assert k.pr_huber_k == pytest.approx(0.2)
    assert k.doppler_huber_k == pytest.approx(0.8)
    assert k.carrier_huber_k == pytest.approx(0.5)
    assert k.motion_sigma_m == pytest.approx(0.01)


def test_per_type_kernel_for_street():
    k = per_type_kernel_for(TRIP_TYPE_STREET, phone="pixel5")
    assert k.pr_huber_k == pytest.approx(0.1)
    assert k.doppler_huber_k == pytest.approx(0.4)
    assert k.motion_sigma_m == pytest.approx(0.05)


def test_per_type_kernel_mi8_overrides_motion_sigma():
    k_highway_mi8 = per_type_kernel_for(TRIP_TYPE_HIGHWAY, phone="mi8")
    assert k_highway_mi8.motion_sigma_m == pytest.approx(0.1)  # mi8 override.
    assert k_highway_mi8.pr_huber_k == pytest.approx(0.2)  # Highway PR is untouched.


def test_per_type_kernel_pixel4_overrides_doppler():
    k = per_type_kernel_for(TRIP_TYPE_HIGHWAY, phone="pixel4")
    assert k.doppler_huber_k == pytest.approx(0.2)  # pixel4 override.
    assert k.pr_huber_k == pytest.approx(0.2)


def test_per_type_kernel_unknown_type_falls_back_to_highway():
    k = per_type_kernel_for("UnknownType", phone="pixel5")
    assert k.pr_huber_k == PR_HUBER_K_BY_TYPE[TRIP_TYPE_HIGHWAY]
    assert k.doppler_huber_k == DOPPLER_HUBER_K_BY_TYPE[TRIP_TYPE_HIGHWAY]
    assert k.motion_sigma_m == MOTION_SIGMA_M_BY_TYPE[TRIP_TYPE_HIGHWAY]


def test_load_settings_lookup_handles_minimal_table(tmp_path: Path):
    csv = tmp_path / "settings.csv"
    pd.DataFrame({
        "Course": ["2021-01-04-21-50-us-ca-e1highway280driveroutea", "2022-08-04-20-07-us-ca-sjc-q"],
        "Phone": ["mi8", "pixel5"],
        "Type": ["Highway", "Highway"],
        "Other": [1, 2],
    }).to_csv(csv, index=False)
    lookup = load_settings_lookup(csv)
    assert lookup[("2021-01-04-21-50-us-ca-e1highway280driveroutea", "mi8")] == "Highway"
    assert lookup[("2022-08-04-20-07-us-ca-sjc-q", "pixel5")] == "Highway"


def test_load_settings_lookup_rejects_missing_columns(tmp_path: Path):
    csv = tmp_path / "bad.csv"
    pd.DataFrame({"Course": ["x"], "Other": [1]}).to_csv(csv, index=False)
    with pytest.raises(ValueError):
        load_settings_lookup(csv)


def test_trip_type_from_data_root_uses_settings_csv(tmp_path: Path):
    (tmp_path / "settings_train.csv").write_text(
        "Course,Phone,Type\nA,pixel5,Street\nB,mi8,Highway\n"
    )
    assert trip_type_from_data_root(tmp_path, "train/A/pixel5") == "Street"
    assert trip_type_from_data_root(tmp_path, "train/B/mi8") == "Highway"


def test_trip_type_from_data_root_falls_back_when_missing(tmp_path: Path):
    # No settings file present -> fallback.
    assert trip_type_from_data_root(tmp_path, "train/X/pixel5") == TRIP_TYPE_HIGHWAY
    # Trip not in table -> fallback.
    (tmp_path / "settings_train.csv").write_text("Course,Phone,Type\nA,pixel5,Street\n")
    assert trip_type_from_data_root(tmp_path, "train/Z/pixel5") == TRIP_TYPE_HIGHWAY


def test_trip_type_from_data_root_uses_test_split(tmp_path: Path):
    (tmp_path / "settings_test.csv").write_text("Course,Phone,Type\nT,pixel5,Mix\n")
    assert trip_type_from_data_root(tmp_path, "test/T/pixel5") == "Mix"
