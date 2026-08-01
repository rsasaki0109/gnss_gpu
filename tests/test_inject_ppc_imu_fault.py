from __future__ import annotations

import math

import pytest

from experiments.inject_ppc_imu_fault import inject_imu_fault


def _rows() -> list[dict[str, float | int]]:
    return [
        dict(week=2325, tow=10.0 + 0.01 * index, ax=1, ay=2, az=3,
             gx=4, gy=5, gz=6)
        for index in range(5)
    ]


def test_dropout_and_time_offset_are_bounded_and_deterministic() -> None:
    dropped, affected = inject_imu_fault(
        _rows(), fault="dropout", start_tow=10.01, end_tow=10.02
    )
    assert affected == 2
    assert [row["tow"] for row in dropped] == [10.0, 10.03, 10.04]

    shifted, affected = inject_imu_fault(
        _rows(), fault="time_offset", start_tow=10.02, end_tow=10.02,
        time_offset_s=0.005,
    )
    assert affected == 1
    assert [row["tow"] for row in shifted] == [10.0, 10.01, 10.025, 10.03, 10.04]


def test_bias_jump_and_vibration_change_only_selected_components() -> None:
    biased, affected = inject_imu_fault(
        _rows(), fault="bias_jump", start_tow=10.01, end_tow=10.01,
        accel_bias_mps2=(0.1, 0.2, 0.3), gyro_bias_degps=(1, 2, 3),
    )
    assert affected == 1
    assert [biased[1][key] for key in ("ax", "ay", "az")] == [1.1, 2.2, 3.3]
    assert [biased[1][key] for key in ("gx", "gy", "gz")] == [5.0, 7.0, 9.0]

    vibrated, _ = inject_imu_fault(
        _rows(), fault="vibration", start_tow=10.0, end_tow=10.04,
        vibration_frequency_hz=25.0,
        vibration_accel_mps2=(2, 0, 0), vibration_gyro_degps=(0, 0, 3),
    )
    assert vibrated[0]["ax"] == 1.0
    assert vibrated[1]["ax"] == pytest.approx(3.0)
    assert vibrated[1]["gz"] == pytest.approx(9.0)
    assert math.isfinite(float(vibrated[-1]["ax"]))


def test_fault_rejects_time_collision() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        inject_imu_fault(
            _rows(), fault="time_offset", start_tow=10.01, end_tow=10.01,
            time_offset_s=0.01,
        )
