from __future__ import annotations

import csv
from pathlib import Path

import pytest

from experiments.audit_ppc_imu_fgo_health import audit_health


def _write_shadow(path: Path, values: list[float]) -> None:
    fields = (
        "gps_week",
        "tow",
        "imu_fgo_available",
        "imu_fgo_fault_reason",
        "imu_fgo_recovery_epochs",
        "imu_fgo_factor_nis_per_dof",
        "imu_fgo_pose_correction_m",
        "imu_fgo_accel_bias_step_mps2",
        "imu_fgo_gyro_bias_step_radps",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, value in enumerate(values):
            writer.writerow(
                {
                    "gps_week": 2325,
                    "tow": 10.0 + index,
                    "imu_fgo_available": 1,
                    "imu_fgo_fault_reason": "ok",
                    "imu_fgo_recovery_epochs": 0,
                    "imu_fgo_factor_nis_per_dof": value,
                    "imu_fgo_pose_correction_m": value / 10,
                    "imu_fgo_accel_bias_step_mps2": value / 100,
                    "imu_fgo_gyro_bias_step_radps": value / 1000,
                }
            )


def test_health_audit_uses_truth_free_clean_distribution(tmp_path: Path) -> None:
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    _write_shadow(first, [1.0, 2.0, 3.0, 4.0])
    _write_shadow(second, [2.0, 3.0, 4.0, 5.0])
    result = audit_health({"a": first, "b": second}, rolling_window=3)
    assert result["truth_usage"] == "none"
    assert result["combined"]["clean_complete"] is True
    assert result["combined"]["nis_per_dof"]["count"] == 8
    assert result["provisional_monitor"]["promotion_ready"] is False
    assert result["provisional_monitor"]["threshold_nis_per_dof"] == 5.0


def test_health_audit_rejects_non_increasing_time(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    _write_shadow(path, [1.0, 2.0])
    rows = path.read_text(encoding="utf-8").splitlines()
    rows[-1] = rows[-1].replace("11.0", "10.0")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="non-increasing"):
        audit_health({"bad": path}, rolling_window=2)


def test_health_audit_does_not_roll_across_unavailable_gap(tmp_path: Path) -> None:
    path = tmp_path / "gap.csv"
    _write_shadow(path, [1.0, 2.0, 100.0, 4.0, 5.0, 6.0])
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
        fields = tuple(rows[0])
    rows[2]["imu_fgo_available"] = "0"
    rows[2]["imu_fgo_fault_reason"] = "imu_coverage_gap"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    result = audit_health({"gap": path}, rolling_window=3)
    route = result["routes"]["gap"]
    assert route["clean_nis_segments"] == 2
    assert route["nis_per_dof"]["count"] == 5
    assert route["rolling_median_nis_per_dof"]["count"] == 1
    assert route["rolling_median_nis_per_dof"]["maximum"] == 5.0
