from __future__ import annotations

from pathlib import Path

import pandas as pd

from experiments.audit_gsdc2023_base_readiness import (
    base_readiness_row,
    discover_gsdc_trips,
    next_action_for_status,
)


def test_discover_gsdc_trips_requires_device_gnss(tmp_path: Path) -> None:
    data_root = tmp_path / "sdc2023"
    good = data_root / "train" / "course-a" / "pixel5"
    bad = data_root / "train" / "course-b" / "pixel5"
    good.mkdir(parents=True)
    bad.mkdir(parents=True)
    (good / "device_gnss.csv").write_text("utcTimeMillis\n1\n", encoding="utf-8")

    assert discover_gsdc_trips(data_root, splits=("train",)) == ["train/course-a/pixel5"]


def test_next_action_for_status_maps_known_blockers() -> None:
    assert next_action_for_status("settings_csv_missing") == "generate_settings"
    assert next_action_for_status("base1_missing") == "assign_or_merge_base1"
    assert next_action_for_status("base_correction_ready") == "ready"
    assert next_action_for_status("new_status") == "inspect"


def test_base_readiness_row_reports_ready_base_files(tmp_path: Path) -> None:
    data_root = tmp_path / "gsdc2023" / "sdc2023"
    trip_dir = data_root / "train" / "2023-05-25-19-10-us-ca-sjc-be2" / "pixel5"
    trip_dir.mkdir(parents=True)
    (trip_dir / "device_gnss.csv").write_text("utcTimeMillis\n1\n", encoding="utf-8")
    (trip_dir / "ground_truth.csv").write_text("LatitudeDegrees,LongitudeDegrees\n0,0\n", encoding="utf-8")
    (trip_dir.parent / "brdc.23n").write_text("", encoding="utf-8")
    base_obs = trip_dir.parent / "SLAC_rnx3.obs"
    base_obs.write_text("rinex\n", encoding="utf-8")
    pd.DataFrame(
        [
            {
                "Course": "2023-05-25-19-10-us-ca-sjc-be2",
                "Phone": "pixel5",
                "Base1": "SLAC",
                "RINEX": "V3",
            },
        ],
    ).to_csv(data_root / "settings_train.csv", index=False)
    base_dir = data_root.parent.parent / "base"
    base_dir.mkdir(parents=True)
    (base_dir / "base_position.csv").write_text("Base,Year,X,Y,Z\nSLAC,2023,1,2,3\n", encoding="utf-8")
    (base_dir / "base_offset.csv").write_text("Base,E,N,U\nSLAC,0,0,0\n", encoding="utf-8")

    row = base_readiness_row(data_root, "train/2023-05-25-19-10-us-ca-sjc-be2/pixel5")

    assert row["status"] == "base_correction_ready"
    assert row["ready"] is True
    assert row["next_action"] == "ready"
    assert row["base_obs_file_present"] is True
    assert row["base_obs_size_bytes"] == base_obs.stat().st_size
