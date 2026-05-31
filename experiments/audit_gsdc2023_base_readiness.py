#!/usr/bin/env python3
"""Audit GSDC2023 base-station readiness for carrier/DD candidate work."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from experiments.gsdc2023_raw_bridge import DEFAULT_ROOT, collect_matlab_parity_audit  # noqa: E402


DEFAULT_OUTPUT = Path("experiments/results/gsdc2023_base_readiness_20260519.csv")


def discover_gsdc_trips(data_root: Path, splits: tuple[str, ...] = ("train", "test")) -> list[str]:
    trips: list[str] = []
    for split in splits:
        split_dir = data_root / split
        if not split_dir.is_dir():
            continue
        for course_dir in sorted(split_dir.iterdir()):
            if not course_dir.is_dir() or course_dir.name.startswith("."):
                continue
            for phone_dir in sorted(course_dir.iterdir()):
                if not phone_dir.is_dir() or phone_dir.name.startswith("."):
                    continue
                if (phone_dir / "device_gnss.csv").is_file():
                    trips.append(f"{split}/{course_dir.name}/{phone_dir.name}")
    return trips


def next_action_for_status(status: str) -> str:
    return {
        "settings_csv_missing": "generate_settings",
        "setting_row_missing": "repair_settings_rows",
        "base1_missing": "assign_or_merge_base1",
        "base_metadata_missing": "generate_base_metadata",
        "base_obs_missing": "fetch_base_obs",
        "broadcast_nav_missing": "fetch_broadcast_nav",
        "base_correction_ready": "ready",
    }.get(str(status), "inspect")


def base_readiness_row(data_root: Path, trip: str) -> dict[str, object]:
    audit = collect_matlab_parity_audit(data_root, trip, include_imu_sync=False)
    expected_base_obs = audit.get("expected_base_obs")
    base_obs_size_bytes = 0
    if expected_base_obs:
        path = Path(str(expected_base_obs))
        if path.is_file():
            base_obs_size_bytes = int(path.stat().st_size)

    status = str(audit.get("base_correction_status", "unknown"))
    return {
        "trip": trip,
        "split": audit.get("dataset_split"),
        "course": audit.get("course"),
        "phone": audit.get("phone"),
        "status": status,
        "ready": bool(audit.get("base_correction_ready", False)),
        "next_action": next_action_for_status(status),
        "settings_csv_present": bool(audit.get("settings_csv_present", False)),
        "setting_row_present": bool(audit.get("setting_row_present", False)),
        "base_name": audit.get("base_name"),
        "rinex_type": audit.get("rinex_type"),
        "base_position_csv_present": bool(audit.get("base_position_csv_present", False)),
        "base_offset_csv_present": bool(audit.get("base_offset_csv_present", False)),
        "expected_base_obs": expected_base_obs,
        "base_obs_file_present": bool(audit.get("base_obs_file_present", False)),
        "base_obs_size_bytes": base_obs_size_bytes,
        "broadcast_nav_present": bool(audit.get("broadcast_nav_present", False)),
        "ground_truth_present": bool(audit.get("ground_truth_present", False)),
        "ref_height_present": bool(audit.get("ref_height_present", False)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--split", action="append", choices=("train", "test"), default=[])
    parser.add_argument("--trip", action="append", default=[], help="train/.../phone or test/.../phone; repeatable")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    splits = tuple(args.split) if args.split else ("train", "test")
    trips = discover_gsdc_trips(args.data_root, splits=splits)
    if args.trip:
        wanted = set(args.trip)
        trips = [trip for trip in trips if trip in wanted]
    if args.limit > 0:
        trips = trips[: args.limit]
    if not trips:
        raise RuntimeError("no GSDC trips found")

    rows = [base_readiness_row(args.data_root, trip) for trip in trips]
    frame = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"wrote: {args.output}")
    print(f"trips: {len(frame)} ready: {int(frame['ready'].sum())}")
    print(frame["status"].value_counts().to_string())


if __name__ == "__main__":
    main()
