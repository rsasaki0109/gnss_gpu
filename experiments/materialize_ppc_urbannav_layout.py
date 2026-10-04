#!/usr/bin/env python3
"""Materialize PPC-Dataset runs in the UrbanNav layout read by exp_pf_smoother_eval.

PPC run directories hold rover.obs / base.obs / base.nav / reference.csv /
imu.csv. The PF smoother expects rover_trimble.obs / base_trimble.obs and an
UrbanNav-style imu.csv. The IMU differs in two ways, checked against the
reference heading rate (UrbanNav +0.965, PPC -0.0169 per gyro unit):

- PPC is z-up, UrbanNav z-down: rotate 180 deg about x (negate y and z).
- PPC gyro is in deg/s, UrbanNav in rad/s.

PPC has no wheel odometer, so the wheel column is left empty; run the PF with
``--imu-speed-source doppler`` (preset ``rtk_anchored_doppler``).

Usage:
    python experiments/materialize_ppc_urbannav_layout.py \\
        --ppc-root E:/datasets/PPC-Dataset-data --out-root <dir>
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
from pathlib import Path

RUNS = tuple(f"{city}/run{i}" for city in ("tokyo", "nagoya") for i in (1, 2, 3))
COPIES = (
    ("rover.obs", "rover_trimble.obs"),
    ("base.obs", "base_trimble.obs"),
    ("base.nav", "base.nav"),
    ("reference.csv", "reference.csv"),
)
IMU_HEADER = (
    "GPS TOW (s)",
    "GPS Week",
    "Acceleration X (m/s^2)",
    "Acceleration Y (m/s^2)",
    "Acceleration Z (m/s^2)",
    "Angular rate X (rad/s)",
    "Angular rate Y (rad/s)",
    "Angular rate Z (rad/s)",
    "Wheel velocity (m/s)",
)


def convert_imu_row(row: list[str]) -> list[str]:
    """PPC imu.csv row (z-up, deg/s) -> UrbanNav row (z-down, rad/s, no wheel)."""

    tow, week, ax, ay, az, gx, gy, gz = (float(v) for v in row[:8])
    return [
        f"{tow:.3f}",
        str(int(week)),
        f"{ax:.6f}",
        f"{-ay:.6f}",
        f"{-az:.6f}",
        f"{math.radians(gx):.9f}",
        f"{-math.radians(gy):.9f}",
        f"{-math.radians(gz):.9f}",
        "",
    ]


def materialize_run(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for name, target in COPIES:
        if not (dst / target).exists():
            shutil.copyfile(src / name, dst / target)
    with (src / "imu.csv").open(newline="") as fin, (dst / "imu.csv").open(
        "w", newline=""
    ) as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout)
        next(reader)
        writer.writerow(IMU_HEADER)
        for row in reader:
            if len(row) >= 8:
                writer.writerow(convert_imu_row(row))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ppc-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args(argv)
    for run in RUNS:
        dst = args.out_root / run.replace("/", "_")
        materialize_run(args.ppc_root / run, dst)
        print(f"materialized {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
