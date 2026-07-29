#!/usr/bin/env python3
"""Export a PF trajectory CSV as a gnssplusplus per-epoch ECEF seed file."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def export_seed_pos(input_path: Path, output_path: Path, *, gps_week: int) -> int:
    with input_path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    required = {"tow", "ecef_x", "ecef_y", "ecef_z"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"trajectory must contain {sorted(required)}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("% gps_week tow_s ecef_x_m ecef_y_m ecef_z_m\n")
        for row in rows:
            stream.write(
                f"{gps_week:d} {float(row['tow']):.9f} "
                f"{float(row['ecef_x']):.9f} {float(row['ecef_y']):.9f} "
                f"{float(row['ecef_z']):.9f}\n"
            )
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--gps-week", type=int, required=True)
    args = parser.parse_args()
    count = export_seed_pos(args.input, args.output, gps_week=args.gps_week)
    print(f"exported {count} PF seed rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
