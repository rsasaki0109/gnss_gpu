#!/usr/bin/env python3
"""Build a truth-free median DDPR-FDE offset seed against a frozen trajectory."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def build(anchor_csv: Path, trajectory_csv: Path, *, segment: tuple[int, int]) -> dict:
    with trajectory_csv.open(newline="", encoding="utf-8-sig") as fh:
        trajectory = list(csv.DictReader(fh))
    start, end = segment
    if not 0 <= start < end <= len(trajectory):
        raise ValueError("seed segment is outside the trajectory")
    by_tow = {
        round(float(row["tow"]), 1): np.asarray(
            [float(row["ecef_x"]), float(row["ecef_y"]), float(row["ecef_z"])]
        )
        for row in trajectory[start:end]
    }
    offsets = []
    with anchor_csv.open(newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            if row.get("status") != "accepted":
                continue
            position = by_tow.get(round(float(row["tow"]), 1))
            if position is None:
                continue
            anchor = np.asarray(
                [float(row["anchor_x_m"]), float(row["anchor_y_m"]), float(row["anchor_z_m"])]
            )
            if np.isfinite(anchor).all():
                offsets.append(anchor - position)
    if len(offsets) < 3:
        raise ValueError("fewer than three accepted FDE anchors match the trajectory")
    seed = np.median(np.asarray(offsets), axis=0)
    return {
        "schema": "wp61_ddpr_fde_block_seed_v1",
        "production_input_truth": False,
        "truth_usage": "none",
        "runtime_network_required": False,
        "segment": [start, end],
        "source": "median accepted one-row-exclusion DDPR-FDE anchor minus frozen trajectory",
        "accepted_anchor_epochs": len(offsets),
        "seeds": [{"offset_ecef_m": seed.tolist()}],
        "input_sha256": {
            "anchor_csv": hashlib.sha256(anchor_csv.read_bytes()).hexdigest(),
            "trajectory": hashlib.sha256(trajectory_csv.read_bytes()).hexdigest(),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("anchor_csv", type=Path)
    parser.add_argument("trajectory_csv", type=Path)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.anchor_csv, args.trajectory_csv, segment=(args.start, args.end))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
