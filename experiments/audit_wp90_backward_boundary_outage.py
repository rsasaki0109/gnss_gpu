#!/usr/bin/env python3
"""Audit one frozen predecessor outage using an independent promoted boundary."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "python"))
sys.path.insert(0, str(_ROOT / "experiments"))

from apply_wp42_moving_block_offset import apply_offset  # noqa: E402
from gnss_gpu.io.ppc import PPCDatasetLoader  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("boundary_promotion", type=Path)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--block-length", type=int, default=55)
    parser.add_argument("--output-trajectory", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    args = parser.parse_args()

    promotion_bytes = args.boundary_promotion.read_bytes()
    promotion = json.loads(promotion_bytes)
    if (
        bool(promotion.get("production_input_truth", True))
        or not bool(promotion.get("production_promoted", False))
        or promotion.get("reason") != "unique_singlebasis_road_carrier_cppr_mode"
    ):
        raise RuntimeError("outage boundary is not an independent truth-free promotion")
    boundary = int(promotion["segment"][0])
    start, end = boundary - int(args.block_length), boundary
    if start < 0 or args.block_length <= 0:
        raise ValueError("backward outage block is invalid")

    rows = list(csv.DictReader(args.trajectory.open(newline="", encoding="utf-8-sig")))
    epochs = np.asarray([int(row["epoch"]) for row in rows])
    if not np.array_equal(epochs, np.arange(len(rows))):
        raise RuntimeError("trajectory epochs are not contiguous from zero")
    positions = np.asarray(
        [[float(row[key]) for key in ("ecef_x", "ecef_y", "ecef_z")] for row in rows]
    )
    output = apply_offset(
        positions,
        start=start,
        end=end,
        offset=np.asarray(promotion["offset_ecef_m"]),
    )

    # The outage profile is frozen above. Truth is loaded only for shadow audit.
    truth_times, truth_positions = PPCDatasetLoader(args.data_dir).load_ground_truth()
    times = np.asarray([float(row["tow"]) for row in rows])
    truth = np.asarray(
        [truth_positions[int(np.argmin(np.abs(truth_times - tow)))] for tow in times]
    )
    before_errors = np.linalg.norm(positions - truth, axis=1)
    after_errors = np.linalg.norm(output - truth, axis=1)
    before = before_errors < 0.5
    after = after_errors < 0.5
    for index, row in enumerate(rows):
        row["ecef_x"], row["ecef_y"], row["ecef_z"] = (
            repr(float(value)) for value in output[index]
        )
        row["error_m"] = repr(float(after_errors[index]))
        row["sub50cm"] = str(int(after[index]))
    args.output_trajectory.parent.mkdir(parents=True, exist_ok=True)
    with args.output_trajectory.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "schema": "wp90_backward_boundary_outage_shadow_audit_v1",
        "production_input_truth": False,
        "truth_usage": "post_selection_full_denominator_audit_only",
        "production_promoted": False,
        "full_denominator_epochs": len(rows),
        "segment": [start, end],
        "boundary_epoch": boundary,
        "boundary_offset_ecef_m": promotion["offset_ecef_m"],
        "before_sub50cm_epochs": int(np.count_nonzero(before)),
        "after_sub50cm_epochs": int(np.count_nonzero(after)),
        "after_sub50cm_pct": float(100.0 * np.mean(after)),
        "gained_epochs": int(np.count_nonzero(after & ~before)),
        "lost_epochs": int(np.count_nonzero(before & ~after)),
        "segment_after_sub50cm_epochs": int(np.count_nonzero(after[start:end])),
        "input_sha256": {
            "trajectory": hashlib.sha256(args.trajectory.read_bytes()).hexdigest(),
            "boundary_promotion": hashlib.sha256(promotion_bytes).hexdigest(),
        },
    }
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
