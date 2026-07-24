#!/usr/bin/env python3
"""Apply a frozen WP87 fixed-boundary affine selection for shadow audit only."""

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

from gnss_gpu.io.ppc import PPCDatasetLoader  # noqa: E402


def apply_fixed_affine_profile(
    positions: np.ndarray,
    *,
    start: int,
    end: int,
    start_offset: np.ndarray,
    boundary_offset: np.ndarray,
) -> np.ndarray:
    output = np.asarray(positions, dtype=np.float64).copy()
    left = np.asarray(start_offset, dtype=np.float64).reshape(3)
    right = np.asarray(boundary_offset, dtype=np.float64).reshape(3)
    if (
        output.ndim != 2
        or output.shape[1] != 3
        or not 0 <= start < end < len(output)
        or not np.isfinite(left).all()
        or not np.isfinite(right).all()
    ):
        raise ValueError("fixed affine profile inputs are invalid")
    scales = np.asarray([(end - epoch) / (end - start) for epoch in range(start, end)])
    output[start:end] += scales[:, None] * left + (1.0 - scales[:, None]) * right
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("selection", type=Path)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-trajectory", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    args = parser.parse_args()

    selection_bytes = args.selection.read_bytes()
    selection = json.loads(selection_bytes)
    model = selection.get("offset_model", {})
    if (
        bool(selection.get("production_input_truth", True))
        or not bool(selection.get("accepted", False))
        or selection.get("reason") != "unique_singlebasis_road_carrier_cppr_mode"
        or model.get("mode") != "right_boundary_affine_fixed"
    ):
        raise RuntimeError("WP87 selection is not an accepted truth-free profile")
    rows = list(csv.DictReader(args.trajectory.open(newline="", encoding="utf-8-sig")))
    positions = np.asarray(
        [[float(row[key]) for key in ("ecef_x", "ecef_y", "ecef_z")] for row in rows]
    )
    epochs = np.asarray([int(row["epoch"]) for row in rows])
    if not np.array_equal(epochs, np.arange(len(rows))):
        raise RuntimeError("trajectory epochs are not contiguous from zero")
    start, end = (int(value) for value in selection["segment"])
    if int(model["boundary_epoch"]) != end:
        raise RuntimeError("fixed affine boundary is not the segment end")
    output = apply_fixed_affine_profile(
        positions,
        start=start,
        end=end,
        start_offset=np.asarray(selection["selected_profile"]["offset_ecef_m"]),
        boundary_offset=np.asarray(model["boundary_offset_ecef_m"]),
    )

    # The selected profile is immutable above. Truth is loaded only for audit.
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
        "schema": "wp87_fixed_affine_singlebasis_shadow_audit_v1",
        "production_input_truth": False,
        "truth_usage": "post_selection_full_denominator_audit_only",
        "production_promoted": False,
        "full_denominator_epochs": len(rows),
        "segment": [start, end],
        "before_sub50cm_epochs": int(np.count_nonzero(before)),
        "after_sub50cm_epochs": int(np.count_nonzero(after)),
        "after_sub50cm_pct": float(100.0 * np.mean(after)),
        "gained_epochs": int(np.count_nonzero(after & ~before)),
        "lost_epochs": int(np.count_nonzero(before & ~after)),
        "segment_after_sub50cm_epochs": int(np.count_nonzero(after[start:end])),
        "input_sha256": {
            "trajectory": hashlib.sha256(args.trajectory.read_bytes()).hexdigest(),
            "selection": hashlib.sha256(selection_bytes).hexdigest(),
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
