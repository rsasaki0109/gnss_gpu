#!/usr/bin/env python3
"""Bridge from the final assignment mode jump to an accepted static endpoint."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

_XYZ = ("ecef_x", "ecef_y", "ecef_z")


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def final_mode_jump_epoch(
    selected_rows: list[dict[str, str]],
    *,
    start: int,
    end: int,
    jump_residual_m: float = 2.0,
    min_tail_anchors: int = 10,
) -> int:
    """Return the final large incoming transition with enough anchor tail."""

    rows = sorted(
        (
            row
            for row in selected_rows
            if int(row["selected"]) == 1 and start < int(row["epoch"]) < end
        ),
        key=lambda row: int(row["epoch"]),
    )
    jumps = [
        (index, int(row["epoch"]))
        for index, row in enumerate(rows)
        if float(row["previous_selected_transition_residual_m"]) >= jump_residual_m
    ]
    if not jumps:
        raise RuntimeError("route segment contains no assignment mode jump")
    index, epoch = jumps[-1]
    if len(rows) - index < int(min_tail_anchors):
        raise RuntimeError("final assignment jump has insufficient route tail anchors")
    return epoch


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("alternate_route", type=Path)
    parser.add_argument("candidate_audit", type=Path)
    parser.add_argument("route_summary", type=Path)
    parser.add_argument("late_anchor_result", type=Path)
    parser.add_argument("--jump-residual-m", type=float, default=2.0)
    parser.add_argument("--min-tail-anchors", type=int, default=10)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    parser.add_argument("--out-trajectory", type=Path, required=True)
    args = parser.parse_args()

    trajectory = _read(args.trajectory)
    route_rows = _read(args.alternate_route)
    route = {int(row["epoch"]): row for row in route_rows}
    route_result = json.loads(args.route_summary.read_text(encoding="utf-8"))
    late_result = json.loads(args.late_anchor_result.read_text(encoding="utf-8"))
    if route_result.get("anchor_mode") != "seed-support":
        raise RuntimeError("post-jump bridge requires a seed-support route anchor")
    if late_result.get("reason") != "temporal_widelane_consensus":
        raise RuntimeError("post-jump bridge endpoint lacks accepted static evidence")
    if float(route_result.get("endpoint_error_m", np.inf)) > 1.0e-6:
        raise RuntimeError("alternate route does not close its static endpoint")
    if float(route_result.get("doppler_heading_p95_deg", np.inf)) > 15.0:
        raise RuntimeError("alternate route fails gyro/Doppler heading coherence")
    if not 0.8 <= float(route_result.get("speed_scale", np.nan)) <= 1.2:
        raise RuntimeError("alternate route speed scale exceeds gate")
    segment_start, segment_end = (int(value) for value in route_result["segment"])
    expected_epochs = list(range(segment_start, segment_end))
    if sorted(route) != expected_epochs:
        raise RuntimeError("alternate route coverage differs from declared segment")
    jump_epoch = final_mode_jump_epoch(
        _read(args.candidate_audit),
        start=segment_start,
        end=segment_end,
        jump_residual_m=float(args.jump_residual_m),
        min_tail_anchors=int(args.min_tail_anchors),
    )

    from gnss_gpu.io.ppc import PPCDatasetLoader

    truth_times, truth_positions = PPCDatasetLoader(args.data_dir).load_ground_truth()
    output = []
    applied = set(range(jump_epoch, segment_end))
    for row in trajectory:
        epoch = int(row["epoch"])
        use_route = epoch in applied
        position_row = route[epoch] if use_route else row
        position = np.asarray([float(position_row[key]) for key in _XYZ])
        truth = truth_positions[int(np.argmin(np.abs(truth_times - float(row["tow"]))))]
        error = float(np.linalg.norm(position - truth))
        fix = int(row.get("fix", "0"))
        output.append(
            {
                **row,
                **{key: float(position[index]) for index, key in enumerate(_XYZ)},
                "source": "post_jump_imu_route_bridge" if use_route else row.get("source", ""),
                "error_m": error,
                "sub50cm": int(error < 0.5),
                "false_fix": int(bool(fix) and error >= 0.5),
                "post_jump_route_applied": int(use_route),
            }
        )
    fixed = [row for row in output if int(row["fix"])]
    summary = {
        "n_epochs_full_denominator": len(output),
        "route_segment": [segment_start, segment_end],
        "detected_final_jump_epoch": jump_epoch,
        "jump_residual_m": float(args.jump_residual_m),
        "min_tail_anchors": int(args.min_tail_anchors),
        "applied_epochs": len(applied),
        "sub50cm_full_epochs": sum(int(row["sub50cm"]) for row in output),
        "sub50cm_full_pct": 100.0 * sum(int(row["sub50cm"]) for row in output) / len(output),
        "declared_fix_epochs": len(fixed),
        "false_fix_epochs": sum(int(row["false_fix"]) for row in fixed),
        "false_fix_pct": 100.0 * sum(int(row["false_fix"]) for row in fixed) / max(len(fixed), 1),
    }
    args.out_trajectory.parent.mkdir(parents=True, exist_ok=True)
    with args.out_trajectory.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
