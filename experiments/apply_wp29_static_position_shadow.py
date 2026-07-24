#!/usr/bin/env python3
"""Apply an accepted static-position shadow result to a trajectory segment."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "python"))

from gnss_gpu.io.ppc import PPCDatasetLoader  # noqa: E402


def accepted_static_position(result: dict) -> tuple[int, int, np.ndarray]:
    if result.get("reason") != "height_temporal_road_consensus":
        raise RuntimeError("static position result is not accepted")
    start, end = (int(value) for value in result["segment"])
    position = np.asarray(result["position_ecef"], dtype=np.float64).reshape(3)
    if not np.isfinite(position).all() or end <= start:
        raise RuntimeError("static position result is invalid")
    return start, end, position


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("--static-json", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    parser.add_argument("--out-trajectory", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.static_json.read_text(encoding="utf-8"))
    start, end, static_position = accepted_static_position(result)
    with args.trajectory.open(newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    times, truth_ecef = PPCDatasetLoader(args.data_dir).load_ground_truth()
    output = []
    for row in rows:
        epoch = int(row["epoch"])
        position = (
            static_position.copy()
            if start <= epoch < end
            else np.asarray(
                [float(row["ecef_x"]), float(row["ecef_y"]), float(row["ecef_z"])],
                dtype=np.float64,
            )
        )
        tow = float(row["tow"])
        truth = np.asarray(truth_ecef[int(np.argmin(np.abs(times - tow)))])
        error = float(np.linalg.norm(position - truth))
        fix = int(row.get("fix", "0"))
        output.append(
            {
                **row,
                "ecef_x": float(position[0]),
                "ecef_y": float(position[1]),
                "ecef_z": float(position[2]),
                "source": (
                    "static_height_temporal_shadow"
                    if start <= epoch < end
                    else row.get("source", "")
                ),
                "error_m": error,
                "sub50cm": int(error < 0.5),
                "false_fix": int(bool(fix) and error >= 0.5),
                "static_position_applied": int(start <= epoch < end),
            }
        )
    fixed = [row for row in output if int(row["fix"])]
    summary = {
        "n_epochs_full_denominator": len(output),
        "segment": [start, end],
        "selected_candidate_id": result.get("selected_candidate_id"),
        "static_position_ecef": static_position.tolist(),
        "static_override_epochs": end - start,
        "static_override_sub50cm_epochs": sum(
            int(row["sub50cm"]) for row in output[start:end]
        ),
        "sub50cm_full_epochs": sum(int(row["sub50cm"]) for row in output),
        "sub50cm_full_pct": 100.0
        * sum(int(row["sub50cm"]) for row in output)
        / max(len(output), 1),
        "declared_fix_epochs": len(fixed),
        "false_fix_epochs": sum(int(row["false_fix"]) for row in fixed),
        "false_fix_pct": 100.0
        * sum(int(row["false_fix"]) for row in fixed)
        / max(len(fixed), 1),
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
