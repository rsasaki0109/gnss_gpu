#!/usr/bin/env python3
"""Apply an accepted moving-offset wide-lane shadow selection."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "python"))

from gnss_gpu.io.ppc import PPCDatasetLoader  # noqa: E402


def selected_offset(
    candidates: dict[str, Any], selection: dict[str, Any]
) -> tuple[int, int, int, np.ndarray]:
    if selection.get("selection_reason") != "regularized_widelane_consensus":
        raise RuntimeError("moving-offset selection is not accepted")
    selected_id = selection.get("selected_candidate_id")
    if selected_id is None:
        raise RuntimeError("moving-offset selection has no candidate")
    if list(selection.get("segment", [])) != list(candidates.get("segment", [])):
        raise RuntimeError("candidate and selection segments differ")
    matches = [
        row
        for row in candidates.get("candidates", [])
        if int(row.get("candidate_id", -1)) == int(selected_id)
    ]
    if len(matches) != 1:
        raise RuntimeError("selected moving-offset candidate is absent or duplicated")
    start, end = (int(value) for value in candidates["segment"])
    return start, end, int(selected_id), np.asarray(
        matches[0]["offset_ecef_m"], dtype=np.float64
    ).reshape(3)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("--candidates-json", type=Path, required=True)
    parser.add_argument("--selection-json", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    parser.add_argument("--out-trajectory", type=Path, required=True)
    args = parser.parse_args()

    candidates = json.loads(args.candidates_json.read_text(encoding="utf-8"))
    selection = json.loads(args.selection_json.read_text(encoding="utf-8"))
    start, end, selected_id, offset = selected_offset(candidates, selection)
    with args.trajectory.open(newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    data = PPCDatasetLoader(args.data_dir).load_experiment_data(
        max_epochs=len(rows), systems=("G", "R", "E", "C", "J")
    )
    output: list[dict[str, Any]] = []
    for row in rows:
        epoch = int(row["epoch"])
        position = np.asarray(
            [float(row["ecef_x"]), float(row["ecef_y"]), float(row["ecef_z"])],
            dtype=np.float64,
        )
        applied = start <= epoch < end
        if applied:
            position += offset
        error = float(
            np.linalg.norm(position - np.asarray(data["ground_truth"][epoch], dtype=np.float64))
        )
        fix = int(row.get("fix", "0"))
        output.append(
            {
                **row,
                "ecef_x": float(position[0]),
                "ecef_y": float(position[1]),
                "ecef_z": float(position[2]),
                "source": (
                    "moving_offset_widelane_shadow" if applied else row.get("source", "")
                ),
                "error_m": error,
                "sub50cm": int(error < 0.5),
                "false_fix": int(bool(fix) and error >= 0.5),
                "moving_offset_applied": int(applied),
            }
        )
    fixed = [row for row in output if int(row["fix"])]
    summary = {
        "n_epochs_full_denominator": len(output),
        "segment": [start, end],
        "selected_candidate_id": selected_id,
        "offset_ecef_m": offset.tolist(),
        "moving_offset_epochs": end - start,
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
