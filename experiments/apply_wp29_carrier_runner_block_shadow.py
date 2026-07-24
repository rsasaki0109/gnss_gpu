#!/usr/bin/env python3
"""Apply contiguous absolute-carrier runner-up blocks to a saved trajectory."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "python"))

_XYZ = ("ecef_x", "ecef_y", "ecef_z")


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def contiguous_anchor_blocks(
    epochs: list[int], *, stride: int, min_anchors: int
) -> list[list[int]]:
    """Return deterministic consecutive anchor blocks that pass a length gate."""

    blocks: list[list[int]] = []
    current: list[int] = []
    for epoch in sorted(set(int(value) for value in epochs)):
        if current and epoch != current[-1] + int(stride):
            if len(current) >= int(min_anchors):
                blocks.append(current)
            current = []
        current.append(epoch)
    if len(current) >= int(min_anchors):
        blocks.append(current)
    return blocks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("candidate_audit", type=Path)
    parser.add_argument("absolute_evidence", type=Path)
    parser.add_argument("--anchor-stride", type=int, default=5)
    parser.add_argument("--min-block-anchors", type=int, default=5)
    parser.add_argument("--min-carrier-rows", type=int, default=8)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    parser.add_argument("--out-trajectory", type=Path, required=True)
    args = parser.parse_args()

    trajectory = _read(args.trajectory)
    scope_end = len(trajectory) if args.end is None else int(args.end)
    if not (0 <= int(args.start) < scope_end <= len(trajectory)):
        raise RuntimeError("carrier runner scope is invalid")
    audit = _read(args.candidate_audit)
    evidence = {
        (int(row["epoch"]), row["basin_id"]): row
        for row in _read(args.absolute_evidence)
    }
    by_epoch: dict[int, list[dict[str, str]]] = {}
    for row in audit:
        if not (int(args.start) <= int(row["epoch"]) < scope_end):
            continue
        key = (int(row["epoch"]), row["basin_id"])
        if key not in evidence:
            continue
        prepared = {**row, **evidence[key]}
        if int(prepared["carrier_rows"]) < int(args.min_carrier_rows):
            continue
        by_epoch.setdefault(int(row["epoch"]), []).append(prepared)

    runner_winners: dict[int, dict[str, str]] = {}
    changed_epochs: list[int] = []
    for epoch, rows in by_epoch.items():
        top_two = sorted(rows, key=lambda row: int(row["max_marginal_rank"]))[:2]
        if len(top_two) != 2 or int(top_two[-1]["max_marginal_rank"]) > 2:
            continue
        selected = min(top_two, key=lambda row: float(row["carrier_cost"]))
        if int(selected["max_marginal_rank"]) == 1:
            continue
        runner_winners[epoch] = selected
        changed_epochs.append(epoch)
    blocks = contiguous_anchor_blocks(
        changed_epochs,
        stride=int(args.anchor_stride),
        min_anchors=int(args.min_block_anchors),
    )
    accepted_epochs = {epoch for block in blocks for epoch in block}
    positions = np.asarray(
        [[float(row[key]) for key in _XYZ] for row in trajectory], dtype=np.float64
    )
    output_positions = positions.copy()
    applied_epochs: set[int] = set()
    for block in blocks:
        left, right = block[0], block[-1]
        anchor_positions = np.asarray(
            [[float(runner_winners[epoch][key]) for key in _XYZ] for epoch in block]
        )
        query = np.arange(left, right + 1)
        for axis in range(3):
            output_positions[left : right + 1, axis] = np.interp(
                query, block, anchor_positions[:, axis]
            )
        applied_epochs.update(query.tolist())

    from gnss_gpu.io.ppc import PPCDatasetLoader

    times, truth = PPCDatasetLoader(args.data_dir).load_ground_truth()
    output = []
    for row, position in zip(trajectory, output_positions):
        epoch = int(row["epoch"])
        reference = truth[int(np.argmin(np.abs(times - float(row["tow"]))))]
        error = float(np.linalg.norm(position - reference))
        fix = int(row.get("fix", "0"))
        output.append(
            {
                **row,
                **{key: float(position[index]) for index, key in enumerate(_XYZ)},
                "source": (
                    "carrier_runner_block_shadow"
                    if epoch in applied_epochs
                    else row.get("source", "")
                ),
                "error_m": error,
                "sub50cm": int(error < 0.5),
                "false_fix": int(bool(fix) and error >= 0.5),
                "carrier_runner_applied": int(epoch in applied_epochs),
            }
        )
    fixed = [row for row in output if int(row["fix"])]
    summary = {
        "n_epochs_full_denominator": len(output),
        "scope": [int(args.start), scope_end],
        "changed_anchor_epochs": changed_epochs,
        "accepted_anchor_blocks": blocks,
        "accepted_anchor_epochs": sorted(accepted_epochs),
        "applied_epochs": len(applied_epochs),
        "sub50cm_full_epochs": sum(int(row["sub50cm"]) for row in output),
        "sub50cm_full_pct": 100.0
        * sum(int(row["sub50cm"]) for row in output)
        / len(output),
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
