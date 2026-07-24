#!/usr/bin/env python3
"""Build a sparse truth-free basin trace from ranked static-grid hypotheses."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


FIELDS = (
    "epoch",
    "tow",
    "basin_id",
    "assignment_id",
    "assignment_json",
    "epoch_log_likelihood",
    "cumulative_log_marginal",
    "log_weight",
    "ecef_x",
    "ecef_y",
    "ecef_z",
    "velocity_x",
    "velocity_y",
    "velocity_z",
    "birth_epoch",
    "lineage",
    "proposal_sources",
)


def select_candidate_ids(
    integrity: dict[str, Any], *, score_name: str, top_k: int
) -> list[int]:
    candidates = list(integrity["candidates"])
    candidates.sort(key=lambda row: float(row[score_name]))
    return [int(row["candidate_id"]) for row in candidates[: int(top_k)]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("grid_candidates_json", type=Path)
    parser.add_argument("integrity_json", type=Path)
    parser.add_argument("epoch_diagnostics", type=Path)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--score-name", default="carrier_temporal_arc_cauchy_mean"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    grid = json.loads(args.grid_candidates_json.read_text(encoding="utf-8"))
    integrity = json.loads(args.integrity_json.read_text(encoding="utf-8"))
    selected = select_candidate_ids(
        integrity, score_name=str(args.score_name), top_k=int(args.top_k)
    )
    by_id = {int(row["candidate_id"]): row for row in grid["candidates"]}
    score_by_id = {
        int(row["candidate_id"]): float(row[str(args.score_name)])
        for row in integrity["candidates"]
    }
    with args.epoch_diagnostics.open(newline="", encoding="utf-8-sig") as fh:
        tow_by_epoch = {int(row["epoch"]): float(row["tow"]) for row in csv.DictReader(fh)}
    rows: list[dict[str, Any]] = []
    for epoch in range(int(args.start), int(args.end), int(args.stride)):
        for candidate_id in selected:
            source = by_id[candidate_id]
            position = source["position_ecef"]
            score = score_by_id[candidate_id]
            token = f"{int(args.start)}:static_grid_temporal:{candidate_id}"
            rows.append(
                {
                    "epoch": epoch,
                    "tow": tow_by_epoch[epoch],
                    "basin_id": f"grid-{candidate_id}",
                    "assignment_id": f"grid-{candidate_id}",
                    "assignment_json": "[]",
                    "epoch_log_likelihood": -score,
                    "cumulative_log_marginal": -score,
                    "log_weight": -score,
                    "ecef_x": position[0],
                    "ecef_y": position[1],
                    "ecef_z": position[2],
                    "velocity_x": 0.0,
                    "velocity_y": 0.0,
                    "velocity_z": 0.0,
                    "birth_epoch": int(args.start),
                    "lineage": token,
                    "proposal_sources": token,
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"selected_candidate_ids": selected, "rows": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
