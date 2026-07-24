#!/usr/bin/env python3
"""Extract a fail-closed static parent from a production motion trajectory."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def motion_parent(
    rows: list[dict[str, str]],
    *,
    start: int,
    end: int,
    min_epochs: int = 30,
    max_p95_deviation_m: float = 0.25,
    max_deviation_m: float = 0.5,
) -> dict[str, object]:
    if end <= start:
        raise ValueError("end must be greater than start")
    selected = [row for row in rows if start <= int(row["epoch"]) < end]
    epochs = [int(row["epoch"]) for row in selected]
    if epochs != list(range(start, end)):
        raise RuntimeError("trajectory does not provide contiguous segment coverage")
    if len(selected) < min_epochs:
        raise RuntimeError("segment has insufficient motion-parent epochs")
    positions = np.asarray(
        [[float(row[f"ecef_{axis}"]) for axis in "xyz"] for row in selected],
        dtype=np.float64,
    )
    if not np.isfinite(positions).all():
        raise RuntimeError("segment contains non-finite positions")
    center = np.median(positions, axis=0)
    deviations = np.linalg.norm(positions - center, axis=1)
    p95 = float(np.percentile(deviations, 95.0))
    maximum = float(np.max(deviations))
    if p95 > max_p95_deviation_m or maximum > max_deviation_m:
        raise RuntimeError("motion parent fails static-position spread gate")
    return {
        "segment": [int(start), int(end)],
        "n_epochs": len(selected),
        "position_ecef": center.tolist(),
        "median_deviation_m": float(np.median(deviations)),
        "p95_deviation_m": p95,
        "max_deviation_m": maximum,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("production_summary", type=Path)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--min-epochs", type=int, default=30)
    parser.add_argument("--max-p95-deviation-m", type=float, default=0.25)
    parser.add_argument("--max-deviation-m", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    summary_bytes = args.production_summary.read_bytes()
    summary = json.loads(summary_bytes)
    if bool(summary.get("development_anchor_used", True)):
        parser.error("source summary used a development anchor")
    if not bool(summary.get("production_promoted", False)):
        parser.error("source summary is not production-promoted")
    trajectory_bytes = args.trajectory.read_bytes()
    rows = list(csv.DictReader(trajectory_bytes.decode("utf-8").splitlines()))
    try:
        parent = motion_parent(
            rows,
            start=args.start,
            end=args.end,
            min_epochs=args.min_epochs,
            max_p95_deviation_m=args.max_p95_deviation_m,
            max_deviation_m=args.max_deviation_m,
        )
    except (RuntimeError, ValueError) as error:
        parser.error(str(error))
    result = {
        "schema": "wp32_motion_parent_v1",
        "production_input_truth": False,
        "production_source": True,
        "trajectory_sha256": hashlib.sha256(trajectory_bytes).hexdigest(),
        "production_summary_sha256": hashlib.sha256(summary_bytes).hexdigest(),
        "config": {
            "min_epochs": args.min_epochs,
            "max_p95_deviation_m": args.max_p95_deviation_m,
            "max_deviation_m": args.max_deviation_m,
        },
        **parent,
        "candidates": [
            {
                "candidate_id": 0,
                "proposal_kind": "production_motion_parent",
                "position_ecef": parent["position_ecef"],
            }
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**result, "candidates": "1 candidate"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
