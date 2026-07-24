#!/usr/bin/env python3
"""Select a fine-grid child cluster supported by a prior truth-free motion path."""

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


def select_motion_child_cluster(
    candidates: list[dict[str, Any]],
    predicted_position_ecef: np.ndarray,
    *,
    min_members: int = 5,
    max_members: int = 12,
    max_spread_m: float = 0.5,
) -> dict[str, Any]:
    children = [
        row
        for row in candidates
        if row.get("proposal_kind") == "offset_seed"
        or (
            "proposal_kind" not in row
            and int(row.get("coverage_epochs", 0)) == 0
            and int(row.get("members", 0)) == 0
        )
    ]
    base = {
        "selected_candidate_id": None,
        "reason": "motion_child_cluster_rejected",
        "child_candidates": len(children),
    }
    if len(children) < int(min_members):
        return {**base, "reason": "insufficient_child_candidates"}
    predicted = np.asarray(predicted_position_ecef, dtype=np.float64).reshape(3)
    ordered = sorted(
        children,
        key=lambda row: float(
            np.linalg.norm(np.asarray(row["position_ecef"]) - predicted)
        ),
    )
    selected: tuple[int, np.ndarray, float] | None = None
    limit = min(int(max_members), len(ordered))
    for count in range(int(min_members), limit + 1):
        positions = np.asarray(
            [row["position_ecef"] for row in ordered[:count]], dtype=np.float64
        )
        center = np.mean(positions, axis=0)
        spread = float(np.max(np.linalg.norm(positions - center, axis=1)))
        if spread <= float(max_spread_m):
            selected = count, center, spread
    if selected is None:
        return {**base, "reason": "no_compact_motion_prefix"}
    count, center, spread = selected
    selected_rows = ordered[:count]
    distances = [
        float(np.linalg.norm(np.asarray(row["position_ecef"]) - predicted))
        for row in selected_rows
    ]
    representative = min(
        selected_rows,
        key=lambda row: float(np.linalg.norm(np.asarray(row["position_ecef"]) - center)),
    )
    return {
        **base,
        "selected_candidate_id": int(representative["candidate_id"]),
        "reason": "motion_supported_child_cluster",
        "cluster_member_ids": [int(row["candidate_id"]) for row in selected_rows],
        "cluster_members": count,
        "cluster_spread_m": spread,
        "motion_distance_min_m": min(distances),
        "motion_distance_max_m": max(distances),
        "position_ecef": center.tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidates_json", type=Path)
    parser.add_argument("motion_trajectory", type=Path)
    parser.add_argument("--min-members", type=int, default=5)
    parser.add_argument("--max-members", type=int, default=12)
    parser.add_argument("--max-spread-m", type=float, default=0.5)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.candidates_json.read_text(encoding="utf-8"))
    start, end = (int(value) for value in source["segment"])
    with args.motion_trajectory.open(newline="", encoding="utf-8-sig") as fh:
        motion_rows = {int(row["epoch"]): row for row in csv.DictReader(fh)}
    if start not in motion_rows:
        raise RuntimeError("motion trajectory does not contain the segment start")
    predicted = np.asarray(
        [float(motion_rows[start][key]) for key in ("ecef_x", "ecef_y", "ecef_z")]
    )
    result = select_motion_child_cluster(
        list(source["candidates"]),
        predicted,
        min_members=args.min_members,
        max_members=args.max_members,
        max_spread_m=args.max_spread_m,
    )
    result["segment"] = [start, end]
    result["motion_epoch"] = start
    result["motion_position_ecef"] = predicted.tolist()
    if result.get("position_ecef") is not None and args.data_dir is not None:
        _times, truth = PPCDatasetLoader(args.data_dir).load_ground_truth()
        segment_truth = np.asarray(truth[start:end], dtype=np.float64)
        segment_truth = segment_truth[np.isfinite(segment_truth).all(axis=1)]
        if not len(segment_truth):
            raise RuntimeError("static segment has no finite audit truth")
        truth_position = np.median(segment_truth, axis=0)
        result["selected_audit_error_m"] = float(
            np.linalg.norm(np.asarray(result["position_ecef"]) - truth_position)
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
