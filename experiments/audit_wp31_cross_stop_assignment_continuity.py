#!/usr/bin/env python3
"""Audit versioned carrier-assignment continuity between two static stops."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def _load_candidates(path: Path, ids: set[int] | None = None) -> dict[int, np.ndarray]:
    source = json.loads(path.read_text(encoding="utf-8"))
    result = {}
    for row in source.get("candidates", []):
        candidate_id = int(row["candidate_id"])
        if ids is None or candidate_id in ids:
            result[candidate_id] = np.asarray(row["position_ecef"], dtype=np.float64)
    if not result:
        raise ValueError("candidate selection is empty")
    return result


def collect_assignments(
    basin_csv: Path,
    left_candidates: dict[int, np.ndarray],
    right_candidates: dict[int, np.ndarray],
    *,
    left_segment: tuple[int, int],
    right_segment: tuple[int, int],
    radius_m: float,
) -> tuple[dict[int, set[tuple[str, str, int, int]]], dict[int, set[tuple[str, str, int, int]]], dict[str, int]]:
    left = {candidate_id: set() for candidate_id in left_candidates}
    right = {candidate_id: set() for candidate_id in right_candidates}
    matched_rows = {"left": 0, "right": 0}
    with basin_csv.open(newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            epoch = int(row["epoch"])
            side = None
            candidates = None
            target = None
            if left_segment[0] <= epoch < left_segment[1]:
                side, candidates, target = "left", left_candidates, left
            elif right_segment[0] <= epoch < right_segment[1]:
                side, candidates, target = "right", right_candidates, right
            elif epoch >= right_segment[1]:
                break
            else:
                continue
            position = np.asarray([float(row[f"ecef_{axis}"]) for axis in "xyz"])
            distances = {
                candidate_id: float(np.linalg.norm(position - candidate_position))
                for candidate_id, candidate_position in candidates.items()
            }
            candidate_id, distance = min(distances.items(), key=lambda item: item[1])
            if distance > radius_m:
                continue
            assignment = json.loads(row["assignment_json"])
            for item in assignment:
                if len(item) != 5:
                    raise ValueError("assignment item must contain ref, sat, wavelength, generation, integer")
                target[candidate_id].add((str(item[0]), str(item[1]), int(item[2]), int(item[3])))
            matched_rows[side] += 1
    return left, right, matched_rows


def audit_continuity(
    left: dict[int, set[tuple[str, str, int, int]]],
    right: dict[int, set[tuple[str, str, int, int]]],
) -> list[dict[str, Any]]:
    rows = []
    for left_id, left_keys in left.items():
        left_raw = {(ref, sat, wavelength) for ref, sat, wavelength, _generation in left_keys}
        for right_id, right_keys in right.items():
            right_raw = {(ref, sat, wavelength) for ref, sat, wavelength, _generation in right_keys}
            shared_raw = left_raw & right_raw
            shared_versioned = left_keys & right_keys
            generation_deltas = []
            for raw_key in shared_raw:
                left_generations = {key[3] for key in left_keys if key[:3] == raw_key}
                right_generations = {key[3] for key in right_keys if key[:3] == raw_key}
                generation_deltas.append(min(abs(r - l) for l in left_generations for r in right_generations))
            rows.append(
                {
                    "left_candidate_id": left_id,
                    "right_candidate_id": right_id,
                    "left_versioned_keys": len(left_keys),
                    "right_versioned_keys": len(right_keys),
                    "shared_raw_keys": len(shared_raw),
                    "shared_versioned_keys": len(shared_versioned),
                    "min_generation_delta": min(generation_deltas) if generation_deltas else None,
                    "max_generation_delta": max(generation_deltas) if generation_deltas else None,
                    "continuous": bool(shared_versioned),
                }
            )
    return rows


def _segment(value: str) -> tuple[int, int]:
    start, end = (int(item) for item in value.split(":"))
    if start < 0 or end <= start:
        raise argparse.ArgumentTypeError("segment must satisfy START < END")
    return start, end


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("basin_csv", type=Path)
    parser.add_argument("--left-candidates", type=Path, required=True)
    parser.add_argument("--left-id", type=int, required=True)
    parser.add_argument("--left-segment", type=_segment, required=True)
    parser.add_argument("--right-candidates", type=Path, required=True)
    parser.add_argument("--right-ids", required=True)
    parser.add_argument("--right-segment", type=_segment, required=True)
    parser.add_argument("--radius-m", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    right_ids = {int(value) for value in args.right_ids.split(",") if value.strip()}
    left_candidates = _load_candidates(args.left_candidates, {args.left_id})
    right_candidates = _load_candidates(args.right_candidates, right_ids)
    left, right, matched = collect_assignments(
        args.basin_csv, left_candidates, right_candidates,
        left_segment=args.left_segment, right_segment=args.right_segment,
        radius_m=args.radius_m,
    )
    rows = audit_continuity(left, right)
    result = {
        "schema": "wp31_cross_stop_assignment_continuity_v1",
        "truth_free": True,
        "basin_csv_sha256": hashlib.sha256(args.basin_csv.read_bytes()).hexdigest(),
        "left_segment": list(args.left_segment),
        "right_segment": list(args.right_segment),
        "radius_m": args.radius_m,
        "matched_basin_rows": matched,
        "candidate_pairs": rows,
        "continuous_pair_count": sum(int(row["continuous"]) for row in rows),
        "production_selected_candidate_id": None,
        "production_reason": "cross_stop_continuity_available" if any(row["continuous"] for row in rows) else "no_continuous_versioned_assignment",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
