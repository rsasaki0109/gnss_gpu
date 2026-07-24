#!/usr/bin/env python3
"""Build a lightweight ECEF shell candidate artifact without DD optimization."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "experiments"))

from analyze_wp29_static_reanchor_shadow import (  # noqa: E402
    offset_seed_candidates,
    parse_ecef,
)


def build_shell_candidates(
    center_ecef: np.ndarray,
    radii_m: tuple[float, ...],
    *,
    directions: str = "cube26",
    include_center: bool = True,
) -> list[dict[str, object]]:
    center = np.asarray(center_ecef, dtype=np.float64).reshape(3)
    if not np.isfinite(center).all():
        raise ValueError("center ECEF must be finite")
    if not radii_m or any(float(radius) <= 0.0 for radius in radii_m):
        raise ValueError("shell radii must be positive")
    positions: list[np.ndarray] = []
    kinds: list[str] = []
    if include_center:
        positions.append(center.copy())
        kinds.append("shell_center")
    for row in offset_seed_candidates(center, radii_m, directions=directions):
        positions.append(np.asarray(row["position_ecef"], dtype=np.float64))
        kinds.append("offset_seed")
    return [
        {
            "candidate_id": candidate_id,
            "proposal_kind": kind,
            "position_ecef": position.tolist(),
        }
        for candidate_id, (kind, position) in enumerate(zip(kinds, positions))
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--center-ecef", required=True)
    parser.add_argument("--radii-m", required=True)
    parser.add_argument("--directions", choices=("axes", "cube26"), default="cube26")
    parser.add_argument("--exclude-center", action="store_true")
    parser.add_argument("--center-source", default="external_truth_free_ecef")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.end <= args.start:
        parser.error("--end must be greater than --start")
    radii = tuple(float(value) for value in args.radii_m.split(",") if value.strip())
    center = parse_ecef(args.center_ecef)
    candidates = build_shell_candidates(
        center,
        radii,
        directions=args.directions,
        include_center=not args.exclude_center,
    )
    result = {
        "schema": "wp31_static_shell_candidates_v1",
        "segment": [args.start, args.end],
        "seed_center_source": args.center_source,
        "seed_center_ecef": center.tolist(),
        "seed_radii_m": list(radii),
        "seed_directions": args.directions,
        "include_center": not args.exclude_center,
        "candidates": candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**result, "candidates": f"{len(candidates)} candidates"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
