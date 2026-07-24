#!/usr/bin/env python3
"""Build an ENU-horizontal ring around an independently height-corrected center."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "experiments"))

from build_wp29_imu_heading_route_seed_trace import local_enu_basis  # noqa: E402
from analyze_wp29_static_reanchor_shadow import parse_ecef  # noqa: E402


def build_horizontal_shell(
    center_ecef: np.ndarray, radii_m: tuple[float, ...], *, directions: int = 8
) -> list[dict[str, object]]:
    center = np.asarray(center_ecef, dtype=np.float64).reshape(3)
    if not np.isfinite(center).all() or not radii_m or any(radius <= 0 for radius in radii_m):
        raise ValueError("center and positive radii are required")
    if directions < 4 or directions % 4:
        raise ValueError("directions must be a multiple of four")
    basis = local_enu_basis(center)
    rows: list[dict[str, object]] = [
        {"candidate_id": 0, "proposal_kind": "horizontal_shell_center", "east_m": 0.0, "north_m": 0.0, "position_ecef": center.tolist()}
    ]
    for radius in radii_m:
        for index in range(directions):
            angle = 2.0 * np.pi * index / directions
            east = float(radius * np.cos(angle))
            north = float(radius * np.sin(angle))
            position = center + east * basis[0] + north * basis[1]
            rows.append(
                {
                    "candidate_id": len(rows),
                    "proposal_kind": "horizontal_ring",
                    "radius_m": float(radius),
                    "direction_index": index,
                    "east_m": east,
                    "north_m": north,
                    "position_ecef": position.tolist(),
                }
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--center-ecef", required=True)
    parser.add_argument("--radii-m", required=True)
    parser.add_argument("--directions", type=int, default=8)
    parser.add_argument("--center-source", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.end <= args.start:
        parser.error("--end must be greater than --start")
    center = parse_ecef(args.center_ecef)
    radii = tuple(float(value) for value in args.radii_m.split(",") if value.strip())
    candidates = build_horizontal_shell(center, radii, directions=args.directions)
    result = {
        "schema": "wp31_horizontal_shell_candidates_v1",
        "segment": [args.start, args.end],
        "seed_center_source": args.center_source,
        "seed_center_ecef": center.tolist(),
        "seed_radii_m": list(radii),
        "horizontal_directions": args.directions,
        "candidates": candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**result, "candidates": f"{len(candidates)} candidates"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
