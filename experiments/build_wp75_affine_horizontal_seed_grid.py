#!/usr/bin/env python3
"""Build a truth-free EN seed grid around an affine float-ambiguity solution."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def build_grid(
    source: dict[str, Any],
    trajectory_rows: list[dict[str, str]],
    *,
    radius_m: float,
    step_m: float,
) -> dict[str, Any]:
    if bool(source.get("production_input_truth", True)):
        raise ValueError("affine candidate source is not production-safe")
    model = source.get("offset_model", {})
    if model.get("mode") not in {
        "constant",
        "right_boundary_affine_zero",
        "right_boundary_affine_fixed",
    }:
        raise ValueError("candidate source offset model is unsupported")
    segment = [int(value) for value in source.get("segment", [])]
    if len(segment) != 2:
        raise ValueError("candidate source segment is invalid")
    rows = [
        row for row in trajectory_rows if segment[0] <= int(row["epoch"]) < segment[1]
    ]
    if len(rows) != segment[1] - segment[0]:
        raise ValueError("trajectory does not cover affine source segment")
    if radius_m <= 0.0 or step_m <= 0.0:
        raise ValueError("horizontal grid radius and step must be positive")

    positions = np.asarray(
        [
            [float(row["ecef_x"]), float(row["ecef_y"]), float(row["ecef_z"])]
            for row in rows
        ]
    )
    representative = np.median(positions, axis=0)
    up = representative / np.linalg.norm(representative)
    east = np.asarray([-up[1], up[0], 0.0])
    east /= np.linalg.norm(east)
    north = np.cross(up, east)

    diagnostics = source.get("float_ambiguity_diagnostics", {})
    center = np.asarray(diagnostics.get("float_offset_ecef_m"), dtype=np.float64)
    if center.shape != (3,) or not np.all(np.isfinite(center)):
        raise ValueError("candidate source lacks a finite float offset")
    gsi = source.get("gsi_height_prior") or {}
    up_center_key = (
        "up_prior_center_m"
        if model.get("mode") == "constant"
        else "affine_reference_up_prior_center_m"
    )
    up_center = gsi.get(up_center_key)
    if up_center is None or not np.isfinite(float(up_center)):
        raise ValueError(f"candidate source lacks GSI Up center {up_center_key}")
    center += (float(up_center) - float(np.dot(center, up))) * up

    values = np.arange(-radius_m, radius_m + 0.5 * step_m, step_m)
    seeds = []
    for east_m in values:
        for north_m in values:
            offset = center + float(east_m) * east + float(north_m) * north
            seeds.append(
                {
                    "east_delta_m": float(east_m),
                    "north_delta_m": float(north_m),
                    "offset_ecef_m": offset.tolist(),
                }
            )
    return {
        "schema": "wp75_affine_horizontal_seed_grid_v1",
        "production_input_truth": False,
        "truth_usage": "none",
        "segment": segment,
        "offset_model": model,
        "grid": {
            "radius_m": float(radius_m),
            "step_m": float(step_m),
            "seed_count": len(seeds),
        },
        "float_center_gsi_up_normalized_ecef_m": center.tolist(),
        "gsi_reference_up_center_m": float(up_center),
        "gsi_reference_up_center_key": up_center_key,
        "local_basis_ecef": {
            "east": east.tolist(),
            "north": north.tolist(),
            "up": up.tolist(),
        },
        "seeds": seeds,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("--radius-m", type=float, default=1.5)
    parser.add_argument("--step-m", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source_bytes = args.source.read_bytes()
    trajectory_bytes = args.trajectory.read_bytes()
    source = json.loads(source_bytes)
    with args.trajectory.open(newline="", encoding="utf-8-sig") as fh:
        trajectory_rows = list(csv.DictReader(fh))
    result = build_grid(
        source,
        trajectory_rows,
        radius_m=args.radius_m,
        step_m=args.step_m,
    )
    result["input_sha256"] = {
        "source": hashlib.sha256(source_bytes).hexdigest(),
        "trajectory": hashlib.sha256(trajectory_bytes).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
