#!/usr/bin/env python3
"""Build a horizontal shell after a unique PF parent is corrected to cached GSI height."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from pyproj import Transformer

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "experiments"))

from build_wp31_static_shell_candidates import build_shell_candidates  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_corrected_shell(
    candidate_source: dict[str, Any],
    height_result: dict[str, Any],
    radii_m: tuple[float, ...],
    *,
    min_parent_height_gap_m: float = 0.1,
    max_height_correction_m: float = 5.0,
) -> dict[str, Any]:
    if height_result.get("reason") != "weak_absolute_height_match":
        raise ValueError("height result must be a rejected absolute-height proposal")
    if height_result.get("selected_candidate_id") is not None:
        raise ValueError("height result must not already select a candidate")
    if float(height_result["runner_gap_m"]) < float(min_parent_height_gap_m):
        raise ValueError("height parent is not separated")
    correction = float(height_result["best_height_residual_m"])
    if correction > float(max_height_correction_m):
        raise ValueError("required height correction exceeds proposal gate")
    parent_id = int(height_result["best_candidate_id"])
    matches = [
        row
        for row in candidate_source.get("candidates", [])
        if int(row["candidate_id"]) == parent_id
    ]
    if len(matches) != 1:
        raise ValueError("height parent candidate is missing")
    parent = np.asarray(matches[0]["position_ecef"], dtype=np.float64)
    to_lla = Transformer.from_crs("EPSG:4978", "EPSG:4979", always_xy=True)
    to_ecef = Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True)
    longitude, latitude, _height = to_lla.transform(*parent)
    target_height = float(height_result["predicted_antenna_ellipsoid_height_m"])
    center = np.asarray(to_ecef.transform(longitude, latitude, target_height))
    if not np.isfinite(center).all():
        raise ValueError("height-corrected center is nonfinite")
    candidates = build_shell_candidates(center, radii_m, include_center=True)
    return {
        "schema": "wp31_gsi_height_corrected_shell_v1",
        "segment": [int(value) for value in candidate_source["segment"]],
        "parent_candidate_id": parent_id,
        "parent_position_ecef": parent.tolist(),
        "parent_height_runner_gap_m": float(height_result["runner_gap_m"]),
        "height_correction_m": correction,
        "target_ellipsoid_height_m": target_height,
        "seed_center_source": "unique_pf_parent_cached_gsi_height_correction",
        "seed_center_ecef": center.tolist(),
        "seed_radii_m": list(radii_m),
        "seed_directions": "cube26",
        "include_center": True,
        "candidates": candidates,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidates_json", type=Path)
    parser.add_argument("height_result_json", type=Path)
    parser.add_argument("--radii-m", default="0.5,1.0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.candidates_json.read_text(encoding="utf-8"))
    height = json.loads(args.height_result_json.read_text(encoding="utf-8"))
    radii = tuple(float(value) for value in args.radii_m.split(",") if value.strip())
    result = build_corrected_shell(source, height, radii)
    result["candidate_source_sha256"] = _sha256(args.candidates_json)
    result["height_result_sha256"] = _sha256(args.height_result_json)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**result, "candidates": f"{len(result['candidates'])} candidates"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
