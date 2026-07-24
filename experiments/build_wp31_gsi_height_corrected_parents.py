#!/usr/bin/env python3
"""Project every recurring PF parent to one cached GSI antenna height."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from pyproj import Transformer


def build_corrected_parents(
    candidate_source: dict[str, Any], height_result: dict[str, Any]
) -> dict[str, Any]:
    if height_result.get("reason") not in (
        "weak_absolute_height_match",
        "height_winner_not_separated",
    ):
        raise ValueError("height result is not a valid proposal result")
    target_height = float(height_result["predicted_antenna_ellipsoid_height_m"])
    to_lla = Transformer.from_crs("EPSG:4978", "EPSG:4979", always_xy=True)
    to_ecef = Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True)
    output = []
    for row in candidate_source.get("candidates", []):
        position = np.asarray(row["position_ecef"], dtype=np.float64)
        longitude, latitude, original_height = to_lla.transform(*position)
        corrected = np.asarray(to_ecef.transform(longitude, latitude, target_height))
        output.append(
            {
                "candidate_id": int(row["candidate_id"]),
                "proposal_kind": "gsi_height_corrected_parent",
                "coverage_epochs": int(row.get("coverage_epochs", 0)),
                "members": int(row.get("members", 0)),
                "original_position_ecef": position.tolist(),
                "original_ellipsoid_height_m": float(original_height),
                "signed_height_correction_m": float(target_height - original_height),
                "absolute_height_correction_m": abs(float(target_height - original_height)),
                "position_ecef": corrected.tolist(),
            }
        )
    if len(output) < 2:
        raise ValueError("at least two recurring parents are required")
    output.sort(key=lambda row: int(row["candidate_id"]))
    return {
        "schema": "wp31_gsi_height_corrected_parents_v1",
        "segment": [int(value) for value in candidate_source["segment"]],
        "target_ellipsoid_height_m": target_height,
        "height_rank_best_candidate_id": int(height_result["best_candidate_id"]),
        "height_rank_runner_candidate_id": int(height_result["runner_candidate_id"]),
        "height_rank_runner_gap_m": float(height_result["runner_gap_m"]),
        "candidates": output,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidates_json", type=Path)
    parser.add_argument("height_result_json", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_bytes = args.candidates_json.read_bytes()
    height_bytes = args.height_result_json.read_bytes()
    result = build_corrected_parents(
        json.loads(source_bytes), json.loads(height_bytes)
    )
    result["candidate_source_sha256"] = hashlib.sha256(source_bytes).hexdigest()
    result["height_result_sha256"] = hashlib.sha256(height_bytes).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**result, "candidates": f"{len(result['candidates'])} candidates"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
