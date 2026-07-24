#!/usr/bin/env python3
"""Recompute and promote a holdout-validated WP76 affine profile."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from pyproj import Transformer
from shapely.geometry import LineString
from shapely.strtree import STRtree

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "experiments"))

from select_wp76_affine_multibasis_road_carrier import select_clusters  # noqa: E402

_M4_HASHES = {
    Path("internal_docs/wp30_m4_production_config.json"): (
        "66A5FF3F1919C4B0F9ED95A5EFA38865B518C9E03E6FD2652B7A0456A1F89486"
    ),
    Path("internal_docs/wp30_m4_tokyo_evidence_ledger.json"): (
        "9D756F447304C30B73694225F1CEEA1A82DE864F8D968D449928662582DF098C"
    ),
}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_route(path: Path, *, start: int, end: int) -> np.ndarray:
    with path.open(newline="", encoding="utf-8-sig") as fh:
        rows = [
            row
            for row in csv.DictReader(fh)
            if start <= int(row["epoch"]) < end
        ]
    if len(rows) != end - start:
        raise ValueError("trajectory does not cover promotion segment")
    return np.asarray(
        [[float(row[key]) for key in ("ecef_x", "ecef_y", "ecef_z")] for row in rows]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", type=Path, nargs=3)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--osm-cache", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--nagoya-transfer", type=Path, required=True)
    parser.add_argument("--tokyo-unsafe", type=Path, required=True)
    parser.add_argument("--road-lower-m", type=float, required=True)
    parser.add_argument("--road-upper-m", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source_bytes = [path.read_bytes() for path in args.sources]
    sources = [json.loads(value) for value in source_bytes]
    start, end = (int(value) for value in sources[0]["segment"])
    cache_bytes = args.osm_cache.read_bytes()
    cache = json.loads(cache_bytes)
    epsg = int(cache["epsg"])
    road = STRtree([LineString(row) for row in cache["projected_road_lines"]])
    transformer = Transformer.from_crs("EPSG:4978", f"EPSG:{epsg}", always_xy=True)
    route = _read_route(args.trajectory, start=start, end=end)
    boundary = int(sources[0]["offset_model"]["boundary_epoch"])
    scales = np.asarray(
        [(boundary - epoch) / (boundary - start) for epoch in range(start, end)]
    )
    recomputed = select_clusters(
        sources,
        route,
        road,
        transformer,
        scales=scales,
        road_lower_m=args.road_lower_m,
        road_upper_m=args.road_upper_m,
    )
    selection_bytes = args.selection.read_bytes()
    selection = json.loads(selection_bytes)
    comparable_keys = (
        "accepted",
        "reason",
        "selected_profile",
        "winner",
        "runner",
        "runner_margin",
        "family_rank_limit",
        "family_rank_pass",
        "runner_margin_pass",
        "cluster_count",
        "basis_mode_counts",
        "clusters",
    )
    if any(recomputed.get(key) != selection.get(key) for key in comparable_keys):
        raise RuntimeError("stored WP76 selection does not match recomputation")
    if not recomputed["accepted"]:
        raise RuntimeError("recomputed WP76 target does not pass")
    expected_inputs = selection.get("input_sha256", {})
    if expected_inputs.get("sources") != [_sha256(value) for value in source_bytes]:
        raise RuntimeError("WP76 source hashes do not match selection")
    if expected_inputs.get("trajectory") != _sha256(args.trajectory.read_bytes()):
        raise RuntimeError("WP76 trajectory hash does not match selection")
    if expected_inputs.get("osm_cache") != _sha256(cache_bytes):
        raise RuntimeError("WP76 OSM cache hash does not match selection")

    validation = {}
    for label, path in (
        ("nagoya_transfer", args.nagoya_transfer),
        ("tokyo_unsafe", args.tokyo_unsafe),
    ):
        value = path.read_bytes()
        payload = json.loads(value)
        if bool(payload.get("production_input_truth", True)) or bool(
            payload.get("accepted", True)
        ):
            raise RuntimeError(f"{label} validation did not fail closed")
        if payload.get("reason") != "fewer_than_two_three_basis_clusters":
            raise RuntimeError(f"{label} validation failed for an unexpected reason")
        validation[label] = {
            "path": str(path),
            "sha256": _sha256(value),
            "cluster_count": int(payload["cluster_count"]),
            "reason": payload["reason"],
        }

    preserved = {}
    for path, expected in _M4_HASHES.items():
        actual = hashlib.sha256(path.read_bytes()).hexdigest().upper()
        if actual != expected:
            raise RuntimeError(f"M4 artifact changed: {path}")
        preserved[str(path)] = actual

    profile = recomputed["selected_profile"]
    result = {
        "schema": "wp76_affine_multibasis_promotion_v1",
        "production_input_truth": False,
        "truth_usage": "none",
        "production_promoted": True,
        "reason": "unique_affine_multibasis_road_carrier_cluster",
        "segment": [start, end],
        "profile_mode": "right_boundary_affine_zero",
        "boundary_epoch": boundary,
        "offset_ecef_m": profile["offset_ecef_m"],
        "block_offsets_ecef_m": profile["block_offsets_ecef_m"],
        "member_candidate_ids": profile["member_candidate_ids"],
        "cluster_diameter_m": profile["cluster_diameter_m"],
        "family_ranks": profile["family_ranks"],
        "runner_margin": recomputed["runner_margin"],
        "selection": {"path": str(args.selection), "sha256": _sha256(selection_bytes)},
        "validation": validation,
        "input_sha256": {
            "sources": [_sha256(value) for value in source_bytes],
            "trajectory": _sha256(args.trajectory.read_bytes()),
            "osm_cache": _sha256(cache_bytes),
        },
        "m4_preserved_sha256": preserved,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
