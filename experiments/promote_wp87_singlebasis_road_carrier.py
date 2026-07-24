#!/usr/bin/env python3
"""Recompute and promote a holdout-validated WP87 fixed affine profile."""

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

from select_wp76_affine_multibasis_road_carrier import (  # noqa: E402
    affine_baseline_route,
    profile_scales,
)
from select_wp87_singlebasis_road_carrier import select_single_basis  # noqa: E402

_M4_HASHES = {
    Path("internal_docs/wp30_m4_production_config.json"): (
        "66A5FF3F1919C4B0F9ED95A5EFA38865B518C9E03E6FD2652B7A0456A1F89486"
    ),
    Path("internal_docs/wp30_m4_tokyo_evidence_ledger.json"): (
        "9D756F447304C30B73694225F1CEEA1A82DE864F8D968D449928662582DF098C"
    ),
}
_COMPARABLE_KEYS = (
    "accepted",
    "reason",
    "selected_profile",
    "winner",
    "runner",
    "runner_margin",
    "family_rank_limit",
    "family_rank_pass",
    "runner_margin_pass",
    "mode_count",
    "modes",
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_route(path: Path, *, start: int, end: int) -> np.ndarray:
    with path.open(newline="", encoding="utf-8-sig") as fh:
        rows = [row for row in csv.DictReader(fh) if start <= int(row["epoch"]) < end]
    if len(rows) != end - start:
        raise ValueError("trajectory does not cover promotion segment")
    return np.asarray(
        [[float(row[key]) for key in ("ecef_x", "ecef_y", "ecef_z")] for row in rows]
    )


def _recompute(
    source_path: Path,
    trajectory_path: Path,
    osm_path: Path,
    *,
    lower_m: float,
    upper_m: float,
) -> tuple[dict, dict[str, str]]:
    source_bytes = source_path.read_bytes()
    source = json.loads(source_bytes)
    start, end = (int(value) for value in source["segment"])
    model = source["offset_model"]
    scales = profile_scales(start, end, model)
    route = affine_baseline_route(
        _read_route(trajectory_path, start=start, end=end), scales, model
    )
    cache_bytes = osm_path.read_bytes()
    cache = json.loads(cache_bytes)
    road = STRtree([LineString(row) for row in cache["projected_road_lines"]])
    transformer = Transformer.from_crs(
        "EPSG:4978", f"EPSG:{int(cache['epsg'])}", always_xy=True
    )
    result = select_single_basis(
        source,
        route,
        road,
        transformer,
        scales=scales,
        road_lower_m=lower_m,
        road_upper_m=upper_m,
    )
    return result, {
        "source": _sha256(source_bytes),
        "trajectory": _sha256(trajectory_path.read_bytes()),
        "osm_cache": _sha256(cache_bytes),
    }


def _verify_stored(
    recomputed: dict, selection_path: Path, hashes: dict[str, str]
) -> dict:
    selection_bytes = selection_path.read_bytes()
    stored = json.loads(selection_bytes)
    if any(recomputed.get(key) != stored.get(key) for key in _COMPARABLE_KEYS):
        raise RuntimeError(
            f"stored selection does not match recomputation: {selection_path}"
        )
    expected = stored.get("input_sha256", {})
    if any(expected.get(key) != value for key, value in hashes.items()):
        raise RuntimeError(f"selection input hashes do not match: {selection_path}")
    return {
        "path": str(selection_path),
        "sha256": _sha256(selection_bytes),
        "accepted": bool(stored["accepted"]),
        "reason": stored["reason"],
        "mode_count": int(stored["mode_count"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--osm-cache", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--nagoya-transfer-source", type=Path, required=True)
    parser.add_argument("--nagoya-transfer-trajectory", type=Path, required=True)
    parser.add_argument("--nagoya-transfer-osm-cache", type=Path, required=True)
    parser.add_argument("--nagoya-transfer-selection", type=Path, required=True)
    parser.add_argument("--tokyo-unsafe-source", type=Path, required=True)
    parser.add_argument("--tokyo-unsafe-trajectory", type=Path, required=True)
    parser.add_argument("--tokyo-unsafe-osm-cache", type=Path, required=True)
    parser.add_argument("--tokyo-unsafe-selection", type=Path, required=True)
    parser.add_argument("--road-lower-m", type=float, required=True)
    parser.add_argument("--road-upper-m", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    target, target_hashes = _recompute(
        args.source,
        args.trajectory,
        args.osm_cache,
        lower_m=args.road_lower_m,
        upper_m=args.road_upper_m,
    )
    target_stored = _verify_stored(target, args.selection, target_hashes)
    if not target["accepted"]:
        raise RuntimeError("recomputed WP87 target does not pass")

    validations = {}
    for label, source, trajectory, osm, selection in (
        (
            "nagoya_transfer",
            args.nagoya_transfer_source,
            args.nagoya_transfer_trajectory,
            args.nagoya_transfer_osm_cache,
            args.nagoya_transfer_selection,
        ),
        (
            "tokyo_unsafe",
            args.tokyo_unsafe_source,
            args.tokyo_unsafe_trajectory,
            args.tokyo_unsafe_osm_cache,
            args.tokyo_unsafe_selection,
        ),
    ):
        recomputed, hashes = _recompute(
            source,
            trajectory,
            osm,
            lower_m=args.road_lower_m,
            upper_m=args.road_upper_m,
        )
        validations[label] = _verify_stored(recomputed, selection, hashes)
        if recomputed["accepted"]:
            raise RuntimeError(f"{label} did not fail closed")

    preserved = {}
    for path, expected in _M4_HASHES.items():
        actual = hashlib.sha256(path.read_bytes()).hexdigest().upper()
        if actual != expected:
            raise RuntimeError(f"M4 artifact changed: {path}")
        preserved[str(path)] = actual

    source = json.loads(args.source.read_bytes())
    start, end = (int(value) for value in source["segment"])
    model = source["offset_model"]
    profile = target["selected_profile"]
    is_constant = model.get("mode") == "constant"
    is_wp93_constant = is_constant and [start, end] == [660, 715]
    result = {
        "schema": (
            (
                "wp93_constant_singlebasis_road_carrier_promotion_v1"
                if is_wp93_constant
                else "wp_constant_singlebasis_road_carrier_promotion_v1"
            )
            if is_constant
            else "wp87_singlebasis_road_carrier_promotion_v1"
        ),
        "production_input_truth": False,
        "truth_usage": "none",
        "production_promoted": True,
        "reason": "unique_singlebasis_road_carrier_cppr_mode",
        "segment": [start, end],
        "profile_mode": "constant" if is_constant else "right_boundary_affine_fixed",
        "offset_ecef_m": profile["offset_ecef_m"],
        "block_offsets_ecef_m": profile["block_offsets_ecef_m"],
        "candidate_id": profile["candidate_id"],
        "family_ranks": profile["family_ranks"],
        "runner_margin": target["runner_margin"],
        "selection": target_stored,
        "validation": validations,
        "input_sha256": target_hashes,
        "m4_preserved_sha256": preserved,
    }
    if not is_constant:
        result.update(
            {
                "boundary_epoch": int(model["boundary_epoch"]),
                "boundary_offset_ecef_m": model["boundary_offset_ecef_m"],
                "boundary_profile_sha256": model["boundary_profile_sha256"],
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
