#!/usr/bin/env python3
"""Materialize one static-stop node from the hashed multi-stop PF cache."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def materialize_node(source: dict[str, Any], start: int) -> dict[str, Any]:
    matches = [
        node for node in source.get("nodes", []) if int(node["segment"][0]) == int(start)
    ]
    if len(matches) != 1:
        raise ValueError(f"segment start {start} must select exactly one node")
    node = matches[0]
    candidates = list(node.get("candidates", []))
    if not candidates or int(node.get("candidate_count", -1)) != len(candidates):
        raise ValueError("selected node candidate count is invalid")
    return {
        "schema": "wp31_static_pf_stop_node_v1",
        "segment": [int(value) for value in node["segment"]],
        "source_epoch_count": int(node["source_epoch_count"]),
        "candidate_count": len(candidates),
        "basin_csv": source["basin_csv"],
        "basin_csv_sha256": source["basin_csv_sha256"],
        "sample_stride_epochs": int(source["sample_stride_epochs"]),
        "radius_m": float(source["radius_m"]),
        "dedup_radius_m": float(source["dedup_radius_m"]),
        "max_candidates": int(source["max_candidates"]),
        "candidates": candidates,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache_json", type=Path)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.cache_json.read_text(encoding="utf-8"))
    result = materialize_node(source, args.start)
    result["source_cache"] = str(args.cache_json).replace("\\", "/")
    result["source_cache_sha256"] = _sha256(args.cache_json)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "candidates"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
