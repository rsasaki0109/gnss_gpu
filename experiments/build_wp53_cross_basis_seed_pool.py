#!/usr/bin/env python3
"""Strip audit fields and export every truth-free hypothesis as a seed pool."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def build_pool(source: Path, *, reference_rank: int) -> dict[str, object]:
    source_bytes = source.read_bytes()
    payload = json.loads(source_bytes.decode("utf-8"))
    if bool(payload.get("production_input_truth", True)):
        raise ValueError("source hypothesis artifact is not production-safe")
    hypotheses = payload.get("hypotheses")
    if not isinstance(hypotheses, list) or not hypotheses:
        raise ValueError("source hypothesis artifact has no hypotheses")
    seeds = []
    for row in hypotheses:
        if not isinstance(row, dict):
            continue
        value = row.get("offset_ecef_m")
        if not isinstance(value, list) or len(value) != 3:
            continue
        seeds.append(
            {
                "source_seed_id": int(row["seed_id"]),
                "offset_ecef_m": [float(component) for component in value],
            }
        )
    if not seeds:
        raise ValueError("source hypothesis artifact has no valid offsets")
    return {
        "schema": "wp53_cross_basis_seed_pool_v1",
        "production_input_truth": False,
        "truth_usage": "none; audit fields are discarded",
        "source_reference_rank": int(reference_rank),
        "source_artifact": str(source).replace("\\", "/"),
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "seeds": seeds,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--reference-rank", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pool = build_pool(args.source, reference_rank=args.reference_rank)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(pool, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output} seeds={len(pool['seeds'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
