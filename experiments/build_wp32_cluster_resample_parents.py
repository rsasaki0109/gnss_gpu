#!/usr/bin/env python3
"""Materialize every weak multimode component that passes a proposal gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def proposal_parents(
    clusters: list[dict[str, Any]],
    *,
    min_members: int = 3,
    min_score: float = 0.4,
    max_parents: int = 3,
) -> list[dict[str, Any]]:
    eligible = [
        row
        for row in clusters
        if int(row["members"]) >= min_members and float(row["score"]) >= min_score
    ]
    eligible.sort(key=lambda row: (-float(row["score"]), float(row["spread_m"])))
    output = []
    for candidate_id, row in enumerate(eligible[:max_parents]):
        position = np.asarray(row["position_ecef"], dtype=np.float64).reshape(3)
        if not np.isfinite(position).all():
            raise ValueError("cluster center must be finite")
        output.append(
            {
                "candidate_id": candidate_id,
                "proposal_kind": "weak_multimode_component_parent",
                "position_ecef": position.tolist(),
                "source_member_ids": [int(value) for value in row["member_ids"]],
                "source_members": int(row["members"]),
                "source_score": float(row["score"]),
                "source_spread_m": float(row["spread_m"]),
            }
        )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cluster_json", type=Path)
    parser.add_argument("--min-members", type=int, default=3)
    parser.add_argument("--min-score", type=float, default=0.4)
    parser.add_argument("--max-parents", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_bytes = args.cluster_json.read_bytes()
    source = json.loads(source_bytes)
    parents = proposal_parents(
        list(source.get("clusters", [])),
        min_members=args.min_members,
        min_score=args.min_score,
        max_parents=args.max_parents,
    )
    if not parents:
        parser.error("no cluster passes the resample proposal gate")
    result = {
        "schema": "wp32_cluster_resample_parents_v1",
        "production_input_truth": False,
        "production_promoted": False,
        "segment": source["segment"],
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "config": {
            "min_members": args.min_members,
            "min_score": args.min_score,
            "max_parents": args.max_parents,
        },
        "candidates": parents,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
