#!/usr/bin/env python3
"""Select a unique static candidate by primary/secondary/tertiary DDPR ranks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def _family(document: dict[str, Any], expected: str) -> None:
    if bool(document.get("production_input_truth", True)):
        raise ValueError(f"{expected} DDPR artifact is not truth-free")
    declared = document.get("pseudorange_family")
    if expected == "primary" and declared is None:
        if document.get("schema") != "wp31_static_ddpr_integrity_v1":
            raise ValueError("legacy primary DDPR artifact has an unsupported schema")
    elif declared != expected:
        raise ValueError(f"expected {expected} DDPR artifact, got {declared!r}")
    if document.get("calibration") is not None:
        raise ValueError("trifrequency selector requires uncalibrated DDPR evidence")


def select_trifrequency_ddpr_rank(
    primary: dict[str, Any],
    secondary: dict[str, Any],
    tertiary: dict[str, Any],
    *,
    min_evidence_epochs: int = 10,
    max_family_rank_fraction: float = 0.2,
    min_runner_margin: float = 0.2,
) -> dict[str, Any]:
    documents = (primary, secondary, tertiary)
    for document, family in zip(documents, ("primary", "secondary", "tertiary")):
        _family(document, family)
    segment = primary.get("segment")
    if any(document.get("segment") != segment for document in documents[1:]):
        raise ValueError("DDPR artifact segments differ")
    source_hash = primary.get("candidate_source_sha256")
    if not source_hash or any(
        document.get("candidate_source_sha256") != source_hash
        for document in documents[1:]
    ):
        raise ValueError("DDPR candidate provenance differs")
    evidence = [int(document.get("evidence_epochs", 0)) for document in documents]
    if min(evidence) < int(min_evidence_epochs):
        raise ValueError("DDPR evidence is below the fixed minimum")
    if not 0.0 < float(max_family_rank_fraction) <= 1.0:
        raise ValueError("family rank fraction must be in (0, 1]")
    if float(min_runner_margin) < 0.0:
        raise ValueError("runner margin must be nonnegative")

    mappings = [
        {int(row["candidate_id"]): row for row in document.get("candidates", [])}
        for document in documents
    ]
    candidate_ids = set(mappings[0])
    if len(candidate_ids) < 2 or any(set(mapping) != candidate_ids for mapping in mappings[1:]):
        raise ValueError("DDPR candidate sets differ or are too small")
    rows: list[dict[str, Any]] = []
    for candidate_id in sorted(candidate_ids):
        family_rows = [mapping[candidate_id] for mapping in mappings]
        positions = [
            np.asarray(row["position_ecef"], dtype=np.float64).reshape(3)
            for row in family_rows
        ]
        if not all(np.isfinite(position).all() for position in positions):
            raise ValueError("candidate position is nonfinite")
        if any(not np.allclose(positions[0], position, atol=1e-6, rtol=0.0) for position in positions[1:]):
            raise ValueError("candidate positions differ between DDPR families")
        ranks = [int(row["ddpr_median_abs_m_rank"]) for row in family_rows]
        if any(rank < 1 for rank in ranks):
            raise ValueError("DDPR ranks must be one-based positive integers")
        rows.append(
            {
                "candidate_id": candidate_id,
                "position_ecef": positions[0].tolist(),
                "family_ranks": dict(zip(("primary", "secondary", "tertiary"), ranks)),
                "family_medians_m": dict(
                    zip(
                        ("primary", "secondary", "tertiary"),
                        [float(row["ddpr_median_abs_m"]) for row in family_rows],
                    )
                ),
                "rank_sum": int(sum(ranks)),
            }
        )
    rows.sort(key=lambda row: (row["rank_sum"], row["candidate_id"]))
    winner, runner = rows[:2]
    runner_margin = float(
        (runner["rank_sum"] - winner["rank_sum"]) / winner["rank_sum"]
    )
    rank_limit = int(math.ceil(len(rows) * float(max_family_rank_fraction)))
    family_rank_pass = max(winner["family_ranks"].values()) <= rank_limit
    margin_pass = runner_margin >= float(min_runner_margin)
    accepted = family_rank_pass and margin_pass
    return {
        "selected_candidate_id": winner["candidate_id"] if accepted else None,
        "reason": (
            "unique_trifrequency_ddpr_rank_consensus"
            if accepted
            else "trifrequency_ddpr_rank_gate_failed"
        ),
        "position_ecef": winner["position_ecef"] if accepted else None,
        "winner": winner,
        "runner": runner,
        "runner_margin": runner_margin,
        "family_rank_limit": rank_limit,
        "family_rank_pass": family_rank_pass,
        "runner_margin_pass": margin_pass,
        "evidence_epochs_by_family": dict(
            zip(("primary", "secondary", "tertiary"), evidence)
        ),
        "candidate_count": len(rows),
        "candidates": (
            []
            if not accepted
            else [
                {
                    "candidate_id": winner["candidate_id"],
                    "position_ecef": winner["position_ecef"],
                }
            ]
        ),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("primary_json", type=Path)
    parser.add_argument("secondary_json", type=Path)
    parser.add_argument("tertiary_json", type=Path)
    parser.add_argument("--min-evidence-epochs", type=int, default=10)
    parser.add_argument("--max-family-rank-fraction", type=float, default=0.2)
    parser.add_argument("--min-runner-margin", type=float, default=0.2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = (args.primary_json, args.secondary_json, args.tertiary_json)
    documents = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    selected = select_trifrequency_ddpr_rank(
        *documents,
        min_evidence_epochs=args.min_evidence_epochs,
        max_family_rank_fraction=args.max_family_rank_fraction,
        min_runner_margin=args.min_runner_margin,
    )
    result = {
        "schema": "wp38_trifrequency_ddpr_rank_development_v1",
        "production_input_truth": False,
        "production_promoted": False,
        "segment": documents[0]["segment"],
        "config": {
            "min_evidence_epochs": args.min_evidence_epochs,
            "max_family_rank_fraction": args.max_family_rank_fraction,
            "min_runner_margin": args.min_runner_margin,
        },
        "input_sha256": dict(
            zip(("primary", "secondary", "tertiary"), map(_sha256, paths))
        ),
        **selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
