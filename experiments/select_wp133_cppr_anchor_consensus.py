#!/usr/bin/env python3
"""Select a cross-basis mode only when independent CP/PR is the top anchor."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from select_wp131_cross_basis_cppr_consensus import select as fuse


def anchor(
    fused: dict[str, Any],
    *,
    max_other_family_rank_fraction: float = 0.4,
    min_runner_margin: float = 0.2,
) -> dict[str, Any]:
    """Apply the narrow CP/PR-anchor escape hatch to a WP131 fusion result."""
    result = {
        "schema": "wp133_cppr_anchor_consensus_v1",
        "production_input_truth": False,
        "truth_usage": "none",
        "production_promoted": False,
        "segment": fused.get("segment"),
        "accepted": False,
        "reason": "cppr_anchor_evidence_unavailable",
        "selected_candidate_id": None,
        "base_reason": fused.get("reason"),
        "mode_count": int(fused.get("mode_count", 0)),
    }
    if bool(fused.get("production_input_truth", True)):
        raise ValueError("fusion result is not production-safe")
    winner = fused.get("winner")
    runner = fused.get("runner")
    margin = fused.get("runner_margin")
    if not isinstance(winner, dict) or not isinstance(runner, dict) or margin is None:
        return result

    ranks = winner.get("family_ranks")
    mode_count = int(fused["mode_count"])
    if not isinstance(ranks, dict) or mode_count < 2:
        return result
    other_limit = int(
        math.ceil(mode_count * float(max_other_family_rank_fraction))
    )
    cppr_anchor_pass = int(ranks["cppr"]) == 1
    other_family_pass = max(
        int(ranks["cross_consensus"]), int(ranks["carrier_rms"])
    ) <= other_limit
    runner_margin_pass = float(margin) >= float(min_runner_margin)
    accepted = cppr_anchor_pass and other_family_pass and runner_margin_pass
    result.update(
        {
            "accepted": accepted,
            "reason": (
                "unique_cppr_anchor_cross_basis_mode"
                if accepted
                else "cppr_anchor_family_or_margin_gate_failed"
            ),
            "selected_candidate_id": winner["candidate_id"] if accepted else None,
            "winner": winner,
            "runner": runner,
            "runner_margin": float(margin),
            "cppr_anchor_pass": cppr_anchor_pass,
            "other_family_rank_limit": other_limit,
            "other_family_pass": other_family_pass,
            "runner_margin_pass": runner_margin_pass,
            "modes": fused["modes"],
            "gate": {
                "required_cppr_rank": 1,
                "max_other_family_rank_fraction": float(
                    max_other_family_rank_fraction
                ),
                "min_runner_margin": float(min_runner_margin),
                "base_absolute_gates": fused.get("gate"),
            },
        }
    )
    return result


def select(source: dict[str, Any], cross: dict[str, Any]) -> dict[str, Any]:
    return anchor(fuse(source, cross))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--cross-basis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_bytes = args.source.read_bytes()
    cross_bytes = args.cross_basis.read_bytes()
    result = select(
        json.loads(source_bytes.decode("utf-8")),
        json.loads(cross_bytes.decode("utf-8")),
    )
    result["input_sha256"] = {
        "source": hashlib.sha256(source_bytes).hexdigest(),
        "cross_basis": hashlib.sha256(cross_bytes).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
