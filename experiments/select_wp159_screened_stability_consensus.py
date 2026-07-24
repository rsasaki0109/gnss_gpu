#!/usr/bin/env python3
"""Fuse direct cross-basis stability, block stability, carrier, and CP/PR ranks.

Same as WP138 except the dense-ranked stability families used for
ranking/family-rank checks/margin are screened down from four to three,
dropping ``cross_refit_disagreement_m`` from ranking only. The absolute
eligibility gate ``max_cross_refit_disagreement_m <= 0.10`` inside the
WP131 fuse step remains fully in force; this module never touches
eligibility.

WP159 is additionally scoped to DDPR-screened chains: the source
truth-free artifact must carry a non-empty top-level
``ddpr_excluded_satellites`` list (survives ``sanitize_wp55_cppr_candidates``
sanitization unchanged, since that script only strips ``audit_``-prefixed
per-hypothesis keys and two named truth-audit blocks). Chains without
screen evidence fail closed with ``reason == "screen_evidence_required"``
before any ranking is attempted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from select_wp131_cross_basis_cppr_consensus import select as fuse

_FAMILIES = (
    "block_spread_m",
    "max_cross_basis_carrier_rms_cycles",
    "cppr_rank_sum",
)


def _dense_ranks(values: list[float]) -> list[int]:
    ordered = {value: rank for rank, value in enumerate(sorted(set(values)), 1)}
    return [ordered[value] for value in values]


def rerank(
    fused: dict[str, Any],
    *,
    max_family_rank_fraction: float = 0.4,
    min_runner_margin: float = 0.2,
) -> dict[str, Any]:
    if bool(fused.get("production_input_truth", True)):
        raise ValueError("fusion result is not production-safe")
    modes = [dict(row) for row in fused.get("modes", [])]
    excluded = fused.get("ddpr_excluded_satellites")
    screen_eligible = isinstance(excluded, list) and len(excluded) > 0
    base = {
        "schema": "wp159_screened_stability_consensus_v1",
        "production_input_truth": False,
        "truth_usage": "none",
        "production_promoted": False,
        "segment": fused.get("segment"),
        "accepted": False,
        "reason": "stability_cppr_evidence_unavailable",
        "selected_candidate_id": None,
        "base_reason": fused.get("reason"),
        "mode_count": len(modes),
        "ranking_families": list(_FAMILIES),
        "ddpr_excluded_satellites": excluded if isinstance(excluded, list) else [],
    }
    # Eligibility precondition, checked before any ranking: WP159 is scoped
    # to DDPR-screened chains only. A missing/empty exclusion list fails
    # closed here, mirroring the missing-CP/PR abstain shape below.
    if not screen_eligible:
        base["reason"] = "screen_evidence_required"
        base["mode_count"] = 0
        return base
    if len(modes) < 2:
        return base
    for family in _FAMILIES:
        if any(family not in row for row in modes):
            return base
        ranks = _dense_ranks([float(row[family]) for row in modes])
        for row, rank in zip(modes, ranks, strict=True):
            row.setdefault("stability_family_ranks", {})[family] = rank
    for row in modes:
        row["stability_rank_sum"] = int(
            sum(row["stability_family_ranks"].values())
        )
    modes.sort(key=lambda row: (row["stability_rank_sum"], row["candidate_id"]))
    winner, runner = modes[:2]
    margin = float(
        (runner["stability_rank_sum"] - winner["stability_rank_sum"])
        / max(winner["stability_rank_sum"], 1)
    )
    family_limit = int(math.ceil(len(modes) * float(max_family_rank_fraction)))
    family_pass = max(winner["stability_family_ranks"].values()) <= family_limit
    margin_pass = margin >= float(min_runner_margin)
    accepted = family_pass and margin_pass
    base.update(
        {
            "accepted": accepted,
            "reason": (
                "unique_cross_basis_stability_cppr_mode"
                if accepted
                else "stability_cppr_family_or_margin_gate_failed"
            ),
            "selected_candidate_id": winner["candidate_id"] if accepted else None,
            "winner": winner,
            "runner": runner,
            "runner_margin": margin,
            "family_rank_limit": family_limit,
            "family_rank_pass": family_pass,
            "runner_margin_pass": margin_pass,
            "modes": modes,
            "gate": {
                "families": list(_FAMILIES),
                "max_family_rank_fraction": float(max_family_rank_fraction),
                "min_runner_margin": float(min_runner_margin),
                "base_absolute_gates": fused.get("gate"),
            },
        }
    )
    return base


def select(source: dict[str, Any], cross: dict[str, Any]) -> dict[str, Any]:
    fused = dict(fuse(source, cross))
    # wp131's fuse() returns a curated dict that does not carry arbitrary
    # source fields through; thread the DDPR screen evidence across
    # explicitly so rerank() can gate on it.
    fused["ddpr_excluded_satellites"] = source.get("ddpr_excluded_satellites")
    return rerank(fused)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--cross-basis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_bytes = args.source.read_bytes()
    cross_bytes = args.cross_basis.read_bytes()
    result = select(json.loads(source_bytes), json.loads(cross_bytes))
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
