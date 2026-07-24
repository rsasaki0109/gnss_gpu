#!/usr/bin/env python3
"""Promote a holdout-validated cross-basis/CPPR constant offset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from select_wp131_cross_basis_cppr_consensus import select

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
    "selected_candidate_id",
    "winner",
    "runner",
    "runner_margin",
    "family_rank_limit",
    "family_rank_pass",
    "runner_margin_pass",
    "mode_count",
    "modes",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _recompute(source_path: Path, cross_path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    source = json.loads(source_path.read_text(encoding="utf-8"))
    cross = json.loads(cross_path.read_text(encoding="utf-8"))
    return select(source, cross), {
        "source": _sha256(source_path),
        "cross_basis": _sha256(cross_path),
    }


def _verify(
    recomputed: dict[str, Any], selection_path: Path, hashes: dict[str, str]
) -> dict[str, Any]:
    stored = json.loads(selection_path.read_text(encoding="utf-8"))
    if any(stored.get(key) != recomputed.get(key) for key in _COMPARABLE_KEYS):
        raise RuntimeError(f"stored selection does not match: {selection_path}")
    if stored.get("input_sha256") != hashes:
        raise RuntimeError(f"selection input hashes do not match: {selection_path}")
    return {
        "path": str(selection_path),
        "sha256": _sha256(selection_path),
        "accepted": bool(stored["accepted"]),
        "reason": stored["reason"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--cross-basis", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    for name in ("holdout-a", "holdout-b"):
        parser.add_argument(f"--{name}-source", type=Path, required=True)
        parser.add_argument(f"--{name}-cross-basis", type=Path, required=True)
        parser.add_argument(f"--{name}-selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    target, target_hashes = _recompute(args.source, args.cross_basis)
    target_stored = _verify(target, args.selection, target_hashes)
    if not target["accepted"]:
        raise RuntimeError("target cross-basis/CPPR selector does not pass")

    holdouts = {}
    for label in ("holdout_a", "holdout_b"):
        source = getattr(args, f"{label}_source")
        cross = getattr(args, f"{label}_cross_basis")
        selection = getattr(args, f"{label}_selection")
        recomputed, hashes = _recompute(source, cross)
        holdouts[label] = _verify(recomputed, selection, hashes)
        if recomputed["accepted"]:
            raise RuntimeError(f"{label} did not fail closed")

    preserved = {}
    for path, expected in _M4_HASHES.items():
        actual = _sha256(path).upper()
        if actual != expected:
            raise RuntimeError(f"M4 artifact changed: {path}")
        preserved[str(path)] = actual

    source_payload = json.loads(args.source.read_text(encoding="utf-8"))
    winner = target["winner"]
    result = {
        "schema": "wp131_cross_basis_cppr_promotion_v1",
        "production_input_truth": False,
        "truth_usage": "none",
        "production_promoted": True,
        "reason": "unique_cross_basis_cppr_mode",
        "segment": source_payload["segment"],
        "profile_mode": "constant",
        "offset_ecef_m": winner["offset_ecef_m"],
        "block_offsets_ecef_m": winner["block_offsets_ecef_m"],
        "candidate_id": winner["candidate_id"],
        "family_ranks": winner["family_ranks"],
        "runner_margin": target["runner_margin"],
        "selection": target_stored,
        "holdouts": holdouts,
        "input_sha256": target_hashes,
        "m4_preserved_sha256": preserved,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
