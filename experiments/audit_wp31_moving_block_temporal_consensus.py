#!/usr/bin/env python3
"""Audit non-overlapping block support for a frozen primary ambiguity gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def temporal_support_pairs(
    primary_rows: list[dict[str, Any]],
    support_rows: list[dict[str, Any]],
    primary_ids: list[int],
    *,
    max_offset_distance_m: float = 0.5,
    min_support_integer_arcs: int = 4,
    min_support_carrier_rows: int = 8,
    min_support_ddpr_rows: int = 40,
    max_support_carrier_rms_cycles: float = 0.5,
) -> list[dict[str, float | int]]:
    primary_by_id = {int(row["seed_id"]): row for row in primary_rows}
    output = []
    for primary_id in primary_ids:
        primary = primary_by_id[primary_id]
        primary_offset = np.asarray(primary["offset_ecef_m"])
        for support in support_rows:
            if int(support["integer_arcs"]) < min_support_integer_arcs:
                continue
            if int(support["carrier_rows"]) < min_support_carrier_rows:
                continue
            if int(support["ddpr_rows"]) < min_support_ddpr_rows:
                continue
            if float(support["carrier_rms_cycles"]) > max_support_carrier_rms_cycles:
                continue
            distance = float(np.linalg.norm(primary_offset - np.asarray(support["offset_ecef_m"])))
            if distance <= max_offset_distance_m:
                output.append({"primary_seed_id": primary_id, "support_seed_id": int(support["seed_id"]), "offset_distance_m": distance})
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("primary_artifact", type=Path); parser.add_argument("support_artifact", type=Path)
    parser.add_argument("primary_gate_audit", type=Path); parser.add_argument("--scope", choices=("development", "holdout"), required=True)
    parser.add_argument("--output", type=Path, required=True); args = parser.parse_args()
    primary = json.loads(args.primary_artifact.read_text(encoding="utf-8"))
    support = json.loads(args.support_artifact.read_text(encoding="utf-8"))
    gate = json.loads(args.primary_gate_audit.read_text(encoding="utf-8"))
    if list(support["segment"])[1] != list(primary["segment"])[0]:
        raise RuntimeError("support and primary blocks must be exactly adjacent and non-overlapping")
    primary_ids = [int(value) for value in gate["passing_seed_ids"]]
    pairs = temporal_support_pairs(primary["hypotheses"], support["hypotheses"], primary_ids)
    accepted = len(primary_ids) == 1 and len(pairs) == 1 and pairs[0]["primary_seed_id"] == primary_ids[0]
    primary_by_id = {int(row["seed_id"]): row for row in primary["hypotheses"]}
    result = {
        "schema": "wp31_moving_block_temporal_consensus_audit_v1", "scope": args.scope,
        "production_input_truth": False, "truth_usage": "post_consensus_audit_only",
        "support_segment": support["segment"], "primary_segment": primary["segment"],
        "frozen_gate": {"max_offset_distance_m": 0.5, "min_support_integer_arcs": 4, "min_support_carrier_rows": 8, "min_support_ddpr_rows": 40, "max_support_carrier_rms_cycles": 0.5},
        "primary_gate_seed_ids": primary_ids, "support_pairs": pairs,
        "consensus_selected_seed_id": primary_ids[0] if accepted else None,
        "production_promoted": bool(accepted and args.scope == "holdout"),
        "selection_reason": (
            "development_adjacent_block_unique_offset_consensus"
            if accepted and args.scope == "development"
            else "holdout_adjacent_block_unique_offset_consensus"
            if accepted else "temporal_consensus_rejected"
        ),
        "selected_post_consensus_audit": (
            {"audit_median_error_m": primary_by_id[primary_ids[0]]["audit_median_error_m"], "audit_sub50cm_epochs": primary_by_id[primary_ids[0]]["audit_sub50cm_epochs"]}
            if accepted else None
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
