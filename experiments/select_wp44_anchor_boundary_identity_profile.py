#!/usr/bin/env python3
"""Select a small stable identity profile immediately after an accepted anchor."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_boundary_identity_profile(
    source: dict[str, Any],
    anchor_summary: dict[str, Any],
    *,
    block_epochs: int = 55,
    min_evidence_epochs: int = 10,
    min_integer_arcs: int = 4,
    min_carrier_rows: int = 24,
    min_ddpr_rows: int = 40,
    max_carrier_rms_cycles: float = 0.5,
    max_common_offset_norm_m: float = 0.2,
    max_profile_offset_norm_m: float = 0.2,
    max_block_spread_m: float = 0.2,
    min_runner_offset_norm_m: float = 1.0,
) -> dict[str, Any]:
    if bool(source.get("production_input_truth", True)):
        raise ValueError("boundary candidate source is not truth-free")
    if not bool(anchor_summary.get("production_promoted", False)) or bool(
        anchor_summary.get("development_anchor_used", True)
    ):
        raise ValueError("anchor summary is not a production-only trajectory")
    start, end = (int(value) for value in source["segment"])
    if end - start != int(block_epochs):
        raise ValueError("boundary block length differs from the fixed gate")
    anchors = [
        row
        for row in anchor_summary.get("static_anchor_spans", [])
        if int(row["end"]) == start
    ]
    if len(anchors) != 1:
        raise ValueError(
            "boundary block is not adjacent to exactly one accepted anchor"
        )
    hypotheses = list(source.get("hypotheses", []))
    matches = [row for row in hypotheses if int(row.get("seed_id", -1)) == 0]
    if len(matches) != 1 or len(hypotheses) < 2:
        raise ValueError("boundary identity candidate is absent or has no runner")
    identity = matches[0]
    seed = np.asarray(identity["seed_offset_ecef_m"], dtype=np.float64)
    offset = np.asarray(identity["offset_ecef_m"], dtype=np.float64)
    profile = np.asarray(identity["block_offsets_ecef_m"], dtype=np.float64)
    if seed.shape != (3,) or offset.shape != (3,) or profile.shape != (4, 3):
        raise ValueError("boundary identity profile shape is invalid")
    if not all(np.isfinite(value).all() for value in (seed, offset, profile)):
        raise ValueError("boundary identity profile is nonfinite")
    runner_norm = min(
        float(np.linalg.norm(np.asarray(row["offset_ecef_m"], dtype=np.float64)))
        for row in hypotheses
        if int(row.get("seed_id", -1)) != 0
    )
    common_norm = float(np.linalg.norm(offset))
    profile_norm = float(np.max(np.linalg.norm(profile, axis=1)))
    diagnostics = list(source.get("stride_phase_diagnostics", []))
    selected_phase = int(source.get("selected_stride_phase", -1))
    selected_rows = [row for row in diagnostics if int(row["phase"]) == selected_phase]
    availability_pass = (
        source.get("stride_phase_mode") == "auto"
        and len(selected_rows) == 1
        and int(selected_rows[0]["evidence_epochs"])
        == max(int(row["evidence_epochs"]) for row in diagnostics)
    )
    gates = {
        "anchor_adjacency": True,
        "auto_phase_availability": availability_pass,
        "evidence_epochs": int(source.get("evidence_epochs", 0)) >= min_evidence_epochs,
        "zero_seed": float(np.linalg.norm(seed)) <= 1e-9,
        "integer_arcs": int(identity["integer_arcs"]) >= min_integer_arcs,
        "carrier_rows": int(identity["carrier_rows"]) >= min_carrier_rows,
        "ddpr_rows": int(identity["ddpr_rows"]) >= min_ddpr_rows,
        "carrier_rms": float(identity["carrier_rms_cycles"]) <= max_carrier_rms_cycles,
        "common_offset_norm": common_norm <= max_common_offset_norm_m,
        "profile_offset_norm": profile_norm <= max_profile_offset_norm_m,
        "block_spread": float(identity["block_spread_m"]) <= max_block_spread_m,
        "runner_offset_separation": runner_norm >= min_runner_offset_norm_m,
    }
    accepted = all(gates.values())
    return {
        "schema": "wp44_anchor_boundary_identity_profile_v1",
        "production_input_truth": False,
        "production_promoted": accepted,
        "segment": [start, end],
        "selected_candidate_id": 0 if accepted else None,
        "reason": "unique_anchor_boundary_identity_profile"
        if accepted
        else "anchor_boundary_identity_profile_gate_failed",
        "anchor": anchors[0],
        "selected_stride_phase": selected_phase,
        "offset_ecef_m": offset.tolist() if accepted else None,
        "profile_mode": "linear_bootstrap_centers" if accepted else None,
        "block_offsets_ecef_m": profile.tolist() if accepted else None,
        "diagnostics": {
            "common_offset_norm_m": common_norm,
            "max_profile_offset_norm_m": profile_norm,
            "block_spread_m": float(identity["block_spread_m"]),
            "runner_min_offset_norm_m": runner_norm,
            "integer_arcs": int(identity["integer_arcs"]),
            "carrier_rows": int(identity["carrier_rows"]),
            "ddpr_rows": int(identity["ddpr_rows"]),
            "carrier_rms_cycles": float(identity["carrier_rms_cycles"]),
        },
        "gates": gates,
        "config": {
            "block_epochs": block_epochs,
            "min_evidence_epochs": min_evidence_epochs,
            "min_integer_arcs": min_integer_arcs,
            "min_carrier_rows": min_carrier_rows,
            "min_ddpr_rows": min_ddpr_rows,
            "max_carrier_rms_cycles": max_carrier_rms_cycles,
            "max_common_offset_norm_m": max_common_offset_norm_m,
            "max_profile_offset_norm_m": max_profile_offset_norm_m,
            "max_block_spread_m": max_block_spread_m,
            "min_runner_offset_norm_m": min_runner_offset_norm_m,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_json", type=Path)
    parser.add_argument("anchor_summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.candidate_json.read_text(encoding="utf-8"))
    anchors = json.loads(args.anchor_summary.read_text(encoding="utf-8"))
    result = select_boundary_identity_profile(source, anchors)
    result["input_sha256"] = {
        "candidate_source": _sha256(args.candidate_json),
        "anchor_summary": _sha256(args.anchor_summary),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
