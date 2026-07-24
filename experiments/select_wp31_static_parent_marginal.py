#!/usr/bin/env python3
"""Select a compact static parent mode by marginalizing tied wide-lane children."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "python"))

from gnss_gpu.io.ppc import PPCDatasetLoader  # noqa: E402


def select_parent_marginal(
    candidates: list[dict[str, Any]],
    widelane: list[dict[str, Any]],
    *,
    evidence_epochs: int,
    min_evidence_epochs: int = 30,
    members: int = 2,
    max_widelane_m: float = 0.5,
    max_widelane_ratio: float = 1.1,
    max_widelane_gap_m: float = 0.05,
    max_spread_m: float = 0.5,
) -> dict[str, Any]:
    base = {
        "selected_candidate_id": None,
        "reason": "parent_marginal_rejected",
        "evidence_epochs": int(evidence_epochs),
    }
    if int(evidence_epochs) < int(min_evidence_epochs):
        return {**base, "reason": "insufficient_evidence_epochs"}
    positions = {int(row["candidate_id"]): row for row in candidates}
    ranked = sorted(
        (
            row
            for row in widelane
            if np.isfinite(float(row.get("widelane_median_abs_m", np.inf)))
            and int(row["candidate_id"]) in positions
        ),
        key=lambda row: float(row["widelane_median_abs_m"]),
    )
    if len(ranked) < int(members):
        return {**base, "reason": "insufficient_widelane_children"}
    selected = ranked[: int(members)]
    best = float(selected[0]["widelane_median_abs_m"])
    worst = float(selected[-1]["widelane_median_abs_m"])
    ratio = worst / max(best, 1e-9)
    gap = worst - best
    if best > float(max_widelane_m):
        return {**base, "reason": "weak_absolute_widelane", "best_widelane_m": best}
    if ratio > float(max_widelane_ratio) and gap > float(max_widelane_gap_m):
        return {
            **base,
            "reason": "separated_widelane_children",
            "widelane_ratio": ratio,
            "widelane_gap_m": gap,
        }
    child_positions = np.asarray(
        [positions[int(row["candidate_id"])]["position_ecef"] for row in selected],
        dtype=np.float64,
    )
    center = np.mean(child_positions, axis=0)
    spread = float(np.max(np.linalg.norm(child_positions - center, axis=1)))
    if spread > float(max_spread_m):
        return {**base, "reason": "noncompact_widelane_children", "cluster_spread_m": spread}
    representative = min(
        selected,
        key=lambda row: float(
            np.linalg.norm(
                np.asarray(positions[int(row["candidate_id"])]["position_ecef"]) - center
            )
        ),
    )
    return {
        **base,
        "selected_candidate_id": int(representative["candidate_id"]),
        "reason": "compact_widelane_parent_marginal",
        "cluster_member_ids": [int(row["candidate_id"]) for row in selected],
        "cluster_members": len(selected),
        "cluster_spread_m": spread,
        "best_widelane_m": best,
        "widelane_ratio": ratio,
        "widelane_gap_m": gap,
        "position_ecef": center.tolist(),
    }


def require_global_comparison_for_conditioned_parent(
    result: dict[str, Any], source: dict[str, Any]
) -> dict[str, Any]:
    parent_id = source.get("seed_parent_candidate_id")
    if parent_id is None:
        return result
    return {
        "selected_candidate_id": None,
        "reason": "parent_conditioned_requires_global_comparison",
        "evidence_epochs": int(result["evidence_epochs"]),
        "seed_parent_candidate_id": int(parent_id),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidates_json", type=Path)
    parser.add_argument("widelane_json", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.candidates_json.read_text(encoding="utf-8"))
    wl = json.loads(args.widelane_json.read_text(encoding="utf-8"))
    result = select_parent_marginal(
        list(source["candidates"]),
        list(wl["candidates"]),
        evidence_epochs=int(wl["evidence_epochs"]),
    )
    result = require_global_comparison_for_conditioned_parent(result, source)
    result["segment"] = [int(value) for value in source["segment"]]
    if result.get("position_ecef") is not None and args.data_dir is not None:
        start, end = result["segment"]
        _times, truth = PPCDatasetLoader(args.data_dir).load_ground_truth()
        segment_truth = np.asarray(truth[start:end], dtype=np.float64)
        segment_truth = segment_truth[np.isfinite(segment_truth).all(axis=1)]
        if not len(segment_truth):
            raise RuntimeError("static segment has no finite audit truth")
        truth_position = np.median(segment_truth, axis=0)
        result["selected_audit_error_m"] = float(
            np.linalg.norm(np.asarray(result["position_ecef"]) - truth_position)
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
