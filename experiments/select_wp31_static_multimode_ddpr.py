#!/usr/bin/env python3
"""Promote a compact carrier-rank cluster with absolute DDPR consensus."""

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


def select_multimode_ddpr(
    cluster: dict[str, Any],
    ddpr: dict[str, Any],
    *,
    min_evidence_epochs: int = 10,
    max_ddpr_median_m: float = 0.5,
    min_support_members: int = 2,
    max_support_spread_m: float = 0.5,
) -> dict[str, Any]:
    base = {
        "selected_candidate_id": None,
        "selected_candidate_ids": [],
        "reason": "multimode_ddpr_consensus_rejected",
        "production_promoted": False,
    }
    if cluster.get("reason") != "compact_multimode_rank_cluster_development":
        return {**base, "reason": "no_multimode_cluster"}
    if int(ddpr.get("evidence_epochs", 0)) < min_evidence_epochs:
        return {**base, "reason": "insufficient_ddpr_evidence"}
    cluster_ids = {int(value) for value in cluster.get("selected_candidate_ids", [])}
    rows = {
        int(row["candidate_id"]): row for row in ddpr.get("candidates", [])
        if int(row["candidate_id"]) in cluster_ids
    }
    support = sorted(
        candidate_id for candidate_id, row in rows.items()
        if float(row["ddpr_median_abs_m"]) <= max_ddpr_median_m
    )
    if len(support) < min_support_members:
        return {**base, "reason": "insufficient_ddpr_cluster_support", "support_ids": support}
    positions = np.asarray([rows[candidate_id]["position_ecef"] for candidate_id in support])
    center = np.mean(positions, axis=0)
    spread = float(np.max(np.linalg.norm(positions - center, axis=1)))
    if spread > max_support_spread_m:
        return {**base, "reason": "ddpr_cluster_support_not_compact", "support_ids": support, "support_spread_m": spread}
    return {
        **base,
        "selected_candidate_id": int(support[0]),
        "selected_candidate_ids": support,
        "support_ids": support,
        "position_ecef": center.tolist(),
        "reason": "multimode_ddpr_consensus",
        "production_promoted": True,
        "ddpr_evidence_epochs": int(ddpr["evidence_epochs"]),
        "support_ddpr_median_m": {
            str(candidate_id): float(rows[candidate_id]["ddpr_median_abs_m"])
            for candidate_id in support
        },
        "support_spread_m": spread,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cluster_json", type=Path)
    parser.add_argument("ddpr_json", type=Path)
    parser.add_argument("--min-evidence-epochs", type=int, default=10)
    parser.add_argument("--max-ddpr-median-m", type=float, default=0.5)
    parser.add_argument("--min-support-members", type=int, default=2)
    parser.add_argument("--max-support-spread-m", type=float, default=0.5)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cluster = json.loads(args.cluster_json.read_text(encoding="utf-8"))
    ddpr = json.loads(args.ddpr_json.read_text(encoding="utf-8"))
    result = select_multimode_ddpr(
        cluster, ddpr,
        min_evidence_epochs=args.min_evidence_epochs,
        max_ddpr_median_m=args.max_ddpr_median_m,
        min_support_members=args.min_support_members,
        max_support_spread_m=args.max_support_spread_m,
    )
    result["schema"] = "wp31_static_multimode_ddpr_consensus_v1"
    result["segment"] = [int(value) for value in cluster["segment"]]
    result["production_input_truth"] = False
    result["config"] = {
        "min_evidence_epochs": args.min_evidence_epochs,
        "max_ddpr_median_m": args.max_ddpr_median_m,
        "min_support_members": args.min_support_members,
        "max_support_spread_m": args.max_support_spread_m,
    }
    if result.get("position_ecef") is not None and args.data_dir is not None:
        start, end = result["segment"]
        data = PPCDatasetLoader(args.data_dir).load_experiment_data(max_epochs=end)
        truth = np.asarray(data["ground_truth"][start:end], dtype=np.float64)
        truth = truth[np.isfinite(truth).all(axis=1)]
        result["selected_audit_error_m"] = float(
            np.linalg.norm(np.asarray(result["position_ecef"]) - np.median(truth, axis=0))
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
