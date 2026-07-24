#!/usr/bin/env python3
"""Select exactly one resampled posterior with secondary-band DD code."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]


def select_unique_secondary_posterior(
    posteriors: list[dict[str, Any]],
    secondary_results: list[dict[str, Any]],
    *,
    min_evidence_epochs: int = 10,
    top_k: int = 3,
    max_secondary_median_m: float = 0.5,
    max_support_spread_m: float = 0.5,
) -> dict[str, Any]:
    if len(posteriors) != len(secondary_results):
        raise ValueError("posterior and secondary result counts differ")
    eligible = []
    audits = []
    for parent_id, (posterior, secondary) in enumerate(
        zip(posteriors, secondary_results, strict=True)
    ):
        audit: dict[str, Any] = {"parent_id": parent_id, "eligible": False}
        if posterior.get("reason") != "compact_rank_posterior_development":
            audit["reason"] = "no_compact_posterior"
            audits.append(audit)
            continue
        if secondary.get("pseudorange_family") != "secondary":
            audit["reason"] = "not_secondary_pseudorange"
            audits.append(audit)
            continue
        if int(secondary.get("evidence_epochs", 0)) < min_evidence_epochs:
            audit["reason"] = "insufficient_secondary_evidence"
            audits.append(audit)
            continue
        posterior_ids = {
            int(value) for value in posterior.get("selected_candidate_ids", [])
        }
        ranked = sorted(
            (
                row
                for row in secondary.get("candidates", [])
                if int(row["candidate_id"]) in posterior_ids
            ),
            key=lambda row: float(row["ddpr_median_abs_m"]),
        )
        support = ranked[:top_k]
        audit["support_ids"] = [int(row["candidate_id"]) for row in support]
        audit["support_secondary_median_m"] = [
            float(row["ddpr_median_abs_m"]) for row in support
        ]
        if len(support) < top_k:
            audit["reason"] = "insufficient_topk_support"
            audits.append(audit)
            continue
        if float(support[-1]["ddpr_median_abs_m"]) > max_secondary_median_m:
            audit["reason"] = "topk_secondary_gate_failed"
            audits.append(audit)
            continue
        positions = np.asarray([row["position_ecef"] for row in support])
        center = np.mean(positions, axis=0)
        spread = float(np.max(np.linalg.norm(positions - center, axis=1)))
        audit["support_spread_m"] = spread
        if spread > max_support_spread_m:
            audit["reason"] = "topk_support_not_compact"
            audits.append(audit)
            continue
        audit.update(
            {
                "eligible": True,
                "reason": "eligible",
                "position_ecef": center.tolist(),
                "posterior_score": float(posterior["selected_score"]),
            }
        )
        audits.append(audit)
        eligible.append(audit)
    base = {
        "selected_parent_id": None,
        "selected_candidate_ids": [],
        "production_promoted": False,
        "parent_audits": audits,
    }
    if not eligible:
        return {**base, "reason": "no_secondary_posterior"}
    if len(eligible) != 1:
        return {
            **base,
            "reason": "ambiguous_secondary_posteriors",
            "eligible_parent_ids": [int(row["parent_id"]) for row in eligible],
        }
    selected = eligible[0]
    return {
        **base,
        "selected_parent_id": int(selected["parent_id"]),
        "selected_candidate_ids": selected["support_ids"],
        "position_ecef": selected["position_ecef"],
        "support_secondary_median_m": selected["support_secondary_median_m"],
        "support_spread_m": selected["support_spread_m"],
        "reason": "unique_secondary_topk_posterior_development",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--posterior", type=Path, action="append", required=True)
    parser.add_argument("--secondary", type=Path, action="append", required=True)
    parser.add_argument("--min-evidence-epochs", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-secondary-median-m", type=float, default=0.5)
    parser.add_argument("--max-support-spread-m", type=float, default=0.5)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    posteriors = [json.loads(path.read_text(encoding="utf-8")) for path in args.posterior]
    secondary = [json.loads(path.read_text(encoding="utf-8")) for path in args.secondary]
    result = select_unique_secondary_posterior(
        posteriors,
        secondary,
        min_evidence_epochs=args.min_evidence_epochs,
        top_k=args.top_k,
        max_secondary_median_m=args.max_secondary_median_m,
        max_support_spread_m=args.max_support_spread_m,
    )
    result.update(
        {
            "schema": "wp32_unique_secondary_posterior_v1",
            "segment": posteriors[0]["segment"],
            "production_input_truth": False,
            "config": {
                "min_evidence_epochs": args.min_evidence_epochs,
                "top_k": args.top_k,
                "max_secondary_median_m": args.max_secondary_median_m,
                "max_support_spread_m": args.max_support_spread_m,
            },
        }
    )
    if result.get("position_ecef") is not None and args.data_dir is not None:
        import sys

        sys.path.insert(0, str(_ROOT / "python"))
        from gnss_gpu.io.ppc import PPCDatasetLoader

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
