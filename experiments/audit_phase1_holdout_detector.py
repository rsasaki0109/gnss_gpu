#!/usr/bin/env python3
"""Replay truth-free portions of the four mandatory negative holdouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from gnss_gpu.evaluation_contract import MANDATORY_NEGATIVE_HOLDOUTS, write_json
from gnss_gpu.evidence import (
    BasinEvidence,
    EvidenceFamily,
    EvidenceSample,
    UnsafeAcceptanceDetector,
)


def _sample(
    family: EvidenceFamily,
    epoch: int,
    residual: float,
    scale: float,
    *,
    count: int = 1,
) -> EvidenceSample:
    return EvidenceSample(
        family=family,
        epoch=epoch,
        residual=residual,
        scale=scale,
        sample_count=count,
        source="historical_truth_free_lock",
    )


def holdout_basins(holdout_id: str, payload: dict[str, Any]) -> tuple[BasinEvidence, ...]:
    """Adapt only fields that were available to the production selector."""

    target = payload.get("target", payload)
    start, end = target.get("segment", payload.get("segment"))
    if holdout_id == "nagoya_wp53":
        audits = payload["posterior_audits"]
        margin = audits["three_basis_carrier_rms_sum"]["runner_margin_over_winner"]
        return (
            BasinEvidence(
                "wp53_winner",
                (
                    _sample(EvidenceFamily.CARRIER_CONTINUITY, start, 1.0 - margin, 1.0),
                    _sample(EvidenceFamily.SATELLITE_ARC, end, 0.7, 1.0, count=29),
                ),
            ),
        )
    if holdout_id == "tokyo_wp129":
        refinement = target["grid_refinement"]
        margin = refinement["runner_margin"]
        return tuple(
            BasinEvidence(
                f"wp129_mode_{index}",
                (
                    _sample(EvidenceFamily.TDCP, start, 0.30 + index * margin, 1.0),
                    _sample(EvidenceFamily.CARRIER_CONTINUITY, end, 0.25 + index * margin, 1.0),
                    _sample(EvidenceFamily.ROAD_HEIGHT, end, 0.20 + index * margin, 1.0),
                ),
            )
            for index in range(2)
        )
    if holdout_id == "tokyo_wp156":
        count = int(target["gsi_compatible_samples"])
        return (
            BasinEvidence(
                "wp156_selector_winner",
                (
                    _sample(EvidenceFamily.SATELLITE_ARC, start, 0.2, 1.0, count=count),
                    _sample(EvidenceFamily.ROAD_HEIGHT, end, 0.2, 1.0, count=count),
                ),
            ),
        )
    if holdout_id == "tokyo_wp168":
        screen = target["ddpr_screen"]
        return (
            BasinEvidence(
                "wp168_selector_winner",
                (
                    _sample(
                        EvidenceFamily.CARRIER_CONTINUITY,
                        start,
                        0.15,
                        1.0,
                        count=int(screen["evidence_epochs"]),
                    ),
                    _sample(EvidenceFamily.LOS_NLOS, end, 0.15, 1.0),
                ),
            ),
        )
    raise KeyError(f"unsupported historical holdout: {holdout_id}")


def audit(repo_root: Path) -> dict[str, Any]:
    detector = UnsafeAcceptanceDetector()
    results = []
    for spec in MANDATORY_NEGATIVE_HOLDOUTS:
        payload = json.loads((repo_root / spec.lock_path).read_text(encoding="utf-8"))
        decision = detector.decide(holdout_basins(spec.holdout_id, payload))
        results.append(
            {
                "holdout_id": spec.holdout_id,
                "accepted": decision.accepted,
                "reason": decision.reason,
                "unsafe_reasons": list(decision.unsafe_reasons),
            }
        )
    rejected = sum(not result["accepted"] for result in results)
    return {
        "schema": "gnss_gpu_phase1_holdout_detector_audit_v1",
        "production_input_truth": False,
        "rejected_holdouts": rejected,
        "minimum_required_rejections": 2,
        "passed": rejected >= 2,
        "results": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = audit(args.repo_root.resolve())
    if args.output is not None:
        write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
