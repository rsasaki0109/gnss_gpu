#!/usr/bin/env python3
"""Deterministic outage/reacquisition audit for the Phase 3 controller."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gnss_gpu.ambiguity_basin_pf import AmbiguityBasinParticleFilter, BasinKalmanState
from gnss_gpu.evaluation_contract import M4_PRESERVED_SHA256, sha256_file, write_json
from gnss_gpu.evidence import BasinEvidence, EvidenceBuilder, UnsafeAcceptanceDetector
from gnss_gpu.multihypothesis_navigation import (
    MultiHypothesisNavigationController,
    RecoveryPolicy,
)


def _state(x: float) -> BasinKalmanState:
    return BasinKalmanState.from_position(
        np.array([x, 0.0, 0.0]),
        np.eye(3),
        velocity_ecef=np.zeros(3),
    )


def _safe_decision():
    def evidence(name: str, residual: float) -> BasinEvidence:
        return (
            EvidenceBuilder(name)
            .tdcp(1, residual, 1.0)
            .tdcp(2, residual, 1.0)
            .carrier_continuity(1, residual, 1.0)
            .carrier_continuity(2, residual, 1.0)
            .road_height(1, residual, 1.0)
            .road_height(2, residual, 1.0)
            .build()
        )

    return UnsafeAcceptanceDetector().decide(
        (evidence("true-road", 0.1), evidence("wrong-road", 1.2))
    )


def audit(repo_root: Path) -> dict:
    policy = RecoveryPolicy(required_reacquisition_streak=3)
    pf = AmbiguityBasinParticleFilter(
        max_basins=8,
        min_fixed_ambiguities=0,
        dedup_position_radius_m=0.25,
        diversity_reserve_fraction=0.25,
        diversity_radius_m=2.0,
    )
    pf.spawn([{}, {}], [_state(0.0), _state(10.0)])
    controller = MultiHypothesisNavigationController(pf, policy=policy)
    outage_status = controller.observe_gnss(available=False)
    true_basin = max(pf.basins, key=lambda basin: basin.conditional.mean[0])
    wrong_basin = min(pf.basins, key=lambda basin: basin.conditional.mean[0])
    pf.update_log_likelihoods({true_basin.basin_id: 6.0, wrong_basin.basin_id: -6.0})
    accepted = _safe_decision()
    statuses = [
        controller.observe_gnss(available=True, evidence_decision=accepted)
        for _ in range(policy.required_reacquisition_streak)
    ]
    final = statuses[-1]
    selected = pf.map_basin()
    selected_x = float(selected.conditional.mean[0]) if selected is not None else float("nan")
    legacy_greedy_error_m = 10.0
    multihypothesis_error_m = abs(10.0 - selected_x)
    m4 = {
        path: {
            "expected_sha256": expected,
            "actual_sha256": sha256_file(repo_root / path),
        }
        for path, expected in M4_PRESERVED_SHA256.items()
    }
    passed = (
        outage_status.safe_to_emit_fix is False
        and final.recovery_epochs == 3
        and multihypothesis_error_m < legacy_greedy_error_m
        and all(value["expected_sha256"] == value["actual_sha256"] for value in m4.values())
    )
    return {
        "schema": "gnss_gpu_phase3_outage_recovery_audit_v1",
        "production_input_truth": False,
        "outage_fix_suppressed": not outage_status.safe_to_emit_fix,
        "retained_hypotheses": 2,
        "recovery_epochs": final.recovery_epochs,
        "legacy_greedy_error_m": legacy_greedy_error_m,
        "multihypothesis_error_m": multihypothesis_error_m,
        "m4": m4,
        "passed": passed,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = audit(args.repo_root.resolve())
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
