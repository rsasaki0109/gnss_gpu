from __future__ import annotations

import json
from pathlib import Path


EVIDENCE = Path("internal_docs/ppc_quality_imu_gap2_promotion_evidence_2026_08_02.json")


def test_promoted_policy_evidence_is_internally_consistent() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    routes = list(evidence["routes"].values())
    totals = evidence["totals"]

    assert len(routes) == 6
    assert sum(route["total_epochs"] for route in routes) == totals["total_epochs"]
    assert sum(route["correct_fix"] for route in routes) == totals["correct_fix"]
    assert sum(route["false_fix"] for route in routes) == 0
    assert sum(route["false_fix_above_1m"] for route in routes) == 0
    assert totals["fixed"] == totals["correct_fix"] == 9964
    assert totals["total_epochs"] == 48778
    assert abs(totals["correct_fix_rate_full_denominator"] - 9964 / 48778) < 1e-15


def test_promoted_policy_preserves_validation_and_fault_gates() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))

    assert evidence["route_blocked_selection"]["passed"] is True
    assert (
        evidence["route_blocked_selection"]["validation_gap2_correct_fix"]
        > (evidence["route_blocked_selection"]["validation_gap1_correct_fix"])
    )
    assert evidence["gates"] == {
        "six_route_complete": True,
        "route_blocked_validation": True,
        "all_routes_improved_vs_original": True,
        "fault_matrix": True,
        "zero_false_fix": True,
        "zero_false_fix_above_1m": True,
    }
    assert all(
        fault["false_fix"] == fault["false_fix_above_1m"] == 0
        for fault in evidence["fault_audits"].values()
    )
    assert evidence["promotion_ready"] is True
    assert evidence["sota_claim"] is False
