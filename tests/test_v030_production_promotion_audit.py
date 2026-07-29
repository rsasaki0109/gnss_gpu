from __future__ import annotations

from pathlib import Path

from tools.audit_v030_production_promotion import audit_promotion


REPO_ROOT = Path(__file__).parents[1]
CONTRACT = REPO_ROOT / "configs/evaluation/v030_production_promotion.json"


def test_promotion_audit_passes_all_locked_requirements() -> None:
    result = audit_promotion(REPO_ROOT, CONTRACT)
    assert result["gate_count"] == 12
    assert result["passed_gate_count"] == 12
    assert result["promotion_allowed"] is True
    assert result["failed_gates"] == []


def test_every_gate_has_authoritative_evidence_and_expectation() -> None:
    result = audit_promotion(REPO_ROOT, CONTRACT)
    assert all(gate["evidence"] for gate in result["gates"])
    assert all("actual" in gate and "expected" in gate for gate in result["gates"])
