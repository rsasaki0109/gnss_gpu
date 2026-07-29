from __future__ import annotations

from pathlib import Path

from tools.audit_v030_production_promotion import audit_promotion


REPO_ROOT = Path(__file__).parents[1]
CONTRACT = REPO_ROOT / "configs/evaluation/v030_production_promotion.json"


def test_promotion_audit_fails_closed_only_on_unmet_tokyo_kpi() -> None:
    result = audit_promotion(REPO_ROOT, CONTRACT)
    assert result["gate_count"] == 11
    assert result["passed_gate_count"] == 10
    assert result["promotion_allowed"] is False
    assert result["failed_gates"] == ["tokyo_sub50cm_target"]


def test_every_gate_has_authoritative_evidence_and_expectation() -> None:
    result = audit_promotion(REPO_ROOT, CONTRACT)
    assert all(gate["evidence"] for gate in result["gates"])
    assert all("actual" in gate and "expected" in gate for gate in result["gates"])
