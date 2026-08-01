from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "experiments/evaluate_ppc_official_suite.py"
SPEC = importlib.util.spec_from_file_location("evaluate_ppc_official_suite", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_aggregate_routes_uses_global_distance_and_safety_gate() -> None:
    rows = {}
    for index, (_, _, route) in enumerate(MODULE.ROUTES):
        rows[route] = {
            "pass_distance_m": 8.0 if index == 0 else 9.0,
            "total_distance_m": 10.0 if index == 0 else 20.0,
            "ppc_score_pct": 80.0 if index == 0 else 45.0,
            "fixed_epochs": 2,
            "correct_fix_epochs": 2,
            "false_fix_epochs": 0,
            "false_fix_above_1m_epochs": 0,
        }

    result = MODULE.aggregate_routes(rows)

    assert result["ppc_score_pct"] == pytest.approx(305.0 / 6.0)
    assert result["pooled_ppc_score_pct_diagnostic"] == pytest.approx(5300.0 / 110.0)
    assert result["fixed_epochs"] == 12
    assert result["safety_gate_passed"] is True
    assert result["targets"]["first_70_pct"] is False


def test_aggregate_routes_rejects_incomplete_suite() -> None:
    with pytest.raises(ValueError, match="exactly the six"):
        MODULE.aggregate_routes({})
