from __future__ import annotations

import json
from pathlib import Path


def test_committed_float_selector_evidence_is_safe_and_internally_consistent() -> None:
    path = (
        Path(__file__).parents[1] / "docs" / "ppc_causal_float_selector_evidence.json"
    )
    evidence = json.loads(path.read_text(encoding="utf-8"))

    assert evidence["truth_contract"] == {
        "production_input_truth": False,
        "truth_usage": "post_estimator_scoring_only",
        "forward_only": True,
    }
    assert evidence["split_contract"]["retuned_after_validation"] is False
    assert evidence["split_contract"]["retuned_after_sealed_evaluation"] is False
    aggregate = evidence["aggregate"]
    assert aggregate["selected_score_pct"] > aggregate["baseline_score_pct"]
    assert aggregate["false_fix_epochs"] == 0
    assert aggregate["false_fix_above_1m_epochs"] == 0
    assert aggregate["safety_gate_passed"] is True
    assert len(evidence["routes"]) == 6
    for route in evidence["routes"].values():
        assert route["selected_score_pct"] > route["baseline_score_pct"]
        assert route["runtime_p95_ms"] < 100.0
        assert route["runtime_under_100ms_pct"] >= 99.8
