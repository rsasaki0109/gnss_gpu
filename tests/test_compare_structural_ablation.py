from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "experiments/compare_structural_ablation.py"
SPEC = importlib.util.spec_from_file_location("compare_structural_ablation", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_compare_summaries_matches_scope_and_computes_candidate_minus_baseline():
    baseline = pd.DataFrame(
        {
            "scope": ["run_holdout", "run_holdout"],
            "scope_id": ["tokyo_run1_after_development", "tokyo_run2_after_development"],
            "evaluation_role": ["holdout", "holdout"],
            "epochs": [10, 20],
            "pass_3m": [0.4, 0.7],
            "error_p95_m": [5.0, 3.0],
        }
    )
    candidate = pd.DataFrame(
        {
            "scope": ["run_holdout"],
            "scope_id": ["tokyo_run1_after_development"],
            "evaluation_role": ["holdout"],
            "epochs": [10],
            "pass_3m": [0.6],
            "error_p95_m": [4.5],
        }
    )

    result = MODULE.compare_summaries(baseline, candidate)

    assert len(result) == 1
    assert result.loc[0, "epochs_delta"] == 0
    assert result.loc[0, "pass_3m_delta"] == pytest.approx(0.2)
    assert result.loc[0, "error_p95_m_delta"] == pytest.approx(-0.5)


def test_compare_summaries_rejects_disjoint_scopes():
    baseline = pd.DataFrame(
        {"scope": ["run"], "scope_id": ["a"], "evaluation_role": ["diagnostic"]}
    )
    candidate = pd.DataFrame(
        {"scope": ["run"], "scope_id": ["b"], "evaluation_role": ["diagnostic"]}
    )

    with pytest.raises(ValueError, match="no identical scopes"):
        MODULE.compare_summaries(baseline, candidate)
