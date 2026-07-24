import json

import pytest

from experiments.promote_wp32_secondary_posterior import validate_and_promote
from experiments.run_wp29_tdcp_anchor_smoother import _load_static_position_override


def _inputs():
    config = {
        "proposal_min_members": 3,
        "proposal_min_score": 0.4,
        "secondary_min_evidence_epochs": 10,
        "secondary_top_k": 3,
        "secondary_max_median_m": 0.5,
        "secondary_max_support_spread_m": 0.5,
    }
    manifest = {
        "schema": "wp32_secondary_posterior_validation_v1",
        "production_approved": True,
        "selection_config": config,
        "development": {
            "required_reason": "unique_secondary_topk_posterior_development",
            "max_audit_error_m": 0.5,
        },
        "positive_holdout": {
            "required_reason": "multimode_ddpr_consensus",
            "required_support_members": 3,
            "max_audit_error_m": 0.5,
        },
        "late_negative_holdouts": [{"required_reason": "no_secondary_posterior"}],
        "early_negative_holdouts": [{}],
    }
    development = {
        "reason": "unique_secondary_topk_posterior_development",
        "production_input_truth": False,
        "selected_candidate_ids": [1, 2, 3],
        "position_ecef": [1.0, 2.0, 3.0],
        "segment": [10, 20],
        "support_secondary_median_m": [0.3, 0.4, 0.45],
        "support_spread_m": 0.2,
        "selected_audit_error_m": 0.4,
        "config": {
            "min_evidence_epochs": 10,
            "top_k": 3,
            "max_secondary_median_m": 0.5,
            "max_support_spread_m": 0.5,
        },
    }
    artifacts = {
        "development": development,
        "positive_holdout": {
            "reason": "multimode_ddpr_consensus",
            "selected_candidate_ids": [4, 5, 6],
            "selected_audit_error_m": 0.3,
        },
        "late_negative_holdouts": [{"reason": "no_secondary_posterior"}],
        "early_negative_holdouts": [{"clusters": [{"members": 2, "score": 1.0}]}],
    }
    return manifest, artifacts


def test_promotes_only_after_positive_and_negative_validation():
    manifest, artifacts = _inputs()
    result = validate_and_promote(manifest, artifacts)
    assert result["production_promoted"] is True
    assert result["reason"] == "unique_secondary_topk_posterior"


def test_rejects_early_negative_that_passes_proposal_gate():
    manifest, artifacts = _inputs()
    artifacts["early_negative_holdouts"][0]["clusters"][0]["members"] = 3
    with pytest.raises(RuntimeError, match="early negative"):
        validate_and_promote(manifest, artifacts)


def test_production_reason_is_accepted_as_position_override(tmp_path):
    path = tmp_path / "anchor.json"
    path.write_text(
        json.dumps(
            {
                "selected_candidate_id": 37,
                "position_ecef": [1.0, 2.0, 3.0],
                "segment": [805, 923],
                "reason": "unique_secondary_topk_posterior",
                "production_promoted": True,
            }
        )
    )
    assert _load_static_position_override(path)[3:] == (
        37,
        "unique_secondary_topk_posterior",
    )
