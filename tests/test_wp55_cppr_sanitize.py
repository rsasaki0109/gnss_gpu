from __future__ import annotations

import json

import pytest

from experiments.sanitize_wp55_cppr_candidates import sanitize


def test_sanitize_removes_all_post_selection_audits() -> None:
    source = {
        "production_input_truth": False,
        "truth_usage": "post_selection_audit_only",
        "truth_seeded_oracle_diagnostic": {"audit_median_error_m": 0.1},
        "osm_road_local_supply_audit": [{"audit_sub50cm_epochs": 4}],
        "hypotheses": [
            {"seed_id": 0, "offset_ecef_m": [1, 2, 3], "audit_median_error_m": 1}
        ],
    }

    result = sanitize(json.dumps(source).encode())

    assert result["truth_usage"] == "none"
    assert result["hypotheses"] == [{"seed_id": 0, "offset_ecef_m": [1, 2, 3]}]
    assert "audit_" not in json.dumps(result)
    assert "truth_seeded_oracle_diagnostic" not in result


def test_sanitize_rejects_truth_seeded_generation() -> None:
    with pytest.raises(ValueError, match="truth-seeded"):
        sanitize(
            json.dumps(
                {"production_input_truth": True, "hypotheses": [{"seed_id": 0}]}
            ).encode()
        )
