from __future__ import annotations

import hashlib
import json

import pytest

from experiments.promote_wp55_cppr_rank import promote


def _inputs() -> tuple[bytes, bytes]:
    candidate = {
        "seed_id": 2,
        "offset_ecef_m": [1.0, 2.0, 3.0],
        "block_offsets_ecef_m": [[1.0, 2.0, 3.0], [1.1, 2.0, 3.0]],
        "carrier_rms_cycles": 0.2,
        "block_spread_m": 0.1,
        "cp_pr_consistency": {"checked_pairs": 100, "bad_pairs": 1},
        "audit_median_error_m": 0.1,
    }
    source = {
        "production_input_truth": False,
        "hypotheses": [candidate],
    }
    source_bytes = json.dumps(source).encode()
    validation = {
        "production_input_truth": False,
        "input_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "segment": [10, 20],
        "selected_candidate_id": 2,
        "reason": "unique_cppr_rank_consensus",
        "family_rank_pass": True,
        "runner_margin_pass": True,
        "absolute_gate_pass": True,
        "runner_margin": 1.0,
        "absolute_gate": {"block_spread_pass": True},
        "winner": {
            **{
                key: candidate[key]
                for key in (
                    "offset_ecef_m",
                    "block_offsets_ecef_m",
                    "carrier_rms_cycles",
                    "block_spread_m",
                )
            },
            "family_ranks": {"median": 1},
            "rank_sum": 3,
        },
    }
    return source_bytes, json.dumps(validation).encode()


def test_promote_strips_audit_and_links_both_inputs() -> None:
    source, validation = _inputs()

    result = promote(source, validation)

    assert result["production_promoted"] is True
    assert result["selected_candidate_id"] == 2
    assert "audit" not in str(result)
    assert result["input_sha256"]["validation"] == hashlib.sha256(
        validation
    ).hexdigest()


def test_promote_rejects_unlinked_validation() -> None:
    source, validation = _inputs()
    payload = json.loads(validation)
    payload["input_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="hash-linked"):
        promote(source, json.dumps(payload).encode())
