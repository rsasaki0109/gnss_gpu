from __future__ import annotations

import hashlib
import json

import pytest

from experiments.promote_wp57_precursor_boundary import promote


def _inputs() -> tuple[bytes, bytes, bytes, bytes]:
    candidate = {
        "seed_id": 4,
        "offset_ecef_m": [1.0, 2.0, 3.0],
        "block_offsets_ecef_m": [[1.0, 2.0, 3.0]] * 4,
        "block_spread_m": 0.1,
        "carrier_rms_cycles": 0.2,
        "cp_pr_consistency": {"checked_pairs": 100, "bad_pairs": 0},
    }
    source = {"production_input_truth": False, "hypotheses": [candidate]}
    cppr = {"production_input_truth": False}
    anchor = {
        "production_input_truth": False,
        "production_promoted": True,
        "segment": [55, 275],
        "reason": "unique_cppr_rank_consensus",
    }
    source_bytes, cppr_bytes, anchor_bytes = (
        json.dumps(source).encode(),
        json.dumps(cppr).encode(),
        json.dumps(anchor).encode(),
    )
    hashes = {
        "candidate_source": hashlib.sha256(source_bytes).hexdigest(),
        "cppr_validation": hashlib.sha256(cppr_bytes).hexdigest(),
        "right_anchor": hashlib.sha256(anchor_bytes).hexdigest(),
    }
    validation = {
        "production_input_truth": False,
        "segment": [0, 55],
        "selected_candidate_id": 4,
        "reason": "unique_long_cppr_precursor_boundary",
        "distance_pass": True,
        "runner_margin_pass": True,
        "runner_margin": 2.0,
        "winner": {
            **{
                key: candidate[key]
                for key in (
                    "offset_ecef_m",
                    "block_offsets_ecef_m",
                    "block_spread_m",
                )
            },
            "boundary_distance_m": 0.1,
        },
        "runner": {"boundary_distance_m": 0.3},
        "input_sha256": hashes,
    }
    return source_bytes, cppr_bytes, json.dumps(validation).encode(), anchor_bytes


def test_promotes_hash_linked_winner_without_audit() -> None:
    source, cppr, validation, anchor = _inputs()

    result = promote(source, cppr, validation, anchor)

    assert result["production_promoted"] is True
    assert result["selected_candidate_id"] == 4
    assert result["anchor_lineage"]["may_seed_another_boundary_promotion"] is False
    assert "audit" not in str(result)


def test_rejects_modified_anchor() -> None:
    source, cppr, validation, anchor = _inputs()

    with pytest.raises(ValueError, match="hashes"):
        promote(source, cppr, validation, anchor + b" ")
