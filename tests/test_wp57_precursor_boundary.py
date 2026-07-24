from __future__ import annotations

import hashlib
import json

import pytest

from experiments.select_wp57_precursor_boundary import select


def _inputs() -> tuple[bytes, bytes, bytes]:
    rows = []
    ranked = []
    for candidate_id, boundary in ((0, 0.1), (1, 0.5)):
        rows.append(
            {
                "seed_id": candidate_id,
                "offset_ecef_m": [boundary, 0.0, 0.0],
                "block_offsets_ecef_m": [[boundary, 0.0, 0.0]] * 4,
            }
        )
        ranked.append(
            {
                "candidate_id": candidate_id,
                "family_ranks": {"median": 1, "p95": 1, "bad": 1},
                "rank_sum": 3,
                "checked_pairs": 100,
                "bad_pair_fraction": 0.0,
                "block_spread_m": 0.1,
            }
        )
    source = {"production_input_truth": False, "segment": [0, 55], "hypotheses": rows}
    source_bytes = json.dumps(source).encode()
    cppr = {
        "production_input_truth": False,
        "input_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "family_rank_limit": 1,
        "absolute_gate": {
            "min_checked_pairs": 40,
            "max_bad_pair_fraction": 0.05,
            "max_block_spread_m": 0.5,
        },
        "candidates": ranked,
    }
    anchor = {
        "production_input_truth": False,
        "production_promoted": True,
        "reason": "unique_cppr_rank_consensus",
        "segment": [55, 275],
        "block_offsets_ecef_m": [[0.0, 0.0, 0.0]] * 4,
    }
    return source_bytes, json.dumps(cppr).encode(), json.dumps(anchor).encode()


def test_selects_unique_cppr_eligible_boundary_winner() -> None:
    source, cppr, anchor = _inputs()

    result = select(source, cppr, anchor)

    assert result["selected_candidate_id"] == 0
    assert result["winner"]["boundary_distance_m"] == pytest.approx(0.1)
    assert result["runner_margin"] == pytest.approx(4.0)
    assert "audit" not in str(result)


def test_rejects_short_recursive_anchor() -> None:
    source, cppr, anchor = _inputs()
    payload = json.loads(anchor)
    payload["segment"] = [55, 110]

    with pytest.raises(ValueError, match="too short"):
        select(source, cppr, json.dumps(payload).encode())
