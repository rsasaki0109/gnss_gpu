from __future__ import annotations

import hashlib
import json

import pytest

from experiments.promote_wp60_two_block_path import promote
from experiments.select_wp60_two_block_path import select


def _payloads() -> tuple[list[bytes], list[bytes], bytes, bytes, bytes, bytes]:
    left_sources, left_cpprs = [], []
    for rank, offset in enumerate((0.10, 0.12, 1.0)):
        source = {
            "production_input_truth": False,
            "carrier_reference_rank": rank,
            "segment": [0, 55],
            "hypotheses": [
                {
                    "seed_id": rank,
                    "offset_ecef_m": [offset, 0.0, 0.0],
                    "block_offsets_ecef_m": [[offset, 0.0, 0.0]] * 4,
                }
            ],
        }
        source_bytes = json.dumps(source).encode()
        cppr = {
            "production_input_truth": False,
            "input_sha256": hashlib.sha256(source_bytes).hexdigest(),
            "selected_candidate_id": rank,
        }
        left_sources.append(source_bytes)
        left_cpprs.append(json.dumps(cppr).encode())
    right_source = {
        "production_input_truth": False,
        "segment": [55, 110],
        "hypotheses": [],
    }
    right_cppr = {"production_input_truth": False}
    anchor = {
        "production_input_truth": False,
        "reason": "unique_cppr_rank_consensus",
    }
    right_source_bytes = json.dumps(right_source).encode()
    right_cppr_bytes = json.dumps(right_cppr).encode()
    anchor_bytes = json.dumps(anchor).encode()
    right_boundary = {
        "production_input_truth": False,
        "selected_candidate_id": 5,
        "winner": {
            "candidate_id": 5,
            "block_offsets_ecef_m": [[0.15, 0.0, 0.0]] * 4,
        },
        "runner": {
            "candidate_id": 6,
            "block_offsets_ecef_m": [[0.8, 0.0, 0.0]] * 4,
        },
        "input_sha256": {
            "candidate_source": hashlib.sha256(right_source_bytes).hexdigest(),
            "cppr_validation": hashlib.sha256(right_cppr_bytes).hexdigest(),
            "right_anchor": hashlib.sha256(anchor_bytes).hexdigest(),
        },
    }
    return (
        left_sources,
        left_cpprs,
        right_source_bytes,
        right_cppr_bytes,
        json.dumps(right_boundary).encode(),
        anchor_bytes,
    )


def test_selects_unique_two_basis_path_and_promotes_it() -> None:
    payloads = _payloads()

    result = select(*payloads)
    validation = json.dumps(result).encode()
    promoted = promote(validation, *payloads)

    assert result["selected"] is True
    assert result["left_consensus"]["basis_support"] == 2
    assert result["left_consensus"]["max_cross_basis_distance_m"] == pytest.approx(
        0.02
    )
    assert promoted["production_promoted"] is True
    assert promoted["lineage"]["may_seed_another_path_promotion"] is False
    assert "audit" not in str(promoted)


def test_promotion_rejects_modified_source() -> None:
    payloads = _payloads()
    validation = json.dumps(select(*payloads)).encode()
    left_sources = list(payloads[0])
    left_sources[0] += b" "

    with pytest.raises(ValueError, match="hashes"):
        promote(validation, left_sources, *payloads[1:])
