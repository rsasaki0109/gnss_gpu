from __future__ import annotations

import hashlib
import json

from experiments.promote_wp62_backward_outage_path import promote
import experiments.select_wp62_backward_outage_path as selector


def _predecessors() -> tuple[list[bytes], list[bytes]]:
    sources, cpprs = [], []
    for rank in (0, 1):
        source = {
            "production_input_truth": False,
            "carrier_reference_rank": rank,
            "segment": [-55, 0],
            "hypotheses": [
                {
                    "seed_id": rank,
                    "block_offsets_ecef_m": [
                        [5.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [0.02, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                    ],
                }
            ],
        }
        source_bytes = json.dumps(source).encode()
        cppr = {
            "production_input_truth": False,
            "input_sha256": hashlib.sha256(source_bytes).hexdigest(),
            "family_rank_pass": True,
            "runner_margin_pass": True,
            "absolute_gate": {
                "checked_pairs_pass": True,
                "bad_pair_fraction_pass": True,
                "block_spread_pass": False,
            },
            "winner": {"candidate_id": rank},
        }
        sources.append(source_bytes)
        cpprs.append(json.dumps(cppr).encode())
    return sources, cpprs


def test_selects_and_promotes_two_basis_leading_instability(monkeypatch) -> None:
    base = {
        "production_input_truth": False,
        "selected": True,
        "segment": [0, 110],
        "block_offsets_ecef_m": [[0.1, 0.0, 0.0]] * 8,
    }
    monkeypatch.setattr(selector, "select_base_path", lambda *args: base)
    predecessor_sources, predecessor_cpprs = _predecessors()
    base_bytes = json.dumps(base).encode()
    dummy = json.dumps({"production_input_truth": False}).encode()

    result = selector.select(
        predecessor_sources,
        predecessor_cpprs,
        base_bytes,
        [],
        [],
        dummy,
        dummy,
        dummy,
        dummy,
    )
    validation = json.dumps(result).encode()
    inputs = [
        *predecessor_sources,
        *predecessor_cpprs,
        base_bytes,
        dummy,
        dummy,
        dummy,
        dummy,
    ]
    promoted = promote(validation, inputs)

    assert result["selected"] is True
    assert result["basis_support"] == 2
    assert len(result["block_offsets_ecef_m"]) == 12
    assert promoted["production_promoted"] is True
    assert promoted["lineage"]["may_seed_another_outage_or_path_promotion"] is False
    assert "audit" not in str(promoted)
