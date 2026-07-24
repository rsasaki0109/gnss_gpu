from __future__ import annotations

from experiments.select_wp53_cross_basis_consensus import select


def _hypothesis(seed_id: int, seed: list[float], offset: list[float]) -> dict:
    return {
        "seed_id": seed_id,
        "seed_offset_ecef_m": seed,
        "offset_ecef_m": offset,
        "carrier_rms_cycles": 0.2,
        "block_spread_m": 0.1,
        "block_offsets_ecef_m": [offset, offset],
    }


def test_select_uses_three_basis_distance_and_margin_only() -> None:
    source_rows = [
        _hypothesis(10, [0, 0, 0], [0, 0, 0]),
        _hypothesis(11, [2, 0, 0], [2, 0, 0]),
    ]
    pool = {
        "production_input_truth": False,
        "source_reference_rank": 1,
        "seeds": [
            {"source_seed_id": 10, "offset_ecef_m": [0, 0, 0]},
            {"source_seed_id": 11, "offset_ecef_m": [2, 0, 0]},
        ],
    }
    source = {
        "production_input_truth": False,
        "carrier_reference_rank": 1,
        "segment": [5, 10],
        "hypotheses": source_rows,
    }
    cross0 = {
        "production_input_truth": False,
        "carrier_reference_rank": 0,
        "segment": [5, 10],
        "hypotheses": [
            _hypothesis(1, [0, 0, 0], [0.2, 0, 0]),
            _hypothesis(2, [2, 0, 0], [2.8, 0, 0]),
        ],
    }
    cross2 = {
        "production_input_truth": False,
        "carrier_reference_rank": 2,
        "segment": [5, 10],
        "hypotheses": [
            _hypothesis(1, [0, 0, 0], [0.21, 0, 0]),
            _hypothesis(2, [2, 0, 0], [2.79, 0, 0]),
        ],
    }

    result = select(pool, source, cross0, cross2)

    assert result["selected_candidate_id"] == 10
    assert result["reason"] == "unique_three_reference_basis_consensus"
    assert result["runner_margin"] > 0.2
    assert "audit" not in str(result)


def test_select_accepts_rank2_as_the_source_basis() -> None:
    source_rows = [
        _hypothesis(10, [0, 0, 0], [0, 0, 0]),
        _hypothesis(11, [2, 0, 0], [2, 0, 0]),
    ]
    pool = {
        "production_input_truth": False,
        "source_reference_rank": 2,
        "seeds": [
            {"source_seed_id": 10, "offset_ecef_m": [0, 0, 0]},
            {"source_seed_id": 11, "offset_ecef_m": [2, 0, 0]},
        ],
    }
    source = {
        "production_input_truth": False,
        "carrier_reference_rank": 2,
        "segment": [5, 10],
        "hypotheses": source_rows,
    }
    cross0 = {
        "production_input_truth": False,
        "carrier_reference_rank": 0,
        "segment": [5, 10],
        "hypotheses": [
            _hypothesis(1, [0, 0, 0], [0.2, 0, 0]),
            _hypothesis(2, [2, 0, 0], [2.8, 0, 0]),
        ],
    }
    cross1 = {
        "production_input_truth": False,
        "carrier_reference_rank": 1,
        "segment": [5, 10],
        "hypotheses": [
            _hypothesis(1, [0, 0, 0], [0.21, 0, 0]),
            _hypothesis(2, [2, 0, 0], [2.79, 0, 0]),
        ],
    }

    result = select(pool, source, cross0, cross1)

    assert result["selected_candidate_id"] == 10
    assert result["source_reference_rank"] == 2
    assert result["cross_reference_ranks"] == [0, 1]
