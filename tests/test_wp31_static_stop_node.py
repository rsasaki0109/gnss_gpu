from __future__ import annotations

import pytest

from experiments.materialize_wp31_static_stop_node import materialize_node


def _source() -> dict:
    return {
        "basin_csv": "basin.csv",
        "basin_csv_sha256": "abc",
        "sample_stride_epochs": 5,
        "radius_m": 0.2,
        "dedup_radius_m": 0.2,
        "max_candidates": 24,
        "nodes": [
            {
                "segment": [10, 20],
                "source_epoch_count": 10,
                "candidate_count": 1,
                "candidates": [{"candidate_id": 0, "position_ecef": [1, 2, 3]}],
            }
        ],
    }


def test_materialize_node_preserves_provenance() -> None:
    result = materialize_node(_source(), 10)

    assert result["segment"] == [10, 20]
    assert result["basin_csv_sha256"] == "abc"
    assert result["candidates"][0]["candidate_id"] == 0


def test_materialize_node_rejects_missing_segment() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        materialize_node(_source(), 11)


def test_materialize_node_rejects_count_mismatch() -> None:
    source = _source()
    source["nodes"][0]["candidate_count"] = 2
    with pytest.raises(ValueError, match="count"):
        materialize_node(source, 10)
