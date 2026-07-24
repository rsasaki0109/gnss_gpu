from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "experiments"))

from analyze_wp47_local_pool_temporal_trifrequency import sanitize_pool  # noqa: E402


def test_sanitize_pool_assigns_stable_candidate_ids() -> None:
    source = {
        "schema": "wp31_moving_block_truth_free_local_pool_v1",
        "production_input_truth": False,
        "candidates": [
            {
                "parent_road_seed": 4,
                "local_delta_xyh_m": [1, 2, -1],
                "offset_ecef_m": [3, 4, 5],
                "integer_arcs": 6,
                "retained_carrier_rows": 40,
                "carrier_rms_cycles": 0.2,
                "proposal_score": 0.3,
            }
        ],
    }
    rows = sanitize_pool(source)
    assert rows[0]["candidate_id"] == 0
    assert rows[0]["carrier_rows"] == 40


def test_sanitize_pool_rejects_truth_input() -> None:
    with pytest.raises(ValueError, match="not truth-free"):
        sanitize_pool(
            {
                "schema": "wp31_moving_block_truth_free_local_pool_v1",
                "production_input_truth": True,
                "candidates": [],
            }
        )
