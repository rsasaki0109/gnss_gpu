from __future__ import annotations

import numpy as np
import pytest

from experiments.select_wp29_static_height_temporal_shadow import (
    select_height_temporal_candidate,
)
from experiments.apply_wp29_static_position_shadow import accepted_static_position
from experiments.run_wp29_tdcp_anchor_smoother import _load_static_position_override


def test_height_temporal_selection_adjusts_only_height() -> None:
    position = np.asarray([-3961598.0, 3349408.8, 3698135.1])
    temporal = {
        "candidates": [
            {
                "candidate_id": 36,
                "position_ecef": position.tolist(),
                "carrier_temporal_arc_cauchy_mean": 0.019,
                "carrier_temporal_arcs": 32,
            },
            {
                "candidate_id": 59,
                "position_ecef": (position + 1.0).tolist(),
                "carrier_temporal_arc_cauchy_mean": 0.021,
                "carrier_temporal_arcs": 32,
            },
        ]
    }
    road = {
        "candidates": [
            {
                "candidate_id": 36,
                "road_distance_m": 0.4,
                "position_ecef": position.tolist(),
            }
        ]
    }
    result = select_height_temporal_candidate(
        temporal,
        road,
        [position - np.asarray([0.0, 0.0, 0.4]), position - np.asarray([0.0, 0.0, 0.2])],
        max_temporal_ratio=0.95,
        max_road_distance_m=0.5,
        min_temporal_arcs=30,
        max_prior_height_spread_m=1.0,
    )
    assert result["selected_candidate_id"] == 36
    assert result["reason"] == "height_temporal_road_consensus"
    assert np.isfinite(result["position_ecef"]).all()


def test_height_temporal_selection_rejects_unseparated_winner() -> None:
    temporal = {
        "candidates": [
            {"candidate_id": 1, "position_ecef": [1e6, 2e6, 3e6], "carrier_temporal_arc_cauchy_mean": 1.0, "carrier_temporal_arcs": 32},
            {"candidate_id": 2, "position_ecef": [1e6, 2e6, 3e6], "carrier_temporal_arc_cauchy_mean": 1.01, "carrier_temporal_arcs": 32},
        ]
    }
    road = {
        "candidates": [
            {
                "candidate_id": 1,
                "road_distance_m": 0.1,
                "position_ecef": [1e6, 2e6, 3e6],
            }
        ]
    }
    with pytest.raises(RuntimeError, match="not separated"):
        select_height_temporal_candidate(
            temporal,
            road,
            [np.asarray([1e6, 2e6, 3e6]), np.asarray([1e6, 2e6, 3e6])],
            max_temporal_ratio=0.95,
            max_road_distance_m=0.5,
            min_temporal_arcs=30,
            max_prior_height_spread_m=1.0,
        )


def test_accepted_static_position_fails_closed() -> None:
    with pytest.raises(RuntimeError, match="not accepted"):
        accepted_static_position(
            {"reason": "insufficient_evidence", "segment": [10, 20], "position_ecef": [1, 2, 3]}
        )


def test_runtime_static_position_override_requires_accepted_reason(tmp_path) -> None:
    path = tmp_path / "override.json"
    path.write_text(
        '{"reason":"height_temporal_road_consensus","selected_candidate_id":36,'
        '"segment":[878,1100],"position_ecef":[1,2,3]}',
        encoding="utf-8",
    )
    start, end, position, candidate_id, reason = _load_static_position_override(path)
    assert (start, end, candidate_id) == (878, 1100, 36)
    np.testing.assert_allclose(position, [1, 2, 3])
    assert reason == "height_temporal_road_consensus"
