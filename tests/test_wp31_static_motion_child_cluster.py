from __future__ import annotations

import numpy as np

from experiments.select_wp31_static_motion_child_cluster import (
    select_motion_child_cluster,
)


def _child(candidate_id: int, position: tuple[float, float, float]) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "position_ecef": list(position),
        "coverage_epochs": 0,
        "members": 0,
        "proposal_kind": "offset_seed",
    }


def test_motion_child_cluster_selects_largest_compact_prefix() -> None:
    rows = [
        _child(10 + index, (10.0 + 0.08 * index, 0.02 * (index % 2), 0.0))
        for index in range(7)
    ]
    rows.extend(_child(20 + index, (12.0 + index, 0.0, 0.0)) for index in range(3))

    result = select_motion_child_cluster(rows, np.zeros(3))

    assert result["reason"] == "motion_supported_child_cluster"
    assert result["cluster_members"] == 7
    assert result["cluster_spread_m"] <= 0.5


def test_motion_child_cluster_fails_closed_without_compact_five_member_prefix() -> None:
    rows = [_child(index, (float(index), 0.0, 0.0)) for index in range(8)]

    result = select_motion_child_cluster(rows, np.zeros(3))

    assert result["selected_candidate_id"] is None
    assert result["reason"] == "no_compact_motion_prefix"
