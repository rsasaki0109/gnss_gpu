from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from select_wp31_static_tdcp_joint_pair import select_tdcp_joint_pair


def _row(candidate_id: int, x: float) -> dict:
    return {"candidate_id": candidate_id, "position_ecef": [x, 0.0, 0.0]}


def test_selects_unique_pair_on_tdcp_only_edge() -> None:
    result = select_tdcp_joint_pair(
        [_row(1, 0.0), _row(2, 5.0)],
        [_row(3, 10.0), _row(4, 18.0)],
        np.asarray([[10.0, 0.0, 0.0]]),
        ["tdcp"],
    )
    assert result["reason"] == "tdcp_joint_pair_unique"
    assert (result["left_selected_candidate_id"], result["right_selected_candidate_id"]) == (1, 3)


def test_rejects_common_translation_ambiguity() -> None:
    result = select_tdcp_joint_pair(
        [_row(1, 0.0), _row(2, 5.0)],
        [_row(3, 10.0), _row(4, 15.0)],
        np.asarray([[10.0, 0.0, 0.0]]),
        ["tdcp"],
    )
    assert result["reason"] == "joint_pair_not_unique"
    assert result["selected"] is False


def test_rejects_gap_filled_edge() -> None:
    result = select_tdcp_joint_pair(
        [_row(1, 0.0)],
        [_row(2, 10.0), _row(3, 12.0)],
        np.asarray([[10.0, 0.0, 0.0]]),
        ["gyro_doppler_gap_fill"],
    )
    assert result["reason"] == "insufficient_tdcp_edge_fraction"


def test_road_continuity_breaks_common_translation_tie() -> None:
    left = [_row(1, 0.0), _row(2, 5.0)]
    right = [_row(3, 10.0), _row(4, 15.0)]
    left_road = [
        {**left[0], "road_distance_m": 4.9},
        {**left[1], "road_distance_m": 2.0},
    ]
    right_road = [
        {**right[0], "road_distance_m": 4.8},
        {**right[1], "road_distance_m": 2.4},
    ]
    result = select_tdcp_joint_pair(
        left,
        right,
        np.asarray([[10.0, 0.0, 0.0]]),
        ["tdcp"],
        left_road_candidates=left_road,
        right_road_candidates=right_road,
    )
    assert result["reason"] == "tdcp_gsi_road_continuity_unique"
    assert (result["left_selected_candidate_id"], result["right_selected_candidate_id"]) == (1, 3)
