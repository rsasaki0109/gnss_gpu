from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.build_wp31_static_anchor_imu_route import resolve_accepted_anchor


def test_resolve_standard_position_anchor(tmp_path: Path) -> None:
    path = tmp_path / "position.json"
    path.write_text(
        json.dumps(
            {
                "reason": "gsi_height_osm_unique_gate",
                "selected_candidate_id": 41,
                "segment": [100, 120],
                "position_ecef": [1.0, 2.0, 3.0],
            }
        ),
        encoding="utf-8",
    )

    anchor = resolve_accepted_anchor(position_path=path)

    assert anchor[:2] == (100, 120)
    np.testing.assert_allclose(anchor[2], [1.0, 2.0, 3.0])
    assert anchor[3:] == (41, "gsi_height_osm_unique_gate")


def test_resolve_joint_anchor_side(tmp_path: Path) -> None:
    path = tmp_path / "joint.json"
    path.write_text(
        json.dumps(
            {
                "selected": True,
                "reason": "tdcp_gsi_road_continuity_unique",
                "left_segment": [10, 20],
                "right_segment": [25, 30],
                "left_selected_candidate_id": 16,
                "right_selected_candidate_id": 17,
                "left_position_ecef": [1.0, 2.0, 3.0],
                "right_position_ecef": [4.0, 5.0, 6.0],
            }
        ),
        encoding="utf-8",
    )

    anchor = resolve_accepted_anchor(joint_path=path, joint_side="right")

    assert anchor[:2] == (25, 30)
    np.testing.assert_allclose(anchor[2], [4.0, 5.0, 6.0])
    assert anchor[3:] == (17, "tdcp_gsi_road_continuity_unique")


def test_resolve_rejects_ambiguous_sources(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        resolve_accepted_anchor(position_path=tmp_path / "a", joint_path=tmp_path / "b")
