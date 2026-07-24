import json

from experiments.promote_wp34_relative_secondary import (
    _relative_margin,
    three_radius_ranking,
)
from experiments.run_wp29_tdcp_anchor_smoother import _load_static_position_override


def test_three_radius_ranking_uses_only_complete_road_groups():
    selection = {"road_eligible_candidate_ids": [0, 2, 3, 5, 6, 8]}
    secondary = {
        "candidates": [
            {"candidate_id": candidate_id, "ddpr_median_abs_m": value}
            for candidate_id, value in enumerate([1.0, 9.0, 2.0, 1.1, 9.0, 2.1, 1.2, 9.0, 2.2])
        ]
    }
    audit = {
        "candidates": [
            {"candidate_id": candidate_id, "audit_error_m": float(candidate_id)}
            for candidate_id in range(9)
        ]
    }

    ranking = three_radius_ranking(
        selection, secondary, audit, direction_count=3
    )

    assert [row["ids"] for row in ranking] == [[0, 3, 6], [2, 5, 8]]
    assert _relative_margin(ranking) > 0.8


def test_production_reason_is_accepted_by_smoother(tmp_path):
    path = tmp_path / "anchor.json"
    path.write_text(
        json.dumps(
            {
                "segment": [6945, 7076],
                "selected_candidate_id": 0,
                "reason": "unique_relative_secondary_parent_primary_compact",
                "position_ecef": [1.0, 2.0, 3.0],
            }
        ),
        encoding="utf-8",
    )

    assert _load_static_position_override(path)[3:] == (
        0,
        "unique_relative_secondary_parent_primary_compact",
    )
