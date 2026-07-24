from __future__ import annotations

from experiments.analyze_wp29_static_grid_integrity_shadow import filter_candidate_ids


def test_candidate_filter_preserves_source_order() -> None:
    rows = [{"candidate_id": 4}, {"candidate_id": 0}, {"candidate_id": 23}]
    assert filter_candidate_ids(rows, "0,23") == rows[1:]


def test_empty_candidate_filter_keeps_all() -> None:
    rows = [{"candidate_id": 1}]
    assert filter_candidate_ids(rows, "") is rows
