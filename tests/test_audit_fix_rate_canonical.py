import pytest

from experiments.audit_fix_rate_canonical import audit_route, build_audit


def _row(tow: float, **overrides: str) -> dict[str, str]:
    row = {
        "tow": str(tow),
        "library_status": "3",
        "primary_pair_count": "12",
        "primary_ffrt_passed": "1",
        "disjoint_a_ffrt_passed": "1",
        "disjoint_b_ffrt_passed": "1",
        "disjoint_hard_separation_passed": "1",
        "disjoint_statistical_separation_passed": "1",
        "disjoint_passed": "1",
        "failure_budget_passed": "1",
        "quality_gate_passed": "1",
        "processing_runtime_ms": "10",
        "causal_arc_resets": "0",
        "causal_arc_ready_pairs": "0",
    }
    row.update(overrides)
    return row


def _position(status: int, x: float) -> dict[str, float | int]:
    return {
        "status": status,
        "ecef_x": x,
        "ecef_y": 0.0,
        "ecef_z": 0.0,
    }


def test_audit_separates_correct_false_and_nonfix_blockers() -> None:
    rows = [
        _row(1.0, library_status="4"),
        _row(2.0, library_status="4"),
        _row(3.0, primary_pair_count="8", primary_ffrt_passed="0"),
        _row(4.0, disjoint_hard_separation_passed="0", disjoint_passed="0"),
    ]
    positions = {
        1.0: _position(4, 0.1),
        2.0: _position(4, 2.0),
        3.0: _position(3, 0.0),
        4.0: _position(3, 0.0),
    }
    truth = {tow: (0.0, 0.0, 0.0) for tow in positions}

    result = audit_route(
        "tokyo/run1", rows, positions, truth, block_count=2
    )

    assert result["fixed_rate"] == pytest.approx(0.5)
    assert result["correct_fix_rate"] == pytest.approx(0.25)
    assert result["false_per_fixed"] == pytest.approx(0.5)
    assert result["false_fixed_above_1m_epochs"] == 1
    assert result["nonfixed_primary_blockers"] == {
        "disjoint_hard_separation_rejected": 1,
        "primary_pairs_below_minimum": 1,
    }
    assert result["contiguous_time_blocks"]["0"]["false_fixed_epochs"] == 1


def test_audit_labels_candidate_oracle_only_after_runtime_selection() -> None:
    rows = [
        _row(
            1.0,
            satellite_par_ffrt_passed="1",
            satellite_par_subset_size="9",
            satellite_par_candidate_ecef_x="0.1",
            satellite_par_candidate_ecef_y="0",
            satellite_par_candidate_ecef_z="0",
        ),
        _row(
            2.0,
            satellite_par_ffrt_passed="1",
            satellite_par_subset_size="8",
            satellite_par_candidate_ecef_x="2",
            satellite_par_candidate_ecef_y="0",
            satellite_par_candidate_ecef_z="0",
        ),
    ]
    positions = {1.0: _position(3, 0.0), 2.0: _position(3, 0.0)}
    truth = {1.0: (0.0, 0.0, 0.0), 2.0: (0.0, 0.0, 0.0)}

    result = audit_route(
        "tokyo/run1", rows, positions, truth, block_count=2
    )
    satellite = result["candidate_sources"]["satellite_par"]

    assert satellite["nonfixed_ffrt_passed_epochs"] == 2
    assert satellite["nonfixed_pair_count_p50"] == pytest.approx(8.5)
    assert satellite["nonfixed_candidate_oracle_correct_epochs"] == 1
    assert satellite["nonfixed_candidate_oracle_wrong_epochs"] == 1


def test_build_audit_aggregates_routes_without_changing_route_metrics() -> None:
    tokyo = (
        "tokyo/run1",
        [_row(1.0, library_status="4")],
        {1.0: _position(4, 0.1)},
        {1.0: (0.0, 0.0, 0.0)},
    )
    nagoya = (
        "nagoya/run1",
        [_row(2.0)],
        {2.0: _position(3, 0.0)},
        {2.0: (0.0, 0.0, 0.0)},
    )

    result = build_audit([tokyo, nagoya], block_count=2)

    assert result["aggregate"]["epochs"] == 2
    assert result["aggregate"]["correct_fix_rate"] == pytest.approx(0.5)
    assert result["aggregate"]["false_fixed_epochs"] == 0
    assert result["routes"]["tokyo/run1"]["correct_fix_rate"] == 1.0
