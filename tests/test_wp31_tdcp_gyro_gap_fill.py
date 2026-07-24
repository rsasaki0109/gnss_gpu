from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.build_wp31_tdcp_gyro_gap_fill import (
    bridge_long_gyro_routes,
    close_anchor_motion_gaps,
    load_joint_position_overrides,
)


def test_load_joint_position_overrides(tmp_path: Path) -> None:
    path = tmp_path / "joint.json"
    path.write_text(
        json.dumps(
            {
                "selected": True,
                "reason": "tdcp_gsi_road_continuity_unique",
                "left_segment": [10, 20],
                "right_segment": [25, 30],
                "left_selected_candidate_id": 16,
                "right_selected_candidate_id": 16,
                "left_position_ecef": [1.0, 2.0, 3.0],
                "right_position_ecef": [4.0, 5.0, 6.0],
            }
        ),
        encoding="utf-8",
    )

    spans = load_joint_position_overrides(path)

    assert [(row[0], row[1], row[3], row[4]) for row in spans] == [
        (10, 20, 16, "tdcp_gsi_road_continuity_unique"),
        (25, 30, 16, "tdcp_gsi_road_continuity_unique"),
    ]
    np.testing.assert_allclose(spans[0][2], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(spans[1][2], [4.0, 5.0, 6.0])


def test_load_joint_position_overrides_rejects_unselected(tmp_path: Path) -> None:
    path = tmp_path / "joint.json"
    path.write_text(json.dumps({"selected": False}), encoding="utf-8")

    with pytest.raises(ValueError, match="not selected"):
        load_joint_position_overrides(path)


def test_motion_gap_closure_preserves_observed_rows_and_closes_endpoint() -> None:
    displacements = [np.zeros(3) for _ in range(8)]
    for epoch in range(2, 7):
        displacements[epoch] = np.array([1.0, 0.0, 0.0])
    rows = [
        {
            "source": "gyro_doppler_gap_fill" if 3 <= epoch < 6 else "tdcp",
            "interval_dt_s": 0.2,
            "dx_m": float(displacements[epoch][0]),
            "dy_m": 0.0,
            "dz_m": 0.0,
            "norm_m": float(np.linalg.norm(displacements[epoch])),
        }
        for epoch in range(8)
    ]
    spans = [
        (0, 2, np.zeros(3), 1, "clear_widelane"),
        (7, 8, np.array([8.0, 0.0, 0.0]), 2, "clear_widelane"),
    ]

    reports = close_anchor_motion_gaps(displacements, rows, spans)

    assert len(reports) == 1
    assert reports[0]["gap_start"] == 3
    assert reports[0]["gap_end"] == 6
    np.testing.assert_allclose(sum(displacements[2:8]), [8.0, 0.0, 0.0])
    np.testing.assert_allclose(displacements[2], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(displacements[6], [1.0, 0.0, 0.0])
    assert all(rows[epoch]["source"] == "anchor_closed_gyro_gap_fill" for epoch in range(3, 6))


def test_all_filled_motion_gap_closure_uses_every_uncertain_run() -> None:
    displacements = [np.zeros(3) for _ in range(9)]
    rows = []
    for epoch in range(9):
        source = "gyro_doppler_gap_fill" if epoch in (2, 3, 6) else "tdcp"
        rows.append(
            {
                "source": source,
                "interval_dt_s": 0.2,
                "dx_m": 0.0,
                "dy_m": 0.0,
                "dz_m": 0.0,
                "norm_m": 0.0,
            }
        )
    spans = [
        (0, 1, np.zeros(3), 1, "clear_widelane"),
        (8, 9, np.array([3.0, 0.0, 0.0]), 2, "clear_widelane"),
    ]

    reports = close_anchor_motion_gaps(displacements, rows, spans, mode="all_filled")

    np.testing.assert_allclose(sum(displacements[1:9]), [3.0, 0.0, 0.0])
    np.testing.assert_allclose(displacements[2], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(displacements[3], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(displacements[6], [1.0, 0.0, 0.0])
    assert reports[0]["selected_gap_runs"] == 2
    assert reports[0]["selected_gap_epochs"] == 3
    assert rows[4]["source"] == "tdcp"


def test_duration_weighted_closure_favors_longer_gap() -> None:
    displacements = [np.zeros(3) for _ in range(9)]
    rows = []
    for epoch in range(9):
        source = "gyro_doppler_gap_fill" if epoch in (2, 3, 6) else "tdcp"
        rows.append({"source": source, "interval_dt_s": 0.2, "dx_m": 0.0, "dy_m": 0.0, "dz_m": 0.0, "norm_m": 0.0})
    spans = [(0, 1, np.zeros(3), 1, "a"), (8, 9, np.array([5.0, 0.0, 0.0]), 2, "b")]

    close_anchor_motion_gaps(displacements, rows, spans, mode="duration_weighted", duration_exponent=2.0)

    np.testing.assert_allclose(sum(displacements[1:9]), [5.0, 0.0, 0.0])
    assert displacements[2][0] == pytest.approx(2.0)
    assert displacements[3][0] == pytest.approx(2.0)
    assert displacements[6][0] == pytest.approx(1.0)


def test_fragmentation_gate_uses_all_filled_without_dominant_run() -> None:
    displacements = [np.zeros(3) for _ in range(9)]
    rows = []
    for epoch in range(9):
        source = "gyro_doppler_gap_fill" if epoch in (2, 3, 6, 7) else "tdcp"
        rows.append({"source": source, "interval_dt_s": 0.2, "dx_m": 0.0, "dy_m": 0.0, "dz_m": 0.0, "norm_m": 0.0})
    spans = [(0, 1, np.zeros(3), 1, "a"), (8, 9, np.array([4.0, 0.0, 0.0]), 2, "b")]

    reports = close_anchor_motion_gaps(displacements, rows, spans, mode="fragmentation_gated")

    np.testing.assert_allclose(displacements[2], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(displacements[3], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(displacements[6], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(displacements[7], [1.0, 0.0, 0.0])
    assert reports[0]["effective_closure_mode"] == "all_filled"
    assert reports[0]["dominant_filled_run_share"] == pytest.approx(0.5)


def test_fragmentation_gate_uses_duration_weighting_for_dominant_run() -> None:
    displacements = [np.zeros(3) for _ in range(9)]
    rows = []
    for epoch in range(9):
        source = "gyro_doppler_gap_fill" if epoch in (2, 3, 4, 6) else "tdcp"
        rows.append({"source": source, "interval_dt_s": 0.2, "dx_m": 0.0, "dy_m": 0.0, "dz_m": 0.0, "norm_m": 0.0})
    spans = [(0, 1, np.zeros(3), 1, "a"), (8, 9, np.array([10.0, 0.0, 0.0]), 2, "b")]

    reports = close_anchor_motion_gaps(displacements, rows, spans, mode="fragmentation_gated")

    np.testing.assert_allclose(displacements[2], [3.0, 0.0, 0.0])
    np.testing.assert_allclose(displacements[3], [3.0, 0.0, 0.0])
    np.testing.assert_allclose(displacements[4], [3.0, 0.0, 0.0])
    np.testing.assert_allclose(displacements[6], [1.0, 0.0, 0.0])
    assert reports[0]["effective_closure_mode"] == "duration_weighted"
    assert reports[0]["dominant_filled_run_share"] == pytest.approx(0.75)


def test_all_interval_bias_closure_updates_observed_rows() -> None:
    displacements = [np.zeros(3) for _ in range(5)]
    rows = [{"source": "tdcp", "interval_dt_s": 0.2, "dx_m": 0.0, "dy_m": 0.0, "dz_m": 0.0, "norm_m": 0.0} for _ in range(5)]
    spans = [(0, 1, np.zeros(3), 1, "a"), (4, 5, np.array([4.0, 0.0, 0.0]), 2, "b")]

    close_anchor_motion_gaps(displacements, rows, spans, mode="all_intervals")

    np.testing.assert_allclose(sum(displacements[1:5]), [4.0, 0.0, 0.0])
    assert all(rows[index]["source"] == "anchor_closed_interval_bias" for index in range(1, 5))


def test_long_gyro_route_bridge_uses_independent_boundaries(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_route(positions, _doppler, _speed, _gyro, *, start, end, **_kwargs):
        assert np.isfinite(_speed[start : end + 1]).all()
        route = np.linspace(positions[start], positions[end], end - start + 1)
        return route, {
            "doppler_heading_p95_deg": 5.0,
            "speed_scale": 1.0,
            "endpoint_error_m": 0.0,
        }

    monkeypatch.setattr(
        "experiments.build_wp31_tdcp_gyro_gap_fill.build_endpoint_closed_route",
        fake_route,
    )
    displacements = [np.zeros(3) for _ in range(12)]
    rows = []
    for epoch in range(12):
        source = "gyro_doppler_gap_fill" if 2 <= epoch < 7 or epoch == 8 else "tdcp"
        rows.append(
            {
                "source": source,
                "interval_dt_s": 0.2,
                "dx_m": 0.0,
                "dy_m": 0.0,
                "dz_m": 0.0,
                "norm_m": 0.0,
            }
        )
    spans = [
        (0, 1, np.zeros(3), 1, "left"),
        (10, 11, np.array([5.0, 0.0, 0.0]), 2, "right"),
    ]

    reports = bridge_long_gyro_routes(
        displacements,
        rows,
        spans,
        times=np.arange(12, dtype=float) * 0.2,
        gyro_increments=np.zeros(11),
        gyro_bias_radps=0.0,
        doppler_displacements=np.zeros((12, 3)),
        min_gap_duration_s=0.5,
        min_longest_runner_ratio=2.0,
    )

    assert reports[0]["applied"] is True
    assert reports[0]["reason"] == "dominant_long_gyro_route_closed"
    np.testing.assert_allclose(sum(displacements[1:11]), [5.0, 0.0, 0.0])
    assert all(rows[epoch]["source"] == "anchor_closed_gyro_route" for epoch in range(2, 7))
    assert rows[8]["source"] == "gyro_doppler_gap_fill"


def test_long_gyro_route_bridge_rejects_nonunique_duration() -> None:
    displacements = [np.zeros(3) for _ in range(10)]
    rows = [
        {
            "source": "gyro_doppler_gap_fill" if epoch in (2, 3, 5, 6) else "tdcp",
            "interval_dt_s": 0.2,
            "dx_m": 0.0,
            "dy_m": 0.0,
            "dz_m": 0.0,
            "norm_m": 0.0,
        }
        for epoch in range(10)
    ]
    spans = [(0, 1, np.zeros(3), 1, "left"), (9, 10, np.ones(3), 2, "right")]

    reports = bridge_long_gyro_routes(
        displacements,
        rows,
        spans,
        times=np.arange(10, dtype=float) * 0.2,
        gyro_increments=np.zeros(9),
        gyro_bias_radps=0.0,
        doppler_displacements=np.zeros((10, 3)),
        min_gap_duration_s=0.2,
        min_longest_runner_ratio=2.0,
    )

    assert reports[0]["applied"] is False
    assert reports[0]["reason"] == "long_gap_not_duration_unique"
