from __future__ import annotations

import pytest

from experiments.build_wp32_motion_parent import motion_parent


def _rows(offsets: list[float]) -> list[dict[str, str]]:
    return [
        {
            "epoch": str(epoch),
            "ecef_x": str(10.0 + offset),
            "ecef_y": "20.0",
            "ecef_z": "30.0",
            "error_m": "999.0",
        }
        for epoch, offset in enumerate(offsets, start=5)
    ]


def test_motion_parent_uses_position_median_and_ignores_truth_audit() -> None:
    result = motion_parent(
        _rows([-0.02, 0.0, 0.01, 0.02]),
        start=5,
        end=9,
        min_epochs=4,
        max_p95_deviation_m=0.1,
        max_deviation_m=0.1,
    )
    assert result["position_ecef"] == pytest.approx([10.005, 20.0, 30.0])
    assert result["n_epochs"] == 4


def test_motion_parent_rejects_noncontiguous_coverage() -> None:
    rows = _rows([0.0, 0.0, 0.0])
    rows[1]["epoch"] = "8"
    with pytest.raises(RuntimeError, match="contiguous"):
        motion_parent(rows, start=5, end=8, min_epochs=3)


def test_motion_parent_rejects_dynamic_segment() -> None:
    with pytest.raises(RuntimeError, match="spread"):
        motion_parent(
            _rows([0.0, 0.0, 1.0, 1.0]),
            start=5,
            end=9,
            min_epochs=4,
            max_p95_deviation_m=0.25,
            max_deviation_m=0.5,
        )
