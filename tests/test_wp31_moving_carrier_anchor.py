from __future__ import annotations

import numpy as np

from experiments.refine_wp31_moving_carrier_anchor import apply_temporal_consensus


def _row(epoch: int, x: float, passed: int = 1) -> dict:
    return {
        "epoch": epoch,
        "position_ecef": [x, 0.0, 0.0],
        "single_epoch_metric_pass": passed,
        "production_selected": 0,
    }


def test_temporal_consensus_rejects_single_epoch_pass() -> None:
    rows = [_row(0, 0.0, 1), _row(5, 1.0, 0), _row(10, 2.0, 0)]
    current = {0: np.array([0.0, 0.0, 0.0]), 5: np.array([1.0, 0.0, 0.0]), 10: np.array([2.0, 0.0, 0.0])}
    apply_temporal_consensus(rows, current, stride=5)
    assert not any(row["production_selected"] for row in rows)


def test_temporal_consensus_accepts_consistent_triple() -> None:
    rows = [_row(0, 0.0), _row(5, 1.1), _row(10, 2.2)]
    current = {0: np.array([0.0, 0.0, 0.0]), 5: np.array([1.0, 0.0, 0.0]), 10: np.array([2.0, 0.0, 0.0])}
    apply_temporal_consensus(rows, current, stride=5)
    assert all(row["production_selected"] for row in rows)


def test_temporal_consensus_rejects_inconsistent_motion() -> None:
    rows = [_row(0, 0.0), _row(5, 3.0), _row(10, 6.0)]
    current = {0: np.array([0.0, 0.0, 0.0]), 5: np.array([1.0, 0.0, 0.0]), 10: np.array([2.0, 0.0, 0.0])}
    apply_temporal_consensus(rows, current, stride=5)
    assert not any(row["production_selected"] for row in rows)
