from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.gsdc2023_ab_gates import Disposition, TripDisposition
from experiments.gsdc2023_ab_revert import (
    compute_delta_stats,
    dispositions_to_keep_map,
    per_trip_shift_proxy,
    simulate_revert,
)


def _disp(trip: str, disp: Disposition) -> TripDisposition:
    return TripDisposition(trip_id=trip, disposition=disp, dd_pass=False, ntdc_pass=False)


def _row_delta(rows: list[tuple[str, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["tripId", "delta_m"])


def test_keep_map_only_revert_zeroes_out():
    keep = dispositions_to_keep_map(
        [
            _disp("a", Disposition.KEPT),
            _disp("b", Disposition.REVERTED),
            _disp("c", Disposition.PASSTHROUGH),
        ]
    )
    assert keep == {"a": True, "b": False, "c": True}


def test_simulate_revert_zeroes_only_reverted_trips():
    row_delta = _row_delta(
        [
            ("a", 1.0),
            ("a", 2.5),
            ("b", 7.5),
            ("b", 0.5),
            ("c", 3.0),
        ]
    )
    sim = simulate_revert(
        row_delta,
        [
            _disp("a", Disposition.KEPT),
            _disp("b", Disposition.REVERTED),
            _disp("c", Disposition.PASSTHROUGH),
        ],
    )
    expected = [1.0, 2.5, 0.0, 0.0, 3.0]
    assert list(sim["sim_delta_m"]) == expected
    # Input must not be mutated.
    assert "sim_delta_m" not in row_delta.columns


def test_simulate_revert_treats_missing_trip_as_passthrough():
    row_delta = _row_delta([("a", 4.0), ("b", 2.0)])
    sim = simulate_revert(row_delta, [_disp("a", Disposition.REVERTED)])
    assert list(sim["sim_delta_m"]) == [0.0, 2.0]


def test_simulate_revert_validates_required_columns():
    row_delta = pd.DataFrame({"tripId": ["a"], "wrong": [1.0]})
    with pytest.raises(ValueError, match="row_delta is missing columns"):
        simulate_revert(row_delta, [])


def test_simulate_revert_supports_custom_sim_column_name():
    row_delta = _row_delta([("a", 1.0), ("b", 5.0)])
    sim = simulate_revert(
        row_delta,
        [_disp("a", Disposition.KEPT), _disp("b", Disposition.REVERTED)],
        sim_column="gated_delta_m",
    )
    assert list(sim["gated_delta_m"]) == [1.0, 0.0]
    assert "sim_delta_m" not in sim.columns


def test_compute_delta_stats_counts_thresholds():
    s = pd.Series([0.0, 0.5, 1.5, 6.0, 10.0])
    stats = compute_delta_stats(s)
    assert stats.changed_rows == 4  # > 1e-9
    assert stats.rows_gt_1m == 3
    assert stats.rows_gt_5m == 2
    assert stats.sum_m == pytest.approx(18.0)
    assert stats.max_m == 10.0


def test_compute_delta_stats_empty_series_returns_zeros():
    stats = compute_delta_stats(pd.Series([], dtype=float))
    assert stats.changed_rows == 0
    assert stats.max_m == 0.0
    assert stats.p95_m == 0.0


def test_per_trip_shift_proxy_uses_p50_p95_midpoint():
    row_delta = _row_delta(
        [
            ("a", 0.0),
            ("a", 0.0),
            ("a", 5.0),  # p50=0, p95~4 -> proxy ~2
            ("b", 1.0),
            ("b", 3.0),
            ("b", 5.0),  # p50=3, p95~4.8 -> proxy ~3.9
        ]
    )
    proxy = per_trip_shift_proxy(row_delta, column="delta_m")
    assert proxy["a"] == pytest.approx(0.5 * (0.0 + float(np.percentile([0.0, 0.0, 5.0], 95))))
    assert proxy["b"] == pytest.approx(0.5 * (3.0 + float(np.percentile([1.0, 3.0, 5.0], 95))))


def test_per_trip_shift_proxy_rejects_missing_column():
    row_delta = _row_delta([("a", 1.0)])
    with pytest.raises(ValueError, match="not in row_delta columns"):
        per_trip_shift_proxy(row_delta, column="bogus")
