"""Unit tests for accel-based submission smoother."""
from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.postprocess_gsdc2023_submission_accel_smooth import (
    apply_accel_smoothing_to_submission,
    smooth_axis_by_accel,
)


def test_smooth_axis_replaces_high_accel_outlier():
    # Uniform 1-Hz, single jump at t=3 of 50m (-> accel ~100 m/s²)
    x = np.array([0.0, 1.0, 2.0, 52.0, 4.0, 5.0, 6.0], dtype=np.float64)
    dt = np.ones(6, dtype=np.float64)
    out, replaced = smooth_axis_by_accel(x, dt, accel_max=5.0)
    assert replaced[3]
    # Replacement = mean of neighbours = (2 + 4) / 2 = 3
    assert abs(out[3] - 3.0) < 1e-9
    # Other rows untouched
    untouched = [i for i in range(len(x)) if i != 3]
    np.testing.assert_allclose(out[untouched], x[untouched])


def test_smooth_axis_preserves_smooth_motion():
    # Constant-velocity motion (accel = 0): nothing flagged.
    x = np.arange(20, dtype=np.float64) * 2.0  # 2 m/s, straight
    dt = np.ones(19, dtype=np.float64)
    out, replaced = smooth_axis_by_accel(x, dt, accel_max=5.0)
    assert not replaced.any()
    np.testing.assert_array_equal(out, x)


def test_smooth_axis_preserves_realistic_acceleration():
    # Acceleration of 2 m/s² (well under threshold) -> untouched
    t = np.arange(20, dtype=np.float64)
    x = 0.5 * 2.0 * t * t  # accel = 2 m/s²
    dt = np.ones(19, dtype=np.float64)
    out, replaced = smooth_axis_by_accel(x, dt, accel_max=5.0)
    assert not replaced.any()


def test_apply_accel_smoothing_per_trip_isolation():
    # Trip A: outlier at index 3.  Trip B: clean.
    rows = []
    rows += [("A", i * 1000, 0.0, 0.0) for i in range(10)]
    # Trip A: lat jump at idx 3 -> ~111 km
    rows[3] = ("A", 3000, 1.0, 0.0)
    # Trip B: clean monotonic motion
    rows += [("B", i * 1000, 37.0 + i * 1e-6, -122.0) for i in range(10)]
    df = pd.DataFrame(rows, columns=["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"])
    out, stats = apply_accel_smoothing_to_submission(df, accel_max=5.0, passes=1)
    assert stats["trips"] == 2
    a = out[out["tripId"] == "A"].sort_values("UnixTimeMillis")
    b = out[out["tripId"] == "B"].sort_values("UnixTimeMillis")
    # A row 3 replaced (no longer 1.0)
    assert abs(a["LatitudeDegrees"].iloc[3]) < 0.5
    # B unchanged
    b_lat = b["LatitudeDegrees"].to_numpy()
    expected = 37.0 + np.arange(10) * 1e-6
    np.testing.assert_allclose(b_lat, expected)


def test_apply_accel_smoothing_iterative_passes_decay():
    # Two adjacent outliers (idx 3, 4) — single pass leaves one, two passes clean.
    n = 12
    rows = []
    for i in range(n):
        lat = 0.0
        if i in (3, 4):
            lat = 0.5  # ~55 km offset
        rows.append(("A", i * 1000, lat, 0.0))
    df = pd.DataFrame(rows, columns=["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"])
    out_p1, stats_p1 = apply_accel_smoothing_to_submission(df, accel_max=5.0, passes=1)
    out_p3, stats_p3 = apply_accel_smoothing_to_submission(df, accel_max=5.0, passes=3)
    # More passes should change >= same number of rows (monotonic accumulation)
    assert int(stats_p3["rows_changed"]) >= int(stats_p1["rows_changed"])
    # After 3 passes, both outliers should be ~0
    a = out_p3[out_p3["tripId"] == "A"].sort_values("UnixTimeMillis")
    assert abs(a["LatitudeDegrees"].iloc[3]) < 0.1
    assert abs(a["LatitudeDegrees"].iloc[4]) < 0.1
