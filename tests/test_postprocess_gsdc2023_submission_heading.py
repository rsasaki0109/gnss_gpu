"""Unit tests for heading-consistency smoother."""
from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.postprocess_gsdc2023_submission_heading import (
    apply_heading_smoothing_to_submission,
    smooth_heading_outliers,
)


def test_smooth_heading_flags_impossible_turn():
    # Straight east-ward motion (10 rows), single bad row that jumps north
    # (large bearing change between edge t-1->t and t->t+1).
    n = 10
    lat = np.full(n, 37.0)
    lng = -122.0 + np.arange(n) * (10.0 / 111_320.0)  # 10 m east per epoch
    # Inject a single lat jump at index 5 (~50m north)
    lat[5] = lat[5] + 50.0 / 111_320.0
    dt = np.ones(n - 1)
    out_lat, out_lng, flagged = smooth_heading_outliers(lat, lng, dt, heading_max_dps=30.0)
    assert flagged[5]
    # Replacement: midpoint of neighbours
    assert abs(out_lat[5] - 37.0) < 1e-7
    # Other rows untouched
    for i in [0, 1, 2, 3, 4, 6, 7, 8, 9]:
        assert out_lat[i] == lat[i]
        assert out_lng[i] == lng[i]


def test_smooth_heading_preserves_smooth_curve():
    # Gentle 5°/s turn at 10 m/s — no flag.
    n = 30
    dt = np.ones(n - 1)
    speed = 10.0  # m/s
    yaw_rate = 5.0  # deg/s
    east = np.zeros(n)
    north = np.zeros(n)
    heading = 0.0
    for i in range(1, n):
        heading += np.radians(yaw_rate) * dt[i - 1]
        east[i] = east[i - 1] + speed * np.sin(heading) * dt[i - 1]
        north[i] = north[i - 1] + speed * np.cos(heading) * dt[i - 1]
    lat0 = 37.0
    lng0 = -122.0
    mlat = 111_320.0
    mlng = 111_320.0 * np.cos(np.radians(lat0))
    lat = lat0 + north / mlat
    lng = lng0 + east / mlng
    out_lat, out_lng, flagged = smooth_heading_outliers(lat, lng, dt, heading_max_dps=30.0)
    assert not flagged.any()


def test_smooth_heading_preserves_straight_motion():
    n = 20
    lat = np.full(n, 37.0) + np.arange(n) * (5.0 / 111_320.0)
    lng = np.full(n, -122.0)
    dt = np.ones(n - 1)
    out_lat, out_lng, flagged = smooth_heading_outliers(lat, lng, dt, heading_max_dps=30.0)
    assert not flagged.any()
    np.testing.assert_array_equal(out_lat, lat)


def test_apply_heading_smoothing_per_trip_isolation():
    # Trip A: zigzag, Trip B: smooth
    n = 12
    rows_a = []
    lat_a = 37.0
    lng_a = -122.0
    for i in range(n):
        # alternating large lat jump
        rows_a.append(("A", i * 1000, lat_a + (0.0005 if i % 2 == 1 else 0.0), lng_a + i * 1e-5))
    rows_b = [
        ("B", i * 1000, 37.0 + i * 5.0 / 111_320.0, -121.0) for i in range(n)
    ]
    df = pd.DataFrame(rows_a + rows_b, columns=["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"])
    out, stats = apply_heading_smoothing_to_submission(df, heading_max_dps=30.0)
    assert stats["trips"] == 2
    b = out[out["tripId"] == "B"].sort_values("UnixTimeMillis")
    # B trajectory unchanged (straight smooth motion)
    expected_lat = 37.0 + np.arange(n) * 5.0 / 111_320.0
    np.testing.assert_allclose(b["LatitudeDegrees"].to_numpy(), expected_lat)


def test_apply_heading_smoothing_no_op_when_threshold_high():
    # Same zigzag, but threshold so high nothing flags.
    n = 12
    rows = []
    for i in range(n):
        rows.append(("A", i * 1000, 37.0 + (0.0005 if i % 2 == 1 else 0.0), -122.0 + i * 1e-5))
    df = pd.DataFrame(rows, columns=["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"])
    out, stats = apply_heading_smoothing_to_submission(df, heading_max_dps=10000.0)
    assert stats["rows_changed"] == 0
