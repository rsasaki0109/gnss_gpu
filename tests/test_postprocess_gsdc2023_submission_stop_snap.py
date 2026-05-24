"""Unit tests for stationary-segment median snap smoother."""
from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.postprocess_gsdc2023_submission_stop_snap import (
    apply_stop_snap_to_submission,
    detect_stationary_runs,
)


def test_detect_runs_single_long_run():
    # 12 stationary edges (all under threshold), one moving edge at idx 12
    d = np.array([0.1] * 12 + [5.0] * 5)
    runs = detect_stationary_runs(d, move_threshold_m=0.5, min_run_length=10)
    assert runs == [(0, 12)]


def test_detect_runs_skips_short_run():
    d = np.array([0.1] * 5 + [5.0] + [0.1] * 11 + [5.0])
    runs = detect_stationary_runs(d, move_threshold_m=0.5, min_run_length=10)
    # First run is only 6 rows long -> skipped.  Second is 12 rows -> kept.
    assert runs == [(6, 17)]


def test_detect_runs_empty_signal():
    assert detect_stationary_runs(np.array([]), move_threshold_m=0.5, min_run_length=10) == []


def test_apply_snap_collapses_noisy_stationary_segment():
    # 15 rows where the car is "parked" with ~1m GPS wobble.  Lat/lng noise
    # should collapse to a single median value.
    lat = 37.0 + np.array([0.0, 1e-6, -1e-6, 2e-6, 0.0, 1e-6, -2e-6, 0.0, 1e-6, -1e-6, 0.0, 1e-6, 0.0, -1e-6, 1e-6])
    lng = -122.0 + np.array([0.0, -1e-6, 1e-6, 0.0, -2e-6, 1e-6, 0.0, -1e-6, 2e-6, 0.0, 1e-6, -1e-6, 0.0, 1e-6, -1e-6])
    df = pd.DataFrame({
        "tripId": ["A"] * 15,
        "UnixTimeMillis": list(range(15)),
        "LatitudeDegrees": lat,
        "LongitudeDegrees": lng,
    })
    out, stats = apply_stop_snap_to_submission(df, move_threshold_m=0.5, min_run_length=10)
    assert stats["runs"] == 1
    a = out[out["tripId"] == "A"].sort_values("UnixTimeMillis")
    # All rows collapsed to the same median
    assert a["LatitudeDegrees"].nunique() == 1
    assert a["LongitudeDegrees"].nunique() == 1


def test_apply_snap_preserves_moving_trajectory():
    # Steady 5 m/s motion -> not stationary -> untouched.
    n = 30
    lat = 37.0 + np.arange(n) * (5.0 / 111_320.0)  # 5 m per epoch in lat
    lng = -122.0 * np.ones(n)
    df = pd.DataFrame({
        "tripId": ["A"] * n,
        "UnixTimeMillis": list(range(n)),
        "LatitudeDegrees": lat,
        "LongitudeDegrees": lng,
    })
    out, stats = apply_stop_snap_to_submission(df, move_threshold_m=0.5, min_run_length=10)
    assert stats["runs"] == 0
    assert stats["rows_changed"] == 0


def test_apply_snap_per_trip_isolation():
    # Trip A: stationary, Trip B: moving
    lat_a = 37.0 + np.random.RandomState(0).uniform(-1e-6, 1e-6, size=20)
    lng_a = -122.0 + np.random.RandomState(1).uniform(-1e-6, 1e-6, size=20)
    lat_b = 37.0 + np.arange(20) * (5.0 / 111_320.0)
    lng_b = -121.0 * np.ones(20)
    df = pd.DataFrame({
        "tripId": ["A"] * 20 + ["B"] * 20,
        "UnixTimeMillis": list(range(20)) * 2,
        "LatitudeDegrees": np.concatenate([lat_a, lat_b]),
        "LongitudeDegrees": np.concatenate([lng_a, lng_b]),
    })
    out, stats = apply_stop_snap_to_submission(df, move_threshold_m=0.5, min_run_length=10)
    assert stats["trips"] == 2
    a = out[out["tripId"] == "A"]
    b = out[out["tripId"] == "B"].sort_values("UnixTimeMillis")
    assert a["LatitudeDegrees"].nunique() == 1
    # B unchanged
    np.testing.assert_allclose(b["LatitudeDegrees"].to_numpy(), lat_b)
