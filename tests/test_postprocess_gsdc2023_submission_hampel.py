"""Unit tests for ``experiments.postprocess_gsdc2023_submission_hampel``.

Synthetic trajectories verify the Hampel filter:

  * isolated outlier replaced by median
  * smooth straight-line motion untouched (no false positives)
  * multiple tripIds processed independently (no cross-trip leakage)
  * mad_floor prevents over-aggressive replacement when MAD is zero
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.postprocess_gsdc2023_submission_hampel import (
    apply_hampel_to_submission,
    hampel_filter_1d,
    main,
)


def test_hampel_1d_replaces_isolated_outlier():
    values = np.array([1.0, 1.0, 1.0, 1.0, 50.0, 1.0, 1.0, 1.0, 1.0])
    out = hampel_filter_1d(values, window=5, k=2.5, mad_floor=0.0)
    assert out[4] == 1.0
    np.testing.assert_array_equal(out[[0, 1, 2, 3, 5, 6, 7, 8]], values[[0, 1, 2, 3, 5, 6, 7, 8]])


def test_hampel_1d_preserves_smooth_motion():
    values = np.arange(20, dtype=np.float64)
    out = hampel_filter_1d(values, window=5, k=2.5, mad_floor=0.0)
    np.testing.assert_array_equal(out, values)


def test_hampel_1d_mad_floor_prevents_over_aggressive_replacement():
    # All-constant series has MAD=0 -> sigma would be 0 -> any tiny deviation flagged.
    values = np.array([1.0, 1.0, 1.0, 1.001, 1.0, 1.0, 1.0])
    out = hampel_filter_1d(values, window=5, k=2.5, mad_floor=0.01)
    np.testing.assert_array_equal(out, values)


def test_apply_hampel_to_submission_per_trip_isolation():
    df = pd.DataFrame({
        "tripId": ["A"] * 8 + ["B"] * 8,
        "UnixTimeMillis": list(range(8)) * 2,
        "LatitudeDegrees": [37.0] * 3 + [37.5] + [37.0] * 4 + [38.0] * 8,
        "LongitudeDegrees": [-122.0] * 8 + [-121.0] * 4 + [-121.5] + [-121.0] * 3,
    })
    out, stats = apply_hampel_to_submission(df, window=5, k=2.5, mad_floor_deg=0.0)
    assert stats["trips"] == 2
    a = out[out["tripId"] == "A"].sort_values("UnixTimeMillis")
    b = out[out["tripId"] == "B"].sort_values("UnixTimeMillis")
    # Trip A: outlier at index 3 replaced
    assert a["LatitudeDegrees"].iloc[3] == 37.0
    # Trip B: outlier at index 4 replaced
    assert b["LongitudeDegrees"].iloc[4] == -121.0


def test_apply_hampel_preserves_columns_other_than_latlng():
    df = pd.DataFrame({
        "tripId": ["A"] * 10,
        "UnixTimeMillis": list(range(10)),
        "LatitudeDegrees": [37.0] * 10,
        "LongitudeDegrees": [-122.0] * 10,
        "SomeMetadata": list("abcdefghij"),
    })
    out, _ = apply_hampel_to_submission(df, window=5, k=2.5, mad_floor_deg=1e-6)
    assert list(out["SomeMetadata"]) == list("abcdefghij")
    assert list(out.columns) == list(df.columns)


def test_apply_hampel_iterative_passes_peel_consecutive_outliers():
    # Two adjacent outliers (indices 9, 10) form a small "plateau" that a
    # single Hampel pass with a small window may treat as the local median.
    # Iterative passes should clean it up.
    lat = [37.0] * 9 + [37.5, 37.5] + [37.0] * 9
    lng = [-122.0] * 20
    df = pd.DataFrame({
        "tripId": ["A"] * 20,
        "UnixTimeMillis": list(range(20)),
        "LatitudeDegrees": lat,
        "LongitudeDegrees": lng,
    })
    out_p1, stats_p1 = apply_hampel_to_submission(df, window=5, k=2.5, mad_floor_deg=0.0, passes=1)
    out_p3, stats_p3 = apply_hampel_to_submission(df, window=5, k=2.5, mad_floor_deg=0.0, passes=3)
    # passes=3 must clean at least as many rows as passes=1 (monotonic).
    assert stats_p3["rows_changed"] >= stats_p1["rows_changed"]
    assert stats_p3["passes"] == 3
    assert len(stats_p3["per_pass_changed"]) == 3
    # After 3 passes both outliers should be replaced.
    assert out_p3["LatitudeDegrees"].iloc[9] == 37.0
    assert out_p3["LatitudeDegrees"].iloc[10] == 37.0


def test_apply_hampel_passes_idempotent_on_clean_input():
    # passes=5 on smooth motion must change nothing.
    df = pd.DataFrame({
        "tripId": ["A"] * 30,
        "UnixTimeMillis": list(range(30)),
        "LatitudeDegrees": np.linspace(37.0, 37.0001, 30).tolist(),
        "LongitudeDegrees": np.linspace(-122.0, -122.0001, 30).tolist(),
    })
    out, stats = apply_hampel_to_submission(df, window=7, k=2.5, mad_floor_deg=1e-6, passes=5)
    assert stats["rows_changed"] == 0
    assert all(p == 0 for p in stats["per_pass_changed"])


def test_apply_hampel_exclude_phone_prefix_skips_pixel_trip_only():
    pixel_lat = [37.0] * 3 + [37.5] + [37.0] * 4
    samsung_lat = [38.0] * 3 + [38.5] + [38.0] * 4
    df = pd.DataFrame({
        "tripId": ["2023-05-01-US-CA/Pixel7Pro"] * 8
        + ["2023-05-01-US-CA/SamsungS22"] * 8,
        "UnixTimeMillis": list(range(8)) * 2,
        "LatitudeDegrees": pixel_lat + samsung_lat,
        "LongitudeDegrees": [-122.0] * 8 + [-123.0] * 8,
    })

    out, stats = apply_hampel_to_submission(
        df,
        window=5,
        k=2.5,
        mad_floor_deg=0.0,
        exclude_phone_prefixes=("pixel",),
    )

    pixel = out[out["tripId"].str.endswith("Pixel7Pro")].sort_values("UnixTimeMillis")
    samsung = out[out["tripId"].str.endswith("SamsungS22")].sort_values("UnixTimeMillis")
    assert pixel["LatitudeDegrees"].iloc[3] == 37.5
    assert samsung["LatitudeDegrees"].iloc[3] == 38.0
    assert stats["rows_changed"] == 1
    assert stats["skipped_trips"] == 1


def test_apply_hampel_without_exclude_prefix_keeps_existing_behavior():
    df = pd.DataFrame({
        "tripId": ["2023-05-01-US-CA/Pixel7Pro"] * 8
        + ["2023-05-01-US-CA/SamsungS22"] * 8,
        "UnixTimeMillis": list(range(8)) * 2,
        "LatitudeDegrees": [37.0] * 3 + [37.5] + [37.0] * 4
        + [38.0] * 3 + [38.5] + [38.0] * 4,
        "LongitudeDegrees": [-122.0] * 8 + [-123.0] * 8,
    })

    out, stats = apply_hampel_to_submission(df, window=5, k=2.5, mad_floor_deg=0.0)

    pixel = out[out["tripId"].str.endswith("Pixel7Pro")].sort_values("UnixTimeMillis")
    samsung = out[out["tripId"].str.endswith("SamsungS22")].sort_values("UnixTimeMillis")
    assert pixel["LatitudeDegrees"].iloc[3] == 37.0
    assert samsung["LatitudeDegrees"].iloc[3] == 38.0
    assert stats["rows_changed"] == 2
    assert "skipped_trips" not in stats


def test_hampel_cli_reports_skipped_trips(tmp_path, capsys):
    df = pd.DataFrame({
        "tripId": ["2023-05-01-US-CA/Pixel7Pro"] * 8
        + ["2023-05-01-US-CA/SamsungS22"] * 8,
        "UnixTimeMillis": list(range(8)) * 2,
        "LatitudeDegrees": [37.0] * 3 + [37.5] + [37.0] * 4
        + [38.0] * 3 + [38.5] + [38.0] * 4,
        "LongitudeDegrees": [-122.0] * 8 + [-123.0] * 8,
    })
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"
    df.to_csv(input_path, index=False)

    rc = main([
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--window",
        "5",
        "--k",
        "2.5",
        "--mad-floor-deg",
        "0",
        "--exclude-phone-prefix",
        "PIXEL",
    ])

    assert rc == 0
    stdout = capsys.readouterr().out
    assert "skipped_trips=1" in stdout
    out = pd.read_csv(output_path)
    pixel = out[out["tripId"].str.endswith("Pixel7Pro")].sort_values("UnixTimeMillis")
    samsung = out[out["tripId"].str.endswith("SamsungS22")].sort_values("UnixTimeMillis")
    assert pixel["LatitudeDegrees"].iloc[3] == 37.5
    assert samsung["LatitudeDegrees"].iloc[3] == 38.0
