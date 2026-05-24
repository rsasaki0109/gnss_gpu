"""Smoke tests for build_gsdc2023_chunk_merged_submission.

The CLI composes two already-tested modules (gsdc2023_ab_chunk_merge for
merge prediction, build_gsdc2023_bridge_submission for sample-CSV patching).
These tests cover the only new logic: per-source column swap and the
``rows_skipped_missing_column`` accounting.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.build_gsdc2023_chunk_merged_submission import (
    _SOURCE_LATITUDE_COLUMNS,
    _apply_merge_to_trip,
)


def _make_bridge_csv(tmp_path: Path) -> Path:
    """200 epoch trip; rows 0..99 selected fgo_no_tdcp, rows 100..199 baseline."""
    n = 200
    times = np.arange(1_000_000, 1_000_000 + n) * 1000
    sources = ["fgo_no_tdcp"] * 100 + ["baseline"] * 100
    base_lat = np.full(n, 35.0)
    base_lon = np.full(n, -120.0)
    fgo_lat = np.full(n, 35.001)
    fgo_lon = np.full(n, -120.001)
    rawwls_lat = np.full(n, 35.002)
    rawwls_lon = np.full(n, -120.002)
    sel_lat = np.where(np.array(sources) == "fgo_no_tdcp", 35.005, base_lat)
    sel_lon = np.where(np.array(sources) == "fgo_no_tdcp", -120.005, base_lon)
    df = pd.DataFrame({
        "UnixTimeMillis": times,
        "SelectedSource": sources,
        "BaselineLatitudeDegrees": base_lat,
        "BaselineLongitudeDegrees": base_lon,
        "BaselineAltitudeMeters": np.full(n, 100.0),
        "RawWlsLatitudeDegrees": rawwls_lat,
        "RawWlsLongitudeDegrees": rawwls_lon,
        "RawWlsAltitudeMeters": np.full(n, 100.2),
        "FgoLatitudeDegrees": fgo_lat,
        "FgoLongitudeDegrees": fgo_lon,
        "FgoAltitudeMeters": np.full(n, 100.1),
        "LatitudeDegrees": sel_lat,
        "LongitudeDegrees": sel_lon,
        "AltitudeMeters": np.full(n, 100.0),
        "GroundTruthLatitudeDegrees": np.full(n, np.nan),
        "GroundTruthLongitudeDegrees": np.full(n, np.nan),
        "GroundTruthAltitudeMeters": np.full(n, np.nan),
    })
    trip_dir = tmp_path / "2099-01-01-99-99-us-ca-tst-x" / "pixel_test"
    trip_dir.mkdir(parents=True)
    csv_path = trip_dir / "bridge_positions.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def test_apply_merge_swaps_to_baseline_uses_per_source_column(tmp_path: Path):
    csv_path = _make_bridge_csv(tmp_path)
    # Bridge has one chunk 0..100 (fgo_no_tdcp) and one chunk 100..200 (baseline).
    chunk_records = [
        {"start_epoch": 0, "end_epoch": 100, "auto_source": "baseline", "gated_source": "fgo_no_tdcp", "candidates": {}},
        {"start_epoch": 100, "end_epoch": 200, "auto_source": "baseline", "gated_source": "baseline", "candidates": {}},
    ]
    # Predict the first chunk should be baseline post-merge (fix #2 recovery direction).
    merged_predictions = [(0, 100, "baseline"), (100, 200, "baseline")]

    df, res = _apply_merge_to_trip(csv_path, chunk_records, merged_predictions)
    assert res.chunks_changed == 1
    assert res.rows_changed == 100
    assert res.rows_skipped_missing_column == 0
    # Rows 0..99 should now reflect Baseline columns (35.0, -120.0), not fgo_no_tdcp (35.005, -120.005).
    swapped = df.iloc[:100]
    assert (swapped["LatitudeDegrees"] == 35.0).all()
    assert (swapped["LongitudeDegrees"] == -120.0).all()
    assert (swapped["SelectedSource"] == "baseline").all()
    # Rows 100..199 untouched.
    assert (df.iloc[100:]["SelectedSource"] == "baseline").all()


def test_apply_merge_skips_when_predicted_source_has_no_column(tmp_path: Path):
    csv_path = _make_bridge_csv(tmp_path)
    chunk_records = [
        {"start_epoch": 0, "end_epoch": 100, "auto_source": "baseline", "gated_source": "fgo_no_tdcp", "candidates": {}},
        {"start_epoch": 100, "end_epoch": 200, "auto_source": "baseline", "gated_source": "baseline", "candidates": {}},
    ]
    # Predict swap toward fgo_no_tdcp -- which has no per-source column.
    merged_predictions = [(100, 200, "fgo_no_tdcp")]
    df, res = _apply_merge_to_trip(csv_path, chunk_records, merged_predictions)
    assert res.chunks_changed == 0
    assert res.rows_changed == 0
    assert res.rows_skipped_missing_column == 100
    # Sources unchanged.
    assert (df.iloc[100:]["SelectedSource"] == "baseline").all()


def test_source_column_mapping_covers_known_per_source_outputs():
    assert set(_SOURCE_LATITUDE_COLUMNS) == {"baseline", "raw_wls", "fgo"}
