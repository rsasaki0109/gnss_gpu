"""Unit tests for ``experiments.postprocess_gsdc2023_submission_guard``."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.postprocess_gsdc2023_submission_guard import (
    _haversine_m,
    apply_deviation_guard_to_submission,
    main,
)


def test_deviation_guard_replaces_rows_over_threshold():
    reference = pd.DataFrame({
        "tripId": ["A"] * 3,
        "UnixTimeMillis": [1000, 2000, 3000],
        "LatitudeDegrees": [37.0, 37.1, 37.2],
        "LongitudeDegrees": [-122.0, -122.1, -122.2],
    })
    candidate = reference.copy()
    candidate.loc[1, "LatitudeDegrees"] = 38.0

    out, stats = apply_deviation_guard_to_submission(
        candidate,
        reference,
        max_deviation_m=100.0,
    )

    assert out.loc[1, "LatitudeDegrees"] == reference.loc[1, "LatitudeDegrees"]
    assert out.loc[1, "LongitudeDegrees"] == reference.loc[1, "LongitudeDegrees"]
    assert out.loc[0, "LatitudeDegrees"] == candidate.loc[0, "LatitudeDegrees"]
    assert stats["rows_total"] == 3
    assert stats["guarded_rows"] == 1
    assert stats["trips_touched"] == 1


def test_deviation_guard_preserves_input_row_order_and_columns():
    candidate = pd.DataFrame({
        "tripId": ["B", "A", "B"],
        "UnixTimeMillis": [3000, 1000, 2000],
        "LatitudeDegrees": [37.0, 38.0, 39.0],
        "LongitudeDegrees": [-122.0, -123.0, -124.0],
        "source": ["b3", "a1", "b2"],
    })
    reference = pd.DataFrame({
        "tripId": ["A", "B", "B"],
        "UnixTimeMillis": [1000, 2000, 3000],
        "LatitudeDegrees": [38.0, 39.0, 37.0],
        "LongitudeDegrees": [-123.0, -124.0, -122.0],
    })

    out, _ = apply_deviation_guard_to_submission(candidate, reference, max_deviation_m=100.0)

    assert list(out.columns) == list(candidate.columns)
    assert out[["tripId", "UnixTimeMillis"]].to_dict("records") == candidate[
        ["tripId", "UnixTimeMillis"]
    ].to_dict("records")
    assert list(out["source"]) == ["b3", "a1", "b2"]


def test_deviation_guard_does_not_replace_threshold_boundary():
    reference = pd.DataFrame({
        "tripId": ["A"],
        "UnixTimeMillis": [1000],
        "LatitudeDegrees": [0.0],
        "LongitudeDegrees": [0.0],
    })
    candidate = pd.DataFrame({
        "tripId": ["A"],
        "UnixTimeMillis": [1000],
        "LatitudeDegrees": [0.0],
        "LongitudeDegrees": [0.001],
    })
    distance = float(
        _haversine_m(
            candidate["LatitudeDegrees"].to_numpy(),
            candidate["LongitudeDegrees"].to_numpy(),
            reference["LatitudeDegrees"].to_numpy(),
            reference["LongitudeDegrees"].to_numpy(),
        )[0]
    )

    out, stats = apply_deviation_guard_to_submission(
        candidate,
        reference,
        max_deviation_m=distance,
    )

    pd.testing.assert_frame_equal(out, candidate)
    assert stats["guarded_rows"] == 0


def test_deviation_guard_rejects_duplicate_keys():
    candidate = pd.DataFrame({
        "tripId": ["A", "A"],
        "UnixTimeMillis": [1000, 1000],
        "LatitudeDegrees": [37.0, 37.1],
        "LongitudeDegrees": [-122.0, -122.1],
    })
    reference = candidate.drop(index=1).reset_index(drop=True)

    with pytest.raises(ValueError, match="input has duplicate"):
        apply_deviation_guard_to_submission(candidate, reference)

    with pytest.raises(ValueError, match="reference has duplicate"):
        apply_deviation_guard_to_submission(reference, candidate)


def test_deviation_guard_rejects_missing_rows_from_either_side():
    candidate = pd.DataFrame({
        "tripId": ["A", "B"],
        "UnixTimeMillis": [1000, 2000],
        "LatitudeDegrees": [37.0, 38.0],
        "LongitudeDegrees": [-122.0, -123.0],
    })
    reference = pd.DataFrame({
        "tripId": ["A", "C"],
        "UnixTimeMillis": [1000, 3000],
        "LatitudeDegrees": [37.0, 39.0],
        "LongitudeDegrees": [-122.0, -124.0],
    })

    with pytest.raises(ValueError, match="input row.*no reference row.*reference row.*no input row"):
        apply_deviation_guard_to_submission(candidate, reference)


def test_deviation_guard_cli_summary_counts(tmp_path, capsys):
    reference = pd.DataFrame({
        "tripId": ["A/Pixel7", "A/Pixel7", "B/SamsungS22", "C/Mi8"],
        "UnixTimeMillis": [1000, 2000, 1000, 1000],
        "LatitudeDegrees": [37.0, 37.1, 38.0, 39.0],
        "LongitudeDegrees": [-122.0, -122.1, -123.0, -124.0],
    })
    candidate = reference.copy()
    candidate.loc[1, "LatitudeDegrees"] = 38.1
    candidate.loc[2, "LongitudeDegrees"] = -124.0
    input_path = tmp_path / "candidate.csv"
    reference_path = tmp_path / "reference.csv"
    output_path = tmp_path / "guarded.csv"
    candidate.to_csv(input_path, index=False)
    reference.to_csv(reference_path, index=False)

    rc = main([
        "--input",
        str(input_path),
        "--reference",
        str(reference_path),
        "--output",
        str(output_path),
        "--max-deviation-m",
        "100",
    ])

    assert rc == 0
    stdout = capsys.readouterr().out
    assert "trip=A/Pixel7 guarded_rows=1" in stdout
    assert "trip=B/SamsungS22 guarded_rows=1" in stdout
    assert "rows_total=4 guarded_rows=2 (50.00%) trips_touched=2" in stdout
    out = pd.read_csv(output_path)
    pd.testing.assert_frame_equal(out, reference)


def _chunked_frames(n_rows: int = 40, bad_lo: int = 20, bad_hi: int = 40, offset_deg: float = 0.001):
    # ~89 m of longitude offset (at lat 37) on rows [bad_lo, bad_hi): below the
    # 100 m row guard but far beyond the healthy-chunk deviation band.
    times = 1000 + np.arange(n_rows) * 1000
    reference = pd.DataFrame({
        "tripId": ["trip-a"] * n_rows,
        "UnixTimeMillis": times,
        "LatitudeDegrees": np.full(n_rows, 37.0),
        "LongitudeDegrees": np.full(n_rows, -122.0),
    })
    candidate = reference.copy()
    candidate.loc[bad_lo:bad_hi - 1, "LongitudeDegrees"] += offset_deg
    return candidate, reference


def test_chunk_fallback_replaces_diverged_chunk_and_keeps_healthy_chunk():
    candidate, reference = _chunked_frames()

    out, stats = apply_deviation_guard_to_submission(
        candidate,
        reference,
        max_deviation_m=100.0,
        chunk_size=20,
        chunk_deviation_p95_m=50.0,
    )

    # diverged chunk (rows 20-39) snapped to reference, healthy chunk untouched
    pd.testing.assert_series_equal(
        out["LongitudeDegrees"], reference["LongitudeDegrees"], check_names=False, check_exact=True
    )
    assert stats["chunk_fallback_rows"] == 20
    assert stats["guarded_rows"] == 20
    chunks = stats["chunk_fallback_chunks"]
    assert len(chunks) == 1
    assert chunks[0]["tripId"] == "trip-a"
    assert chunks[0]["chunk"] == 1
    assert chunks[0]["rows"] == 20


def test_chunk_fallback_disabled_by_default_keeps_sub_threshold_rows():
    candidate, reference = _chunked_frames()

    out, stats = apply_deviation_guard_to_submission(
        candidate,
        reference,
        max_deviation_m=100.0,
    )

    # ~55 m offsets stay below the row guard, so nothing is replaced
    assert stats["guarded_rows"] == 0
    assert stats["chunk_fallback_rows"] == 0
    pd.testing.assert_frame_equal(out, candidate)


def test_chunk_fallback_threshold_boundary_keeps_chunk():
    candidate, reference = _chunked_frames()

    out, stats = apply_deviation_guard_to_submission(
        candidate,
        reference,
        max_deviation_m=100.0,
        chunk_size=20,
        chunk_deviation_p95_m=90.0,
    )

    assert stats["chunk_fallback_rows"] == 0
    pd.testing.assert_frame_equal(out, candidate)
