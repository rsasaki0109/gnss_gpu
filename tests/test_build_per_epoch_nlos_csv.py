"""Unit tests for PLATEAU per-epoch NLOS CSV helpers."""

from __future__ import annotations

import csv
import io
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from build_per_epoch_nlos_csv import (  # noqa: E402
    EXTENDED_NLOS_CSV_HEADER,
    MINIMAL_NLOS_CSV_HEADER,
    _elevation_deg,
    _nearest_position,
    _write_batch_rows,
)
from gnss_gpu.nlos_mask import load_nlos_prn_sets  # noqa: E402


def test_minimal_header_matches_downstream_loader_contract():
    assert MINIMAL_NLOS_CSV_HEADER == ("tow", "epoch_idx", "prn", "is_los")
    assert MINIMAL_NLOS_CSV_HEADER == tuple(EXTENDED_NLOS_CSV_HEADER[:4])


def test_elevation_deg_computes_positive_for_overhead_satellite():
    rx = np.array([0.0, 0.0, 6_371_000.0], dtype=np.float64)
    sat = np.array([[0.0, 0.0, 26_571_000.0]], dtype=np.float64)
    elev = _elevation_deg(rx, sat)
    assert elev[0] == pytest.approx(90.0, abs=1.0)


def test_nearest_position_picks_closest_time_within_tolerance():
    times = np.array([10.0, 20.0, 30.0], dtype=np.float64)
    positions = np.array(
        [
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    pos, delta = _nearest_position(times, positions, 20.05, 0.11)
    assert pos is not None
    np.testing.assert_allclose(pos, [2.0, 0.0, 0.0])
    assert abs(delta) == pytest.approx(0.05)


def test_write_batch_rows_emits_extended_csv_and_loader_roundtrip(tmp_path: Path):
    data = {
        "times": [123.456],
        "used_prns": [["G01", "G02"]],
        "sat_ecef": [
            np.array(
                [
                    [20_000_000.0, 0.0, 0.0],
                    [0.0, 20_000_000.0, 0.0],
                ],
                dtype=np.float64,
            )
        ],
    }
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(list(EXTENDED_NLOS_CSV_HEADER))
    rx = np.array([2_000_000.0, 0.0, 0.0], dtype=np.float64)
    los = np.array([[True, False]], dtype=bool)
    rows, nlos = _write_batch_rows(
        writer,
        data,
        0,
        np.asarray([rx], dtype=np.float64),
        ["reference"],
        [0.0],
        los,
    )
    assert rows == 2
    assert nlos == 1

    csv_text = buffer.getvalue()
    out_path = tmp_path / "roundtrip.csv"
    out_path.write_text(csv_text, encoding="utf-8")
    loaded = load_nlos_prn_sets(out_path)
    assert loaded == {0: {"G02"}}
