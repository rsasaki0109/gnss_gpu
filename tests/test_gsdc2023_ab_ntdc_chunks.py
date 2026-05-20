from __future__ import annotations

import json
from pathlib import Path

from experiments.gsdc2023_ab_ntdc_chunks import (
    PromotedNtdcChunk,
    extract_promoted_ntdc_chunks,
    load_promoted_ntdc_chunks,
)


def _chunk_record(gated: str, *, start: int, end: int, ntdc_qs: float = 0.5, ntdc_mse: float = 8.0) -> dict:
    return {
        "gated_source": gated,
        "start_epoch": start,
        "end_epoch": end,
        "candidates": {
            "fgo_no_tdcp": {
                "quality_score": ntdc_qs,
                "mse_pr": ntdc_mse,
                "baseline_gap_max_m": 10.0,
                "step_p95_m": 20.0,
                "accel_p95_m": 1.5,
            }
        },
    }


def test_n_rows_uses_end_minus_start():
    chunk = PromotedNtdcChunk("a", 100, 300, 0.5, 8.0, 10.0, 20.0, 1.5)
    assert chunk.n_rows == 200


def test_n_rows_clamps_negative_to_zero():
    chunk = PromotedNtdcChunk("a", 300, 100, 0.5, 8.0, 10.0, 20.0, 1.5)
    assert chunk.n_rows == 0


def test_extract_keeps_only_gated_source_fgo_no_tdcp():
    trip_metrics = [
        {
            "trip": "train/foo/pixel5",
            "chunk_selection_records": [
                _chunk_record("fgo", start=0, end=100),
                _chunk_record("fgo_no_tdcp", start=200, end=300, ntdc_qs=0.7, ntdc_mse=9.5),
                _chunk_record("baseline", start=300, end=400),
                _chunk_record("fgo_no_tdcp", start=500, end=600, ntdc_qs=0.65, ntdc_mse=6.5),
            ],
        }
    ]
    out = extract_promoted_ntdc_chunks(trip_metrics)
    assert [(c.start_epoch, c.end_epoch, c.ntdc_quality_score) for c in out] == [
        (200, 300, 0.7),
        (500, 600, 0.65),
    ]


def test_extract_strips_train_test_prefix_in_trip_id():
    trip_metrics = [
        {
            "trip": "test/foo/pixel5",
            "chunk_selection_records": [
                _chunk_record("fgo_no_tdcp", start=0, end=100),
            ],
        }
    ]
    [chunk] = extract_promoted_ntdc_chunks(trip_metrics)
    assert chunk.trip_id == "foo/pixel5"


def test_extract_handles_missing_candidates_subdict():
    trip_metrics = [
        {
            "trip": "train/foo/pixel5",
            "chunk_selection_records": [
                {"gated_source": "fgo_no_tdcp", "start_epoch": 0, "end_epoch": 100},
            ],
        }
    ]
    [chunk] = extract_promoted_ntdc_chunks(trip_metrics)
    assert chunk.ntdc_quality_score == 0.0
    assert chunk.ntdc_mse_pr == 0.0


def test_extract_handles_none_metric_values():
    trip_metrics = [
        {
            "trip": "train/foo/pixel5",
            "chunk_selection_records": [
                {
                    "gated_source": "fgo_no_tdcp",
                    "start_epoch": 0,
                    "end_epoch": 100,
                    "candidates": {"fgo_no_tdcp": {"quality_score": None, "mse_pr": None}},
                }
            ],
        }
    ]
    [chunk] = extract_promoted_ntdc_chunks(trip_metrics)
    assert chunk.ntdc_quality_score == 0.0
    assert chunk.ntdc_mse_pr == 0.0


def test_extract_returns_empty_when_trip_has_no_records():
    trip_metrics = [{"trip": "train/foo/p", "chunk_selection_records": []}]
    assert extract_promoted_ntdc_chunks(trip_metrics) == []


def test_load_from_summary_reads_trip_metrics(tmp_path: Path):
    payload = {
        "trip_metrics": [
            {
                "trip": "train/foo/p",
                "chunk_selection_records": [
                    _chunk_record("fgo_no_tdcp", start=10, end=110, ntdc_qs=0.5, ntdc_mse=7.0)
                ],
            }
        ]
    }
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    [chunk] = load_promoted_ntdc_chunks(path)
    assert chunk.trip_id == "foo/p"
    assert chunk.ntdc_mse_pr == 7.0
