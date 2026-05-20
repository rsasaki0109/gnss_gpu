from __future__ import annotations

import json
from pathlib import Path

from experiments.gsdc2023_ab_dd_signals import (
    DDSignals,
    extract_dd_signals,
    load_dd_signals_from_summary,
)


def _trip_metric(
    trip: str,
    *,
    n_epochs: int,
    anchor: int = 0,
    dd: int = 0,
    snapped: int = 0,
    pairs_mean: float = 0.0,
) -> dict:
    return {
        "trip": trip,
        "n_epochs": n_epochs,
        "dd_carrier_accepted_anchor_epochs": anchor,
        "dd_carrier_dd_epochs": dd,
        "dd_carrier_base_snapped_epochs": snapped,
        "dd_carrier_dd_pairs_mean": pairs_mean,
    }


def test_dd_signals_anchor_coverage_handles_zero_epochs():
    signals = DDSignals(trip_id="a", n_epochs=0, dd_anchor_epochs=0, dd_dd_epochs=0, dd_base_snapped_epochs=0, dd_pairs_mean=0.0)
    assert signals.anchor_coverage == 0.0


def test_dd_signals_anchor_coverage_basic():
    signals = DDSignals(
        trip_id="a",
        n_epochs=1000,
        dd_anchor_epochs=600,
        dd_dd_epochs=900,
        dd_base_snapped_epochs=500,
        dd_pairs_mean=5.5,
    )
    assert signals.anchor_coverage == 0.6


def test_extract_dd_signals_strips_train_test_prefix():
    out = extract_dd_signals(
        [
            _trip_metric("train/foo/pixel5", n_epochs=100, anchor=60, pairs_mean=4.2),
            _trip_metric("test/bar/pixel4", n_epochs=200, anchor=80),
        ]
    )
    assert [s.trip_id for s in out] == ["foo/pixel5", "bar/pixel4"]
    assert out[0].anchor_coverage == 0.6
    assert out[0].dd_pairs_mean == 4.2


def test_extract_dd_signals_defaults_missing_fields_to_zero():
    out = extract_dd_signals([{"trip": "x/y", "n_epochs": 100}])
    assert out == [DDSignals(trip_id="x/y", n_epochs=100, dd_anchor_epochs=0, dd_dd_epochs=0, dd_base_snapped_epochs=0, dd_pairs_mean=0.0)]


def test_extract_dd_signals_handles_none_values():
    out = extract_dd_signals(
        [
            {
                "trip": "x/y",
                "n_epochs": 100,
                "dd_carrier_accepted_anchor_epochs": None,
                "dd_carrier_dd_pairs_mean": None,
            }
        ]
    )
    assert out[0].dd_anchor_epochs == 0
    assert out[0].dd_pairs_mean == 0.0


def test_load_dd_signals_from_summary_reads_trip_metrics(tmp_path: Path):
    path = tmp_path / "summary.json"
    path.write_text(
        json.dumps(
            {
                "trip_metrics": [
                    _trip_metric("train/a/p", n_epochs=200, anchor=120, dd=180, pairs_mean=5.0),
                    _trip_metric("test/b/q", n_epochs=400, anchor=200, dd=300, snapped=180, pairs_mean=4.0),
                ]
            }
        )
    )
    rows = load_dd_signals_from_summary(path)
    assert [(r.trip_id, r.anchor_coverage) for r in rows] == [("a/p", 0.6), ("b/q", 0.5)]


def test_load_dd_signals_from_summary_returns_empty_when_missing(tmp_path: Path):
    path = tmp_path / "summary.json"
    path.write_text("{}", encoding="utf-8")
    assert load_dd_signals_from_summary(path) == []
