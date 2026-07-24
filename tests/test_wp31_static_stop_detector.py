from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from detect_wp31_static_stops import detect_static_stops


def _row(epoch: int, *, speed: float = 0.01, used: int = 10, rms: float = 0.01):
    return {
        "tow": str(1000.0 + 0.2 * epoch),
        "norm_m": str(speed * 0.2),
        "postfit_rms_m": str(rms),
        "n_used": str(used),
    }


def test_detects_stop_and_bridges_short_recorded_time_gap() -> None:
    rows = [_row(i) for i in range(80)]
    for i in range(30, 35):
        rows[i] = _row(i, used=0)
    stops = detect_static_stops(rows, min_stop_epochs=40)
    assert [(row["start"], row["end"]) for row in stops] == [(1, 80)]


def test_does_not_bridge_large_tow_outage() -> None:
    rows = [_row(i) for i in range(100)]
    for i in range(45, 50):
        rows[i] = _row(i, used=0)
    for i in range(50, 100):
        rows[i]["tow"] = str(float(rows[i]["tow"]) + 10.0)
    stops = detect_static_stops(rows, min_stop_epochs=40)
    assert [(row["start"], row["end"]) for row in stops] == [(1, 45), (50, 100)]


def test_rejects_low_purity_bridged_run() -> None:
    rows = [_row(i) for i in range(60)]
    for i in range(10, 25):
        rows[i] = _row(i, used=0)
    for i in range(35, 50):
        rows[i] = _row(i, used=0)
    assert detect_static_stops(rows, min_stop_epochs=40) == []
