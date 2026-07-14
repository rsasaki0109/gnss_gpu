from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "experiments/ppc_distance_score.py"
SPEC = importlib.util.spec_from_file_location("ppc_distance_score", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _reference(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["tow", "week", "lat", "lon", "h", "x", "y", "z"])
        writer.writerows(
            [
                [0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 3, 0, 0],
                [2, 0, 0, 0, 0, 3, 4, 0],
                [3, 0, 0, 0, 0, 3, 4, 12],
            ]
        )


def test_honest_distance_score_counts_missing_epoch_as_failure(tmp_path: Path) -> None:
    reference = tmp_path / "reference.csv"
    _reference(reference)
    result = MODULE.honest_ppc_distance_score(
        {0: 0.1, 1: 0.4, 3: 0.2}, reference
    )
    assert result["pass_distance_m"] == pytest.approx(15.0)
    assert result["total_distance_m"] == pytest.approx(19.0)
    assert result["honest_ppc_score_pct"] == pytest.approx(100.0 * 15.0 / 19.0)


def test_blocked_slice_does_not_count_entry_from_before_span(tmp_path: Path) -> None:
    reference = tmp_path / "reference.csv"
    _reference(reference)
    result = MODULE.honest_ppc_distance_score(
        {1: 0.1, 2: 0.1},
        reference,
        start_epoch=1,
        end_epoch=3,
    )
    assert result["pass_distance_m"] == pytest.approx(4.0)
    assert result["total_distance_m"] == pytest.approx(4.0)
