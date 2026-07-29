from __future__ import annotations

import csv
from pathlib import Path

from experiments.audit_tokyo_candidate_supply import (
    audit_external_research_candidate,
    audit_native_candidates,
)


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_native_audit_computes_union_without_treating_it_as_selection(
    tmp_path: Path,
) -> None:
    production = tmp_path / "production.csv"
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    base_rows = [
        {"epoch": index, "error_m": error}
        for index, error in enumerate([0.1, 1.0, 1.0, 1.0])
    ]
    _write(production, base_rows)
    _write(
        first,
        [
            {"epoch": index, "error_m": error}
            for index, error in enumerate([1.0, 0.1, 1.0, 1.0])
        ],
    )
    _write(
        second,
        [
            {"epoch": index, "error_m": error}
            for index, error in enumerate([1.0, 1.0, 0.1, 1.0])
        ],
    )
    result = audit_native_candidates(
        production, [first, second], target_percent=75.0
    )
    assert result["production_sub50cm_epochs"] == 1
    assert result["oracle_union_sub50cm_epochs"] == 3
    assert result["novel_epochs_needed_beyond_archive"] == 0
    assert all(item["replacement_loss"] == 1 for item in result["candidates"])


def test_external_fgo_candidate_is_diagnostic_only(tmp_path: Path) -> None:
    production = tmp_path / "production.csv"
    candidate = tmp_path / "candidate.csv"
    reference = tmp_path / "reference.csv"
    _write(
        production,
        [
            {"tow": 1.0, "error_m": 1.0},
            {"tow": 2.0, "error_m": 0.1},
        ],
    )
    _write(
        candidate,
        [
            {
                "tow": 1.0,
                "ecef_x": 0.1,
                "ecef_y": 0.0,
                "ecef_z": 0.0,
                "fix": 1,
            },
            {
                "tow": 2.0,
                "ecef_x": 1.0,
                "ecef_y": 0.0,
                "ecef_z": 0.0,
                "fix": 1,
            },
        ],
    )
    _write(
        reference,
        [
            {
                "GPS TOW (s)": 1.0,
                "ECEF X (m)": 0.0,
                "ECEF Y (m)": 0.0,
                "ECEF Z (m)": 0.0,
            },
            {
                "GPS TOW (s)": 2.0,
                "ECEF X (m)": 0.0,
                "ECEF Y (m)": 0.0,
                "ECEF Z (m)": 0.0,
            },
        ],
    )
    result = audit_external_research_candidate(production, candidate, reference)
    assert result["production_eligible"] is False
    assert result["oracle_gain_over_production"] == 1
    assert result["replacement_loss"] == 1
    assert result["false_fix_epochs"] == 1
