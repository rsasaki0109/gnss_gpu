from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from experiments.promote_wp173_lambda_fix_declarations import (
    declare_lambda_fix_epochs,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _candidate(*, ratio: float = 3.0, satellites: int = 12) -> dict[str, float | int]:
    return {"ratio": ratio, "num_satellites": satellites}


def test_declaration_requires_complete_causal_streak() -> None:
    candidates = {
        round(1.0 + 0.2 * index, 3): _candidate()
        for index in range(7)
    }
    declared = declare_lambda_fix_epochs(
        candidates,
        minimum_ratio=3.0,
        minimum_satellites=6,
        minimum_contiguous_epochs=5,
        maximum_epoch_gap_s=0.21,
    )
    assert declared == {1.8, 2.0, 2.2}


def test_ratio_satellite_and_gap_fail_closed_and_reset_streak() -> None:
    candidates = {
        1.0: _candidate(),
        1.2: _candidate(),
        1.4: _candidate(ratio=2.9),
        1.6: _candidate(),
        1.8: _candidate(satellites=5),
        2.4: _candidate(),
        2.6: _candidate(),
        2.8: _candidate(),
        3.0: _candidate(),
        3.2: _candidate(),
    }
    declared = declare_lambda_fix_epochs(
        candidates,
        minimum_ratio=3.0,
        minimum_satellites=6,
        minimum_contiguous_epochs=5,
        maximum_epoch_gap_s=0.21,
    )
    assert declared == {3.2}


def test_missing_candidate_stream_declares_no_fix() -> None:
    assert (
        declare_lambda_fix_epochs(
            {},
            minimum_ratio=3.0,
            minimum_satellites=6,
            minimum_contiguous_epochs=5,
            maximum_epoch_gap_s=0.21,
        )
        == set()
    )


def test_locked_tokyo_lambda_fix_trajectory_is_safe_and_complete() -> None:
    summary = json.loads(
        (
            REPO_ROOT
            / "internal_docs/wp173_tokyo_operational_audit_2026_07_29.json"
        ).read_text(encoding="utf-8")
    )
    trajectory = REPO_ROOT / summary["output_trajectory"]
    canonical = trajectory.read_bytes().replace(b"\r\n", b"\n")
    digest = hashlib.sha256(canonical).hexdigest().upper()
    with trajectory.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    assert digest == summary["output_trajectory_canonical_sha256"]
    assert len(rows) == summary["full_denominator_epochs"] == 11_924
    assert sum(int(row["sub50cm"]) for row in rows) == 5_546
    assert sum(int(row["fix"]) for row in rows) == summary["fix_epochs"] == 1_296
    assert sum(int(row["false_fix"]) for row in rows) == 0
    assert summary["fix_percent"] > 10.0
    assert summary["promotion_allowed"] is True
