from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from experiments.promote_wp172_pf_seeded_rtk_consensus import (
    select_consensus_candidates,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _candidate(
    x: float, *, status: int = 4, residual: float = 1.0
) -> dict[str, float | int]:
    return {
        "ecef_x": x,
        "ecef_y": 0.0,
        "ecef_z": 0.0,
        "status": status,
        "prefit_residual_rms_m": residual,
    }


def test_consensus_gate_requires_fixed_status_residual_and_independent_agreement() -> None:
    seeded = {
        1.0: _candidate(0.0),
        2.0: _candidate(0.0, status=3),
        3.0: _candidate(0.0, residual=4.1),
        4.0: _candidate(0.0),
    }
    independent = {
        1.0: _candidate(0.5),
        2.0: _candidate(0.0),
        3.0: _candidate(0.0),
        4.0: _candidate(1.1),
    }
    selected = select_consensus_candidates(
        seeded,
        independent,
        required_status=4,
        max_disagreement_m=1.0,
        max_prefit_residual_rms_m=4.0,
    )
    assert list(selected) == [1.0]


def test_locked_tokyo_trajectory_matches_promotion_summary() -> None:
    summary = json.loads(
        (
            REPO_ROOT
            / "internal_docs/wp172_tokyo_final_holdout_2026_07_29.json"
        ).read_text(encoding="utf-8")
    )
    trajectory = REPO_ROOT / summary["output_trajectory"]
    digest = hashlib.sha256(trajectory.read_bytes()).hexdigest().upper()
    with trajectory.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    assert digest == summary["output_trajectory_sha256"]
    assert len(rows) == summary["full_denominator_epochs"] == 11_924
    assert sum(int(row["sub50cm"]) for row in rows) == 5_546
    assert sum(int(row["fix"]) for row in rows) == 0
    assert sum(int(row["false_fix"]) for row in rows) == 0
    assert summary["gained_epochs"] == 1_802
    assert summary["lost_epochs"] == 0
    assert summary["promotion_allowed"] is True
