from __future__ import annotations

import json
import sys
from pathlib import Path

_EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
if str(_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS))

from analyze_wp29_widelane_assignment_shadow import _residuals  # noqa: E402


def test_residuals_match_l1_minus_l2_minus_widelane() -> None:
    assignment = json.dumps(
        [
            ["G01@L1_E1_B1", "G02@L1_E1_B1", 190293673, 0, 12],
            ["G01@L2_E5B_B2", "G02@L2_E5B_B2", 244210213, 0, 7],
            ["E01@L1_E1_B1", "E02@L1_E1_B1", 190293673, 0, 99],
        ]
    )

    assert _residuals(assignment, (("G01", "G02", 4),)) == (1,)


def test_residuals_skip_unpaired_assignments() -> None:
    assignment = json.dumps(
        [["G01@L1_E1_B1", "G02@L1_E1_B1", 190293673, 0, 12]]
    )

    assert _residuals(assignment, (("G01", "G02", 4),)) == ()
