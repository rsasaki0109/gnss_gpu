from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from analyze_wp29_source_branch_shadow import branch_rows, source_tokens
from analyze_wp29_oracle_extinction import source_families


def test_source_tokens_and_fullmatch_do_not_confuse_position_indices() -> None:
    rows = [
        {"proposal_sources": "20:0|20:1"},
        {"proposal_sources": "20:10"},
        {"proposal_sources": "20:arc_assignment:1"},
    ]

    assert source_tokens(rows[0]["proposal_sources"]) == ("20:0", "20:1")
    assert branch_rows(rows, re.compile(r"\d+:1")) == rows[:1]


def test_source_families_normalizes_epoch_prefixed_tokens() -> None:
    assert source_families("900:snapshot:0|807:trusted_float_line:2") == (
        "snapshot",
        "trusted_float_line",
    )
