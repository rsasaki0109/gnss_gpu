from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "experiments"))

from scan_wp46_moving_evidence_supply import (  # noqa: E402
    block_spans,
    summarize_block,
)


def test_block_spans_retains_incomplete_tail() -> None:
    assert block_spans(1051, 1180, 55) == [
        (1051, 1106),
        (1106, 1161),
        (1161, 1180),
    ]


def test_summary_chooses_evidence_then_carrier_phase() -> None:
    supply = {}
    for epoch in range(100, 155):
        supply[epoch] = {
            "evidence": int(epoch % 5 in {0, 2}),
            "carrier_rows": 4 if epoch % 5 == 2 else 2,
            "ddpr_epoch": int(epoch % 5 in {0, 2}),
            "ddpr_rows": 5,
        }
    result = summarize_block(
        start=100,
        end=155,
        block_epochs=55,
        stride=5,
        epoch_supply=supply,
        min_evidence_epochs=10,
        min_carrier_rows=24,
        min_ddpr_rows=40,
    )
    assert result["selected_stride_phase"] == 2
    assert result["pre_candidate_supply_pass"] is True


def test_incomplete_block_always_fails_supply() -> None:
    supply = {
        epoch: {
            "evidence": 1,
            "carrier_rows": 10,
            "ddpr_epoch": 1,
            "ddpr_rows": 10,
        }
        for epoch in range(100, 130)
    }
    result = summarize_block(
        start=100,
        end=130,
        block_epochs=55,
        stride=5,
        epoch_supply=supply,
        min_evidence_epochs=1,
        min_carrier_rows=1,
        min_ddpr_rows=1,
    )
    assert result["gates"]["complete_block"] is False
    assert result["pre_candidate_supply_pass"] is False
