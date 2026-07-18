#!/usr/bin/env python3
"""Summarize WP28 proposal supply, survival, and operational neutrality."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def _true_spans(flags: list[bool]) -> list[int]:
    spans: list[int] = []
    start: int | None = None
    for index, value in enumerate(flags + [False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            spans.append(index - start)
            start = None
    return spans


def _summarize(path: Path) -> dict[str, object]:
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    live = [float(row["basin_oracle_min_error_m"]) < 0.5 for row in rows]
    triggered = [row for row in rows if row["respawn_triggered"] == "1"]
    proposal_correct = [
        row
        for row in triggered
        if np.isfinite(float(row["respawn_oracle_min_error_m"]))
        and float(row["respawn_oracle_min_error_m"]) < 0.5
    ]
    proposal_pruned = [
        row
        for row in proposal_correct
        if not np.isfinite(float(row["basin_oracle_min_error_m"]))
        or float(row["basin_oracle_min_error_m"]) >= 0.5
    ]
    proposal_missing = [row for row in triggered if row not in proposal_correct]
    last_reset_epoch: int | None = None
    reset_age_by_epoch: dict[int, int | None] = {}
    for row_index, row in enumerate(rows):
        epoch = int(row.get("epoch", row_index))
        if int(row.get("ambiguities_reset", "0")) > 0:
            last_reset_epoch = epoch
        reset_age_by_epoch[epoch] = (
            None if last_reset_epoch is None else epoch - last_reset_epoch
        )
    spans = _true_spans(live)
    return {
        "epochs": len(rows),
        "oracle_live_sub50cm_epochs": sum(live),
        "oracle_live_pct": 100.0 * sum(live) / max(len(rows), 1),
        "proposal_correct_anchor_epochs": len(proposal_correct),
        "proposal_correct_anchor_pct": (
            100.0 * len(proposal_correct) / max(len(triggered), 1)
        ),
        "proposal_correct_but_not_live_anchor_epochs": len(proposal_pruned),
        "proposal_missing_anchor_epochs": [
            int(row.get("epoch", -1)) for row in proposal_missing
        ],
        "proposal_missing_reset_age_epochs": [
            reset_age_by_epoch.get(int(row.get("epoch", -1)))
            for row in proposal_missing
        ],
        "respawn_anchor_epochs": len(triggered),
        "proposal_correct_ranks": [
            int(row["respawn_oracle_rank"]) for row in proposal_correct
        ],
        "longest_live_span_epochs": max(spans, default=0),
        "live_span_p90_epochs": (
            float(np.percentile(spans, 90)) if spans else 0.0
        ),
        "maximum_live_basins": max(int(row["n_basins"]) for row in rows),
        "maximum_respawn_candidates": max(
            int(row["n_respawn_candidates_born"]) for row in rows
        ),
        "maximum_position_seeds": max(
            int(row.get("n_respawn_position_seeds", "0")) for row in rows
        ),
        "maximum_history_seeds": max(
            int(row.get("n_respawn_history_seeds", "0")) for row in rows
        ),
        "maximum_assignment_candidates": max(
            int(row.get("n_respawn_assignment_candidates", "0")) for row in rows
        ),
        "declared_fix_epochs": sum(row["fix"] == "1" for row in rows),
        "false_fix_epochs": sum(
            row["fix"] == "1" and float(row["output_error_m"]) >= 0.5
            for row in rows
        ),
        "integrity_selected_sub50cm_epochs": sum(
            float(row.get("integrity_map_error_m", "nan")) < 0.5
            for row in rows
        ),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm",
        action="append",
        required=True,
        help="Named input as NAME=PATH; repeat for each arm",
    )
    parser.add_argument("--out-summary", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = {}
    for specification in args.arm:
        name, separator, value = specification.partition("=")
        if not separator or not name or not value:
            parser.error("--arm must use NAME=PATH")
        summary[name] = _summarize(Path(value))
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
