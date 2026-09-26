#!/usr/bin/env python3
"""Summarize WP27 leave-one-satellite-out anchor diagnostics."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path


def _summarize(path: Path) -> dict[str, object]:
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    by_epoch: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_epoch[int(row["epoch"])].append(row)
    selected = [
        max(epoch_rows, key=lambda row: float(row["guard_mean_pair_cost"]))
        for epoch_rows in by_epoch.values()
    ]
    recovery_frequency = Counter(
        row["excluded_satellite"]
        for row in rows
        if row["exclusion_recovers_sub50cm"] == "1"
    )
    selected_frequency = Counter(row["excluded_satellite"] for row in selected)
    full_correct = sum(
        epoch_rows[0]["full_selected_sub50cm"] == "1"
        for epoch_rows in by_epoch.values()
    )
    recoverable = sum(
        any(row["exclusion_recovers_sub50cm"] == "1" for row in epoch_rows)
        for epoch_rows in by_epoch.values()
    )
    return {
        "anchor_epochs": len(by_epoch),
        "full_score_sub50cm_epochs": full_correct,
        "wrong_anchor_epochs": len(by_epoch) - full_correct,
        "wrong_anchors_recoverable_by_one_exclusion": recoverable,
        "max_guard_cost_policy_sub50cm_epochs": sum(
            row["excluded_selected_sub50cm"] == "1" for row in selected
        ),
        "max_guard_cost_policy_recovered_epochs": sum(
            row["exclusion_recovers_sub50cm"] == "1" for row in selected
        ),
        "max_guard_cost_policy_broken_epochs": sum(
            row["exclusion_breaks_sub50cm"] == "1" for row in selected
        ),
        "top_recovery_satellites": recovery_frequency.most_common(10),
        "max_guard_cost_satellite_frequency": selected_frequency.most_common(10),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run1", type=Path, required=True)
    parser.add_argument("--run2", type=Path, required=True)
    parser.add_argument("--run3", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = {
        run: _summarize(getattr(args, run)) for run in ("run1", "run2", "run3")
    }
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
