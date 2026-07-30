#!/usr/bin/env python3
"""Build a truth-free cause ledger for authoritative library non-FIX epochs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def _flag(row: dict[str, str], name: str) -> bool:
    return row.get(name) == "1"


def classify(row: dict[str, str]) -> str:
    """Return one mutually exclusive, pipeline-ordered non-FIX cause."""

    if not _flag(row, "primary_ffrt_passed"):
        return "primary_ffrt_unavailable"
    if not _flag(row, "failure_budget_passed"):
        if (
            not _flag(row, "disjoint_a_ffrt_passed")
            or not _flag(row, "disjoint_b_ffrt_passed")
        ):
            return "disjoint_partition_ffrt_unavailable"
        if not _flag(row, "disjoint_passed"):
            return "disjoint_solution_separation_rejected"
        return "independent_failure_budget_missing"
    if not _flag(row, "disjoint_consensus_declared_fixed"):
        return "causal_consensus_not_declared"
    if not _flag(row, "quality_gate_passed"):
        return "quality_gate_rejected"
    return "other_nonfix"


def analyze(rows: list[dict[str, str]], city: str) -> dict[str, object]:
    nonfixed = [row for row in rows if row.get("library_status") != "4"]
    exclusive = Counter(classify(row) for row in nonfixed)
    multilabel = {
        "primary_ffrt_passed": sum(
            _flag(row, "primary_ffrt_passed") for row in nonfixed
        ),
        "failure_budget_passed": sum(
            _flag(row, "failure_budget_passed") for row in nonfixed
        ),
        "disjoint_passed": sum(
            _flag(row, "disjoint_passed") for row in nonfixed
        ),
        "causal_consensus_declared": sum(
            _flag(row, "disjoint_consensus_declared_fixed")
            for row in nonfixed
        ),
        "quality_gate_passed": sum(
            _flag(row, "quality_gate_passed") for row in nonfixed
        ),
    }
    total = len(rows)
    fixed = total - len(nonfixed)
    return {
        "schema": "gnss_gpu_wp175_nonfix_cause_ledger_v1",
        "city": city,
        "truth_usage": "none",
        "fix_kpi_definition": "gnssplusplus final .pos Status == 4",
        "epochs": total,
        "library_fixed_epochs": fixed,
        "library_fix_rate": fixed / total if total else 0.0,
        "nonfixed_epochs": len(nonfixed),
        "exclusive_pipeline_causes": dict(sorted(exclusive.items())),
        "nonfixed_stage_pass_counts": multilabel,
        "maximum_recoverable_if_all_budgeted_nonfix_passed_quality": (
            fixed + multilabel["failure_budget_passed"]
        ),
        "maximum_recoverable_rate_if_all_budgeted_nonfix_passed_quality": (
            (fixed + multilabel["failure_budget_passed"]) / total
            if total
            else 0.0
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city", choices=("tokyo", "nagoya"), required=True)
    parser.add_argument("--integrity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.integrity.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    payload = analyze(rows, args.city)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
