#!/usr/bin/env python3
"""Truth-joined diagnostic of candidate-supply union; never a selector input."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def _miss_blocks(values: list[bool]) -> list[dict[str, int]]:
    blocks: list[dict[str, int]] = []
    start = None
    for index, available in enumerate(values + [True]):
        if not available and start is None:
            start = index
        elif available and start is not None:
            blocks.append({"start": start, "end": index - 1, "length": index - start})
            start = None
    return sorted(blocks, key=lambda block: -block["length"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--left-name", default="left")
    parser.add_argument("--right-name", default="right")
    parser.add_argument("--target-pct", type=float, required=True)
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--out-report", type=Path, required=True)
    args = parser.parse_args()
    left = _read(args.left)
    right = _read(args.right)
    if len(left) != len(right):
        raise RuntimeError("candidate diagnostics have different denominators")
    for index, (a, b) in enumerate(zip(left, right)):
        if int(a["epoch"]) != index or int(b["epoch"]) != index:
            raise RuntimeError(f"candidate diagnostics are not ordered at {index}")

    left_ok = [bool(int(row["basin_oracle_sub50cm_available"])) for row in left]
    right_ok = [bool(int(row["basin_oracle_sub50cm_available"])) for row in right]
    union = [a or b for a, b in zip(left_ok, right_ok)]
    n_epochs = len(union)
    required = math.ceil(float(args.target_pct) * n_epochs / 100.0)
    classes = {
        "both": sum(a and b for a, b in zip(left_ok, right_ok)),
        f"{args.left_name}_only": sum(a and not b for a, b in zip(left_ok, right_ok)),
        f"{args.right_name}_only": sum(b and not a for a, b in zip(left_ok, right_ok)),
        "neither": sum(not a and not b for a, b in zip(left_ok, right_ok)),
    }
    chunks: list[dict[str, Any]] = []
    for start in range(0, n_epochs, int(args.chunk_size)):
        end = min(start + int(args.chunk_size), n_epochs)
        chunks.append(
            {
                "start": start,
                "end": end - 1,
                "n_epochs": end - start,
                f"{args.left_name}_oracle": sum(left_ok[start:end]),
                f"{args.right_name}_oracle": sum(right_ok[start:end]),
                "union_oracle": sum(union[start:end]),
            }
        )
    union_count = sum(union)
    report = {
        "diagnostic_only_truth_joined": True,
        "n_epochs_full_denominator": n_epochs,
        "target_pct": float(args.target_pct),
        "target_required_epochs": required,
        f"{args.left_name}_oracle_epochs": sum(left_ok),
        f"{args.right_name}_oracle_epochs": sum(right_ok),
        "union_oracle_epochs": union_count,
        "union_oracle_pct": 100.0 * union_count / max(n_epochs, 1),
        "target_candidate_supply_gap_epochs": max(required - union_count, 0),
        "target_possible_from_union": union_count >= required,
        "classes": classes,
        "longest_union_miss_blocks": _miss_blocks(union)[:20],
        "chunks": chunks,
    }
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
