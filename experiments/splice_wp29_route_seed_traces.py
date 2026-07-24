#!/usr/bin/env python3
"""Splice truth-free route seed traces at a declared, evidence-derived epoch."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def splice_rows(
    prefix: list[dict[str, str]], suffix: list[dict[str, str]], switch_epoch: int
) -> list[dict[str, str]]:
    """Use prefix before switch and suffix from switch, rejecting coverage gaps."""

    left = {int(row["epoch"]): row for row in prefix if int(row["epoch"]) < switch_epoch}
    right = {int(row["epoch"]): row for row in suffix if int(row["epoch"]) >= switch_epoch}
    output = {**left, **right}
    epochs = sorted(output)
    if not epochs or epochs != list(range(epochs[0], epochs[-1] + 1)):
        raise RuntimeError("spliced route seed has an epoch coverage gap")
    if switch_epoch not in right or switch_epoch - 1 not in left:
        raise RuntimeError("both sides of the route switch must be present")
    return [output[epoch] for epoch in epochs]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prefix", type=Path)
    parser.add_argument("suffix", type=Path)
    parser.add_argument("--switch-epoch", type=int, required=True)
    parser.add_argument("--base-seeds", type=Path)
    parser.add_argument("--out-seeds", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    args = parser.parse_args()
    route = splice_rows(read_rows(args.prefix), read_rows(args.suffix), args.switch_epoch)
    output = ([] if args.base_seeds is None else read_rows(args.base_seeds)) + route
    summary = {
        "route_segment": [int(route[0]["epoch"]), int(route[-1]["epoch"]) + 1],
        "switch_epoch": int(args.switch_epoch),
        "route_seed_rows": len(route),
        "base_seed_rows": len(output) - len(route),
        "total_seed_rows": len(output),
    }
    args.out_seeds.parent.mkdir(parents=True, exist_ok=True)
    with args.out_seeds.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
