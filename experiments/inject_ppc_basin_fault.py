#!/usr/bin/env python3
"""Inject deterministic truth-free faults into a native basin JSONL stream."""

from __future__ import annotations

import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def inject_fault(
    rows: list[dict[str, Any]],
    *,
    fault: str,
    first_epoch: int,
    last_epoch: int,
    position_bias_m: float = 5.0,
) -> list[dict[str, Any]]:
    if fault not in {"outage", "ambiguous_holdout", "cycle_slip", "nlos"}:
        raise ValueError("unsupported fault")
    if first_epoch < 0 or last_epoch < first_epoch:
        raise ValueError("invalid fault interval")
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["epoch_index"])].append(copy.deepcopy(row))
    output: list[dict[str, Any]] = []
    for epoch, epoch_rows in sorted(grouped.items()):
        if not first_epoch <= epoch <= last_epoch:
            output.extend(epoch_rows)
            continue
        if fault == "outage":
            template = epoch_rows[0]
            outage = {
                "schema": template["schema"],
                "epoch_index": epoch,
                "tow": template["tow"],
                "group_index": -1,
                "rank": -1,
                "evaluated": False,
                "pass": False,
                "reason": "injected_outage",
            }
            if "gps_week" in template:
                outage["gps_week"] = template["gps_week"]
            if "imu_fgo" in template:
                outage["imu_fgo"] = copy.deepcopy(template["imu_fgo"])
            output.append(outage)
            continue
        evaluated = [row for row in epoch_rows if row.get("evaluated") is True]
        if fault == "ambiguous_holdout":
            for row in evaluated[:2]:
                row["pass"] = True
        elif fault == "cycle_slip":
            for row in evaluated:
                row["pass"] = False
                for integer in row.get("fixed_integers", []):
                    integer["fixed_cycles"] = int(integer["fixed_cycles"]) + 1
        else:
            for row in evaluated:
                row["pass"] = False
                position = list(row.get("position_ecef", []))
                if len(position) == 3:
                    position[0] = float(position[0]) + float(position_bias_m)
                    row["position_ecef"] = position
                for residual in row.get("validation_residuals", []):
                    residual["pass"] = False
        output.extend(epoch_rows)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--fault",
        choices=("outage", "ambiguous_holdout", "cycle_slip", "nlos"),
        required=True,
    )
    parser.add_argument("--first-epoch", type=int, required=True)
    parser.add_argument("--last-epoch", type=int, required=True)
    parser.add_argument("--position-bias-m", type=float, default=5.0)
    args = parser.parse_args(argv)
    rows = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    injected = inject_fault(
        rows,
        fault=args.fault,
        first_epoch=args.first_epoch,
        last_epoch=args.last_epoch,
        position_bias_m=args.position_bias_m,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in injected),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
