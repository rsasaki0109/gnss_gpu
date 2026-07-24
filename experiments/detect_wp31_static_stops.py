#!/usr/bin/env python3
"""Detect truth-free static stops from reliable low-speed TDCP intervals."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def detect_static_stops(
    rows: list[dict[str, str]],
    *,
    min_tdcp_sats: int = 8,
    max_postfit_rms_m: float = 0.05,
    max_speed_mps: float = 0.05,
    bridge_gap_epochs: int = 15,
    bridge_gap_seconds: float = 3.0,
    min_stop_epochs: int = 40,
    min_reliable_fraction: float = 0.65,
) -> list[dict[str, Any]]:
    if not rows:
        return []
    tow = np.asarray([float(row["tow"]) for row in rows], dtype=np.float64)
    raw = np.zeros(len(rows), dtype=bool)
    speed = np.full(len(rows), np.nan, dtype=np.float64)
    for epoch in range(1, len(rows)):
        dt = float(tow[epoch] - tow[epoch - 1])
        row = rows[epoch]
        try:
            norm = float(row["norm_m"])
            rms = float(row["postfit_rms_m"])
            used = int(row["n_used"])
        except (KeyError, TypeError, ValueError):
            continue
        speed[epoch] = norm / max(dt, 1e-9)
        raw[epoch] = (
            dt > 0.0
            and used >= int(min_tdcp_sats)
            and np.isfinite(rms)
            and rms <= float(max_postfit_rms_m)
            and speed[epoch] < float(max_speed_mps)
        )
    bridged = raw.copy()
    cursor = 0
    while cursor < len(bridged):
        if bridged[cursor]:
            cursor += 1
            continue
        end = cursor
        while end < len(bridged) and not bridged[end]:
            end += 1
        if (
            cursor > 0
            and end < len(bridged)
            and end - cursor <= int(bridge_gap_epochs)
            and float(tow[end] - tow[cursor - 1]) <= float(bridge_gap_seconds)
        ):
            bridged[cursor:end] = True
        cursor = end
    stops: list[dict[str, Any]] = []
    cursor = 0
    while cursor < len(bridged):
        if not bridged[cursor]:
            cursor += 1
            continue
        end = cursor
        while end < len(bridged) and bridged[end]:
            end += 1
        count = end - cursor
        fraction = float(np.mean(raw[cursor:end]))
        values = speed[cursor:end]
        values = values[np.isfinite(values)]
        if count >= int(min_stop_epochs) and fraction >= float(min_reliable_fraction):
            stops.append(
                {
                    "start": int(cursor),
                    "end": int(end),
                    "epochs": int(count),
                    "duration_s": float(tow[end - 1] - tow[cursor]),
                    "reliable_stationary_fraction": fraction,
                    "median_tdcp_speed_mps": float(np.median(values)),
                    "p95_tdcp_speed_mps": float(np.percentile(values, 95.0)),
                }
            )
        cursor = end
    return stops


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("displacements", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-tdcp-sats", type=int, default=8)
    parser.add_argument("--max-postfit-rms-m", type=float, default=0.05)
    parser.add_argument("--max-speed-mps", type=float, default=0.05)
    parser.add_argument("--bridge-gap-epochs", type=int, default=15)
    parser.add_argument("--bridge-gap-seconds", type=float, default=3.0)
    parser.add_argument("--min-stop-epochs", type=int, default=40)
    parser.add_argument("--min-reliable-fraction", type=float, default=0.65)
    args = parser.parse_args()
    with args.displacements.open(newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    config = {
        "min_tdcp_sats": args.min_tdcp_sats,
        "max_postfit_rms_m": args.max_postfit_rms_m,
        "max_speed_mps": args.max_speed_mps,
        "bridge_gap_epochs": args.bridge_gap_epochs,
        "bridge_gap_seconds": args.bridge_gap_seconds,
        "min_stop_epochs": args.min_stop_epochs,
        "min_reliable_fraction": args.min_reliable_fraction,
    }
    result = {
        "source": str(args.displacements),
        "n_epochs": len(rows),
        "truth_free": True,
        "config": config,
        "stops": detect_static_stops(rows, **config),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
