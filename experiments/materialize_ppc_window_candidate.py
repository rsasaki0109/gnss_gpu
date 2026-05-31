#!/usr/bin/env python3
"""Materialize a TOW-window RTKDiag candidate as a phase-pool directory.

The CT-RBPF/FGO candidate loader expects ``{city}_{run}_full.pos`` and
``{city}_{run}_full.csv`` under ``experiments/results/libgnss_diag_phase10``.
Segment-local RTK resets should not be used outside the segment where their
filter state is valid, so this utility copies only rows inside a TOW window.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _pos_tow(line: str) -> float | None:
    if line.startswith("%") or not line.strip():
        return None
    parts = line.split()
    if len(parts) < 2:
        return None
    try:
        return round(float(parts[1]), 1)
    except ValueError:
        return None


def _copy_pos_window(src: Path, dst: Path, start_tow: float, end_tow: float) -> int:
    kept = 0
    with src.open() as inf, dst.open("w", newline="") as outf:
        for line in inf:
            tow = _pos_tow(line)
            if tow is None:
                outf.write(line)
                continue
            if float(start_tow) <= float(tow) <= float(end_tow):
                outf.write(line)
                kept += 1
    return kept


def _copy_csv_window(src: Path, dst: Path, start_tow: float, end_tow: float) -> int:
    kept = 0
    with src.open(newline="") as inf, dst.open("w", newline="") as outf:
        reader = csv.DictReader(inf)
        if reader.fieldnames is None:
            raise ValueError(f"missing CSV header: {src}")
        writer = csv.DictWriter(outf, fieldnames=reader.fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in reader:
            try:
                tow = round(float(row["tow"]), 1)
            except (KeyError, TypeError, ValueError):
                continue
            if float(start_tow) <= float(tow) <= float(end_tow):
                writer.writerow(row)
                kept += 1
    return kept


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--source-pos", type=Path, required=True)
    parser.add_argument("--source-csv", type=Path, required=True)
    parser.add_argument("--start-tow", type=float, required=True)
    parser.add_argument("--end-tow", type=float, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{args.city}_{args.run}_full"
    out_pos = args.out_dir / f"{prefix}.pos"
    out_csv = args.out_dir / f"{prefix}.csv"
    n_pos = _copy_pos_window(args.source_pos, out_pos, float(args.start_tow), float(args.end_tow))
    n_csv = _copy_csv_window(args.source_csv, out_csv, float(args.start_tow), float(args.end_tow))
    print(f"saved pos: {out_pos} rows={n_pos}")
    print(f"saved csv: {out_csv} rows={n_csv}")


if __name__ == "__main__":
    main()
