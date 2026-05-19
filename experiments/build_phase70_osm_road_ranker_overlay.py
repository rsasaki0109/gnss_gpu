#!/usr/bin/env python3
"""Append rule-driven ranker rows for the Phase70 OSM road candidate."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-predictions", type=Path, required=True)
    parser.add_argument("--trigger-epochs", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--run-id", default="nagoya_run2")
    parser.add_argument("--label", default="xd_gici_osmroad_hs")
    parser.add_argument("--p-pass", type=float, default=999.0)
    parser.add_argument("--penalize-source-label", default="")
    parser.add_argument("--source-p-pass", type=float, default=-999.0)
    args = parser.parse_args()

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.base_predictions.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if not {"run_id", "tow", "label", "p_pass"}.issubset(fieldnames):
        raise SystemExit(f"unexpected prediction columns: {fieldnames}")

    trigger_rows: list[dict[str, str]] = []
    with args.trigger_epochs.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if str(row.get("triggered", "")).lower() != "true":
                continue
            tow = round(float(row["tow"]), 1)
            trigger_rows.append(
                {
                    "run_id": str(args.run_id),
                    "tow": f"{tow:.1f}",
                    "label": str(args.label),
                    "p_pass": f"{float(args.p_pass):.6f}",
                }
            )

    existing = {(row["run_id"], round(float(row["tow"]), 1), row["label"]) for row in rows}
    added = 0
    updated = 0
    if args.penalize_source_label:
        trigger_tows = {round(float(row["tow"]), 1) for row in trigger_rows}
        for row in rows:
            try:
                row_key = (
                    str(row["run_id"]),
                    round(float(row["tow"]), 1),
                    str(row["label"]),
                )
            except (KeyError, ValueError):
                continue
            if row_key[0] == str(args.run_id) and row_key[1] in trigger_tows and row_key[2] == str(args.penalize_source_label):
                row["p_pass"] = f"{float(args.source_p_pass):.6f}"
                updated += 1
    for row in trigger_rows:
        key = (row["run_id"], round(float(row["tow"]), 1), row["label"])
        if key in existing:
            continue
        rows.append(row)
        existing.add(key)
        added += 1

    with args.out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved: {args.out_csv}")
    print(f"base_rows={len(rows) - added} added_rows={added} updated_rows={updated}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
