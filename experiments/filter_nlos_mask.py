"""Filter PLATEAU per-epoch NLOS masks into sparser RTK-safe variants."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _safe_float(value: object, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _keep_nlos(row: dict[str, str], args: argparse.Namespace) -> bool:
    elev = _safe_float(row.get("elevation_deg"))
    if args.min_elevation_deg is not None and elev < float(args.min_elevation_deg):
        return False
    if args.max_elevation_deg is not None and elev > float(args.max_elevation_deg):
        return False
    return True


def filter_mask(args: argparse.Namespace) -> dict[str, float]:
    rows_by_epoch: dict[int, list[dict[str, str]]] = {}
    fieldnames: list[str] = []
    with args.input_csv.open(newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        if "is_los" not in fieldnames:
            raise ValueError(f"missing is_los column: {args.input_csv}")
        for row in reader:
            epoch_idx = _safe_int(row.get("epoch_idx"), -1)
            rows_by_epoch.setdefault(epoch_idx, []).append(dict(row))

    n_rows = 0
    n_nlos_in = 0
    n_nlos_out = 0
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for _epoch_idx, rows in sorted(rows_by_epoch.items()):
            nlos_candidates: list[dict[str, str]] = []
            for row in rows:
                n_rows += 1
                is_nlos = str(row.get("is_los", "")).strip() == "0"
                n_nlos_in += int(is_nlos)
                if is_nlos and _keep_nlos(row, args):
                    nlos_candidates.append(row)

            if args.max_nlos_per_epoch > 0 and len(nlos_candidates) > args.max_nlos_per_epoch:
                # Keep the lowest-elevation NLOS satellites first. They are the
                # most plausible PLATEAU blockers and code variance already
                # downweights them less destructively than masking a whole epoch.
                keep_ids = {
                    id(row)
                    for row in sorted(
                        nlos_candidates,
                        key=lambda r: _safe_float(r.get("elevation_deg"), 90.0),
                    )[: int(args.max_nlos_per_epoch)]
                }
            else:
                keep_ids = {id(row) for row in nlos_candidates}

            for row in rows:
                if str(row.get("is_los", "")).strip() == "0" and id(row) not in keep_ids:
                    row = dict(row)
                    row["is_los"] = "1"
                n_nlos_out += int(str(row.get("is_los", "")).strip() == "0")
                writer.writerow(row)

    frac_in = float(n_nlos_in / n_rows) if n_rows else 0.0
    frac_out = float(n_nlos_out / n_rows) if n_rows else 0.0
    print(
        f"[filter] {args.input_csv} -> {args.output_csv} "
        f"rows={n_rows} nlos={n_nlos_in}->{n_nlos_out} "
        f"frac={frac_in:.4f}->{frac_out:.4f}"
    )
    return {
        "rows": float(n_rows),
        "nlos_in": float(n_nlos_in),
        "nlos_out": float(n_nlos_out),
        "frac_in": frac_in,
        "frac_out": frac_out,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--min-elevation-deg", type=float, default=None)
    parser.add_argument("--max-elevation-deg", type=float, default=None)
    parser.add_argument("--max-nlos-per-epoch", type=int, default=0)
    args = parser.parse_args()
    filter_mask(args)


if __name__ == "__main__":
    main()
