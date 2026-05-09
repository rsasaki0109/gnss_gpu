"""Replace bldg-only ant_* features in the augmented window CSV with bldg+brid
(BRID egm96) values from the per-run per_window CSVs.

Input:
  --window-csv  augmented window CSV with existing ant_* (bldg-only) schema.
  --brid-dir    directory containing
                ppc_antenna_features_BRID_egm96_<city>_<run>_per_window.csv
                files for the 6 PPC routes.

Output: --output-csv with the same columns/order, ant_* values replaced by
the BRID equivalents on rows that match (city, run, window_index).
Unmatched rows keep the original bldg-only values; a count is printed.
"""

import argparse
import csv
from pathlib import Path

ANT_COLS = [
    "eff_db_p10_mean", "eff_db_p10_min", "eff_db_p50_mean", "eff_db_p90_mean",
    "eff_db_max_mean", "eff_db_max_max", "eff_db_mean_mean",
    "usable_count_mean", "usable_count_min", "marginal_count_mean",
    "nlos_at_high_elev_count_mean", "nlos_at_high_elev_count_max",
    "gain_db_mean_mean", "elev_deg_p50_mean",
]


def load_brid(brid_dir: Path):
    """Return dict[(city, run, window_index)] -> {ant_<col>: value}."""
    out = {}
    cities_runs = [
        ("nagoya", "run1"), ("nagoya", "run2"), ("nagoya", "run3"),
        ("tokyo", "run1"), ("tokyo", "run2"), ("tokyo", "run3"),
    ]
    for city, run in cities_runs:
        path = brid_dir / f"ppc_antenna_features_BRID_egm96_{city}_{run}_per_window.csv"
        with open(path) as f:
            for row in csv.DictReader(f):
                key = (row["city"], row["run"], int(row["window_index"]))
                out[key] = {f"ant_{c}": row[c] for c in ANT_COLS}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window-csv", required=True, type=Path)
    ap.add_argument("--brid-dir", required=True, type=Path)
    ap.add_argument("--output-csv", required=True, type=Path)
    args = ap.parse_args()

    brid = load_brid(args.brid_dir)
    print(f"Loaded {len(brid)} BRID per_window rows from {args.brid_dir}")

    with open(args.window_csv) as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    missing_cols = [c for c in (f"ant_{c}" for c in ANT_COLS) if c not in fieldnames]
    if missing_cols:
        raise SystemExit(f"window CSV missing columns: {missing_cols}")

    matched = 0
    unmatched = []
    for row in rows:
        key = (row["city"], row["run"], int(row["window_index"]))
        if key in brid:
            matched += 1
            for col, val in brid[key].items():
                row[col] = val
        else:
            unmatched.append(key)

    print(f"matched {matched}/{len(rows)} window rows")
    if unmatched:
        print(f"unmatched ({len(unmatched)}): {unmatched[:10]}{'...' if len(unmatched) > 10 else ''}")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, "w") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"wrote {args.output_csv}")


if __name__ == "__main__":
    main()
