"""Post-process a GSDC2023 submission CSV with a reference deviation guard.

The FGO submission path can occasionally diverge by a full segment even when
the gated baseline is reliable to the 10-30 m range.  This guard bounds that
failure mode by comparing each candidate row against a row-aligned reference
submission and replacing only rows whose haversine deviation exceeds the
configured threshold.

Input and reference files must both be Kaggle GSDC submission CSVs with unique
``(tripId, UnixTimeMillis)`` keys and identical key sets.  The output preserves
the input row order and input CSV column order.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


KEY_COLUMNS = ["tripId", "UnixTimeMillis"]
LAT_COLUMN = "LatitudeDegrees"
LNG_COLUMN = "LongitudeDegrees"
REQUIRED_COLUMNS = set(KEY_COLUMNS + [LAT_COLUMN, LNG_COLUMN])


def _haversine_m(lat1: np.ndarray, lng1: np.ndarray, lat2: np.ndarray, lng2: np.ndarray) -> np.ndarray:
    radius_m = 6371000.0
    lat1r = np.radians(lat1)
    lat2r = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlng = np.radians(lng2 - lng1)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlng / 2) ** 2
    return 2 * radius_m * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _format_key_examples(keys: pd.DataFrame, *, limit: int = 5) -> str:
    examples = list(keys.head(limit).itertuples(index=False, name=None))
    suffix = "" if len(keys) <= limit else ", ..."
    return f"{examples}{suffix}"


def _validate_required_columns(df: pd.DataFrame, *, name: str) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{name} missing columns: {sorted(missing)}")


def _validate_unique_keys(df: pd.DataFrame, *, name: str) -> None:
    duplicated = df.duplicated(KEY_COLUMNS, keep=False)
    if not duplicated.any():
        return
    duplicate_keys = df.loc[duplicated, KEY_COLUMNS].drop_duplicates()
    raise ValueError(
        f"{name} has duplicate (tripId, UnixTimeMillis) keys: "
        f"{_format_key_examples(duplicate_keys)}"
    )


def _key_frame_from_index(index: pd.MultiIndex) -> pd.DataFrame:
    return index.to_frame(index=False)


def _validate_same_keys(candidate: pd.DataFrame, reference: pd.DataFrame) -> None:
    candidate_keys = pd.MultiIndex.from_frame(candidate[KEY_COLUMNS])
    reference_keys = pd.MultiIndex.from_frame(reference[KEY_COLUMNS])
    missing_reference = candidate_keys.difference(reference_keys)
    missing_candidate = reference_keys.difference(candidate_keys)
    errors: list[str] = []
    if len(missing_reference) > 0:
        errors.append(
            f"{len(missing_reference)} input row(s) have no reference row: "
            f"{_format_key_examples(_key_frame_from_index(missing_reference))}"
        )
    if len(missing_candidate) > 0:
        errors.append(
            f"{len(missing_candidate)} reference row(s) have no input row: "
            f"{_format_key_examples(_key_frame_from_index(missing_candidate))}"
        )
    if errors:
        raise ValueError("; ".join(errors))


def apply_deviation_guard_to_submission(
    df: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    max_deviation_m: float = 100.0,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Return a copy of ``df`` with over-threshold rows replaced by reference coordinates."""
    _validate_required_columns(df, name="input")
    _validate_required_columns(reference, name="reference")
    _validate_unique_keys(df, name="input")
    _validate_unique_keys(reference, name="reference")
    _validate_same_keys(df, reference)

    ref_aligned = df[KEY_COLUMNS].merge(
        reference[KEY_COLUMNS + [LAT_COLUMN, LNG_COLUMN]],
        on=KEY_COLUMNS,
        how="left",
        sort=False,
        validate="one_to_one",
    )
    deviation_m = _haversine_m(
        df[LAT_COLUMN].to_numpy(dtype=np.float64),
        df[LNG_COLUMN].to_numpy(dtype=np.float64),
        ref_aligned[LAT_COLUMN].to_numpy(dtype=np.float64),
        ref_aligned[LNG_COLUMN].to_numpy(dtype=np.float64),
    )
    guarded = deviation_m > max_deviation_m

    out = df.copy()
    out.loc[guarded, LAT_COLUMN] = ref_aligned.loc[guarded, LAT_COLUMN].to_numpy()
    out.loc[guarded, LNG_COLUMN] = ref_aligned.loc[guarded, LNG_COLUMN].to_numpy()

    guarded_rows = int(np.sum(guarded))
    touched = pd.DataFrame({
        "tripId": df["tripId"],
        "guarded": guarded,
        "deviation_m": deviation_m,
    })
    per_trip: list[dict[str, object]] = []
    for trip_id, group in touched.loc[touched["guarded"]].groupby("tripId", sort=False):
        per_trip.append({
            "tripId": str(trip_id),
            "guarded_rows": int(len(group)),
            "max_deviation_m": float(group["deviation_m"].max()),
        })

    stats: dict[str, object] = {
        "rows_total": int(len(out)),
        "guarded_rows": guarded_rows,
        "trips_touched": len(per_trip),
        "max_deviation_m": float(np.max(deviation_m)) if len(deviation_m) else 0.0,
        "per_trip": per_trip,
    }
    return out, stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Guard a GSDC2023 submission against row-level reference deviation.",
    )
    parser.add_argument("--input", type=Path, required=True, help="input submission CSV")
    parser.add_argument("--reference", type=Path, required=True, help="reference submission CSV")
    parser.add_argument("--output", type=Path, required=True, help="output guarded CSV")
    parser.add_argument("--max-deviation-m", type=float, default=100.0)
    args = parser.parse_args(argv)

    if not args.input.is_file():
        print(f"[error] input not found: {args.input}", file=sys.stderr)
        return 1
    if not args.reference.is_file():
        print(f"[error] reference not found: {args.reference}", file=sys.stderr)
        return 1

    try:
        df = pd.read_csv(args.input)
        reference = pd.read_csv(args.reference)
        out, stats = apply_deviation_guard_to_submission(
            df,
            reference,
            max_deviation_m=args.max_deviation_m,
        )
    except ValueError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2

    out.to_csv(args.output, index=False)
    for trip in stats["per_trip"]:  # type: ignore[index]
        print(
            f"trip={trip['tripId']} guarded_rows={trip['guarded_rows']} "
            f"max_deviation_m={trip['max_deviation_m']:.3f}",
            flush=True,
        )
    guarded_rows = int(stats["guarded_rows"])
    rows_total = int(stats["rows_total"])
    print(
        f"rows_total={rows_total} guarded_rows={guarded_rows} "
        f"({100 * guarded_rows / max(1, rows_total):.2f}%) "
        f"trips_touched={stats['trips_touched']} "
        f"max_deviation_m={float(stats['max_deviation_m']):.3f} "
        f"threshold_m={args.max_deviation_m}",
        flush=True,
    )
    print(f"wrote: {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
