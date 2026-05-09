"""Apply audit-only target-trip row source patches to a GSDC2023 submission."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import pandas as pd

from experiments.analyze_gsdc2023_source_ab import KEY_COLUMNS, compare_submissions, comparison_summary


DEFAULT_LAT_COLUMN = "best_reference_source_latitude_degrees"
DEFAULT_LON_COLUMN = "best_reference_source_longitude_degrees"


@dataclass(frozen=True)
class PatchSpec:
    trip_id: str
    row_summary_path: Path
    start_epoch: int
    end_epoch: int


def parse_patch_spec(value: str) -> PatchSpec:
    if "=" not in value or "#" not in value:
        raise ValueError(f"expected TRIP=path#START-END for --patch, got {value!r}")
    trip_id, raw_rhs = value.split("=", 1)
    raw_path, raw_range = raw_rhs.rsplit("#", 1)
    if "-" not in raw_range:
        raise ValueError(f"expected START-END epoch range in {value!r}")
    raw_start, raw_end = raw_range.split("-", 1)
    start_epoch = int(raw_start)
    end_epoch = int(raw_end)
    if not trip_id:
        raise ValueError("patch trip must be non-empty")
    if start_epoch < 0 or end_epoch <= start_epoch:
        raise ValueError(f"invalid epoch range {start_epoch}-{end_epoch}")
    return PatchSpec(
        trip_id=trip_id,
        row_summary_path=Path(raw_path),
        start_epoch=start_epoch,
        end_epoch=end_epoch,
    )


def read_submission(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = set(KEY_COLUMNS + ["LatitudeDegrees", "LongitudeDegrees"])
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if frame[KEY_COLUMNS].duplicated().any():
        raise ValueError(f"{path} contains duplicate submission keys")
    return frame


def _patch_rows(
    frame: pd.DataFrame,
    spec: PatchSpec,
    *,
    lat_column: str,
    lon_column: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    rows = pd.read_csv(spec.row_summary_path)
    required = {"tripId", "UnixTimeMillis", "epoch_index", lat_column, lon_column}
    missing = required.difference(rows.columns)
    if missing:
        raise ValueError(f"{spec.row_summary_path} is missing columns: {sorted(missing)}")
    rows = rows[rows["tripId"] == spec.trip_id].copy()
    rows = rows[(rows["epoch_index"] >= spec.start_epoch) & (rows["epoch_index"] < spec.end_epoch)].copy()
    if rows.empty:
        raise ValueError(f"{spec} matched no patch rows")
    if rows["UnixTimeMillis"].duplicated().any():
        raise ValueError(f"{spec.row_summary_path} contains duplicate patch timestamps")
    if rows[[lat_column, lon_column]].isna().any(axis=None):
        raise ValueError(f"{spec.row_summary_path} contains NaN patch coordinates")

    out = frame.copy()
    target = out[out["tripId"] == spec.trip_id][["tripId", "UnixTimeMillis"]].copy()
    patch_keys = rows[["tripId", "UnixTimeMillis", "epoch_index", lat_column, lon_column]].copy()
    joined = target.merge(patch_keys, on=KEY_COLUMNS, how="inner", validate="one_to_one")
    if len(joined) != len(rows):
        raise ValueError(
            f"patch keys did not match submission for {spec.trip_id}: "
            f"patch_rows={len(rows)} matched={len(joined)}"
        )
    key_index = pd.MultiIndex.from_frame(out[KEY_COLUMNS])
    patch_index = pd.MultiIndex.from_frame(joined[KEY_COLUMNS])
    mask = key_index.isin(patch_index)
    out.loc[mask, "LatitudeDegrees"] = joined[lat_column].to_numpy()
    out.loc[mask, "LongitudeDegrees"] = joined[lon_column].to_numpy()
    return out, {
        "tripId": spec.trip_id,
        "row_summary_path": str(spec.row_summary_path),
        "start_epoch": int(spec.start_epoch),
        "end_epoch": int(spec.end_epoch),
        "rows_replaced": int(len(joined)),
        "first_time_ms": int(joined["UnixTimeMillis"].min()),
        "last_time_ms": int(joined["UnixTimeMillis"].max()),
        "lat_column": lat_column,
        "lon_column": lon_column,
    }


def apply_source_patches(
    submission: pd.DataFrame,
    patch_specs: list[PatchSpec],
    *,
    lat_column: str = DEFAULT_LAT_COLUMN,
    lon_column: str = DEFAULT_LON_COLUMN,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    out = submission.copy()
    summaries: list[dict[str, object]] = []
    patched_keys: set[tuple[str, int]] = set()
    for spec in patch_specs:
        rows = pd.read_csv(spec.row_summary_path, usecols=["tripId", "UnixTimeMillis", "epoch_index"])
        rows = rows[rows["tripId"] == spec.trip_id]
        rows = rows[(rows["epoch_index"] >= spec.start_epoch) & (rows["epoch_index"] < spec.end_epoch)]
        keys = {(str(row.tripId), int(row.UnixTimeMillis)) for row in rows.itertuples(index=False)}
        overlap = patched_keys.intersection(keys)
        if overlap:
            raise ValueError(f"patch specs overlap on {len(overlap)} row(s)")
        patched_keys.update(keys)
        out, summary = _patch_rows(out, spec, lat_column=lat_column, lon_column=lon_column)
        summaries.append(summary)
    return out, summaries


def write_outputs(
    output_dir: Path,
    patched: pd.DataFrame,
    payload: dict[str, object],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    patched_path = output_dir / "submission_with_target_source_patches.csv"
    patched.to_csv(patched_path, index=False)
    payload = {
        **payload,
        "patched_submission_csv": str(patched_path),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return patched_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-submission", type=Path, required=True)
    parser.add_argument("--reference-submission", type=Path, default=None)
    parser.add_argument("--patch", action="append", default=[], help="TRIP=target_trip_source_delta_rows.csv#START-END")
    parser.add_argument("--lat-column", default=DEFAULT_LAT_COLUMN)
    parser.add_argument("--lon-column", default=DEFAULT_LON_COLUMN)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    if not args.patch:
        raise ValueError("at least one --patch is required")
    specs = [parse_patch_spec(value) for value in args.patch]
    candidate = read_submission(args.candidate_submission)
    patched, patch_summaries = apply_source_patches(
        candidate,
        specs,
        lat_column=args.lat_column,
        lon_column=args.lon_column,
    )
    payload: dict[str, object] = {
        "candidate_submission": str(args.candidate_submission),
        "patches": patch_summaries,
        "rows_replaced": int(sum(summary["rows_replaced"] for summary in patch_summaries)),
    }
    if args.reference_submission is not None:
        reference = read_submission(args.reference_submission)
        row_delta, _ = compare_submissions(reference, patched, "patched")
        payload["patched_vs_reference"] = comparison_summary(row_delta).iloc[0].to_dict()
    patched_path = write_outputs(args.output_dir, patched, payload)
    print(f"wrote: {patched_path}")
    print(f"rows replaced: {payload['rows_replaced']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
