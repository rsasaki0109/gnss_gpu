#!/usr/bin/env python3
"""Classify MATLAB-only residual rows that are recovered without the bridge observation mask."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path
import sys

import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from experiments.audit_gsdc2023_residual_value_parity import DEFAULT_RESIDUAL_PARITY_TRIPS  # noqa: E402
from experiments.compare_gsdc2023_residual_values import compare_residual_values  # noqa: E402
from experiments.gsdc2023_audit_cli import (  # noqa: E402
    add_data_root_arg as _add_data_root_arg,
    add_max_epochs_arg as _add_max_epochs_arg,
    add_multi_gnss_arg as _add_multi_gnss_arg,
    add_output_dir_arg as _add_output_dir_arg,
    nonnegative_max_epochs as _nonnegative_max_epochs,
    resolved_output_root as _resolved_output_root,
)
from experiments.gsdc2023_audit_output import (  # noqa: E402
    print_summary_and_output_dir as _print_summary_and_output_dir,
    timestamped_output_dir as _timestamped_output_dir,
    write_summary_json as _write_summary_json,
)
from experiments.gsdc2023_observation_matrix import OBS_MASK_MIN_ELEVATION_DEG  # noqa: E402
from experiments.gsdc2023_raw_bridge import DEFAULT_ROOT  # noqa: E402
from experiments.gsdc2023_residual_audit import RESIDUAL_KEY_COLUMNS  # noqa: E402


CompareFn = Callable[..., tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]]


def _key_set(frame: pd.DataFrame) -> set[tuple[object, ...]]:
    if frame.empty or not set(RESIDUAL_KEY_COLUMNS).issubset(frame.columns):
        return set()
    return {tuple(row) for row in frame[RESIDUAL_KEY_COLUMNS].itertuples(index=False, name=None)}


def _group_reasons(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=group_cols + ["count", "first_epoch", "last_epoch", "unique_svid_count"])
    rows: list[dict[str, object]] = []
    for keys, group in frame.groupby(group_cols, sort=True, dropna=False, observed=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: value for col, value in zip(group_cols, keys)}
        row.update(
            {
                "count": int(len(group)),
                "first_epoch": int(group["epoch_index"].min()),
                "last_epoch": int(group["epoch_index"].max()),
                "unique_svid_count": int(group["svid"].nunique()),
            },
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["count"] + group_cols, ascending=[False] + [True] * len(group_cols))


def _mask_reason(row: pd.Series) -> str:
    if row.get("drop_reason") != "recovered_without_observation_mask":
        return "not_recovered"
    elevation = pd.to_numeric(pd.Series([row.get("matlab_sat_elevation")]), errors="coerce").iloc[0]
    if pd.notna(elevation) and float(elevation) < float(OBS_MASK_MIN_ELEVATION_DEG):
        return "elevation_below_bridge_threshold"
    return "recovered_unknown_mask_predicate"


def residual_mask_drop_audit(
    data_root: Path,
    trips: Sequence[str],
    *,
    max_epochs: int,
    multi_gnss: bool,
    compare_fn: CompareFn = compare_residual_values,
    verbose: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    rows: list[pd.DataFrame] = []
    errors: list[dict[str, object]] = []
    for trip_idx, trip in enumerate(trips, start=1):
        if verbose:
            print(f"[{trip_idx}/{len(trips)}] {trip}", file=sys.stderr, flush=True)
        trip_dir = Path(data_root) / trip
        try:
            masked, _masked_summary, masked_payload = compare_fn(
                trip_dir,
                max_epochs=max_epochs,
                multi_gnss=multi_gnss,
                apply_observation_mask=True,
            )
            unmasked, _unmasked_summary, unmasked_payload = compare_fn(
                trip_dir,
                max_epochs=max_epochs,
                multi_gnss=multi_gnss,
                apply_observation_mask=False,
            )
        except Exception as exc:  # pragma: no cover - CLI integration behavior.
            errors.append({"trip": trip, "error": f"{type(exc).__name__}: {exc}"})
            continue

        matlab_only = masked[masked["side"].eq("matlab_only")].copy()
        if matlab_only.empty:
            continue
        unmasked_bridge_keys = _key_set(unmasked[unmasked["side"].isin(("both", "bridge_only"))])
        matlab_only.insert(0, "trip", trip)
        matlab_only["drop_reason"] = [
            (
                "recovered_without_observation_mask"
                if tuple(item) in unmasked_bridge_keys
                else "still_missing_without_observation_mask"
            )
            for item in matlab_only[RESIDUAL_KEY_COLUMNS].itertuples(index=False, name=None)
        ]
        matlab_only["mask_reason"] = matlab_only.apply(_mask_reason, axis=1)
        matlab_only["masked_total_matlab_only"] = int(masked_payload.get("total_matlab_only", 0) or 0)
        matlab_only["unmasked_total_matlab_only"] = int(unmasked_payload.get("total_matlab_only", 0) or 0)
        rows.append(matlab_only)

    detail = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    by_scope = (
        _group_reasons(detail, ["trip", "drop_reason", "mask_reason", "field", "freq", "sys"])
        if not detail.empty
        else pd.DataFrame()
    )
    by_satellite = (
        _group_reasons(detail, ["trip", "drop_reason", "mask_reason", "field", "freq", "sys", "svid"])
        if not detail.empty
        else pd.DataFrame()
    )
    total_masked_matlab_only = int(len(detail))
    recovered = int((detail["drop_reason"] == "recovered_without_observation_mask").sum()) if "drop_reason" in detail else 0
    payload = {
        "data_root": str(Path(data_root)),
        "trips": list(trips),
        "trip_count": int(len(trips)),
        "completed_trip_count": int(len(trips) - len(errors)),
        "error_count": int(len(errors)),
        "errors": errors,
        "max_epochs": int(max_epochs),
        "multi_gnss": bool(multi_gnss),
        "total_masked_matlab_only": total_masked_matlab_only,
        "recovered_without_observation_mask": recovered,
        "still_missing_without_observation_mask": int(total_masked_matlab_only - recovered),
        "passed": bool(not errors and total_masked_matlab_only == recovered),
    }
    return by_scope.reset_index(drop=True), by_satellite.reset_index(drop=True), detail.reset_index(drop=True), payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _add_data_root_arg(parser, default_root=DEFAULT_ROOT)
    parser.add_argument(
        "--trip",
        action="append",
        dest="trips",
        help="trip in split/course/phone form; repeatable. Defaults to the residual parity trip set.",
    )
    _add_max_epochs_arg(parser, help_text="0 uses each trip's full settings window")
    _add_multi_gnss_arg(parser, default=False)
    parser.add_argument("--verbose", action="store_true", help="print trip progress to stderr")
    _add_output_dir_arg(parser)
    args = parser.parse_args()

    trips = tuple(args.trips) if args.trips else DEFAULT_RESIDUAL_PARITY_TRIPS
    out_dir = _timestamped_output_dir(_resolved_output_root(args), "gsdc2023_residual_mask_drop_audit")
    by_scope, by_satellite, detail, payload = residual_mask_drop_audit(
        Path(args.data_root).resolve(),
        trips,
        max_epochs=_nonnegative_max_epochs(args),
        multi_gnss=bool(args.multi_gnss),
        verbose=bool(args.verbose),
    )
    by_scope.to_csv(out_dir / "mask_drop_by_scope.csv", index=False)
    by_satellite.to_csv(out_dir / "mask_drop_by_satellite.csv", index=False)
    detail.to_csv(out_dir / "mask_drop_detail.csv", index=False)
    _write_summary_json(out_dir, payload)
    _print_summary_and_output_dir(payload, out_dir, label="audit_dir")
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
