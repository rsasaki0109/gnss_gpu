#!/usr/bin/env python3
"""Summarize residual rows that appear on only one side of the MATLAB bridge."""

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
from experiments.gsdc2023_raw_bridge import DEFAULT_ROOT  # noqa: E402


CompareFn = Callable[..., tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]]


def _join_sample(values: pd.Series, *, limit: int = 8) -> str:
    seen: list[str] = []
    for value in values.dropna().tolist():
        text = str(int(value)) if isinstance(value, float) and value.is_integer() else str(value)
        if text not in seen:
            seen.append(text)
        if len(seen) >= limit:
            break
    return " ".join(seen)


def _side_only(frame: pd.DataFrame, trip: str) -> pd.DataFrame:
    if frame.empty or "side" not in frame.columns:
        return pd.DataFrame()
    out = frame[frame["side"].isin(("matlab_only", "bridge_only"))].copy()
    if out.empty:
        return out
    out.insert(0, "trip", trip)
    return out


def _group_side_only(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=group_cols + ["count", "first_epoch", "last_epoch", "unique_svid_count", "sample_svids"])
    rows: list[dict[str, object]] = []
    for keys, group in frame.groupby(group_cols, sort=True, dropna=False, observed=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: value for col, value in zip(group_cols, keys)}
        row.update(
            {
                "count": int(len(group)),
                "first_epoch": int(group["epoch_index"].min()) if "epoch_index" in group else None,
                "last_epoch": int(group["epoch_index"].max()) if "epoch_index" in group else None,
                "unique_svid_count": int(group["svid"].nunique()) if "svid" in group else 0,
                "sample_svids": _join_sample(group["svid"]) if "svid" in group else "",
            },
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["count"] + group_cols, ascending=[False] + [True] * len(group_cols))


def residual_side_only_audit(
    data_root: Path,
    trips: Sequence[str],
    *,
    max_epochs: int,
    multi_gnss: bool,
    compare_fn: CompareFn = compare_residual_values,
    verbose: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    side_only_frames: list[pd.DataFrame] = []
    errors: list[dict[str, object]] = []
    for trip_idx, trip in enumerate(trips, start=1):
        if verbose:
            print(f"[{trip_idx}/{len(trips)}] {trip}", file=sys.stderr, flush=True)
        trip_dir = Path(data_root) / trip
        try:
            merged, _summary, _payload = compare_fn(
                trip_dir,
                max_epochs=max_epochs,
                multi_gnss=multi_gnss,
            )
        except Exception as exc:  # pragma: no cover - CLI integration behavior.
            errors.append({"trip": trip, "error": f"{type(exc).__name__}: {exc}"})
            continue
        side_only = _side_only(merged, trip)
        if not side_only.empty:
            side_only_frames.append(side_only)

    side_only_rows = pd.concat(side_only_frames, ignore_index=True) if side_only_frames else pd.DataFrame()
    scope_cols = ["trip", "side", "field", "freq", "sys"]
    satellite_cols = ["trip", "side", "field", "freq", "sys", "svid"]
    by_scope = _group_side_only(side_only_rows, scope_cols)
    by_satellite = _group_side_only(side_only_rows, satellite_cols)

    examples = pd.DataFrame()
    if not side_only_rows.empty:
        sort_cols = [col for col in ("trip", "side", "field", "freq", "sys", "svid", "epoch_index") if col in side_only_rows]
        examples = (
            side_only_rows.sort_values(sort_cols)
            .groupby(scope_cols, sort=True, dropna=False, observed=False)
            .head(3)
        )

    total_matlab_only = int((side_only_rows["side"] == "matlab_only").sum()) if "side" in side_only_rows else 0
    total_bridge_only = int((side_only_rows["side"] == "bridge_only").sum()) if "side" in side_only_rows else 0
    payload = {
        "data_root": str(Path(data_root)),
        "trips": list(trips),
        "trip_count": int(len(trips)),
        "completed_trip_count": int(len(trips) - len(errors)),
        "error_count": int(len(errors)),
        "errors": errors,
        "max_epochs": int(max_epochs),
        "multi_gnss": bool(multi_gnss),
        "total_matlab_only": total_matlab_only,
        "total_bridge_only": total_bridge_only,
        "total_side_only": int(total_matlab_only + total_bridge_only),
        "passed": bool(not errors and total_matlab_only == 0 and total_bridge_only == 0),
    }
    if not by_scope.empty:
        worst = by_scope.iloc[0]
        payload["largest_scope"] = {
            "trip": str(worst["trip"]),
            "side": str(worst["side"]),
            "field": str(worst["field"]),
            "freq": str(worst["freq"]),
            "sys": int(worst["sys"]) if pd.notna(worst["sys"]) else None,
            "count": int(worst["count"]),
        }
    return by_scope.reset_index(drop=True), by_satellite.reset_index(drop=True), examples.reset_index(drop=True), payload


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
    _add_multi_gnss_arg(parser, default=True)
    parser.add_argument("--verbose", action="store_true", help="print trip progress to stderr")
    _add_output_dir_arg(parser)
    args = parser.parse_args()

    trips = tuple(args.trips) if args.trips else DEFAULT_RESIDUAL_PARITY_TRIPS
    out_dir = _timestamped_output_dir(_resolved_output_root(args), "gsdc2023_residual_side_only_audit")
    by_scope, by_satellite, examples, payload = residual_side_only_audit(
        Path(args.data_root).resolve(),
        trips,
        max_epochs=_nonnegative_max_epochs(args),
        multi_gnss=bool(args.multi_gnss),
        verbose=bool(args.verbose),
    )
    by_scope.to_csv(out_dir / "side_only_by_scope.csv", index=False)
    by_satellite.to_csv(out_dir / "side_only_by_satellite.csv", index=False)
    examples.to_csv(out_dir / "side_only_examples.csv", index=False)
    _write_summary_json(out_dir, payload)
    _print_summary_and_output_dir(payload, out_dir, label="audit_dir")
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
