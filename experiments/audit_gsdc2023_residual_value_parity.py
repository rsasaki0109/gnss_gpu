#!/usr/bin/env python3
"""Run MATLAB residual-value parity checks across multiple GSDC2023 trips."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path
import sys

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from experiments.compare_gsdc2023_residual_values import compare_residual_values  # noqa: E402
from experiments.gsdc2023_residual_audit import RESIDUAL_COMPONENT_SUMMARY_COLUMNS  # noqa: E402
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


DEFAULT_RESIDUAL_PARITY_TRIPS: tuple[str, ...] = (
    "train/2020-06-25-00-34-us-ca-mtv-sb-101/pixel4",
    "train/2020-06-25-00-34-us-ca-mtv-sb-101/pixel4xl",
    "train/2020-07-08-22-28-us-ca/pixel4",
    "train/2020-07-08-22-28-us-ca/pixel4xl",
    "train/2020-07-17-22-27-us-ca-mtv-sf-280/pixel4",
    "train/2020-07-17-23-13-us-ca-sf-mtv-280/pixel4",
    "train/2020-08-04-00-19-us-ca-sb-mtv-101/pixel4",
    "train/2020-08-04-00-20-us-ca-sb-mtv-101/pixel4xl",
    "train/2020-08-04-00-20-us-ca-sb-mtv-101/pixel5",
    "train/2021-12-08-20-28-us-ca-lax-c/pixel5",
    "train/2022-10-06-21-51-us-ca-mtv-n/sm-a205u",
)
DEFAULT_INTERNAL_DELTA_THRESHOLDS: dict[str, float] = {
    "pre_residual_delta": 1.0e-4,
    "common_bias_delta": 1.0e-4,
    "isb_delta": 1.0e-4,
    "observation_delta": 1.0e-4,
    "model_delta": 1.0e-4,
    "sat_position_delta_norm": 1.0e-3,
    "sat_velocity_delta_norm": 1.0e-3,
    "sat_clock_bias_delta": 1.0e-4,
    "sat_clock_drift_delta": 1.0e-6,
    "sat_iono_delta": 1.0e-4,
    "sat_trop_delta": 1.0e-4,
    "sat_elevation_delta": 1.0e-3,
    "rcv_position_delta_norm": 1.0e-3,
    "rcv_velocity_delta_norm": 1.0e-3,
}

CompareFn = Callable[..., tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]]


def _finite_float(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _trip_summary_row(trip: str, merged: pd.DataFrame, payload: dict[str, object]) -> tuple[dict[str, object], dict[str, object]]:
    matched = merged[merged["side"].eq("both")].copy() if "side" in merged.columns else pd.DataFrame()
    max_row: dict[str, object] = {"trip": trip}
    max_field = max_freq = max_epoch = max_svid = None
    if not matched.empty and "delta" in matched.columns:
        matched["abs_delta"] = pd.to_numeric(matched["delta"], errors="coerce").abs()
        max_item = matched.sort_values("abs_delta", ascending=False).iloc[0]
        for column in (
            "field",
            "freq",
            "epoch_index",
            "utcTimeMillis",
            "sys",
            "svid",
            "delta",
            "pre_residual_delta",
            "common_bias_delta",
            "observation_delta",
            "model_delta",
            "sat_position_delta_norm",
            "sat_velocity_delta_norm",
            "sat_clock_drift_delta",
            "sat_clock_bias_delta",
            "sat_iono_delta",
            "sat_trop_delta",
            "sat_elevation_delta",
            "rcv_position_delta_norm",
            "rcv_velocity_delta_norm",
            "isb_delta",
        ):
            if column in matched.columns:
                max_row[column] = max_item.get(column)
        max_field = max_row.get("field")
        max_freq = max_row.get("freq")
        max_epoch = max_row.get("epoch_index")
        max_svid = max_row.get("svid")

    row = {
        "trip": trip,
        "matched_count": int(payload.get("total_matched_count", len(matched)) or 0),
        "median_abs_delta": _finite_float(payload.get("median_abs_delta")),
        "p95_abs_delta": _finite_float(payload.get("p95_abs_delta")),
        "max_abs_delta": _finite_float(payload.get("max_abs_delta")),
        "median_abs_pre_residual_delta": _finite_float(payload.get("median_abs_pre_residual_delta")),
        "p95_abs_pre_residual_delta": _finite_float(payload.get("p95_abs_pre_residual_delta")),
        "max_abs_pre_residual_delta": _finite_float(payload.get("max_abs_pre_residual_delta")),
        "matlab_only_count": int(payload.get("total_matlab_only", 0) or 0),
        "bridge_only_count": int(payload.get("total_bridge_only", 0) or 0),
        "max_field": max_field,
        "max_freq": max_freq,
        "max_epoch": max_epoch,
        "max_svid": max_svid,
    }
    for column in _internal_delta_columns():
        row[f"max_abs_{column}"] = _finite_float(payload.get(f"max_abs_{column}"))
    return row, max_row


def _internal_delta_columns() -> tuple[str, ...]:
    return (
        "pre_residual_delta",
        "common_bias_delta",
        "isb_delta",
        "observation_delta",
        "model_delta",
        *[column for column, _prefix in RESIDUAL_COMPONENT_SUMMARY_COLUMNS],
    )


def _internal_delta_failures(
    trip_summary: pd.DataFrame,
    thresholds: dict[str, float],
) -> list[dict[str, object]]:
    failures: list[dict[str, object]] = []
    if trip_summary.empty:
        return failures
    for column, threshold in thresholds.items():
        summary_column = f"max_abs_{column}"
        if summary_column not in trip_summary.columns:
            continue
        values = pd.to_numeric(trip_summary[summary_column], errors="coerce")
        exceeded = values > float(threshold)
        for _, row in trip_summary.loc[exceeded].iterrows():
            failures.append(
                {
                    "trip": str(row["trip"]),
                    "component": column,
                    "max_abs_delta": float(row[summary_column]),
                    "threshold": float(threshold),
                },
            )
    return failures


def residual_value_parity_audit(
    data_root: Path,
    trips: Sequence[str],
    *,
    max_epochs: int,
    multi_gnss: bool,
    apply_observation_mask: bool = True,
    include_inactive_observations: bool = False,
    max_abs_delta_threshold_m: float = 1.0e-4,
    p95_abs_delta_threshold_m: float | None = None,
    internal_delta_thresholds: dict[str, float] | None = None,
    compare_fn: CompareFn = compare_residual_values,
    verbose: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    rows: list[dict[str, object]] = []
    max_rows: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    for trip_idx, trip in enumerate(trips, start=1):
        if verbose:
            print(f"[{trip_idx}/{len(trips)}] {trip}", file=sys.stderr, flush=True)
        trip_dir = Path(data_root) / trip
        try:
            merged, _summary, payload = compare_fn(
                trip_dir,
                max_epochs=max_epochs,
                multi_gnss=multi_gnss,
                apply_observation_mask=apply_observation_mask,
                include_inactive_observations=include_inactive_observations,
            )
        except Exception as exc:  # pragma: no cover - exercised through CLI behavior.
            errors.append({"trip": trip, "error": f"{type(exc).__name__}: {exc}"})
            continue
        row, max_row = _trip_summary_row(trip, merged, payload)
        rows.append(row)
        max_rows.append(max_row)

    trip_summary = pd.DataFrame(rows)
    max_row_frame = pd.DataFrame(max_rows)
    if not trip_summary.empty:
        trip_summary = trip_summary.sort_values("trip").reset_index(drop=True)
    if not max_row_frame.empty:
        max_row_frame = max_row_frame.sort_values("trip").reset_index(drop=True)

    max_abs = float(trip_summary["max_abs_delta"].max()) if not trip_summary.empty else float("nan")
    p95_abs = float(trip_summary["p95_abs_delta"].max()) if not trip_summary.empty else float("nan")
    pass_max = bool(np.isfinite(max_abs) and max_abs <= float(max_abs_delta_threshold_m))
    pass_p95 = (
        True
        if p95_abs_delta_threshold_m is None
        else bool(np.isfinite(p95_abs) and p95_abs <= float(p95_abs_delta_threshold_m))
    )
    total_matlab_only = int(trip_summary["matlab_only_count"].sum()) if not trip_summary.empty else 0
    total_bridge_only = int(trip_summary["bridge_only_count"].sum()) if not trip_summary.empty else 0
    internal_thresholds = dict(DEFAULT_INTERNAL_DELTA_THRESHOLDS if internal_delta_thresholds is None else internal_delta_thresholds)
    internal_failures = _internal_delta_failures(trip_summary, internal_thresholds)
    payload = {
        "data_root": str(Path(data_root)),
        "trips": list(trips),
        "trip_count": int(len(trips)),
        "completed_trip_count": int(len(trip_summary)),
        "error_count": int(len(errors)),
        "errors": errors,
        "max_epochs": int(max_epochs),
        "multi_gnss": bool(multi_gnss),
        "apply_observation_mask": bool(apply_observation_mask),
        "include_inactive_observations": bool(include_inactive_observations),
        "max_abs_delta_threshold_m": float(max_abs_delta_threshold_m),
        "p95_abs_delta_threshold_m": (
            None if p95_abs_delta_threshold_m is None else float(p95_abs_delta_threshold_m)
        ),
        "overall_max_abs_delta": max_abs,
        "overall_p95_abs_delta_max": p95_abs,
        "total_matlab_only": total_matlab_only,
        "total_bridge_only": total_bridge_only,
        "internal_delta_thresholds": internal_thresholds,
        "internal_delta_failure_count": int(len(internal_failures)),
        "internal_delta_failures": internal_failures,
        "passed": bool(
            not errors
            and pass_max
            and pass_p95
            and total_matlab_only == 0
            and total_bridge_only == 0
            and not internal_failures
        ),
    }
    for column in _internal_delta_columns():
        summary_column = f"max_abs_{column}"
        payload[f"overall_{summary_column}"] = (
            float(pd.to_numeric(trip_summary[summary_column], errors="coerce").max())
            if summary_column in trip_summary and not trip_summary.empty
            else float("nan")
        )
    if not trip_summary.empty:
        worst = trip_summary.loc[trip_summary["max_abs_delta"].idxmax()]
        payload["worst_trip"] = str(worst["trip"])
        payload["worst_field"] = str(worst.get("max_field"))
        payload["worst_freq"] = str(worst.get("max_freq"))
        payload["worst_epoch"] = int(worst["max_epoch"]) if pd.notna(worst.get("max_epoch")) else None
        payload["worst_svid"] = int(worst["max_svid"]) if pd.notna(worst.get("max_svid")) else None
    return trip_summary, max_row_frame, payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _add_data_root_arg(parser, default_root=DEFAULT_ROOT)
    parser.add_argument(
        "--trip",
        action="append",
        dest="trips",
        help="trip in split/course/phone form; repeatable. Defaults to the built-in residual parity trip set.",
    )
    _add_max_epochs_arg(parser, help_text="0 uses each trip's full settings window")
    _add_multi_gnss_arg(parser, default=True)
    parser.add_argument("--observation-mask", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-inactive-observations", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-abs-delta-threshold-m", type=float, default=1.0e-4)
    parser.add_argument("--p95-abs-delta-threshold-m", type=float, default=None)
    parser.add_argument("--verbose", action="store_true", help="print trip progress to stderr")
    _add_output_dir_arg(parser)
    args = parser.parse_args()

    trips = tuple(args.trips) if args.trips else DEFAULT_RESIDUAL_PARITY_TRIPS
    out_dir = _timestamped_output_dir(_resolved_output_root(args), "gsdc2023_residual_value_parity_audit")
    trip_summary, max_rows, payload = residual_value_parity_audit(
        Path(args.data_root).resolve(),
        trips,
        max_epochs=_nonnegative_max_epochs(args),
        multi_gnss=bool(args.multi_gnss),
        apply_observation_mask=bool(args.observation_mask),
        include_inactive_observations=bool(args.include_inactive_observations),
        max_abs_delta_threshold_m=float(args.max_abs_delta_threshold_m),
        p95_abs_delta_threshold_m=args.p95_abs_delta_threshold_m,
        verbose=bool(args.verbose),
    )
    trip_summary.to_csv(out_dir / "trip_summary.csv", index=False)
    max_rows.to_csv(out_dir / "max_rows.csv", index=False)
    _write_summary_json(out_dir, payload)
    _print_summary_and_output_dir(payload, out_dir, label="audit_dir")
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
