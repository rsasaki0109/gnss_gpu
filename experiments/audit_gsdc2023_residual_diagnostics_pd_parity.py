#!/usr/bin/env python3
"""Audit residual-diagnostics P/D sidecar parity across multiple trips."""

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

from experiments.audit_gsdc2023_factor_mask_parity import DEFAULT_FACTOR_MASK_PARITY_TRIPS  # noqa: E402
from experiments.compare_gsdc2023_residual_diagnostics_pd import (  # noqa: E402
    bridge_residual_diagnostics_pd_export_frame,
    compare_residual_diagnostics_pd_values,
)
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


CompareFn = Callable[..., tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]]


def _finite_float(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _trip_summary_row(trip: str, payload: dict[str, object]) -> dict[str, object]:
    return {
        "trip": trip,
        "total_matlab_count": int(payload.get("total_matlab_count", 0) or 0),
        "total_bridge_count": int(payload.get("total_bridge_count", 0) or 0),
        "total_matched_count": int(payload.get("total_matched_count", 0) or 0),
        "total_matlab_only": int(payload.get("total_matlab_only", 0) or 0),
        "total_bridge_only": int(payload.get("total_bridge_only", 0) or 0),
        "median_abs_delta": _finite_float(payload.get("median_abs_delta")),
        "p95_abs_delta": _finite_float(payload.get("p95_abs_delta")),
        "max_abs_delta": _finite_float(payload.get("max_abs_delta")),
        "passed": bool(payload.get("passed", False)),
    }


def _with_trip(trip: str, frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if not out.empty:
        out.insert(0, "trip", trip)
    return out


def _top_side_only_rows(rows: list[dict[str, object]], *, limit: int = 20) -> list[dict[str, object]]:
    return sorted(
        rows,
        key=lambda row: (
            str(row.get("side", "")),
            str(row.get("trip", "")),
            str(row.get("field", "")),
            str(row.get("diagnostics_column", "")),
            str(row.get("freq", "")),
            int(row.get("epoch_index", 0) or 0),
            int(row.get("sys", 0) or 0),
            int(row.get("svid", 0) or 0),
        ),
    )[:limit]


def _write_bridge_subset_export(
    trip: str,
    bridge_values: pd.DataFrame,
    export_dir: Path,
) -> dict[str, object]:
    output_path = export_dir / trip / "phone_data_residual_diagnostics_pd_subset.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subset = bridge_residual_diagnostics_pd_export_frame(bridge_values)
    subset.to_csv(output_path, index=False)
    return {
        "trip": trip,
        "path": str(output_path),
        "row_count": int(len(subset)),
        "value_count": int(len(bridge_values)),
    }


def residual_diagnostics_pd_parity_audit(
    data_root: Path,
    trips: Sequence[str],
    *,
    max_epochs: int,
    multi_gnss: bool,
    apply_observation_mask: bool = True,
    include_inactive_observations: bool = True,
    max_abs_delta_threshold: float = 1.0e-4,
    compare_fn: CompareFn = compare_residual_diagnostics_pd_values,
    bridge_subset_export_dir: Path | None = None,
    verbose: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    data_root = Path(data_root)
    trip_rows: list[dict[str, object]] = []
    column_frames: list[pd.DataFrame] = []
    side_only_rows: list[dict[str, object]] = []
    export_rows: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    for index, trip in enumerate(trips, start=1):
        if verbose:
            print(f"[{index}/{len(trips)}] {trip}", file=sys.stderr, flush=True)
        trip_dir = data_root / trip
        try:
            merged, column_summary, bridge_values, payload = compare_fn(
                trip_dir,
                max_epochs=max_epochs,
                multi_gnss=multi_gnss,
                apply_observation_mask=apply_observation_mask,
                include_inactive_observations=include_inactive_observations,
                max_abs_delta_threshold=max_abs_delta_threshold,
            )
        except Exception as exc:  # pragma: no cover - exercised by CLI usage.
            errors.append({"trip": trip, "error": f"{type(exc).__name__}: {exc}"})
            continue

        trip_rows.append(_trip_summary_row(trip, payload))
        if not column_summary.empty:
            column_frames.append(_with_trip(trip, column_summary))
        if not merged.empty:
            side_only = merged.loc[merged["side"].isin(("matlab_only", "bridge_only"))].copy()
            if not side_only.empty:
                side_only.insert(0, "trip", trip)
                side_only_rows.extend(side_only.to_dict("records"))
        if bridge_subset_export_dir is not None:
            try:
                export_rows.append(_write_bridge_subset_export(trip, bridge_values, bridge_subset_export_dir))
            except Exception as exc:  # pragma: no cover - exercised by CLI usage.
                errors.append({"trip": trip, "error": f"bridge subset export {type(exc).__name__}: {exc}"})

    trip_summary = pd.DataFrame(trip_rows)
    if not trip_summary.empty:
        trip_summary = trip_summary.sort_values("trip").reset_index(drop=True)
    column_summary = pd.concat(column_frames, ignore_index=True) if column_frames else pd.DataFrame()
    side_only_frame = pd.DataFrame(side_only_rows)
    export_summary = pd.DataFrame(export_rows)

    total_matlab_only = int(trip_summary["total_matlab_only"].sum()) if not trip_summary.empty else 0
    total_bridge_only = int(trip_summary["total_bridge_only"].sum()) if not trip_summary.empty else 0
    overall_max_abs_delta = (
        float(pd.to_numeric(trip_summary["max_abs_delta"], errors="coerce").max())
        if not trip_summary.empty
        else float("nan")
    )
    payload = {
        "data_root": str(data_root),
        "trips": list(trips),
        "trip_count": int(len(trips)),
        "completed_trip_count": int(len(trip_summary)),
        "error_count": int(len(errors)),
        "errors": errors,
        "max_epochs": int(max_epochs),
        "multi_gnss": bool(multi_gnss),
        "apply_observation_mask": bool(apply_observation_mask),
        "include_inactive_observations": bool(include_inactive_observations),
        "max_abs_delta_threshold": float(max_abs_delta_threshold),
        "overall_max_abs_delta": overall_max_abs_delta,
        "total_matlab_count": int(trip_summary["total_matlab_count"].sum()) if not trip_summary.empty else 0,
        "total_bridge_count": int(trip_summary["total_bridge_count"].sum()) if not trip_summary.empty else 0,
        "total_matched_count": int(trip_summary["total_matched_count"].sum()) if not trip_summary.empty else 0,
        "total_matlab_only": total_matlab_only,
        "total_bridge_only": total_bridge_only,
        "top_side_only": _top_side_only_rows(side_only_rows),
        "bridge_subset_export_count": int(len(export_summary)),
        "bridge_subset_export_total_rows": int(export_summary["row_count"].sum()) if not export_summary.empty else 0,
        "bridge_subset_export_total_values": int(export_summary["value_count"].sum()) if not export_summary.empty else 0,
        "passed": bool(
            not errors
            and total_matlab_only == 0
            and total_bridge_only == 0
            and np.isfinite(overall_max_abs_delta)
            and overall_max_abs_delta <= float(max_abs_delta_threshold)
        ),
    }
    if not trip_summary.empty:
        worst = trip_summary.sort_values("max_abs_delta", ascending=False).iloc[0]
        payload.update(
            {
                "worst_trip": str(worst["trip"]),
                "worst_trip_max_abs_delta": _finite_float(worst["max_abs_delta"]),
            },
        )
    return trip_summary, column_summary, side_only_frame, export_summary, payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _add_data_root_arg(parser, default_root=DEFAULT_ROOT)
    parser.add_argument(
        "--trip",
        action="append",
        dest="trips",
        help="trip in split/course/phone form; repeatable. Defaults to the 12-trip MATLAB sidecar bundle.",
    )
    _add_max_epochs_arg(parser)
    _add_multi_gnss_arg(parser)
    parser.add_argument("--observation-mask", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-inactive-observations", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-abs-delta-threshold", type=float, default=1.0e-4)
    parser.add_argument("--write-bridge-pd-subsets", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    _add_output_dir_arg(parser)
    args = parser.parse_args()

    trips = tuple(args.trips) if args.trips else DEFAULT_FACTOR_MASK_PARITY_TRIPS
    out_dir = _timestamped_output_dir(_resolved_output_root(args), "gsdc2023_residual_diagnostics_pd_parity_audit")
    export_dir = out_dir / "bridge_residual_diagnostics_pd_subset" if args.write_bridge_pd_subsets else None
    trip_summary, column_summary, side_only, export_summary, payload = residual_diagnostics_pd_parity_audit(
        Path(args.data_root),
        trips,
        max_epochs=_nonnegative_max_epochs(args),
        multi_gnss=bool(args.multi_gnss),
        apply_observation_mask=bool(args.observation_mask),
        include_inactive_observations=bool(args.include_inactive_observations),
        max_abs_delta_threshold=float(args.max_abs_delta_threshold),
        bridge_subset_export_dir=export_dir,
        verbose=bool(args.verbose),
    )
    trip_summary.to_csv(out_dir / "trip_summary.csv", index=False)
    column_summary.to_csv(out_dir / "summary_by_column.csv", index=False)
    side_only.to_csv(out_dir / "side_only.csv", index=False)
    export_summary.to_csv(out_dir / "bridge_subset_exports.csv", index=False)
    _write_summary_json(out_dir, payload)
    _print_summary_and_output_dir(payload, out_dir, label="audit_dir")


if __name__ == "__main__":
    main()
