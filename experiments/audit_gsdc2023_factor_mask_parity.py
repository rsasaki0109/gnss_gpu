#!/usr/bin/env python3
"""Run MATLAB factor-mask parity checks across multiple GSDC2023 trips."""

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

from experiments.compare_gsdc2023_factor_masks import compare_factor_masks  # noqa: E402
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
from experiments.gsdc2023_raw_bridge import (  # noqa: E402
    DEFAULT_ROOT,
    DEFAULT_TDCP_CONSISTENCY_THRESHOLD_M,
    OBS_MASK_DOPPLER_RESIDUAL_THRESHOLD_MPS,
    OBS_MASK_PSEUDORANGE_DOPPLER_THRESHOLD_M,
    OBS_MASK_RESIDUAL_THRESHOLD_M,
    OBS_MASK_RESIDUAL_THRESHOLD_L5_M,
)


DEFAULT_FACTOR_MASK_PARITY_TRIPS: tuple[str, ...] = (
    "train/2020-06-25-00-34-us-ca-mtv-sb-101/pixel4",
    "train/2020-06-25-00-34-us-ca-mtv-sb-101/pixel4xl",
    "train/2020-07-08-22-28-us-ca/pixel4",
    "train/2020-07-08-22-28-us-ca/pixel4xl",
    "train/2020-07-17-22-27-us-ca-mtv-sf-280/pixel4",
    "train/2020-07-17-23-13-us-ca-sf-mtv-280/pixel4",
    "train/2020-07-17-23-13-us-ca-sf-mtv-280/pixel4xl",
    "train/2020-08-04-00-19-us-ca-sb-mtv-101/pixel4",
    "train/2020-08-04-00-20-us-ca-sb-mtv-101/pixel4xl",
    "train/2020-08-04-00-20-us-ca-sb-mtv-101/pixel5",
    "train/2021-12-08-20-28-us-ca-lax-c/pixel5",
    "train/2022-10-06-21-51-us-ca-mtv-n/sm-a205u",
)

CompareFn = Callable[..., tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]]


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
        "side_only_failure_count": int(payload.get("side_only_failure_count", 0) or 0),
        "symmetric_parity": _finite_float(payload.get("symmetric_parity")),
        "jaccard": _finite_float(payload.get("jaccard")),
        "start_epoch": int(payload.get("start_epoch", 0) or 0),
        "max_epochs": int(payload.get("max_epochs", 0) or 0),
    }


def _with_trip(trip: str, rows: object) -> list[dict[str, object]]:
    if not isinstance(rows, list):
        return []
    out: list[dict[str, object]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        item = {"trip": trip}
        item.update(row)
        out.append(item)
    return out


def _top_side_only_failures(rows: list[dict[str, object]], *, limit: int = 20) -> list[dict[str, object]]:
    return sorted(
        rows,
        key=lambda row: (
            str(row.get("side", "")),
            str(row.get("trip", "")),
            str(row.get("field", "")),
            str(row.get("freq", "")),
            int(row.get("epoch_index", 0) or 0),
            int(row.get("sys", 0) or 0),
            int(row.get("svid", 0) or 0),
        ),
    )[:limit]


def _field_side_only_sums(field_summary: pd.DataFrame) -> dict[str, dict[str, dict[str, int]]]:
    out: dict[str, dict[str, dict[str, int]]] = {}
    if field_summary.empty:
        return out
    grouped = field_summary.groupby(["field", "freq"], sort=True)[["matlab_only", "bridge_only"]].sum()
    for (field, freq), row in grouped.iterrows():
        out.setdefault(str(field), {})[str(freq)] = {
            "matlab_only": int(row["matlab_only"]),
            "bridge_only": int(row["bridge_only"]),
        }
    return out


def factor_mask_parity_audit(
    data_root: Path,
    trips: Sequence[str],
    *,
    max_epochs: int,
    multi_gnss: bool,
    min_symmetric_parity: float = 1.0,
    pseudorange_residual_mask_m: float = OBS_MASK_RESIDUAL_THRESHOLD_M,
    pseudorange_residual_mask_l5_m: float = OBS_MASK_RESIDUAL_THRESHOLD_L5_M,
    doppler_residual_mask_mps: float = OBS_MASK_DOPPLER_RESIDUAL_THRESHOLD_MPS,
    tdcp_consistency_threshold_m: float = DEFAULT_TDCP_CONSISTENCY_THRESHOLD_M,
    pseudorange_doppler_mask_m: float = OBS_MASK_PSEUDORANGE_DOPPLER_THRESHOLD_M,
    compare_fn: CompareFn = compare_factor_masks,
    verbose: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    rows: list[dict[str, object]] = []
    field_rows: list[pd.DataFrame] = []
    matlab_only_rows: list[dict[str, object]] = []
    bridge_only_rows: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    for index, trip in enumerate(trips, start=1):
        if verbose:
            print(f"[{index}/{len(trips)}] {trip}", file=sys.stderr, flush=True)
        trip_dir = Path(data_root) / trip
        try:
            _merged, summary_by_field, payload = compare_fn(
                trip_dir,
                max_epochs=max_epochs,
                multi_gnss=multi_gnss,
                pseudorange_residual_mask_m=pseudorange_residual_mask_m,
                pseudorange_residual_mask_l5_m=pseudorange_residual_mask_l5_m,
                doppler_residual_mask_mps=doppler_residual_mask_mps,
                tdcp_consistency_threshold_m=tdcp_consistency_threshold_m,
                pseudorange_doppler_mask_m=pseudorange_doppler_mask_m,
            )
        except Exception as exc:  # pragma: no cover - exercised through CLI behavior.
            errors.append({"trip": trip, "error": f"{type(exc).__name__}: {exc}"})
            continue
        rows.append(_trip_summary_row(trip, payload))
        matlab_only_rows.extend(_with_trip(trip, payload.get("top_matlab_only")))
        bridge_only_rows.extend(_with_trip(trip, payload.get("top_bridge_only")))
        field_frame = summary_by_field.copy()
        if not field_frame.empty:
            field_frame.insert(0, "trip", trip)
            field_rows.append(field_frame)

    trip_summary = pd.DataFrame(rows)
    field_summary = pd.concat(field_rows, ignore_index=True) if field_rows else pd.DataFrame()
    if not trip_summary.empty:
        trip_summary = trip_summary.sort_values("trip").reset_index(drop=True)
    if not field_summary.empty:
        field_summary = field_summary.sort_values(["trip", "field", "freq"]).reset_index(drop=True)

    min_parity = (
        float(pd.to_numeric(trip_summary["symmetric_parity"], errors="coerce").min())
        if not trip_summary.empty
        else float("nan")
    )
    total_matlab_only = int(trip_summary["total_matlab_only"].sum()) if not trip_summary.empty else 0
    total_bridge_only = int(trip_summary["total_bridge_only"].sum()) if not trip_summary.empty else 0
    passed = bool(
        not errors
        and not trip_summary.empty
        and np.isfinite(min_parity)
        and min_parity >= float(min_symmetric_parity)
        and total_matlab_only == 0
        and total_bridge_only == 0
    )
    payload = {
        "data_root": str(Path(data_root)),
        "trips": list(trips),
        "trip_count": int(len(trips)),
        "completed_trip_count": int(len(trip_summary)),
        "error_count": int(len(errors)),
        "errors": errors,
        "max_epochs": int(max_epochs),
        "multi_gnss": bool(multi_gnss),
        "min_symmetric_parity_threshold": float(min_symmetric_parity),
        "overall_min_symmetric_parity": min_parity,
        "total_matlab_only": total_matlab_only,
        "total_bridge_only": total_bridge_only,
        "side_only_failure_count": int(total_matlab_only + total_bridge_only),
        "side_only_by_field_freq": _field_side_only_sums(field_summary),
        "top_matlab_only": _top_side_only_failures(matlab_only_rows),
        "top_bridge_only": _top_side_only_failures(bridge_only_rows),
        "passed": passed,
    }
    if not trip_summary.empty:
        worst = trip_summary.loc[trip_summary["symmetric_parity"].idxmin()]
        payload["worst_trip"] = str(worst["trip"])
        payload["worst_symmetric_parity"] = float(worst["symmetric_parity"])
    return trip_summary, field_summary, payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _add_data_root_arg(parser, default_root=DEFAULT_ROOT)
    parser.add_argument(
        "--trip",
        action="append",
        dest="trips",
        help="trip in split/course/phone form; repeatable. Defaults to the built-in factor-mask parity trip set.",
    )
    _add_max_epochs_arg(parser, help_text="0 uses each trip's full settings window")
    _add_multi_gnss_arg(parser, default=False, help_text="default is GPS-only MATLAB parity scope")
    parser.add_argument("--min-symmetric-parity", type=float, default=1.0)
    parser.add_argument("--pseudorange-residual-mask-m", type=float, default=OBS_MASK_RESIDUAL_THRESHOLD_M)
    parser.add_argument("--pseudorange-residual-mask-l5-m", type=float, default=OBS_MASK_RESIDUAL_THRESHOLD_L5_M)
    parser.add_argument("--doppler-residual-mask-mps", type=float, default=OBS_MASK_DOPPLER_RESIDUAL_THRESHOLD_MPS)
    parser.add_argument("--tdcp-consistency-threshold-m", type=float, default=DEFAULT_TDCP_CONSISTENCY_THRESHOLD_M)
    parser.add_argument("--pseudorange-doppler-mask-m", type=float, default=OBS_MASK_PSEUDORANGE_DOPPLER_THRESHOLD_M)
    parser.add_argument("--verbose", action="store_true", help="print trip progress to stderr")
    _add_output_dir_arg(parser)
    args = parser.parse_args()

    trips = tuple(args.trips) if args.trips else DEFAULT_FACTOR_MASK_PARITY_TRIPS
    out_dir = _timestamped_output_dir(_resolved_output_root(args), "gsdc2023_factor_mask_parity_audit")
    trip_summary, field_summary, payload = factor_mask_parity_audit(
        Path(args.data_root).resolve(),
        trips,
        max_epochs=_nonnegative_max_epochs(args),
        multi_gnss=bool(args.multi_gnss),
        min_symmetric_parity=float(args.min_symmetric_parity),
        pseudorange_residual_mask_m=float(args.pseudorange_residual_mask_m),
        pseudorange_residual_mask_l5_m=float(args.pseudorange_residual_mask_l5_m),
        doppler_residual_mask_mps=float(args.doppler_residual_mask_mps),
        tdcp_consistency_threshold_m=float(args.tdcp_consistency_threshold_m),
        pseudorange_doppler_mask_m=float(args.pseudorange_doppler_mask_m),
        verbose=bool(args.verbose),
    )
    trip_summary.to_csv(out_dir / "trip_summary.csv", index=False)
    field_summary.to_csv(out_dir / "summary_by_trip_field.csv", index=False)
    _write_summary_json(out_dir, payload)
    _print_summary_and_output_dir(payload, out_dir, label="audit_dir")
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
