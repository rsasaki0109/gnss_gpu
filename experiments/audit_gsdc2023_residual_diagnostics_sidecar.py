#!/usr/bin/env python3
"""Inventory MATLAB ``phone_data_residual_diagnostics.csv`` sidecar coverage."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from typing import Iterable

import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from experiments.audit_gsdc2023_factor_mask_parity import DEFAULT_FACTOR_MASK_PARITY_TRIPS  # noqa: E402
from experiments.gsdc2023_audit_cli import (  # noqa: E402
    add_data_root_arg as _add_data_root_arg,
    add_output_dir_arg as _add_output_dir_arg,
    resolved_output_root as _resolved_output_root,
)
from experiments.gsdc2023_audit_output import (  # noqa: E402
    print_summary_and_output_dir as _print_summary_and_output_dir,
    timestamped_output_dir as _timestamped_output_dir,
    write_summary_json as _write_summary_json,
)
from experiments.gsdc2023_raw_bridge import DEFAULT_ROOT  # noqa: E402


DIAGNOSTICS_KEY_COLUMNS = ("freq", "epoch_index", "utcTimeMillis", "sys", "svid", "sat_col")
DIAGNOSTICS_BOOL_COLUMNS = (
    "p_pre_finite",
    "d_pre_finite",
    "l_pre_finite",
    "p_factor_finite",
    "d_factor_finite",
    "l_factor_finite",
)
DIAGNOSTICS_EXPECTED_COLUMNS = (
    *DIAGNOSTICS_KEY_COLUMNS,
    "p_residual_m",
    "d_residual_mps",
    "p_pre_respc_m",
    "d_pre_resd_m",
    "p_corrected_m",
    "p_range_m",
    "d_obs_mps",
    "d_model_mps",
    "sat_x_m",
    "sat_y_m",
    "sat_z_m",
    "sat_vx_mps",
    "sat_vy_mps",
    "sat_vz_mps",
    "sat_clock_bias_m",
    "sat_clock_drift_mps",
    "sat_iono_m",
    "sat_trop_m",
    "sat_range_m",
    "sat_rate_mps",
    "sat_elevation_deg",
    "rcv_x_m",
    "rcv_y_m",
    "rcv_z_m",
    "rcv_vx_mps",
    "rcv_vy_mps",
    "rcv_vz_mps",
    "obs_clk_m",
    "obs_dclk_m",
    "p_isb_m",
    "p_clock_bias_m",
    "d_clock_bias_mps",
    *DIAGNOSTICS_BOOL_COLUMNS,
)


@dataclass(frozen=True)
class DiagnosticsColumnRole:
    column: str
    role: str
    used_by: str
    bridge_status: str
    writer_risk: str


def diagnostics_column_roles() -> list[DiagnosticsColumnRole]:
    roles: dict[str, DiagnosticsColumnRole] = {}

    def add(column: str, role: str, used_by: str, bridge_status: str, writer_risk: str) -> None:
        roles[column] = DiagnosticsColumnRole(column, role, used_by, bridge_status, writer_risk)

    for column in ("freq", "epoch_index", "utcTimeMillis", "sys", "svid"):
        add(column, "key", "all diagnostics joins and factor-mask rebuilds", "implemented", "low")
    add("sat_col", "MATLAB artifact key aid", "byte-compatible sidecar export ordering", "implemented", "low")

    for column in ("p_factor_finite", "d_factor_finite", "l_factor_finite"):
        add(
            column,
            "factor availability",
            "diagnostics mask overlay and factor-mask rebuild",
            "partially implemented from bridge masks",
            "medium",
        )
    for column in ("p_pre_finite", "d_pre_finite", "l_pre_finite"):
        add(
            column,
            "pre-residual availability",
            "residual-value parity and diagnostics-to-factor-mask checks",
            "partially implemented from bridge residual/factor masks",
            "medium",
        )

    for column in ("p_residual_m", "p_pre_respc_m", "p_corrected_m", "p_range_m", "p_isb_m", "p_clock_bias_m"):
        add(column, "pseudorange residual value", "residual-value parity", "implemented in bridge residual audit", "medium")
    for column in ("d_residual_mps", "d_pre_resd_m", "d_obs_mps", "d_model_mps", "d_clock_bias_mps"):
        add(column, "doppler residual value", "residual-value parity", "implemented in bridge residual audit", "medium")

    for column in (
        "sat_x_m",
        "sat_y_m",
        "sat_z_m",
        "sat_vx_mps",
        "sat_vy_mps",
        "sat_vz_mps",
        "sat_clock_bias_m",
        "sat_clock_drift_mps",
        "sat_iono_m",
        "sat_trop_m",
        "sat_range_m",
        "sat_rate_mps",
        "sat_elevation_deg",
        "rcv_x_m",
        "rcv_y_m",
        "rcv_z_m",
        "rcv_vx_mps",
        "rcv_vy_mps",
        "rcv_vz_mps",
        "obs_clk_m",
        "obs_dclk_m",
    ):
        add(column, "internal state component", "internal residual parity thresholds", "implemented in bridge residual audit", "medium")

    return [roles[column] for column in DIAGNOSTICS_EXPECTED_COLUMNS]


def _bool_count(frame: pd.DataFrame, column: str) -> int | None:
    if column not in frame.columns:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").fillna(0)
    return int((values != 0).sum())


def diagnostics_trip_records(data_root: Path, trips: Iterable[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    expected = set(DIAGNOSTICS_EXPECTED_COLUMNS)
    for trip in trips:
        path = data_root / trip / "phone_data_residual_diagnostics.csv"
        row: dict[str, object] = {
            "trip": trip,
            "diagnostics_path": str(path),
            "diagnostics_present": path.is_file(),
        }
        if not path.is_file():
            row.update({"row_count": 0, "column_count": 0, "missing_expected_columns": ",".join(DIAGNOSTICS_EXPECTED_COLUMNS)})
            rows.append(row)
            continue
        frame = pd.read_csv(path)
        columns = set(frame.columns)
        row.update(
            {
                "row_count": int(len(frame)),
                "column_count": int(len(frame.columns)),
                "missing_expected_columns": ",".join(column for column in DIAGNOSTICS_EXPECTED_COLUMNS if column not in columns),
                "extra_columns": ",".join(column for column in frame.columns if column not in expected),
            },
        )
        for column in DIAGNOSTICS_BOOL_COLUMNS:
            row[f"{column}_count"] = _bool_count(frame, column)
        rows.append(row)
    return pd.DataFrame(rows)


def summary_from_records(records: pd.DataFrame) -> dict[str, object]:
    present = records["diagnostics_present"].fillna(False).astype(bool) if "diagnostics_present" in records else pd.Series(dtype=bool)
    row_counts = pd.to_numeric(records.get("row_count", pd.Series(dtype=float)), errors="coerce").fillna(0)
    complete = records.get("missing_expected_columns", pd.Series(dtype=object)).fillna("").astype(str).eq("")
    payload: dict[str, object] = {
        "trip_count": int(len(records)),
        "diagnostics_present_count": int(present.sum()),
        "diagnostics_complete_schema_count": int((present & complete).sum()) if len(records) else 0,
        "total_rows": int(row_counts.sum()),
        "expected_column_count": int(len(DIAGNOSTICS_EXPECTED_COLUMNS)),
    }
    for column in DIAGNOSTICS_BOOL_COLUMNS:
        count_col = f"{column}_count"
        if count_col in records.columns:
            payload[f"total_{count_col}"] = int(pd.to_numeric(records[count_col], errors="coerce").fillna(0).sum())
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _add_data_root_arg(parser, default_root=DEFAULT_ROOT)
    parser.add_argument(
        "--trip",
        action="append",
        dest="trips",
        help="trip in split/course/phone form; repeatable. Defaults to the 12-trip MATLAB sidecar bundle.",
    )
    _add_output_dir_arg(parser)
    args = parser.parse_args()

    trips = tuple(args.trips) if args.trips else DEFAULT_FACTOR_MASK_PARITY_TRIPS
    out_dir = _timestamped_output_dir(_resolved_output_root(args), "gsdc2023_residual_diagnostics_sidecar_audit")
    records = diagnostics_trip_records(Path(args.data_root).resolve(), trips)
    roles = pd.DataFrame([asdict(row) for row in diagnostics_column_roles()])
    summary = summary_from_records(records)
    summary.update({"data_root": str(Path(args.data_root).resolve()), "trips": list(trips)})
    records.to_csv(out_dir / "trip_summary.csv", index=False)
    roles.to_csv(out_dir / "column_roles.csv", index=False)
    _write_summary_json(out_dir, summary)
    _print_summary_and_output_dir(summary, out_dir, label="audit_dir")


if __name__ == "__main__":
    main()
