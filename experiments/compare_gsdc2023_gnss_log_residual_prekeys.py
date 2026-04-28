#!/usr/bin/env python3
"""Compare ``gnss_log`` signal-mask keys with MATLAB pre-residual finite keys.

``phone_data_residual_diagnostics.csv`` records the availability immediately
around MATLAB ``exobs_residuals``.  This tool compares its ``p_pre_finite`` and
``d_pre_finite`` keys against the Python lightweight ``gnss_log`` reader after
the MATLAB ``exobs.m`` signal/status masks.  The remaining deltas isolate keys
lost inside ``gt.Gsat(...).residuals(...)`` before residual thresholding.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from experiments.gsdc2023_audit_output import (
    print_summary_and_output_dir as _print_summary_and_output_dir,
    timestamped_output_dir as _timestamped_output_dir,
    write_summary_json as _write_summary_json,
)
from experiments.gsdc2023_audit_cli import (
    add_data_root_trip_args as _add_data_root_trip_args,
    add_output_dir_arg as _add_output_dir_arg,
    resolve_trip_dir as _resolve_trip_dir,
    resolved_output_root as _resolved_output_root,
)
from experiments.gsdc2023_gnss_log_reader import gnss_log_signal_mask_frame
from experiments.gsdc2023_raw_bridge import DEFAULT_ROOT


_KEY_COLUMNS = ["field", "freq", "epoch_index", "utcTimeMillis", "sys", "svid"]


def _matlab_pre_residual_frame(diagnostics_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(diagnostics_path)
    required = {"freq", "epoch_index", "utcTimeMillis", "sys", "svid", "p_pre_finite", "d_pre_finite"}
    if not required.issubset(frame.columns):
        raise ValueError(f"diagnostics CSV missing required columns: {sorted(required - set(frame.columns))}")

    rows: list[pd.DataFrame] = []
    for field, column in (("P", "p_pre_finite"), ("D", "d_pre_finite")):
        sub = frame.loc[frame[column].astype(bool), ["freq", "epoch_index", "utcTimeMillis", "sys", "svid"]].copy()
        if sub.empty:
            continue
        sub.insert(0, "field", field)
        rows.append(sub)
    if not rows:
        return pd.DataFrame(columns=_KEY_COLUMNS)
    out = pd.concat(rows, ignore_index=True)
    for col in ("epoch_index", "utcTimeMillis", "sys", "svid"):
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(np.int64)
    out["field"] = out["field"].astype(str)
    out["freq"] = out["freq"].astype(str)
    return out.drop_duplicates(_KEY_COLUMNS)


def compare_gnss_log_residual_prekeys(
    trip_dir: Path,
    *,
    diagnostics_path: Path | None = None,
    gps_only: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    if diagnostics_path is None:
        diagnostics_path = trip_dir / "phone_data_residual_diagnostics.csv"
    matlab_keys = _matlab_pre_residual_frame(diagnostics_path)
    log_keys = gnss_log_signal_mask_frame(
        trip_dir / "supplemental" / "gnss_log.txt",
        phone=trip_dir.name,
        gps_only=gps_only,
    )
    log_keys = log_keys[log_keys["field"].isin(("P", "D"))].copy()

    merged = matlab_keys[_KEY_COLUMNS].merge(log_keys[_KEY_COLUMNS], on=_KEY_COLUMNS, how="outer", indicator=True)
    merged["side"] = merged["_merge"].map(
        {
            "left_only": "matlab_only",
            "right_only": "gnss_log_only",
            "both": "both",
        },
    )
    merged = merged.drop(columns=["_merge"])

    summary_rows: list[dict[str, object]] = []
    for (field, freq), group in merged.groupby(["field", "freq"], sort=True):
        both = int(np.count_nonzero(group["side"] == "both"))
        matlab_only = int(np.count_nonzero(group["side"] == "matlab_only"))
        log_only = int(np.count_nonzero(group["side"] == "gnss_log_only"))
        matlab_count = both + matlab_only
        log_count = both + log_only
        denom = max(matlab_count, log_count)
        summary_rows.append(
            {
                "field": field,
                "freq": freq,
                "matlab_pre_count": matlab_count,
                "gnss_log_signal_count": log_count,
                "matched_count": both,
                "matlab_only": matlab_only,
                "gnss_log_only": log_only,
                "symmetric_parity": float(both / denom) if denom > 0 else None,
            },
        )
    summary = pd.DataFrame(summary_rows)
    total_both = int(np.count_nonzero(merged["side"] == "both"))
    total_matlab_only = int(np.count_nonzero(merged["side"] == "matlab_only"))
    total_log_only = int(np.count_nonzero(merged["side"] == "gnss_log_only"))
    payload = {
        "trip_dir": str(trip_dir),
        "diagnostics_path": str(diagnostics_path),
        "gps_only": bool(gps_only),
        "total_matlab_pre_count": int(total_both + total_matlab_only),
        "total_gnss_log_signal_count": int(total_both + total_log_only),
        "total_matched_count": total_both,
        "total_matlab_only": total_matlab_only,
        "total_gnss_log_only": total_log_only,
        "symmetric_parity": (
            float(total_both / max(total_both + total_matlab_only, total_both + total_log_only))
            if total_both + total_matlab_only + total_log_only > 0
            else None
        ),
    }
    return merged, summary, payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _add_data_root_trip_args(parser, default_root=DEFAULT_ROOT)
    parser.add_argument("--diagnostics", type=Path, default=None)
    parser.add_argument("--gps-only", action=argparse.BooleanOptionalAction, default=True)
    _add_output_dir_arg(parser)
    args = parser.parse_args()

    trip_dir = _resolve_trip_dir(args)
    out_dir = _timestamped_output_dir(_resolved_output_root(args), "gsdc2023_gnss_log_residual_prekey_parity")

    merged, summary, payload = compare_gnss_log_residual_prekeys(
        trip_dir,
        diagnostics_path=args.diagnostics,
        gps_only=args.gps_only,
    )
    merged.to_csv(out_dir / "prekey_join.csv", index=False)
    merged[merged["side"] == "matlab_only"].to_csv(out_dir / "matlab_only.csv", index=False)
    merged[merged["side"] == "gnss_log_only"].to_csv(out_dir / "gnss_log_only.csv", index=False)
    summary.to_csv(out_dir / "summary_by_field.csv", index=False)
    _write_summary_json(out_dir, payload)
    _print_summary_and_output_dir(payload, out_dir)


if __name__ == "__main__":
    main()
