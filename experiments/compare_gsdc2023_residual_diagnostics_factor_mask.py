#!/usr/bin/env python3
"""Rebuild MATLAB factor keys from residual diagnostics and compare masks."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from experiments.gsdc2023_raw_bridge import DEFAULT_ROOT
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
from experiments.gsdc2023_factor_mask import (
    build_factor_mask_from_residual_diagnostics,
    factor_mask_side_summary as _factor_mask_side_summary,
    merge_factor_mask_keys as _merge_factor_mask_keys,
    normalize_factor_mask_frame as _normalize_factor_mask_frame,
)


def compare_residual_diagnostics_factor_mask(
    trip_dir: Path,
    *,
    diagnostics_path: Path | None = None,
    factor_mask_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    if diagnostics_path is None:
        diagnostics_path = trip_dir / "phone_data_residual_diagnostics.csv"
    if factor_mask_path is None:
        factor_mask_path = trip_dir / "phone_data_factor_mask.csv"

    expected = _normalize_factor_mask_frame(
        pd.read_csv(factor_mask_path),
        keep_extra_columns=False,
        missing_label="factor mask",
    )
    rebuilt = _normalize_factor_mask_frame(
        build_factor_mask_from_residual_diagnostics(diagnostics_path),
        keep_extra_columns=False,
        missing_label="factor mask",
    )
    merged = _merge_factor_mask_keys(
        expected,
        rebuilt,
        left_only_side="factor_mask_only",
        right_only_side="diagnostics_only",
    )
    summary, side_payload = _factor_mask_side_summary(
        merged,
        left_name="factor_mask",
        right_name="diagnostics",
        left_only_side="factor_mask_only",
        right_only_side="diagnostics_only",
    )
    payload = {
        "trip_dir": str(trip_dir),
        "diagnostics_path": str(diagnostics_path),
        "factor_mask_path": str(factor_mask_path),
        "total_factor_mask_count": side_payload["total_factor_mask_count"],
        "total_diagnostics_count": side_payload["total_diagnostics_count"],
        "total_matched_count": side_payload["total_matched_count"],
        "total_factor_mask_only": side_payload["total_factor_mask_only"],
        "total_diagnostics_only": side_payload["total_diagnostics_only"],
        "symmetric_parity": side_payload["symmetric_parity"],
    }
    return merged, summary, payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _add_data_root_trip_args(parser, default_root=DEFAULT_ROOT)
    parser.add_argument("--diagnostics", type=Path, default=None)
    parser.add_argument("--factor-mask", type=Path, default=None)
    _add_output_dir_arg(parser)
    args = parser.parse_args()

    trip_dir = _resolve_trip_dir(args)
    out_dir = _timestamped_output_dir(
        _resolved_output_root(args),
        "gsdc2023_residual_diagnostics_factor_mask_parity",
    )

    merged, summary, payload = compare_residual_diagnostics_factor_mask(
        trip_dir,
        diagnostics_path=args.diagnostics,
        factor_mask_path=args.factor_mask,
    )
    merged.to_csv(out_dir / "factor_mask_join.csv", index=False)
    merged[merged["side"] == "factor_mask_only"].to_csv(out_dir / "factor_mask_only.csv", index=False)
    merged[merged["side"] == "diagnostics_only"].to_csv(out_dir / "diagnostics_only.csv", index=False)
    summary.to_csv(out_dir / "summary_by_field.csv", index=False)
    _write_summary_json(out_dir, payload)
    _print_summary_and_output_dir(payload, out_dir)


if __name__ == "__main__":
    main()
