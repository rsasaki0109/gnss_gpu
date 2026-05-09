#!/usr/bin/env python3
"""Compare MATLAB ``phone_data`` raw observation counts with ``gnss_log.txt``.

This is an audit tool for the first stage of the MATLAB preprocessing path:
``gnss_log.txt`` -> ``GobsPhone`` raw L1/L5 P/D/L availability.  It intentionally
stops before navigation residual masking.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

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
from experiments.gsdc2023_gnss_log_reader import gnss_log_observation_counts
from experiments.gsdc2023_raw_bridge import DEFAULT_ROOT


_FIELDS = ("P", "D", "L")
_FREQS = ("L1", "L5")


def _load_phone_observation_counts(path: Path) -> dict[str, dict[str, int]]:
    frame = pd.read_csv(path)
    required = {"freq", "field", "count"}
    if not required.issubset(frame.columns):
        raise ValueError(f"observation count CSV must contain columns {sorted(required)}: {path}")
    counts: dict[str, dict[str, int]] = {freq: {} for freq in _FREQS}
    for row in frame.itertuples(index=False):
        freq = str(getattr(row, "freq", "")).strip()
        field = str(getattr(row, "field", "")).strip()
        if freq in _FREQS and field in _FIELDS:
            counts[freq][field] = int(getattr(row, "count"))
    return counts


def compare_gnss_log_observation_counts(
    trip_dir: Path,
    *,
    phone_counts_path: Path | None = None,
    gps_only: bool = False,
    apply_signal_mask: bool = False,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if phone_counts_path is None:
        phone_counts_path = trip_dir / "phone_data_observation_counts.csv"
    phone_counts = _load_phone_observation_counts(phone_counts_path)
    log_path = trip_dir / "supplemental" / "gnss_log.txt"
    log_counts = gnss_log_observation_counts(
        log_path,
        gps_only=gps_only,
        apply_signal_mask=apply_signal_mask,
        phone=trip_dir.name,
    )

    rows: list[dict[str, object]] = []
    phone_total = 0
    log_total = 0
    abs_delta = 0
    for freq in _FREQS:
        for field in _FIELDS:
            phone_count = int(phone_counts.get(freq, {}).get(field, 0))
            parsed_count = int(log_counts.get(freq, {}).get(field, 0))
            delta = parsed_count - phone_count
            rows.append(
                {
                    "freq": freq,
                    "field": field,
                    "phone_count": phone_count,
                    "gnss_log_count": parsed_count,
                    "count_delta": delta,
                },
            )
            phone_total += phone_count
            log_total += parsed_count
            abs_delta += abs(delta)

    summary = {
        "trip_dir": str(trip_dir),
        "gnss_log_path": str(log_path),
        "phone_counts_path": str(phone_counts_path),
        "gps_only": bool(gps_only),
        "apply_signal_mask": bool(apply_signal_mask),
        "matched_rows": int(len(rows)),
        "phone_count_total": int(phone_total),
        "gnss_log_count_total": int(log_total),
        "matched_abs_delta_total": int(abs_delta),
        "count_parity_ratio": float(max(0.0, 1.0 - abs_delta / phone_total)) if phone_total > 0 else None,
    }
    return pd.DataFrame(rows), summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _add_data_root_trip_args(parser, default_root=DEFAULT_ROOT)
    parser.add_argument("--phone-counts", type=Path, default=None)
    parser.add_argument("--gps-only", action="store_true")
    parser.add_argument(
        "--signal-mask",
        action="store_true",
        help="apply MATLAB exobs.m signal/status masks before counting",
    )
    _add_output_dir_arg(parser)
    args = parser.parse_args()

    trip_dir = _resolve_trip_dir(args)
    out_dir = _timestamped_output_dir(_resolved_output_root(args), "gsdc2023_gnss_log_observation_count_parity")

    comparison, summary = compare_gnss_log_observation_counts(
        trip_dir,
        phone_counts_path=args.phone_counts,
        gps_only=args.gps_only,
        apply_signal_mask=args.signal_mask,
    )
    comparison.to_csv(out_dir / "count_comparison.csv", index=False)
    _write_summary_json(out_dir, summary)
    _print_summary_and_output_dir(summary, out_dir)


if __name__ == "__main__":
    main()
