#!/usr/bin/env python3
"""Normalize WP174 partial-AR shadow rows for the FFRT calibration analyzer."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def normalize(row: dict[str, str]) -> dict[str, str]:
    output = dict(row)
    output["pair_count"] = row.get("lambda_src_par_shadow_subset_size", "")
    output["shadow_best_sub50cm"] = row.get("src_par_best_sub50cm", "")
    output["shadow_best_error_m"] = row.get("src_par_best_error_m", "")
    output["lambda_shadow_ratio"] = row.get(
        "lambda_src_par_shadow_ratio", ""
    )
    output["lambda_shadow_bsr"] = row.get("lambda_src_par_shadow_bsr", "")
    output["lambda_shadow_bsr_qscale2"] = ""
    output["lambda_shadow_bsr_qscale4"] = ""
    output["lambda_shadow_bsr_qscale8"] = ""
    output["lambda_shadow_bsr_qscale16"] = row.get(
        "lambda_src_par_shadow_bsr", ""
    )
    output["lambda_shadow_second_position_delta_m"] = row.get(
        "lambda_src_par_shadow_second_position_delta_m", ""
    )
    for axis in "xyz":
        output[f"lambda_shadow_best_ecef_{axis}"] = row.get(
            f"lambda_src_par_shadow_best_ecef_{axis}", ""
        )
        output[f"lambda_shadow_best_correction_{axis}"] = row.get(
            f"lambda_src_par_shadow_best_correction_{axis}", ""
        )
    return output


def normalize_satellite(row: dict[str, str]) -> dict[str, str]:
    output = dict(row)
    output["pair_count"] = row.get(
        "lambda_satellite_par_shadow_subset_size", ""
    )
    output["shadow_best_sub50cm"] = row.get(
        "satellite_par_best_sub50cm", ""
    )
    output["shadow_best_error_m"] = row.get(
        "satellite_par_best_error_m", ""
    )
    output["lambda_shadow_attempted"] = row.get(
        "lambda_satellite_par_shadow_attempted", "0"
    )
    output["lambda_shadow_solved"] = row.get(
        "lambda_satellite_par_shadow_solved", "0"
    )
    output["lambda_shadow_ratio"] = row.get(
        "lambda_satellite_par_shadow_ratio", ""
    )
    output["lambda_shadow_bsr"] = ""
    output["lambda_shadow_bsr_qscale2"] = ""
    output["lambda_shadow_bsr_qscale4"] = ""
    output["lambda_shadow_bsr_qscale8"] = ""
    output["lambda_shadow_bsr_qscale16"] = row.get(
        "lambda_satellite_par_shadow_bsr", ""
    )
    output["lambda_shadow_second_position_delta_m"] = row.get(
        "lambda_satellite_par_shadow_second_position_delta_m", ""
    )
    output["lambda_shadow_runtime_ms"] = row.get(
        "lambda_satellite_par_shadow_runtime_ms", ""
    )
    for axis in "xyz":
        output[f"lambda_shadow_best_ecef_{axis}"] = row.get(
            f"lambda_satellite_par_shadow_best_ecef_{axis}", ""
        )
        output[f"lambda_shadow_best_correction_{axis}"] = row.get(
            f"lambda_satellite_par_shadow_best_correction_{axis}", ""
        )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source",
        choices=("src", "satellite"),
        default="src",
    )
    args = parser.parse_args()
    with args.input.open(newline="", encoding="utf-8-sig") as stream:
        normalizer = (
            normalize_satellite if args.source == "satellite" else normalize
        )
        rows = [normalizer(row) for row in csv.DictReader(stream)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"normalized {len(rows)} {args.source} PAR audit rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
