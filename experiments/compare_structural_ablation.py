#!/usr/bin/env python3
"""Compare two structural-ablation summaries on identical evaluation scopes."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


KEYS = ("scope", "scope_id", "evaluation_role")
METRICS = (
    "epochs",
    "reference_coverage",
    "pass_0p5",
    "pass_1m",
    "pass_3m",
    "error_p50_m",
    "error_p95_m",
    "error_p99_m",
    "ms_per_epoch",
    "mode_abstention_rate",
    "ffbsi_abstention_rate",
    "doppler_update_rate",
    "doppler_known_rows_mean",
    "doppler_unknown_rows_mean",
    "doppler_clock_groups_mean",
    "doppler_clock_fit_rms_p95_mps",
    "doppler_clock_drift_span_p95_mps",
)


def compare_summaries(baseline: pd.DataFrame, candidate: pd.DataFrame) -> pd.DataFrame:
    missing = [key for key in KEYS if key not in baseline or key not in candidate]
    if missing:
        raise ValueError(f"summary is missing comparison keys: {missing}")
    merged = baseline.merge(candidate, on=list(KEYS), how="inner", suffixes=("_baseline", "_candidate"))
    if merged.empty:
        raise ValueError("summaries have no identical scopes")
    output = merged.loc[:, list(KEYS)].copy()
    for metric in METRICS:
        baseline_name = f"{metric}_baseline"
        candidate_name = f"{metric}_candidate"
        if baseline_name not in merged or candidate_name not in merged:
            continue
        baseline_values = pd.to_numeric(merged[baseline_name], errors="coerce")
        candidate_values = pd.to_numeric(merged[candidate_name], errors="coerce")
        output[baseline_name] = baseline_values
        output[candidate_name] = candidate_values
        output[f"{metric}_delta"] = candidate_values - baseline_values
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = compare_summaries(pd.read_csv(args.baseline), pd.read_csv(args.candidate))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Saved {len(result)} matched scopes: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
