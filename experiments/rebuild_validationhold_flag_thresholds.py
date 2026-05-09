#!/usr/bin/env python3
"""Rebuild validationhold lift/reject flags in a window summary CSV using
configurable thresholds.

Input is `ppc_validationhold_window_summary.csv` (which already has
`base_pred_fix_rate_pct`, `validation_*_frac`, `hold_*_frac`,
`validation_block_score_*`, and the original `validationhold_*_signal`
columns).  Output is a modified summary CSV with new `*_signal` columns.

The original §7.7 flags used hand-tuned thresholds (validation_pass_frac
>= 0.60, hold_ready_frac >= 0.55, hold_strict_ready_frac >= 0.45,
validation_block_score_p90 <= 1.25).  §7.8 flagged that these thresholds
were informed by Tokyo run2 behavior.

Downstream: pass the output CSV to
`augment_ppc_windows_with_validationhold_features.py --validationhold-csv`
to build an augmented window CSV with the modified flags.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _compute_flags(
    df: pd.DataFrame,
    *,
    pass_thr: float,
    hold_ready_thr: float,
    hold_strict_thr: float,
    block_p90_thr: float,
    reject_block_thr: float,
    high_pred_thr: float = 50.0,
    low_pred_max: float = 30.0,
    low_pred_min: float = 12.0,
    reject_block_score_max_thr: float = 10.0,
    reject_pass_max: float = 0.05,
    reject_hold_max: float = 0.05,
    reject_block_frac_min: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    base_pred = df["base_pred_fix_rate_pct"].to_numpy(dtype=np.float64)
    block_score_max = df["validation_block_score_max"].to_numpy(dtype=np.float64)
    reject_block_frac = df["validation_reject_block_frac"].to_numpy(dtype=np.float64)
    pass_frac = df["validation_pass_frac"].to_numpy(dtype=np.float64)
    hold_ready_frac = df["hold_ready_frac"].to_numpy(dtype=np.float64)
    hold_strict_frac = df["hold_strict_ready_frac"].to_numpy(dtype=np.float64)
    block_p90 = df["validation_block_score_p90"].to_numpy(dtype=np.float64)

    high_pred = base_pred >= high_pred_thr
    low_pred = (base_pred <= low_pred_max) & (base_pred >= low_pred_min)

    reject = high_pred & (
        (block_score_max >= reject_block_score_max_thr)
        | (
            (reject_block_frac >= reject_block_frac_min)
            & (pass_frac <= reject_pass_max)
            & (hold_ready_frac <= reject_hold_max)
        )
    )
    lift = (
        low_pred
        & (pass_frac >= pass_thr)
        & (hold_ready_frac >= hold_ready_thr)
        & (hold_strict_frac >= hold_strict_thr)
        & (reject_block_frac <= reject_block_thr)
        & (block_p90 <= block_p90_thr)
    )
    return reject.astype(np.float64), lift.astype(np.float64)


PRESETS = {
    "strict": dict(pass_thr=0.65, hold_ready_thr=0.65, hold_strict_thr=0.55, block_p90_thr=1.0, reject_block_thr=0.0),
    "current": dict(pass_thr=0.60, hold_ready_thr=0.55, hold_strict_thr=0.45, block_p90_thr=1.25, reject_block_thr=0.0),
    "relaxed": dict(pass_thr=0.50, hold_ready_thr=0.45, hold_strict_thr=0.35, block_p90_thr=1.5, reject_block_thr=0.0),
    "mid": dict(pass_thr=0.60, hold_ready_thr=0.60, hold_strict_thr=0.50, block_p90_thr=1.15, reject_block_thr=0.0),
    "current_tight_block": dict(pass_thr=0.60, hold_ready_thr=0.55, hold_strict_thr=0.45, block_p90_thr=1.00, reject_block_thr=0.0),
    "current_tight_hold": dict(pass_thr=0.60, hold_ready_thr=0.60, hold_strict_thr=0.45, block_p90_thr=1.25, reject_block_thr=0.0),
    "hold_ready_0p57": dict(pass_thr=0.60, hold_ready_thr=0.57, hold_strict_thr=0.45, block_p90_thr=1.25, reject_block_thr=0.0),
    "hold_ready_0p58": dict(pass_thr=0.60, hold_ready_thr=0.58, hold_strict_thr=0.45, block_p90_thr=1.25, reject_block_thr=0.0),
    "hold_ready_0p62": dict(pass_thr=0.60, hold_ready_thr=0.62, hold_strict_thr=0.45, block_p90_thr=1.25, reject_block_thr=0.0),
    # Variants tightening one more dimension on top of the §7.13 current_tight_hold baseline
    "tight_hold_pass65": dict(pass_thr=0.65, hold_ready_thr=0.60, hold_strict_thr=0.45, block_p90_thr=1.25, reject_block_thr=0.0),
    "tight_hold_strict55": dict(pass_thr=0.60, hold_ready_thr=0.60, hold_strict_thr=0.55, block_p90_thr=1.25, reject_block_thr=0.0),
    "tight_hold_blockp90_1p00": dict(pass_thr=0.60, hold_ready_thr=0.60, hold_strict_thr=0.45, block_p90_thr=1.00, reject_block_thr=0.0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild validationhold flag thresholds on the window summary CSV")
    parser.add_argument("--input-csv", type=Path, default=RESULTS_DIR / "ppc_validationhold_window_summary.csv")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="current")
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_csv)
    params = PRESETS[args.preset]
    reject, lift = _compute_flags(df, **params)
    df["validationhold_high_pred_reject_signal"] = reject
    df["validationhold_low_pred_lift_signal"] = lift
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"preset={args.preset} params={params}")
    print(f"reject fires on {int(reject.sum())} windows, lift fires on {int(lift.sum())} windows")
    print(f"saved: {args.output_csv}")


if __name__ == "__main__":
    main()
