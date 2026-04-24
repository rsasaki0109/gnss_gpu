#!/usr/bin/env python3
"""Merge validationhold window-aggregated features into a PPC window CSV.

This produces a window_predictions-style CSV that carries the original base
prediction + all deployable simulator/RINEX features, plus validation/hold
state aggregates from `ppc_validationhold_window_summary.csv`, so that the
existing strict nested transition surrogate stack can see those signals as
features.

Only numeric validation/hold columns are merged.  Diagnostic prediction
columns (`validationhold_*_pred_pct`) and label-like columns
(`hidden_high_case`, `false_high_case`, `base_abs_error_pp`) are dropped so
they cannot leak into training.

`--mode` selects which feature package to emit:

  - flags (default): the hand-tuned binary reject/lift flags plus the 22
    continuous validation/hold aggregates.  Reproduces §7.7.
  - noflags: same as flags but drops the two binary flag columns.
    Reproduces the §7.8 ablation.
  - compound: drops the flags and adds six continuous engineered compound
    features that try to reproduce the AND/OR decision surface of the
    flag rules through products and differences.
  - compound_flags: adds the compound features on top of the flags.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_WINDOW_CSV = (
    RESULTS_DIR
    / "ppc_window_fix_rate_model_stride1_stat_sim_rinex_phasejump_t0p25_gf0p2_simloscont_focused_simadop_nowt_windowopt_window_predictions.csv"
)
DEFAULT_VH_CSV = RESULTS_DIR / "ppc_validationhold_window_summary.csv"
MODE_OUTPUTS = {
    "flags": RESULTS_DIR
    / "ppc_window_fix_rate_model_stride1_stat_sim_rinex_phasejump_t0p25_gf0p2_simloscont_focused_simadop_nowt_windowopt_validationhold_window_predictions.csv",
    "noflags": RESULTS_DIR
    / "ppc_window_fix_rate_model_stride1_stat_sim_rinex_phasejump_t0p25_gf0p2_simloscont_focused_simadop_nowt_windowopt_validationhold_noflags_window_predictions.csv",
    "compound": RESULTS_DIR
    / "ppc_window_fix_rate_model_stride1_stat_sim_rinex_phasejump_t0p25_gf0p2_simloscont_focused_simadop_nowt_windowopt_validationhold_compound_window_predictions.csv",
    "compound_flags": RESULTS_DIR
    / "ppc_window_fix_rate_model_stride1_stat_sim_rinex_phasejump_t0p25_gf0p2_simloscont_focused_simadop_nowt_windowopt_validationhold_compound_flags_window_predictions.csv",
}

# keep only deployable validation/hold state aggregates
VH_KEEP_COLUMNS = [
    "validation_pass_frac",
    "validation_soft_pass_frac",
    "validation_hard_block_frac",
    "validation_severe_block_frac",
    "validation_reject_block_frac",
    "validation_block_spike_frac",
    "validation_block_score_mean",
    "validation_block_score_p90",
    "validation_block_score_max",
    "validation_block_ewma30_mean",
    "validation_block_cooldown_max",
    "validation_reject_recent_max",
    "validation_quality_mean",
    "validation_quality_p90",
    "hold_state_mean",
    "hold_state_max",
    "hold_ready_frac",
    "hold_strict_ready_frac",
    "hold_carry_score_mean",
    "hold_carry_score_max",
    "first_validation_pass_rel_s",
    "first_hold_ready_rel_s",
    "clean_streak_s_at_start",
    "clean_streak_s_mean",
    "clean_streak_s_p50",
    "clean_streak_s_p90",
    "clean_streak_s_max",
    "strict_clean_streak_s_at_start",
    "strict_clean_streak_s_mean",
    "strict_clean_streak_s_p50",
    "strict_clean_streak_s_p90",
    "strict_clean_streak_s_max",
    # Section 7.17 null: hold_age_s / hold_since_reset_s window aggregates are
    # still emitted at window-summary level for diagnostics, but intentionally
    # NOT forwarded into the augmented window CSV because they dilute ridge
    # feature sampling and degrade strict-LORO metrics.
    "validationhold_high_pred_reject_signal",
    "validationhold_low_pred_lift_signal",
]

FLAG_COLUMNS = [
    "validationhold_high_pred_reject_flag",
    "validationhold_low_pred_lift_flag",
]

# continuous compound features trying to reproduce §7.4 flag logic without
# discrete thresholds.  All inputs are deployable per-window aggregates; the
# compounds never use cross-window statistics or demo5 solver labels.
COMPOUND_FEATURES = [
    "vh_readiness_net",
    "vh_readiness_carry",
    "vh_readiness_depth",
    "vh_validation_clean",
    "vh_phase_deep",
    "vh_phase_jump_risk",
]


def _compute_compound_features(df: pd.DataFrame) -> dict[str, np.ndarray]:
    def col(name: str) -> np.ndarray:
        if name in df.columns:
            values = df[name].to_numpy(dtype=np.float64)
            values[~np.isfinite(values)] = 0.0
            return values
        print(f"compound input missing (substituting zero): {name}")
        return np.zeros(len(df), dtype=np.float64)

    hold_strict = col("hold_strict_ready_frac")
    hold_ready = col("hold_ready_frac")
    hold_carry_mean = col("hold_carry_score_mean")
    pass_frac = col("validation_pass_frac")
    reject_frac = col("validation_reject_block_frac")
    spike_frac = col("validation_block_spike_frac")
    adop_depth_min = col("sim_adop_cont_ge90p0s_count_min")
    phase_streak_depth = col("rinex_phase_streak_ge60p0s_fraction_p10_past_mean")
    phase_jump_risk = col("rinex_phase_jump_ge0p25cy_count_max_past_mean")

    return {
        "vh_readiness_net": hold_strict * pass_frac * (1.0 - np.clip(reject_frac, 0.0, 1.0)),
        "vh_readiness_carry": hold_strict * hold_carry_mean,
        "vh_readiness_depth": hold_strict * adop_depth_min,
        "vh_validation_clean": pass_frac - reject_frac - spike_frac,
        "vh_phase_deep": phase_streak_depth * hold_strict,
        "vh_phase_jump_risk": phase_jump_risk * (1.0 - hold_strict),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge validationhold features into PPC window CSV")
    parser.add_argument("--window-csv", type=Path, default=DEFAULT_WINDOW_CSV)
    parser.add_argument("--validationhold-csv", type=Path, default=DEFAULT_VH_CSV)
    parser.add_argument("--mode", choices=sorted(MODE_OUTPUTS), default="flags")
    parser.add_argument("--output-csv", type=Path, default=None, help="override mode-default output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = pd.read_csv(args.window_csv)
    vh = pd.read_csv(args.validationhold_csv)

    available = [col for col in VH_KEEP_COLUMNS if col in vh.columns]
    missing = [col for col in VH_KEEP_COLUMNS if col not in vh.columns]
    if missing:
        print(f"missing in validationhold CSV (skipped): {missing}")

    key_cols = ["city", "run", "window_index"]
    vh_features = vh[key_cols + available].copy()
    renames = {
        "validationhold_high_pred_reject_signal": "validationhold_high_pred_reject_flag",
        "validationhold_low_pred_lift_signal": "validationhold_low_pred_lift_flag",
    }
    vh_features = vh_features.rename(columns=renames)

    feature_names = [renames.get(name, name) for name in available]

    overlap = [col for col in feature_names if col in base.columns]
    if overlap:
        print(f"columns already present in base CSV (keeping base values): {overlap}")
        vh_features = vh_features.drop(columns=overlap)
        feature_names = [name for name in feature_names if name not in overlap]

    merged = base.merge(vh_features, on=key_cols, how="left")
    # first_*_rel_s columns carry a semantic NaN meaning "event never occurred in this window";
    # zero-fill is applied for those but they are excluded from the data-quality check below.
    semantic_nan_cols = {"first_validation_pass_rel_s", "first_hold_ready_rel_s"}
    quality_cols = [c for c in feature_names if c not in semantic_nan_cols]
    if quality_cols:
        nan_rows = merged[quality_cols].isna().any(axis=1).sum()
        nan_fraction = nan_rows / max(len(merged), 1)
        if nan_fraction > 0.10:
            print(
                f"WARNING: {nan_rows}/{len(merged)} rows ({100*nan_fraction:.1f}%) are missing "
                f"at least one validationhold feature (excluding semantic-NaN first_*_rel_s) "
                f"after the left-join with {args.validationhold_csv.name}.  Zeros will be "
                f"filled in but this may degrade training.  Check whether the validationhold "
                f"summary CSV covers the same (city, run, window_index) set as the base window CSV."
            )
        elif nan_rows > 0:
            print(f"note: {nan_rows} rows have missing validationhold features, filling with 0")
    merged[feature_names] = merged[feature_names].fillna(0.0)

    drop_flags = args.mode in {"noflags", "compound"}
    add_compound = args.mode in {"compound", "compound_flags"}

    if drop_flags:
        drop = [col for col in FLAG_COLUMNS if col in merged.columns]
        if drop:
            merged = merged.drop(columns=drop)
            feature_names = [name for name in feature_names if name not in drop]
            print(f"dropped flag columns: {drop}")

    compound_added: list[str] = []
    if add_compound:
        compounds = _compute_compound_features(merged)
        for name, values in compounds.items():
            if name in merged.columns:
                print(f"compound column already present (skipping): {name}")
                continue
            merged[name] = values
            compound_added.append(name)
        print(f"added compound features: {compound_added}")

    output_path = args.output_csv or MODE_OUTPUTS[args.mode]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    print(f"saved: {output_path}")
    print(f"rows={len(merged)} base_columns={len(base.columns)} vh_added={len(feature_names)} compound_added={len(compound_added)}")


if __name__ == "__main__":
    main()
