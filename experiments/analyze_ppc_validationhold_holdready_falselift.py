#!/usr/bin/env python3
"""Contrast deployable features between hold-ready actual-high and actual-low windows.

This is the §7.6 false-lift contrast analogous to the Tokyo run2 onset scan.
We partition windows that look "hold-ready" according to the validationhold
surrogate into three outcome groups based on demo5 actual FIX rate:

  - actual_high (actual_fix_rate_pct >= 75.0)
  - actual_mid  (20.0 < actual_fix_rate_pct < 75.0)
  - actual_low  (actual_fix_rate_pct <= 20.0)

Two hold-ready subsets are used:

  - lift_signal: windows where validationhold_low_pred_lift_signal fires
  - hold_ready_broad: hold_ready_frac >= 0.45 and base_pred_fix_rate_pct <= 30.0

For each subset we emit per-window rows, a small group summary, and a
feature-level separation rank between actual_high and actual_low within that
subset.  Actual labels are used as diagnostic targets only; ranked features
are restricted to deployable simulator / RINEX / validationhold features.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd

from _common import _is_metadata_or_label


RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_VH_WINDOW_CSV = RESULTS_DIR / "ppc_validationhold_window_summary.csv"
DEFAULT_BASE_WINDOW_CSV = (
    RESULTS_DIR
    / "ppc_window_fix_rate_model_stride1_stat_sim_rinex_phasejump_t0p25_gf0p2_simloscont_focused_simadop_nowt_windowopt_window_predictions.csv"
)
DEFAULT_PREFIX = "ppc_validationhold_holdready_falselift"

VH_FEATURE_COLS = [
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
]

SNAPSHOT_FEATURES = [
    "hold_ready_frac",
    "hold_strict_ready_frac",
    "hold_carry_score_mean",
    "validation_pass_frac",
    "validation_reject_block_frac",
    "validation_block_spike_frac",
    "validation_block_score_max",
    "validation_block_score_p90",
    "validation_quality_mean",
    "first_validation_pass_rel_s",
    "first_hold_ready_rel_s",
]


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        print(f"saved empty: {path}")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved: {path} ({len(rows)} rows)")


def _key_frame(df: pd.DataFrame) -> pd.Series:
    return df["city"].astype(str) + "\t" + df["run"].astype(str) + "\t" + df["window_index"].astype(int).astype(str)


def _safe_mean(values: pd.Series) -> float:
    arr = values.to_numpy(dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def _safe_std(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.std(arr)) if arr.size else float("nan")


def _effect_size(pos: pd.Series, neg: pd.Series) -> float:
    p = pos.to_numpy(dtype=np.float64)
    n = neg.to_numpy(dtype=np.float64)
    p = p[np.isfinite(p)]
    n = n[np.isfinite(n)]
    if p.size == 0 or n.size == 0:
        return float("nan")
    pooled = np.sqrt((np.var(p) + np.var(n)) / 2.0)
    if pooled <= 1e-12:
        return 0.0
    return float((np.mean(p) - np.mean(n)) / pooled)


def _family(name: str) -> str:
    if name.startswith("validation_") or name.startswith("hold_") or name.startswith("validationhold_") or name.startswith("first_validation") or name.startswith("first_hold"):
        return "validationhold"
    if name.startswith("rinex_"):
        return "rinex"
    if name.startswith("sim_"):
        return "sim"
    for prefix in ("phase", "los", "nlos", "residual", "sat", "proxy", "adop"):
        if name.startswith(prefix):
            return prefix
    return "other"


def _actual_group(actual_pct: float, low_thr: float, high_thr: float) -> str:
    if not np.isfinite(actual_pct):
        return "unknown"
    if actual_pct <= low_thr:
        return "actual_low"
    if actual_pct >= high_thr:
        return "actual_high"
    return "actual_mid"


def _compute_separation(
    df: pd.DataFrame,
    feature_names: list[str],
    mask: pd.Series,
    actual_col: str,
    low_thr: float,
    high_thr: float,
) -> list[dict[str, object]]:
    subset = df.loc[mask].copy()
    if subset.empty:
        return []
    actual = subset[actual_col].to_numpy(dtype=np.float64)
    low_mask = actual <= low_thr
    high_mask = actual >= high_thr
    low = subset.loc[low_mask]
    high = subset.loc[high_mask]
    rows: list[dict[str, object]] = []
    for name in feature_names:
        if name not in subset.columns:
            continue
        if not pd.api.types.is_numeric_dtype(subset[name]):
            continue
        values = subset[name]
        high_vals = high[name] if not high.empty else pd.Series(dtype=float)
        low_vals = low[name] if not low.empty else pd.Series(dtype=float)
        high_mean = _safe_mean(high_vals)
        low_mean = _safe_mean(low_vals)
        pooled = _safe_std(values.to_numpy(dtype=np.float64))
        if np.isfinite(pooled) and pooled > 1e-12:
            norm_diff = (high_mean - low_mean) / pooled
        else:
            norm_diff = 0.0
        rows.append(
            {
                "feature": name,
                "family": _family(name),
                "subset_n": int(len(subset)),
                "actual_high_n": int(high_mask.sum()),
                "actual_low_n": int(low_mask.sum()),
                "actual_high_mean": high_mean,
                "actual_low_mean": low_mean,
                "mean_diff_high_minus_low": high_mean - low_mean,
                "pooled_std": pooled,
                "separation_score": norm_diff,
                "effect_size": _effect_size(high_vals, low_vals),
                "actual_high_min": float(np.min(high_vals.to_numpy(dtype=np.float64))) if not high.empty and np.isfinite(high_vals.to_numpy(dtype=np.float64)).any() else float("nan"),
                "actual_high_max": float(np.max(high_vals.to_numpy(dtype=np.float64))) if not high.empty and np.isfinite(high_vals.to_numpy(dtype=np.float64)).any() else float("nan"),
                "actual_low_min": float(np.min(low_vals.to_numpy(dtype=np.float64))) if not low.empty and np.isfinite(low_vals.to_numpy(dtype=np.float64)).any() else float("nan"),
                "actual_low_max": float(np.max(low_vals.to_numpy(dtype=np.float64))) if not low.empty and np.isfinite(low_vals.to_numpy(dtype=np.float64)).any() else float("nan"),
            }
        )
    rows.sort(key=lambda row: abs(row["separation_score"]) if np.isfinite(row["separation_score"]) else 0.0, reverse=True)
    return rows


def _group_summary(
    subset_label: str,
    df: pd.DataFrame,
    mask: pd.Series,
    actual_col: str,
    low_thr: float,
    high_thr: float,
) -> list[dict[str, object]]:
    subset = df.loc[mask]
    if subset.empty:
        return []
    groups = {
        "actual_high": subset[subset[actual_col] >= high_thr],
        "actual_low": subset[subset[actual_col] <= low_thr],
        "actual_mid": subset[(subset[actual_col] > low_thr) & (subset[actual_col] < high_thr)],
        "all": subset,
    }
    out: list[dict[str, object]] = []
    for name, group in groups.items():
        if group.empty:
            out.append({
                "subset": subset_label,
                "group": name,
                "count": 0,
            })
            continue
        out.append(
            {
                "subset": subset_label,
                "group": name,
                "count": int(len(group)),
                "actual_fix_rate_pct_mean": _safe_mean(group[actual_col]),
                "base_pred_fix_rate_pct_mean": _safe_mean(group["base_pred_fix_rate_pct"]),
                "hold_ready_frac_mean": _safe_mean(group["hold_ready_frac"]),
                "hold_strict_ready_frac_mean": _safe_mean(group["hold_strict_ready_frac"]),
                "hold_carry_score_mean_mean": _safe_mean(group["hold_carry_score_mean"]),
                "validation_pass_frac_mean": _safe_mean(group["validation_pass_frac"]),
                "validation_reject_block_frac_mean": _safe_mean(group["validation_reject_block_frac"]),
                "validation_block_spike_frac_mean": _safe_mean(group["validation_block_spike_frac"]),
                "validation_block_score_p90_mean": _safe_mean(group["validation_block_score_p90"]),
                "validation_block_score_max_mean": _safe_mean(group["validation_block_score_max"]),
                "validation_quality_mean_mean": _safe_mean(group["validation_quality_mean"]),
                "first_validation_pass_rel_s_mean": _safe_mean(group["first_validation_pass_rel_s"]),
                "first_hold_ready_rel_s_mean": _safe_mean(group["first_hold_ready_rel_s"]),
            }
        )
    return out


def _per_window_rows(
    df: pd.DataFrame,
    mask: pd.Series,
    actual_col: str,
    low_thr: float,
    high_thr: float,
    extra_features: list[str],
) -> list[dict[str, object]]:
    subset = df.loc[mask].copy()
    if subset.empty:
        return []
    subset = subset.sort_values([actual_col, "city", "run", "window_index"], ascending=[True, True, True, True])
    rows: list[dict[str, object]] = []
    base_cols = [
        "city",
        "run",
        "window_index",
        "actual_fix_rate_pct",
        "base_pred_fix_rate_pct",
        "validationhold_low_pred_lift_signal",
        "validationhold_high_pred_reject_signal",
        "validationhold_diag_pred_pct",
        "validationhold_diag_reason",
    ]
    for _, row in subset.iterrows():
        out: dict[str, object] = {}
        for col in base_cols:
            if col in row:
                out[col] = row[col]
        out["actual_group"] = _actual_group(float(row[actual_col]), low_thr, high_thr)
        for name in SNAPSHOT_FEATURES + extra_features:
            if name in row.index:
                try:
                    out[name] = float(row[name])
                except (TypeError, ValueError):
                    out[name] = row[name]
        rows.append(out)
    return rows


def _merge_base_features(vh: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    base = base.copy()
    base["__key__"] = _key_frame(base)
    vh = vh.copy()
    vh["__key__"] = _key_frame(vh)
    overlap = [col for col in base.columns if col in vh.columns and col not in {"__key__", "city", "run", "window_index"}]
    base = base.drop(columns=overlap)
    merged = vh.merge(base, on="__key__", how="left", suffixes=("", "_base"))
    merged = merged.drop(columns=["__key__"])
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Contrast hold-ready windows by actual FIX outcome")
    parser.add_argument("--validationhold-window-csv", type=Path, default=DEFAULT_VH_WINDOW_CSV)
    parser.add_argument("--base-window-csv", type=Path, default=DEFAULT_BASE_WINDOW_CSV)
    parser.add_argument("--results-prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--actual-low-thr", type=float, default=20.0)
    parser.add_argument("--actual-high-thr", type=float, default=75.0)
    parser.add_argument("--broad-hold-ready-thr", type=float, default=0.45)
    parser.add_argument("--broad-base-pred-max", type=float, default=30.0)
    parser.add_argument("--top-n", type=int, default=60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    vh = pd.read_csv(args.validationhold_window_csv)
    base = pd.read_csv(args.base_window_csv)
    df = _merge_base_features(vh, base)

    deployable_names = [
        name
        for name in df.columns
        if pd.api.types.is_numeric_dtype(df[name]) and not _is_metadata_or_label(name)
    ]
    validationhold_names = [name for name in VH_FEATURE_COLS if name in df.columns]
    # merge + dedupe while keeping validationhold features first
    feature_names: list[str] = []
    for name in validationhold_names + deployable_names:
        if name not in feature_names:
            feature_names.append(name)

    lift_mask = df["validationhold_low_pred_lift_signal"].astype(float) > 0.5
    broad_mask = (df["hold_ready_frac"].astype(float) >= args.broad_hold_ready_thr) & (
        df["base_pred_fix_rate_pct"].astype(float) <= args.broad_base_pred_max
    )

    prefix = RESULTS_DIR / args.results_prefix

    lift_feature_rank = _compute_separation(
        df, feature_names, lift_mask, "actual_fix_rate_pct", args.actual_low_thr, args.actual_high_thr
    )
    broad_feature_rank = _compute_separation(
        df, feature_names, broad_mask, "actual_fix_rate_pct", args.actual_low_thr, args.actual_high_thr
    )

    extra_features = [
        row["feature"] for row in (lift_feature_rank[: args.top_n])[:30] if row.get("feature")
    ]
    extra_features = [name for name in extra_features if name not in SNAPSHOT_FEATURES]

    lift_windows = _per_window_rows(df, lift_mask, "actual_fix_rate_pct", args.actual_low_thr, args.actual_high_thr, extra_features)
    broad_windows = _per_window_rows(df, broad_mask, "actual_fix_rate_pct", args.actual_low_thr, args.actual_high_thr, extra_features)

    summary_rows = _group_summary("lift_signal", df, lift_mask, "actual_fix_rate_pct", args.actual_low_thr, args.actual_high_thr)
    summary_rows += _group_summary("hold_ready_broad", df, broad_mask, "actual_fix_rate_pct", args.actual_low_thr, args.actual_high_thr)

    _write_rows(prefix.with_name(prefix.name + "_lift_signal_windows.csv"), lift_windows)
    _write_rows(prefix.with_name(prefix.name + "_hold_ready_broad_windows.csv"), broad_windows)
    _write_rows(prefix.with_name(prefix.name + "_group_summary.csv"), summary_rows)
    _write_rows(
        prefix.with_name(prefix.name + "_lift_signal_feature_separation.csv"),
        lift_feature_rank[: args.top_n * 3],
    )
    _write_rows(
        prefix.with_name(prefix.name + "_hold_ready_broad_feature_separation.csv"),
        broad_feature_rank[: args.top_n * 3],
    )

    print("group_summary:")
    print(pd.DataFrame(summary_rows).to_string(index=False))

    print("\ntop separating features within lift_signal subset (actual_high vs actual_low):")
    if lift_feature_rank:
        print(
            pd.DataFrame(lift_feature_rank[:20])[
                [
                    "feature",
                    "family",
                    "actual_high_n",
                    "actual_low_n",
                    "actual_high_mean",
                    "actual_low_mean",
                    "separation_score",
                    "effect_size",
                ]
            ].to_string(index=False)
        )

    print("\ntop separating features within hold_ready_broad subset (actual_high vs actual_low):")
    if broad_feature_rank:
        print(
            pd.DataFrame(broad_feature_rank[:20])[
                [
                    "feature",
                    "family",
                    "actual_high_n",
                    "actual_low_n",
                    "actual_high_mean",
                    "actual_low_mean",
                    "separation_score",
                    "effect_size",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
