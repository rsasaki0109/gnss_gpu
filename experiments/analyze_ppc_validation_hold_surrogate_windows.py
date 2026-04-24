#!/usr/bin/env python3
"""Aggregate validation/hold surrogate epoch features to PPC windows."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import pandas as pd


RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_WINDOW_CSV = (
    RESULTS_DIR
    / "ppc_window_fix_rate_model_stride1_stat_sim_rinex_phasejump_t0p25_gf0p2_simloscont_focused_simadop_nowt_windowopt_window_predictions.csv"
)
DEFAULT_EPOCH_CSV = (
    RESULTS_DIR
    / "ppc_demo5_fix_rate_compare_fullsim_stride1_phase_proxy_stat_rinex_phasejump_t0p25_gf0p2_simloscont_focused_simadop_nowt_validationhold_epochs.csv"
)
DEFAULT_PREFIX = "ppc_validationhold"


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
    print(f"saved: {path}")


def _prediction_column(df: pd.DataFrame) -> str:
    for name in (
        "corrected_pred_fix_rate_pct",
        "extra_trees_cal_to_proxy_window_mae_rate_bounded_pred_fix_rate_pct",
        "pred_fix_rate_pct",
        "extra_trees_pred_fix_rate_pct",
    ):
        if name in df.columns:
            return name
    candidates = [name for name in df.columns if name.endswith("_pred_fix_rate_pct")]
    if candidates:
        return candidates[0]
    raise ValueError("missing prediction column")


def _finite(values: pd.Series) -> np.ndarray:
    arr = values.to_numpy(dtype=np.float64)
    return arr[np.isfinite(arr)]


def _mean(values: pd.Series) -> float:
    arr = _finite(values)
    return float(np.mean(arr)) if arr.size else float("nan")


def _max(values: pd.Series) -> float:
    arr = _finite(values)
    return float(np.max(arr)) if arr.size else float("nan")


def _q(values: pd.Series, quantile: float) -> float:
    arr = _finite(values)
    return float(np.quantile(arr, quantile)) if arr.size else float("nan")


def _first_rel_s(group: pd.DataFrame, mask_col: str, start_tow: float) -> float:
    if mask_col not in group.columns:
        return float("nan")
    rows = group[group[mask_col].astype(float) > 0.0]
    if rows.empty:
        return float("nan")
    return float(rows["gps_tow"].iloc[0] - start_tow)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    finite = np.isfinite(a) & np.isfinite(b)
    if finite.sum() < 2:
        return float("nan")
    return float(np.corrcoef(a[finite], b[finite])[0, 1])


def _metrics(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    df = pd.DataFrame(rows)
    if df.empty:
        return []
    actual = df["actual_fix_rate_pct"].to_numpy(dtype=np.float64)
    weight = df["epoch_count"].to_numpy(dtype=np.float64)
    weight = np.where(np.isfinite(weight) & (weight > 0.0), weight, 1.0)
    out: list[dict[str, object]] = []
    targets = [
        ("base", "base_pred_fix_rate_pct"),
        ("validationhold_reject_only", "validationhold_reject_only_pred_pct"),
        ("validationhold_diag", "validationhold_diag_pred_pct"),
    ]
    for name, col in targets:
        pred = df[col].to_numpy(dtype=np.float64)
        abs_err = np.abs(pred - actual)
        out.append(
            {
                "target": name,
                "weighted_window_mae_pp": float(np.average(abs_err, weights=weight)),
                "window_rmse_pp": float(np.sqrt(np.average((pred - actual) ** 2, weights=weight))),
                "window_corr": _corr(pred, actual),
                "over_20pp_windows": int(np.count_nonzero(abs_err > 20.0)),
            }
        )

    run_rows: list[dict[str, object]] = []
    for (city, run), group in df.groupby(["city", "run"], sort=True):
        w = group["epoch_count"].to_numpy(dtype=np.float64)
        w = np.where(np.isfinite(w) & (w > 0.0), w, 1.0)
        run_rows.append(
            {
                "city": city,
                "run": run,
                "actual": float(np.average(group["actual_fix_rate_pct"].to_numpy(dtype=np.float64), weights=w)),
                "base": float(np.average(group["base_pred_fix_rate_pct"].to_numpy(dtype=np.float64), weights=w)),
                "validationhold_reject_only": float(
                    np.average(group["validationhold_reject_only_pred_pct"].to_numpy(dtype=np.float64), weights=w)
                ),
                "validationhold_diag": float(
                    np.average(group["validationhold_diag_pred_pct"].to_numpy(dtype=np.float64), weights=w)
                ),
            }
        )
    rdf = pd.DataFrame(run_rows)
    for target in ("base", "validationhold_reject_only", "validationhold_diag"):
        out.append(
            {
                "target": f"{target}_run",
                "weighted_window_mae_pp": float(np.mean(np.abs(rdf[target] - rdf["actual"]))),
                "window_rmse_pp": float(np.sqrt(np.mean((rdf[target] - rdf["actual"]) ** 2))),
                "window_corr": _corr(rdf[target].to_numpy(dtype=np.float64), rdf["actual"].to_numpy(dtype=np.float64)),
                "over_20pp_windows": int(np.count_nonzero(np.abs(rdf[target] - rdf["actual"]) > 20.0)),
            }
        )
    return out


def _diagnostic_prediction(row: dict[str, object]) -> tuple[float, str]:
    base = float(row["base_pred_fix_rate_pct"])
    reject = bool(row["validationhold_high_pred_reject_signal"])
    lift = bool(row["validationhold_low_pred_lift_signal"])
    if reject:
        return min(base, 5.0), "high_pred_reject"
    if lift:
        hold = float(row["hold_ready_frac"])
        strict = float(row["hold_strict_ready_frac"])
        carry = float(row["hold_carry_score_mean"])
        lifted = max(base, min(95.0, 20.0 + 70.0 * hold + 20.0 * strict + 2.0 * min(carry, 5.0)))
        return lifted, "low_pred_hold_lift"
    return base, "base"


def _window_rows(windows: pd.DataFrame, epochs: pd.DataFrame, pred_col: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    epoch_groups = {key: group.sort_values("gps_tow") for key, group in epochs.groupby(["city", "run"], sort=False)}
    for _, window in windows.sort_values(["city", "run", "window_index"]).iterrows():
        city = str(window["city"])
        run = str(window["run"])
        start = float(window["window_start_tow"])
        end = float(window["window_end_tow"])
        group = epoch_groups.get((city, run), pd.DataFrame())
        if group.empty:
            g = group
        else:
            g = group[(group["gps_tow"] >= start - 1e-6) & (group["gps_tow"] <= end + 1e-6)]

        row: dict[str, object] = {
            "city": city,
            "run": run,
            "window_index": int(window["window_index"]),
            "window_start_tow": start,
            "window_end_tow": end,
            "epoch_count": int(len(g)),
            "actual_fix_rate_pct": float(window["actual_fix_rate_pct"]),
            "base_pred_fix_rate_pct": float(window[pred_col]),
        }
        if g.empty:
            for name in (
                "demo5_fix_rate_pct",
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
                "hold_age_s_at_start",
                "hold_age_s_mean",
                "hold_age_s_p90",
                "hold_age_s_max",
                "hold_since_reset_s_at_start",
                "hold_since_reset_s_mean",
                "hold_since_reset_s_p90",
                "hold_since_reset_s_max",
            ):
                row[name] = float("nan")
        else:
            row.update(
                {
                    "demo5_fix_rate_pct": 100.0 * _mean(g["actual_fixed"]),
                    "validation_pass_frac": _mean(g["validation_pass"]),
                    "validation_soft_pass_frac": _mean(g["validation_soft_pass"]),
                    "validation_hard_block_frac": _mean(g["validation_hard_block"]),
                    "validation_severe_block_frac": _mean(g["validation_severe_block"]),
                    "validation_reject_block_frac": _mean(g["validation_reject_block"]),
                    "validation_block_spike_frac": _mean(g["validation_block_spike"]),
                    "validation_block_score_mean": _mean(g["validation_block_score"]),
                    "validation_block_score_p90": _q(g["validation_block_score"], 0.90),
                    "validation_block_score_max": _max(g["validation_block_score"]),
                    "validation_block_ewma30_mean": _mean(g["validation_block_ewma_30s"]),
                    "validation_block_cooldown_max": _max(g["validation_block_cooldown_s"]),
                    "validation_reject_recent_max": _max(g["validation_reject_recent_s"]),
                    "validation_quality_mean": _mean(g["validation_quality_score"]),
                    "validation_quality_p90": _q(g["validation_quality_score"], 0.90),
                    "hold_state_mean": _mean(g["hold_state"]),
                    "hold_state_max": _max(g["hold_state"]),
                    "hold_ready_frac": _mean(g["hold_ready"]),
                    "hold_strict_ready_frac": _mean(g["hold_strict_ready"]),
                    "hold_carry_score_mean": _mean(g["hold_carry_score"]),
                    "hold_carry_score_max": _max(g["hold_carry_score"]),
                    "first_validation_pass_rel_s": _first_rel_s(g, "validation_pass", start),
                    "first_hold_ready_rel_s": _first_rel_s(g, "hold_ready", start),
                    "clean_streak_s_at_start": float(g["clean_streak_s"].iloc[0]) if "clean_streak_s" in g.columns else float("nan"),
                    "clean_streak_s_mean": _mean(g["clean_streak_s"]) if "clean_streak_s" in g.columns else float("nan"),
                    "clean_streak_s_p50": _q(g["clean_streak_s"], 0.50) if "clean_streak_s" in g.columns else float("nan"),
                    "clean_streak_s_p90": _q(g["clean_streak_s"], 0.90) if "clean_streak_s" in g.columns else float("nan"),
                    "clean_streak_s_max": _max(g["clean_streak_s"]) if "clean_streak_s" in g.columns else float("nan"),
                    "strict_clean_streak_s_at_start": float(g["strict_clean_streak_s"].iloc[0]) if "strict_clean_streak_s" in g.columns else float("nan"),
                    "strict_clean_streak_s_mean": _mean(g["strict_clean_streak_s"]) if "strict_clean_streak_s" in g.columns else float("nan"),
                    "strict_clean_streak_s_p50": _q(g["strict_clean_streak_s"], 0.50) if "strict_clean_streak_s" in g.columns else float("nan"),
                    "strict_clean_streak_s_p90": _q(g["strict_clean_streak_s"], 0.90) if "strict_clean_streak_s" in g.columns else float("nan"),
                    "strict_clean_streak_s_max": _max(g["strict_clean_streak_s"]) if "strict_clean_streak_s" in g.columns else float("nan"),
                    "hold_age_s_at_start": float(g["hold_age_s"].iloc[0]) if "hold_age_s" in g.columns else float("nan"),
                    "hold_age_s_mean": _mean(g["hold_age_s"]) if "hold_age_s" in g.columns else float("nan"),
                    "hold_age_s_p90": _q(g["hold_age_s"], 0.90) if "hold_age_s" in g.columns else float("nan"),
                    "hold_age_s_max": _max(g["hold_age_s"]) if "hold_age_s" in g.columns else float("nan"),
                    "hold_since_reset_s_at_start": float(g["hold_since_reset_s"].iloc[0]) if "hold_since_reset_s" in g.columns else float("nan"),
                    "hold_since_reset_s_mean": _mean(g["hold_since_reset_s"]) if "hold_since_reset_s" in g.columns else float("nan"),
                    "hold_since_reset_s_p90": _q(g["hold_since_reset_s"], 0.90) if "hold_since_reset_s" in g.columns else float("nan"),
                    "hold_since_reset_s_max": _max(g["hold_since_reset_s"]) if "hold_since_reset_s" in g.columns else float("nan"),
                }
            )

        high_pred = float(row["base_pred_fix_rate_pct"]) >= 50.0
        low_pred = float(row["base_pred_fix_rate_pct"]) <= 30.0
        reject_signal = high_pred and (
            float(row["validation_block_score_max"]) >= 10.0
            or (
                float(row["validation_reject_block_frac"]) >= 0.01
                and float(row["validation_pass_frac"]) <= 0.05
                and float(row["hold_ready_frac"]) <= 0.05
            )
        )
        lift_signal = (
            low_pred
            and float(row["base_pred_fix_rate_pct"]) >= 12.0
            and float(row["validation_pass_frac"]) >= 0.60
            and float(row["hold_ready_frac"]) >= 0.55
            and float(row["hold_strict_ready_frac"]) >= 0.45
            and float(row["validation_reject_block_frac"]) <= 0.0
            and float(row["validation_block_score_p90"]) <= 1.25
        )
        row["validationhold_high_pred_reject_signal"] = 1.0 if reject_signal else 0.0
        row["validationhold_low_pred_lift_signal"] = 1.0 if lift_signal else 0.0
        row["validationhold_reject_only_pred_pct"] = min(float(row["base_pred_fix_rate_pct"]), 5.0) if reject_signal else float(
            row["base_pred_fix_rate_pct"]
        )
        diag, reason = _diagnostic_prediction(row)
        row["validationhold_diag_pred_pct"] = diag
        row["validationhold_diag_reason"] = reason
        row["base_abs_error_pp"] = abs(float(row["base_pred_fix_rate_pct"]) - float(row["actual_fix_rate_pct"]))
        row["validationhold_diag_abs_error_pp"] = abs(diag - float(row["actual_fix_rate_pct"]))
        row["validationhold_diag_delta_abs_error_pp"] = row["validationhold_diag_abs_error_pp"] - row["base_abs_error_pp"]
        row["hidden_high_case"] = 1.0 if float(row["actual_fix_rate_pct"]) >= 75.0 and low_pred else 0.0
        row["false_high_case"] = 1.0 if float(row["actual_fix_rate_pct"]) <= 5.0 and high_pred else 0.0
        rows.append(row)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate validation/hold surrogate features to PPC windows")
    parser.add_argument("--window-csv", type=Path, default=DEFAULT_WINDOW_CSV)
    parser.add_argument("--epoch-csv", type=Path, default=DEFAULT_EPOCH_CSV)
    parser.add_argument("--results-prefix", default=DEFAULT_PREFIX)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    windows = pd.read_csv(args.window_csv)
    pred_col = _prediction_column(windows)
    needed = {
        "city",
        "run",
        "gps_tow",
        "actual_fixed",
        "validation_pass",
        "validation_soft_pass",
        "validation_hard_block",
        "validation_severe_block",
        "validation_reject_block",
        "validation_block_spike",
        "validation_block_score",
        "validation_block_ewma_30s",
        "validation_block_cooldown_s",
        "validation_reject_recent_s",
        "validation_quality_score",
        "hold_state",
        "hold_ready",
        "hold_strict_ready",
        "hold_carry_score",
        "hold_age_s",
        "hold_since_reset_s",
        "clean_streak_s",
        "strict_clean_streak_s",
    }
    epochs = pd.read_csv(args.epoch_csv, usecols=lambda col: col in needed)
    rows = _window_rows(windows, epochs, pred_col)

    prefix = RESULTS_DIR / args.results_prefix
    all_path = prefix.with_name(prefix.name + "_window_summary.csv")
    focus_path = prefix.with_name(prefix.name + "_tokyo_run2_focus_window_summary.csv")
    extreme_path = prefix.with_name(prefix.name + "_extreme_window_summary.csv")
    metrics_path = prefix.with_name(prefix.name + "_diagnostic_rule_metrics.csv")
    _write_rows(all_path, rows)
    focus = [
        row
        for row in rows
        if row["city"] == "tokyo" and row["run"] == "run2" and 0 <= int(row["window_index"]) <= 30
    ]
    _write_rows(focus_path, focus)
    extreme = [row for row in rows if row["hidden_high_case"] or row["false_high_case"]]
    _write_rows(extreme_path, extreme)
    metric_rows = _metrics(rows)
    _write_rows(metrics_path, metric_rows)
    print(pd.DataFrame(metric_rows).to_string(index=False))


if __name__ == "__main__":
    main()
