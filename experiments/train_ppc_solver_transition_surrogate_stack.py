#!/usr/bin/env python3
"""Train binary solver-transition surrogates and stack them into fix-rate correction.

This script uses an existing full window-prediction CSV that already contains
the window aggregates produced by ``train_ppc_window_fix_rate_model.py``.  It
trains deployable-style surrogate classifiers from simulator/RINEX features to
binary demo5/RTK transition targets such as high ambiguity ratio and rising
lock count, then uses the cross-fit probabilities to correct the current
fix-rate prediction.

The evaluation is cross-fit leave-one-run-out and lightweight.  It is a signal
check before implementing a stricter nested production candidate.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _float(value: object, default: float = float("nan")) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
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


def _parse_float_list(spec: str) -> list[float]:
    return [float(part.strip()) for part in spec.split(",") if part.strip()]


def _key_frame(df: pd.DataFrame) -> pd.Series:
    return df["city"].astype(str) + "\t" + df["run"].astype(str) + "\t" + df["window_index"].astype(int).astype(str)


def _read_reference_predictions(prefix: str, prediction_column: str) -> dict[str, float]:
    path = RESULTS_DIR / f"{prefix}_window_predictions.csv"
    rows = pd.read_csv(path)
    requested = prediction_column
    if prediction_column not in rows.columns:
        fallback_candidates = ["corrected_pred_fix_rate_pct", "pred_fix_rate_pct"]
        available = [c for c in fallback_candidates if c in rows.columns]
        if not available:
            raise ValueError(
                f"missing prediction column '{requested}' in {path.name}; "
                f"none of the fallbacks {fallback_candidates} are present either. "
                f"Available prediction columns: {[c for c in rows.columns if c.endswith('_pred_fix_rate_pct')]}"
            )
        prediction_column = available[0]
        print(
            f"note: requested prediction column '{requested}' not in {path.name}; "
            f"falling back to '{prediction_column}'"
        )
    return {
        key: float(value) / 100.0
        for key, value in zip(_key_frame(rows), rows[prediction_column].astype(float))
    }


from _common import _is_metadata_or_label  # canonical helper; re-exported for callers


def _sim_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    feature_names: list[str] = []
    for name in df.columns:
        if _is_metadata_or_label(name):
            continue
        if pd.api.types.is_numeric_dtype(df[name]):
            feature_names.append(name)
    x = df[feature_names].to_numpy(dtype=np.float64)
    x[~np.isfinite(x)] = 0.0
    return x, feature_names


def _append_classifier_meta(
    *,
    df: pd.DataFrame,
    x: np.ndarray,
    names: list[str],
    base: np.ndarray,
    include_base: bool,
    include_city: bool,
    include_run_position: bool,
) -> tuple[np.ndarray, list[str]]:
    columns = [x]
    out_names = list(names)
    if include_base:
        columns.append(base.reshape(-1, 1))
        columns.append(_logit(base).reshape(-1, 1))
        out_names.extend(["meta_base_pred", "meta_base_logit"])
    if include_run_position:
        run_pos = np.zeros(len(df), dtype=np.float64)
        run_frac = np.zeros(len(df), dtype=np.float64)
        run_remaining = np.zeros(len(df), dtype=np.float64)
        run_len = np.zeros(len(df), dtype=np.float64)
        for (_city, _run), group in df.groupby(["city", "run"], sort=False):
            indices = list(group.sort_values("window_index").index)
            denom = max(len(indices) - 1, 1)
            for pos, idx in enumerate(indices):
                run_pos[idx] = float(pos)
                run_frac[idx] = float(pos / denom)
                run_remaining[idx] = float((len(indices) - 1 - pos) / denom)
                run_len[idx] = float(len(indices))
        columns.extend(
            [
                run_pos.reshape(-1, 1),
                run_frac.reshape(-1, 1),
                run_remaining.reshape(-1, 1),
                run_len.reshape(-1, 1),
            ]
        )
        out_names.extend(["meta_run_pos", "meta_run_fraction", "meta_run_remaining_fraction", "meta_run_window_count"])
    if include_city:
        for city in sorted(df["city"].astype(str).unique()):
            columns.append((df["city"].astype(str).to_numpy() == city).astype(np.float64).reshape(-1, 1))
            out_names.append(f"meta_city_{city}")
    return np.column_stack(columns), out_names


def _groups(df: pd.DataFrame) -> tuple[np.ndarray, list[tuple[str, str]]]:
    lookup: dict[tuple[str, str], int] = {}
    labels: list[tuple[str, str]] = []
    values: list[int] = []
    for city, run in zip(df["city"].astype(str), df["run"].astype(str)):
        key = (city, run)
        if key not in lookup:
            lookup[key] = len(labels)
            labels.append(key)
        values.append(lookup[key])
    return np.asarray(values, dtype=np.int64), labels


def _transition_targets(df: pd.DataFrame, *, include_focused_targets: bool = False) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    specs = {
        "ratio_mean_ge3": df["solver_demo5_ratio_mean"] >= 3.0,
        "ratio_mean_ge5": df["solver_demo5_ratio_mean"] >= 5.0,
        "ratio_p90_ge10": df["solver_demo5_ratio_p90"] >= 10.0,
        "ratio_p95_ge15": df["solver_demo5_ratio_p95"] >= 15.0,
        "ratio_rise_mean_ge2": df["solver_demo5_ratio_mean_past_delta"] >= 2.0,
        "lock_p90p50_ge100": df["rtk_lock_p90_p50"] >= 100.0,
        "lock_rise_p90p50_ge50": df["rtk_lock_p90_p50_past_delta"] >= 50.0,
        "lock_carry_or_rise": (df["rtk_lock_p90_p50"] >= 100.0) | (df["rtk_lock_p90_p50_past_delta"] >= 50.0),
        "ratio_or_lock_carry": (
            (df["solver_demo5_ratio_mean"] >= 3.0)
            | (df["rtk_lock_p90_p50"] >= 100.0)
            | (df["rtk_lock_p90_p50_past_delta"] >= 50.0)
        ),
    }
    if include_focused_targets:
        specs.update(
            {
                "ratio_mean_ge2": df["solver_demo5_ratio_mean"] >= 2.0,
                "ratio_p90_ge6": df["solver_demo5_ratio_p90"] >= 6.0,
                "ratio_p95_ge10": df["solver_demo5_ratio_p95"] >= 10.0,
                "ratio_rise_mean_ge1": df["solver_demo5_ratio_mean_past_delta"] >= 1.0,
                "lock_p90p50_ge150": df["rtk_lock_p90_p50"] >= 150.0,
                "lock_rise_p90p50_ge100": df["rtk_lock_p90_p50_past_delta"] >= 100.0,
                "ratio_mean_ge3_and_lock_p90p50_ge100": (
                    (df["solver_demo5_ratio_mean"] >= 3.0) & (df["rtk_lock_p90_p50"] >= 100.0)
                ),
                "ratio_p95_ge10_or_lock_rise_ge100": (
                    (df["solver_demo5_ratio_p95"] >= 10.0) | (df["rtk_lock_p90_p50_past_delta"] >= 100.0)
                ),
                "ratio2_or_lock_carry": (
                    (df["solver_demo5_ratio_mean"] >= 2.0)
                    | (df["rtk_lock_p90_p50"] >= 100.0)
                    | (df["rtk_lock_p90_p50_past_delta"] >= 50.0)
                ),
                "ratio_rise_or_lock_rise": (
                    (df["solver_demo5_ratio_mean_past_delta"] >= 1.0)
                    | (df["rtk_lock_p90_p50_past_delta"] >= 50.0)
                ),
                "solver_quiet_ratio_lt2_lock_lt50": (
                    (df["solver_demo5_ratio_mean"] < 2.0)
                    & (df["rtk_lock_p90_p50"] < 50.0)
                    & (df["rtk_lock_p90_p50_past_delta"] < 10.0)
                ),
            }
        )
    for name, mask in specs.items():
        out[name] = np.asarray(mask, dtype=np.int64)
    return out


def _classifier(random_state: int, estimators: int, min_leaf: int, max_features: float) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=estimators,
        min_samples_leaf=min_leaf,
        max_features=max_features,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )


def _positive_probability(model: ExtraTreesClassifier, x: np.ndarray) -> np.ndarray:
    if len(model.classes_) == 1:
        return np.full(x.shape[0], 1.0 if int(model.classes_[0]) == 1 else 0.0, dtype=np.float64)
    class_index = list(model.classes_).index(1)
    return np.asarray(model.predict_proba(x)[:, class_index], dtype=np.float64)


def _loro_probabilities(
    *,
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    groups: np.ndarray,
    random_state: int,
    estimators: int,
    min_leaf: int,
    max_features: float,
) -> np.ndarray:
    out = np.full(y.shape, np.nan, dtype=np.float64)
    for group_id in sorted(set(int(value) for value in groups)):
        train = groups != group_id
        test = groups == group_id
        clf = _classifier(random_state + group_id, estimators, min_leaf, max_features)
        clf.fit(x[train], y[train], sample_weight=weights[train])
        out[test] = _positive_probability(clf, x[test])
    out[~np.isfinite(out)] = 0.0
    return out


def _add_temporal_prob_features(df: pd.DataFrame, prob_matrix: np.ndarray, names: list[str]) -> tuple[np.ndarray, list[str]]:
    grouped: dict[tuple[str, str], list[int]] = {}
    for idx, row in df.iterrows():
        grouped.setdefault((str(row["city"]), str(row["run"])), []).append(idx)
    for indices in grouped.values():
        indices.sort(key=lambda idx: int(df.at[idx, "window_index"]))

    cols = [prob_matrix]
    out_names = list(names)
    for col_idx, name in enumerate(names):
        values = prob_matrix[:, col_idx]
        prev = values.copy()
        delta = np.zeros_like(values)
        past_mean = values.copy()
        for indices in grouped.values():
            for pos, idx in enumerate(indices):
                if pos > 0:
                    prev[idx] = values[indices[pos - 1]]
                    delta[idx] = values[idx] - values[indices[pos - 1]]
                    start = max(0, pos - 3)
                    past_mean[idx] = float(np.mean(values[indices[start:pos]]))
        cols.extend([prev.reshape(-1, 1), delta.reshape(-1, 1), past_mean.reshape(-1, 1)])
        out_names.extend([f"{name}_prev", f"{name}_delta_prev", f"{name}_past_mean3"])
    return np.column_stack(cols), out_names


def _logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 1e-4, 1.0 - 1e-4)
    return np.log(clipped / (1.0 - clipped))


def _residual_model(name: str, random_state: int, estimators: int, min_leaf: int, max_features: float):
    if name == "ridge":
        return Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=10.0))])
    if name == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=estimators,
            min_samples_leaf=min_leaf,
            max_features=max_features,
            random_state=random_state,
            n_jobs=-1,
        )
    if name == "gradient_boosting":
        return HistGradientBoostingRegressor(
            max_iter=max(estimators, 100),
            learning_rate=0.05,
            max_depth=3,
            min_samples_leaf=max(min_leaf, 8),
            l2_regularization=1.0,
            random_state=random_state,
        )
    if name == "elastic_net":
        return Pipeline([
            ("scale", StandardScaler()),
            ("model", ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=20000, random_state=random_state)),
        ])
    raise ValueError(f"unknown residual model: {name}")


def _fit_residual_loro(
    *,
    x: np.ndarray,
    residual: np.ndarray,
    weights: np.ndarray,
    groups: np.ndarray,
    model_name: str,
    estimators: int,
    min_leaf: int,
    max_features: float,
    random_state: int,
) -> np.ndarray:
    out = np.full(residual.shape, np.nan, dtype=np.float64)
    for group_id in sorted(set(int(value) for value in groups)):
        train = groups != group_id
        test = groups == group_id
        model = _residual_model(model_name, random_state + group_id, estimators, min_leaf, max_features)
        if isinstance(model, Pipeline):
            model.fit(x[train], residual[train], model__sample_weight=weights[train])
        else:
            model.fit(x[train], residual[train], sample_weight=weights[train])
        out[test] = np.asarray(model.predict(x[test]), dtype=np.float64)
    return out


def _weighted_rate(values: np.ndarray, weights: np.ndarray) -> float:
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(mask):
        return float("nan")
    return float(np.average(values[mask], weights=weights[mask]))


def _metric_row(
    *,
    name: str,
    y: np.ndarray,
    pred: np.ndarray,
    weights: np.ndarray,
    groups: np.ndarray,
    group_labels: list[tuple[str, str]],
    params: dict[str, object],
) -> dict[str, object]:
    valid = np.isfinite(pred)
    yv = y[valid]
    pv = pred[valid]
    wv = weights[valid]
    err_pp = 100.0 * (pv - yv)
    run_abs: list[float] = []
    for group_id, _label in enumerate(group_labels):
        mask = (groups == group_id) & valid
        if not np.any(mask):
            continue
        actual = _weighted_rate(y[mask], weights[mask])
        predicted = _weighted_rate(pred[mask], weights[mask])
        run_abs.append(abs(100.0 * (predicted - actual)))
    ss_res = float(np.sum((pv - yv) ** 2))
    ss_tot = float(np.sum((yv - np.mean(yv)) ** 2))
    corr = float(np.corrcoef(yv, pv)[0, 1]) if yv.size > 1 and np.std(yv) > 0.0 and np.std(pv) > 0.0 else float("nan")
    row: dict[str, object] = {
        "model": name,
        "windows": int(np.count_nonzero(valid)),
        "actual_fix_rate_pct": 100.0 * _weighted_rate(yv, wv),
        "pred_fix_rate_pct": 100.0 * _weighted_rate(pv, wv),
        "aggregate_error_pp": 100.0 * (_weighted_rate(pv, wv) - _weighted_rate(yv, wv)),
        "window_mae_pp": float(np.mean(np.abs(err_pp))),
        "window_weighted_mae_pp": float(np.average(np.abs(err_pp), weights=wv)),
        "window_rmse_pp": float(np.sqrt(np.mean(err_pp**2))),
        "epoch_weighted_rmse_pp": float(np.sqrt(np.average(err_pp**2, weights=wv))),
        "run_mae_pp": float(np.mean(run_abs)) if run_abs else float("nan"),
        "run_max_abs_pp": float(np.max(run_abs)) if run_abs else float("nan"),
        "r2_window": 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan"),
        "corr_window": corr,
        "under_20pp_count": int(np.count_nonzero(err_pp <= -20.0)),
        "over_20pp_count": int(np.count_nonzero(err_pp >= 20.0)),
    }
    row.update(params)
    return row


def _target_metric_rows(targets: dict[str, np.ndarray], probabilities: dict[str, np.ndarray], weights: np.ndarray) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name, y in targets.items():
        prob = probabilities[name]
        try:
            auc = float(roc_auc_score(y, prob, sample_weight=weights))
        except ValueError:
            auc = float("nan")
        try:
            ap = float(average_precision_score(y, prob, sample_weight=weights))
        except ValueError:
            ap = float("nan")
        rows.append(
            {
                "target": name,
                "positive_windows": int(np.count_nonzero(y)),
                "windows": int(y.size),
                "positive_rate": float(np.mean(y)),
                "prob_mean": float(np.average(prob, weights=weights)),
                "roc_auc": auc,
                "average_precision": ap,
                "brier": float(brier_score_loss(y, prob, sample_weight=weights)),
            }
        )
    return rows


def _prediction_rows(
    *,
    df: pd.DataFrame,
    y: np.ndarray,
    base: np.ndarray,
    residual_pred: np.ndarray,
    corrected: np.ndarray,
    probabilities: dict[str, np.ndarray],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for idx, source in df.reset_index(drop=True).iterrows():
        base_error = 100.0 * (base[idx] - y[idx])
        corrected_error = 100.0 * (corrected[idx] - y[idx])
        out: dict[str, object] = {
            "city": source["city"],
            "run": source["run"],
            "window_index": source["window_index"],
            "sim_matched_epochs": source.get("sim_matched_epochs", ""),
            "actual_fix_rate_pct": 100.0 * float(y[idx]),
            "base_pred_fix_rate_pct": 100.0 * float(base[idx]),
            "residual_pred_pp": 100.0 * float(residual_pred[idx]),
            "corrected_pred_fix_rate_pct": 100.0 * float(corrected[idx]),
            "base_error_pp": base_error,
            "corrected_error_pp": corrected_error,
            "abs_error_gain_pp": abs(base_error) - abs(corrected_error),
            "solver_demo5_ratio_mean": source.get("solver_demo5_ratio_mean", ""),
            "rtk_lock_p90_p50": source.get("rtk_lock_p90_p50", ""),
            "rtk_lock_p90_p50_past_delta": source.get("rtk_lock_p90_p50_past_delta", ""),
        }
        for name, prob in probabilities.items():
            out[f"{name}_prob"] = float(prob[idx])
        rows.append(out)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train solver-transition surrogate stack")
    parser.add_argument("--window-csv", type=Path, required=True)
    parser.add_argument("--base-prefix", required=True)
    parser.add_argument("--base-prediction-column", default="corrected_pred_fix_rate_pct")
    parser.add_argument("--classifier-estimators", type=int, default=80)
    parser.add_argument("--classifier-min-leaf", type=int, default=4)
    parser.add_argument("--classifier-max-features", type=float, default=0.75)
    parser.add_argument("--classifier-include-base", action="store_true")
    parser.add_argument("--classifier-include-city", action="store_true")
    parser.add_argument("--classifier-include-run-position", action="store_true")
    parser.add_argument("--include-focused-targets", action="store_true")
    parser.add_argument("--residual-models", default="ridge,extra_trees")
    parser.add_argument("--residual-estimators", type=int, default=80)
    parser.add_argument("--residual-min-leaf", type=int, default=3)
    parser.add_argument("--residual-max-features", type=float, default=0.75)
    parser.add_argument("--alphas", default="0,0.1,0.2,0.3,0.5,0.75,1.0")
    parser.add_argument("--residual-clip-pp", default="5,10,15,20,30,50,100")
    parser.add_argument("--max-run-mae-pp", type=float, default=0.0)
    parser.add_argument("--max-abs-aggregate-error-pp", type=float, default=0.0)
    parser.add_argument("--random-state", type=int, default=2032)
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--results-prefix", default="ppc_solver_transition_surrogate_stack")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.window_csv)
    df = df.sort_values(["city", "run", "window_index"]).reset_index(drop=True)
    groups, group_labels = _groups(df)
    weights = df["sim_matched_epochs"].to_numpy(dtype=np.float64)
    y = df["actual_fix_rate_pct"].to_numpy(dtype=np.float64) / 100.0
    reference = _read_reference_predictions(args.base_prefix, args.base_prediction_column)
    base = np.asarray([reference[key] for key in _key_frame(df)], dtype=np.float64)
    x, feature_names = _sim_feature_matrix(df)
    x, feature_names = _append_classifier_meta(
        df=df,
        x=x,
        names=feature_names,
        base=base,
        include_base=args.classifier_include_base,
        include_city=args.classifier_include_city,
        include_run_position=args.classifier_include_run_position,
    )
    targets = _transition_targets(df, include_focused_targets=args.include_focused_targets)

    probabilities: dict[str, np.ndarray] = {}
    for idx, (name, target) in enumerate(targets.items()):
        probabilities[name] = _loro_probabilities(
            x=x,
            y=target,
            weights=weights,
            groups=groups,
            random_state=args.random_state + idx * 100,
            estimators=args.classifier_estimators,
            min_leaf=args.classifier_min_leaf,
            max_features=args.classifier_max_features,
        )
    prob_names = list(probabilities)
    prob_matrix = np.column_stack([probabilities[name] for name in prob_names])
    prob_x, prob_feature_names = _add_temporal_prob_features(df, prob_matrix, prob_names)
    meta_x = np.column_stack([base, _logit(base), prob_x])
    residual_target = y - base
    print(f"classifier_features={len(feature_names)} transition_targets={len(targets)} meta_features={meta_x.shape[1]}")

    metric_rows = [
        _metric_row(
            name="base",
            y=y,
            pred=base,
            weights=weights,
            groups=groups,
            group_labels=group_labels,
            params={"residual_model": "base", "alpha": 0.0, "residual_clip_pp": 0.0},
        )
    ]
    corrected_by_name: dict[str, np.ndarray] = {"base": base}
    residual_by_name: dict[str, np.ndarray] = {"base": np.zeros_like(base)}
    residual_models = [part.strip() for part in args.residual_models.split(",") if part.strip()]
    alphas = _parse_float_list(args.alphas)
    clip_values = _parse_float_list(args.residual_clip_pp)
    for model_name in residual_models:
        residual_pred = _fit_residual_loro(
            x=meta_x,
            residual=residual_target,
            weights=weights,
            groups=groups,
            model_name=model_name,
            estimators=args.residual_estimators,
            min_leaf=args.residual_min_leaf,
            max_features=args.residual_max_features,
            random_state=args.random_state + 20_000,
        )
        for alpha in alphas:
            for clip_pp in clip_values:
                clip = clip_pp / 100.0
                clipped_residual = np.clip(residual_pred, -clip, clip)
                pred = np.clip(base + alpha * clipped_residual, 0.0, 1.0)
                name = f"{model_name}_alpha{alpha:g}_clip{clip_pp:g}pp"
                corrected_by_name[name] = pred
                residual_by_name[name] = residual_pred
                metric_rows.append(
                    _metric_row(
                        name=name,
                        y=y,
                        pred=pred,
                        weights=weights,
                        groups=groups,
                        group_labels=group_labels,
                        params={
                            "residual_model": model_name,
                            "alpha": alpha,
                            "residual_clip_pp": clip_pp,
                        },
                    )
                )

    metric_rows.sort(key=lambda row: (float(row["window_weighted_mae_pp"]), float(row["run_mae_pp"])))
    candidate_rows = [
        row
        for row in metric_rows
        if (args.max_run_mae_pp <= 0.0 or float(row["run_mae_pp"]) <= args.max_run_mae_pp)
        and (
            args.max_abs_aggregate_error_pp <= 0.0
            or abs(float(row["aggregate_error_pp"])) <= args.max_abs_aggregate_error_pp
        )
    ]
    if not candidate_rows:
        print("selection constraints matched no rows; falling back to unconstrained best")
        candidate_rows = metric_rows
    best = candidate_rows[0]
    best["selection_max_run_mae_pp"] = args.max_run_mae_pp
    best["selection_max_abs_aggregate_error_pp"] = args.max_abs_aggregate_error_pp
    best_pred = corrected_by_name[str(best["model"])]
    best_residual = residual_by_name[str(best["model"])]
    prediction_rows = _prediction_rows(
        df=df,
        y=y,
        base=base,
        residual_pred=best_residual,
        corrected=best_pred,
        probabilities=probabilities,
    )

    prefix = RESULTS_DIR / args.results_prefix
    _write_rows(prefix.with_name(prefix.name + "_target_metrics.csv"), _target_metric_rows(targets, probabilities, weights))
    _write_rows(prefix.with_name(prefix.name + "_grid_metrics.csv"), metric_rows)
    _write_rows(prefix.with_name(prefix.name + "_best_model.csv"), [best])
    _write_rows(prefix.with_name(prefix.name + "_window_predictions.csv"), prediction_rows)
    _write_rows(
        prefix.with_name(prefix.name + "_top_gains.csv"),
        sorted(prediction_rows, key=lambda row: -float(row["abs_error_gain_pp"]))[: args.top_n],
    )
    _write_rows(
        prefix.with_name(prefix.name + "_top_regressions.csv"),
        sorted(prediction_rows, key=lambda row: float(row["abs_error_gain_pp"]))[: args.top_n],
    )
    print(
        "best model: "
        f"{best['model']} run_mae={float(best['run_mae_pp']):.3f}pp "
        f"window_weighted_mae={float(best['window_weighted_mae_pp']):.3f}pp "
        f"corr={float(best['corr_window']):.3f}"
    )


if __name__ == "__main__":
    main()
