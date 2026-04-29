#!/usr/bin/env python3
"""Fit and run the saved product inference model for PPC FIX-rate prediction.

The research trainer emits strict nested LORO predictions.  This module turns
the adopted configuration into a deployable single-model artifact:

- ``fit`` trains transition-surrogate classifiers on all training windows and
  fits the adopted residual corrector from calibration probabilities.
- ``infer`` loads that artifact and scores a new pre-augmented window CSV
  without retraining.  With ``--online``, it scores the same artifact in
  causal route order using a planned route window count for run-position
  meta features.

The input window CSV for inference must already contain the deployable window
features used by the adopted product model, including validationhold features.
It does not need demo5 actual labels or solver internals.
"""

from __future__ import annotations

import argparse
import gzip
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.pipeline import Pipeline

from train_ppc_solver_transition_surrogate_stack import (
    RESULTS_DIR,
    _add_temporal_prob_features,
    _append_classifier_meta,
    _classifier,
    _key_frame,
    _logit,
    _positive_probability,
    _read_reference_predictions,
    _residual_model,
    _sim_feature_matrix,
    _transition_targets,
    _write_rows,
)


SCHEMA_VERSION = 1
DEFAULT_MODEL_PATH = (
    RESULTS_DIR
    / "ppc_window_fix_rate_model_stride1_stat_sim_rinex_phasejump_t0p25_gf0p2_simloscont_focused_simadop_nowt_solver_transition_surrogate_nested_et80_validationhold_current_tight_hold_carry_alpha75_isotonic75_meta_run45_product_model.pkl.gz"
)
KEY_COLUMNS = {"city", "run", "window_index"}
REQUIRED_FIT_COLUMNS = KEY_COLUMNS | {
    "sim_matched_epochs",
    "actual_fix_rate_pct",
    "solver_demo5_ratio_mean",
    "solver_demo5_ratio_p90",
    "solver_demo5_ratio_p95",
    "solver_demo5_ratio_mean_past_delta",
    "rtk_lock_p90_p50",
    "rtk_lock_p90_p50_past_delta",
}


def _load_pickle_gz(path: Path) -> dict[str, object]:
    with gzip.open(path, "rb") as fh:
        return pickle.load(fh)


def _save_pickle_gz(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"saved: {path}")


def _require_columns(df: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"{label} is missing required columns: {', '.join(missing)}")


def _sort_windows(df: pd.DataFrame) -> pd.DataFrame:
    _require_columns(df, KEY_COLUMNS, "window CSV")
    return df.sort_values(["city", "run", "window_index"]).reset_index(drop=True)


def _feature_matrix_for_names(df: pd.DataFrame, feature_names: list[str]) -> np.ndarray:
    missing = sorted(name for name in feature_names if name not in df.columns)
    if missing:
        preview = ", ".join(missing[:20])
        suffix = f" ... (+{len(missing) - 20} more)" if len(missing) > 20 else ""
        raise SystemExit(f"window CSV is missing model feature columns: {preview}{suffix}")
    x = df[feature_names].to_numpy(dtype=np.float64)
    x[~np.isfinite(x)] = 0.0
    return x


def _base_from_frame_or_reference(
    df: pd.DataFrame,
    *,
    base_prefix: str | None,
    base_prediction_column: str,
) -> np.ndarray:
    if base_prefix:
        reference = _read_reference_predictions(base_prefix, base_prediction_column)
        values: list[float] = []
        missing: list[str] = []
        for key in _key_frame(df):
            if key in reference:
                values.append(reference[key])
            else:
                missing.append(key)
        if missing:
            preview = ", ".join(missing[:5])
            suffix = f" ... (+{len(missing) - 5} more)" if len(missing) > 5 else ""
            raise SystemExit(
                f"{len(missing)} input windows are missing from the base prediction CSV. "
                f"First missing keys: {preview}{suffix}"
            )
        return np.asarray(values, dtype=np.float64)

    if "base_pred_fix_rate_pct" not in df.columns:
        raise SystemExit("window CSV needs base_pred_fix_rate_pct or --base-prefix")
    base = df["base_pred_fix_rate_pct"].to_numpy(dtype=np.float64) / 100.0
    base[~np.isfinite(base)] = 0.0
    return base


def _fit_classifier_models(
    *,
    x: np.ndarray,
    targets: dict[str, np.ndarray],
    weights: np.ndarray,
    classifier_estimators: int,
    classifier_min_leaf: int,
    classifier_max_features: float,
    random_state: int,
) -> dict[str, object]:
    classifiers: dict[str, object] = {}
    for target_idx, (name, y) in enumerate(targets.items()):
        model = _classifier(
            random_state + target_idx * 1000,
            classifier_estimators,
            classifier_min_leaf,
            classifier_max_features,
        )
        model.fit(x, y, sample_weight=weights)
        classifiers[name] = model
    return classifiers


def _probabilities_from_models(classifiers: dict[str, object], x: np.ndarray) -> dict[str, np.ndarray]:
    return {name: _positive_probability(model, x) for name, model in classifiers.items()}


def _append_classifier_meta_online(
    *,
    df: pd.DataFrame,
    x: np.ndarray,
    names: list[str],
    base: np.ndarray,
    include_base: bool,
    include_city: bool,
    include_run_position: bool,
    planned_window_counts: dict[tuple[str, str], int],
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
        for (city, run), group in df.groupby(["city", "run"], sort=False):
            key = (str(city), str(run))
            planned = int(planned_window_counts.get(key, 0))
            if planned <= 0:
                raise SystemExit(
                    "online inference requires planned window counts for every route; "
                    f"missing {city}/{run}"
                )
            indices = list(group.sort_values("window_index").index)
            if len(indices) > planned:
                raise SystemExit(
                    f"online inference received {len(indices)} rows for {city}/{run}, "
                    f"but planned_window_count is {planned}"
                )
            denom = max(planned - 1, 1)
            for pos, idx in enumerate(indices):
                run_pos[idx] = float(pos)
                run_frac[idx] = float(pos / denom)
                run_remaining[idx] = float(max(planned - 1 - pos, 0) / denom)
                run_len[idx] = float(planned)
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


def _planned_counts_from_input(df: pd.DataFrame) -> dict[tuple[str, str], int]:
    return {
        (str(city), str(run)): int(len(group))
        for (city, run), group in df.groupby(["city", "run"], sort=False)
    }


def _planned_counts_from_column(df: pd.DataFrame, column: str) -> dict[tuple[str, str], int]:
    if column not in df.columns:
        raise SystemExit(
            "online inference needs a planned route length. "
            f"Add column '{column}', pass --planned-window-count, or use --online-use-input-run-length."
        )
    counts: dict[tuple[str, str], int] = {}
    for (city, run), group in df.groupby(["city", "run"], sort=False):
        values = pd.to_numeric(group[column], errors="coerce").dropna().unique()
        if len(values) != 1:
            raise SystemExit(f"{column} must have exactly one value for {city}/{run}")
        value = float(values[0])
        if not np.isfinite(value) or value <= 0.0 or not value.is_integer():
            raise SystemExit(f"{column} must be a positive integer for {city}/{run}")
        counts[(str(city), str(run))] = int(value)
    return counts


def _online_planned_window_counts(args: argparse.Namespace, df: pd.DataFrame, artifact: dict[str, object]) -> dict[tuple[str, str], int] | None:
    if not args.online:
        return None
    if not bool(artifact["classifier_include_run_position"]):
        return {}
    if args.online_use_input_run_length:
        return _planned_counts_from_input(df)
    if args.planned_window_count is not None:
        count = int(args.planned_window_count)
        if count <= 0:
            raise SystemExit("--planned-window-count must be positive")
        return {key: count for key in _planned_counts_from_input(df)}
    return _planned_counts_from_column(df, args.planned_window_count_column)


def _calibration_probabilities(
    *,
    calibration_prediction_csv: Path | None,
    df: pd.DataFrame,
    target_names: list[str],
    fallback: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], str]:
    if calibration_prediction_csv is None:
        return fallback, "full_fit_in_sample"

    cal = _sort_windows(pd.read_csv(calibration_prediction_csv))
    if list(_key_frame(cal)) != list(_key_frame(df)):
        raise SystemExit(
            "calibration prediction CSV keys do not match the training window CSV; "
            "use the adopted LORO prediction CSV for this training set"
        )
    probs: dict[str, np.ndarray] = {}
    missing: list[str] = []
    for name in target_names:
        col = f"{name}_prob"
        if col not in cal.columns:
            missing.append(col)
        else:
            values = cal[col].to_numpy(dtype=np.float64)
            values[~np.isfinite(values)] = 0.0
            probs[name] = values
    if missing:
        raise SystemExit(f"calibration prediction CSV is missing probability columns: {', '.join(missing)}")
    return probs, str(calibration_prediction_csv)


def _fit_residual_corrector(
    *,
    df: pd.DataFrame,
    base: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    probabilities: dict[str, np.ndarray],
    residual_model_name: str,
    residual_estimators: int,
    residual_min_leaf: int,
    residual_max_features: float,
    random_state: int,
) -> object:
    prob_x, _ = _add_temporal_prob_features(df, np.column_stack([probabilities[name] for name in probabilities]), list(probabilities))
    x_meta = np.column_stack([base, _logit(base), prob_x])
    residual_target = y - base
    model = _residual_model(
        residual_model_name,
        random_state + 50_000,
        residual_estimators,
        residual_min_leaf,
        residual_max_features,
    )
    if isinstance(model, Pipeline):
        model.fit(x_meta, residual_target, model__sample_weight=weights)
    else:
        model.fit(x_meta, residual_target, sample_weight=weights)
    return model


def _fit_final_calibrator(
    *,
    df: pd.DataFrame,
    calibration_prediction_csv: Path | None,
    calibrator_name: str,
) -> tuple[object | None, str]:
    if calibrator_name == "none":
        return None, ""
    if calibrator_name != "isotonic":
        raise SystemExit(f"unsupported final calibrator: {calibrator_name}")
    if calibration_prediction_csv is None:
        raise SystemExit("--final-calibrator isotonic requires --calibration-prediction-csv")

    cal = _sort_windows(pd.read_csv(calibration_prediction_csv))
    if list(_key_frame(cal)) != list(_key_frame(df)):
        raise SystemExit(
            "calibration prediction CSV keys do not match the training window CSV; "
            "use the adopted LORO prediction CSV for this training set"
        )
    _require_columns(
        cal,
        {"actual_fix_rate_pct", "corrected_pred_fix_rate_pct", "sim_matched_epochs"},
        "calibration prediction CSV",
    )
    x = cal["corrected_pred_fix_rate_pct"].to_numpy(dtype=np.float64) / 100.0
    y = cal["actual_fix_rate_pct"].to_numpy(dtype=np.float64) / 100.0
    weights = cal["sim_matched_epochs"].to_numpy(dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(weights) & (weights > 0.0)
    if np.count_nonzero(finite) < 2:
        raise SystemExit("calibration prediction CSV has too few finite rows for isotonic calibration")
    order = np.argsort(x[finite])
    model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    model.fit(x[finite][order], y[finite][order], sample_weight=weights[finite][order])
    return model, str(calibration_prediction_csv)


def _apply_final_calibrator(corrected: np.ndarray, artifact: dict[str, object]) -> np.ndarray:
    name = str(artifact.get("final_calibrator_name", "none"))
    if name in ("", "none"):
        return corrected
    if name != "isotonic":
        raise SystemExit(f"unsupported final calibrator in product model: {name}")
    model = artifact.get("final_calibrator")
    if model is None:
        raise SystemExit("product model is missing final_calibrator")
    blend = float(artifact.get("final_calibrator_blend", 1.0))
    if not np.isfinite(blend) or blend < 0.0 or blend > 1.0:
        raise SystemExit("product model final_calibrator_blend must be in [0, 1]")
    calibrated = np.asarray(model.predict(corrected), dtype=np.float64)
    values = (1.0 - blend) * corrected + blend * calibrated
    return np.clip(values, 0.0, 1.0)


def fit_model(args: argparse.Namespace) -> None:
    df = _sort_windows(pd.read_csv(args.window_csv))
    _require_columns(df, REQUIRED_FIT_COLUMNS, "training window CSV")
    weights = df["sim_matched_epochs"].to_numpy(dtype=np.float64)
    weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 1.0)
    y = df["actual_fix_rate_pct"].to_numpy(dtype=np.float64) / 100.0
    base = _base_from_frame_or_reference(
        df,
        base_prefix=args.base_prefix,
        base_prediction_column=args.base_prediction_column,
    )
    x_raw, raw_feature_names = _sim_feature_matrix(df)
    x_classifier, classifier_feature_names = _append_classifier_meta(
        df=df,
        x=x_raw,
        names=raw_feature_names,
        base=base,
        include_base=args.classifier_include_base,
        include_city=args.classifier_include_city,
        include_run_position=args.classifier_include_run_position,
    )
    targets = _transition_targets(df, include_focused_targets=args.include_focused_targets)
    classifiers = _fit_classifier_models(
        x=x_classifier,
        targets=targets,
        weights=weights,
        classifier_estimators=args.classifier_estimators,
        classifier_min_leaf=args.classifier_min_leaf,
        classifier_max_features=args.classifier_max_features,
        random_state=args.random_state,
    )
    full_fit_probs = _probabilities_from_models(classifiers, x_classifier)
    calibration_probs, calibration_source = _calibration_probabilities(
        calibration_prediction_csv=args.calibration_prediction_csv,
        df=df,
        target_names=list(targets),
        fallback=full_fit_probs,
    )
    residual_model = _fit_residual_corrector(
        df=df,
        base=base,
        y=y,
        weights=weights,
        probabilities=calibration_probs,
        residual_model_name=args.residual_model,
        residual_estimators=args.residual_estimators,
        residual_min_leaf=args.residual_min_leaf,
        residual_max_features=args.residual_max_features,
        random_state=args.random_state,
    )
    final_calibrator, final_calibrator_source = _fit_final_calibrator(
        df=df,
        calibration_prediction_csv=args.calibration_prediction_csv,
        calibrator_name=args.final_calibrator,
    )
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "training_rows": int(len(df)),
        "raw_feature_names": raw_feature_names,
        "classifier_feature_names": classifier_feature_names,
        "target_names": list(targets),
        "classifiers": classifiers,
        "residual_model": residual_model,
        "residual_model_name": args.residual_model,
        "residual_alpha": float(args.residual_alpha),
        "residual_clip_pp": float(args.residual_clip_pp),
        "classifier_include_base": bool(args.classifier_include_base),
        "classifier_include_city": bool(args.classifier_include_city),
        "classifier_include_run_position": bool(args.classifier_include_run_position),
        "classifier_estimators": int(args.classifier_estimators),
        "classifier_min_leaf": int(args.classifier_min_leaf),
        "classifier_max_features": float(args.classifier_max_features),
        "residual_estimators": int(args.residual_estimators),
        "residual_min_leaf": int(args.residual_min_leaf),
        "residual_max_features": float(args.residual_max_features),
        "random_state": int(args.random_state),
        "base_prediction_column": args.base_prediction_column,
        "calibration_source": calibration_source,
        "final_calibrator_name": args.final_calibrator,
        "final_calibrator": final_calibrator,
        "final_calibrator_source": final_calibrator_source,
        "final_calibrator_blend": float(args.final_calibrator_blend),
    }
    _save_pickle_gz(args.model_output, payload)


def predict_frame(
    df: pd.DataFrame,
    artifact: dict[str, object],
    base: np.ndarray,
    *,
    online_planned_window_counts: dict[tuple[str, str], int] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    if int(artifact.get("schema_version", -1)) != SCHEMA_VERSION:
        raise SystemExit(f"unsupported product model schema: {artifact.get('schema_version')}")
    raw_feature_names = list(artifact["raw_feature_names"])
    x_raw = _feature_matrix_for_names(df, raw_feature_names)
    if online_planned_window_counts is None:
        x_classifier, classifier_feature_names = _append_classifier_meta(
            df=df,
            x=x_raw,
            names=raw_feature_names,
            base=base,
            include_base=bool(artifact["classifier_include_base"]),
            include_city=bool(artifact["classifier_include_city"]),
            include_run_position=bool(artifact["classifier_include_run_position"]),
        )
    else:
        x_classifier, classifier_feature_names = _append_classifier_meta_online(
            df=df,
            x=x_raw,
            names=raw_feature_names,
            base=base,
            include_base=bool(artifact["classifier_include_base"]),
            include_city=bool(artifact["classifier_include_city"]),
            include_run_position=bool(artifact["classifier_include_run_position"]),
            planned_window_counts=online_planned_window_counts,
        )
    expected_names = list(artifact["classifier_feature_names"])
    if classifier_feature_names != expected_names:
        raise SystemExit("classifier feature schema mismatch; check city/meta options and input columns")
    classifiers = artifact["classifiers"]
    probabilities = _probabilities_from_models(classifiers, x_classifier)
    target_names = list(artifact["target_names"])
    prob_x, _ = _add_temporal_prob_features(df, np.column_stack([probabilities[name] for name in target_names]), target_names)
    x_meta = np.column_stack([base, _logit(base), prob_x])
    residual_model = artifact["residual_model"]
    residual = np.asarray(residual_model.predict(x_meta), dtype=np.float64)
    clip = float(artifact["residual_clip_pp"]) / 100.0
    alpha = float(artifact["residual_alpha"])
    corrected = np.clip(base + alpha * np.clip(residual, -clip, clip), 0.0, 1.0)
    corrected = _apply_final_calibrator(corrected, artifact)
    return residual, corrected, probabilities


def _prediction_rows(
    *,
    df: pd.DataFrame,
    base: np.ndarray,
    residual: np.ndarray,
    corrected: np.ndarray,
    probabilities: dict[str, np.ndarray],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    has_actual = "actual_fix_rate_pct" in df.columns
    actual = df["actual_fix_rate_pct"].to_numpy(dtype=np.float64) / 100.0 if has_actual else None
    for idx, source in df.reset_index(drop=True).iterrows():
        out: dict[str, object] = {
            "city": source["city"],
            "run": source["run"],
            "window_index": int(source["window_index"]),
            "sim_matched_epochs": source.get("sim_matched_epochs", ""),
            "base_pred_fix_rate_pct": 100.0 * float(base[idx]),
            "residual_pred_pp": 100.0 * float(residual[idx]),
            "corrected_pred_fix_rate_pct": 100.0 * float(corrected[idx]),
        }
        if has_actual and actual is not None:
            base_error = 100.0 * (base[idx] - actual[idx])
            corrected_error = 100.0 * (corrected[idx] - actual[idx])
            out.update(
                {
                    "actual_fix_rate_pct": 100.0 * float(actual[idx]),
                    "base_error_pp": base_error,
                    "corrected_error_pp": corrected_error,
                    "abs_error_gain_pp": abs(base_error) - abs(corrected_error),
                }
            )
        for name, prob in probabilities.items():
            out[f"{name}_prob"] = float(prob[idx])
        rows.append(out)
    return rows


def _route_rows(window_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    df = pd.DataFrame(window_rows)
    for (city, run), group in df.groupby(["city", "run"], sort=True):
        weights = group["sim_matched_epochs"].to_numpy(dtype=np.float64) if "sim_matched_epochs" in group else np.ones(len(group))
        weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 1.0)
        base = float(np.average(group["base_pred_fix_rate_pct"].to_numpy(dtype=np.float64), weights=weights))
        corrected = float(np.average(group["corrected_pred_fix_rate_pct"].to_numpy(dtype=np.float64), weights=weights))
        out: dict[str, object] = {
            "city": city,
            "run": run,
            "window_count": int(len(group)),
            "baseline_pred_fix_rate_pct": round(base, 3),
            "adopted_pred_fix_rate_pct": round(corrected, 3),
        }
        if "actual_fix_rate_pct" in group.columns:
            actual = float(np.average(group["actual_fix_rate_pct"].to_numpy(dtype=np.float64), weights=weights))
            out.update(
                {
                    "actual_fix_rate_pct": round(actual, 3),
                    "adopted_abs_error_pp": round(abs(corrected - actual), 3),
                    "adopted_signed_error_pp": round(corrected - actual, 3),
                }
            )
        rows.append(out)
    return rows


def run_inference(args: argparse.Namespace) -> None:
    artifact = _load_pickle_gz(args.model)
    df = _sort_windows(pd.read_csv(args.window_csv))
    _require_columns(df, KEY_COLUMNS, "inference window CSV")
    base = _base_from_frame_or_reference(
        df,
        base_prefix=args.base_prefix,
        base_prediction_column=args.base_prediction_column,
    )
    planned_counts = _online_planned_window_counts(args, df, artifact)
    residual, corrected, probabilities = predict_frame(
        df,
        artifact,
        base,
        online_planned_window_counts=planned_counts,
    )
    window_rows = _prediction_rows(
        df=df,
        base=base,
        residual=residual,
        corrected=corrected,
        probabilities=probabilities,
    )
    output_prefix = args.output_prefix
    _write_rows(output_prefix.with_name(output_prefix.name + "_window_predictions.csv"), window_rows)
    _write_rows(output_prefix.with_name(output_prefix.name + "_route_predictions.csv"), _route_rows(window_rows))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit or run the PPC product inference model")
    sub = parser.add_subparsers(dest="cmd", required=True)

    fit = sub.add_parser("fit", help="fit and save a product inference model artifact")
    fit.add_argument("--window-csv", type=Path, required=True)
    fit.add_argument("--base-prefix", required=True)
    fit.add_argument("--base-prediction-column", default="corrected_pred_fix_rate_pct")
    fit.add_argument("--calibration-prediction-csv", type=Path)
    fit.add_argument("--model-output", type=Path, default=DEFAULT_MODEL_PATH)
    fit.add_argument("--classifier-estimators", type=int, default=80)
    fit.add_argument("--classifier-min-leaf", type=int, default=4)
    fit.add_argument("--classifier-max-features", type=float, default=0.75)
    fit.add_argument("--classifier-include-base", action="store_true")
    fit.add_argument("--classifier-include-city", action="store_true")
    fit.add_argument("--classifier-include-run-position", action="store_true")
    fit.add_argument("--include-focused-targets", action="store_true")
    fit.add_argument("--residual-model", default="ridge")
    fit.add_argument("--residual-alpha", type=float, default=0.75)
    fit.add_argument("--residual-clip-pp", type=float, default=50.0)
    fit.add_argument("--residual-estimators", type=int, default=80)
    fit.add_argument("--residual-min-leaf", type=int, default=3)
    fit.add_argument("--residual-max-features", type=float, default=0.75)
    fit.add_argument("--final-calibrator", choices=("none", "isotonic"), default="none")
    fit.add_argument("--final-calibrator-blend", type=float, default=1.0)
    fit.add_argument("--random-state", type=int, default=2034)

    infer = sub.add_parser("infer", help="score a window CSV with a saved product model")
    infer.add_argument("--window-csv", type=Path, required=True)
    infer.add_argument("--base-prefix")
    infer.add_argument("--base-prediction-column", default="corrected_pred_fix_rate_pct")
    infer.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    infer.add_argument("--output-prefix", type=Path, required=True)
    infer.add_argument(
        "--online",
        action="store_true",
        help="score in causal online-compatible mode using planned route window counts",
    )
    infer.add_argument(
        "--planned-window-count",
        type=int,
        help="planned window count for every route when --online is used",
    )
    infer.add_argument(
        "--planned-window-count-column",
        default="planned_window_count",
        help="per-route planned window count column used by --online",
    )
    infer.add_argument(
        "--online-use-input-run-length",
        action="store_true",
        help="offline smoke/backfill mode: use input row count as planned route length",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        if args.cmd == "fit":
            fit_model(args)
        elif args.cmd == "infer":
            run_inference(args)
        else:
            raise AssertionError(args.cmd)
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
