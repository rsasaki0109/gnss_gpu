#!/usr/bin/env python3
"""Summarize PPC structural ablations without tuning on the evaluated rows.

The same fixed metrics are emitted for the pooled data, every official run, and
the predeclared blocked spans. Missing method-specific diagnostics stay empty;
they are never interpreted as zeros.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_SPANS = Path(__file__).with_name("blocked_span_manifest.csv")


def _numeric(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce")


def _truth(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame:
        return pd.Series(False, index=frame.index, dtype=bool)
    values = frame[name]
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    return values.astype(str).str.strip().str.lower().isin({"1", "true", "yes"})


def _finite(values: pd.Series) -> np.ndarray:
    array = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    return array[np.isfinite(array)]


def _quantile(values: pd.Series, q: float) -> float:
    finite = _finite(values)
    return float(np.quantile(finite, q)) if finite.size else np.nan


def _mean(values: pd.Series) -> float:
    finite = _finite(values)
    return float(np.mean(finite)) if finite.size else np.nan


def _rate(mask: pd.Series, eligible: pd.Series | None = None) -> float:
    selected = mask if eligible is None else mask[eligible]
    return float(selected.mean()) if len(selected) else np.nan


def summarize_scope(frame: pd.DataFrame, *, scope: str, scope_id: str) -> dict[str, object]:
    error = _numeric(frame, "emit_to_ref_m")
    error_valid = error.notna()
    mode_eval = _numeric(frame, "pf_mode_count").notna()
    mode_accept = _truth(frame, "pf_mode_selection_accepted")
    mode_count = _numeric(frame, "pf_mode_count")
    selected_xyz = np.column_stack(
        [
            _numeric(frame, "pf_mode_selected_x"),
            _numeric(frame, "pf_mode_selected_y"),
            _numeric(frame, "pf_mode_selected_z"),
        ]
    )
    reference_xyz = np.column_stack(
        [_numeric(frame, "ref_x"), _numeric(frame, "ref_y"), _numeric(frame, "ref_z")]
    )
    selected_error = pd.Series(
        np.linalg.norm(selected_xyz - reference_xyz, axis=1), index=frame.index
    )
    pf_source = (
        frame["emitted_source"].fillna("").astype(str).str.startswith("pf")
        if "emitted_source" in frame
        else pd.Series(True, index=frame.index)
    )
    mode_emit_eligible = (
        _truth(frame, "pf_mode_emit_epoch_eligible")
        if "pf_mode_emit_epoch_eligible" in frame
        else mode_eval
    )
    mode_counterfactual = (
        mode_accept & mode_emit_eligible & pf_source & selected_error.notna() & error_valid
    )
    mode_delta = selected_error - error
    doppler_known = _numeric(frame, "doppler_signal_wavelength_known_count")
    doppler_unknown = _numeric(frame, "doppler_signal_wavelength_unknown_count")
    doppler_diag = doppler_known.notna()
    ffbsi_available = _truth(frame, "pf_ffbsi_available")
    ffbsi_applied = _truth(frame, "pf_ffbsi_applied")
    ffbsi_error = _numeric(frame, "pf_ffbsi_to_ref_m")
    if "pf_ffbsi_baseline_to_ref_m" in frame:
        ffbsi_baseline_error = _numeric(frame, "pf_ffbsi_baseline_to_ref_m")
    elif "pf_before_emit_to_ref_m" in frame:
        # Compatibility for full-six artifacts written before the dedicated
        # counterfactual column existed. FFBSi is only eligible for PF output,
        # so this is the exact pre-replacement position error in those files.
        ffbsi_baseline_error = _numeric(frame, "pf_before_emit_to_ref_m")
    else:
        ffbsi_baseline_error = error
    ffbsi_counterfactual = ffbsi_applied & ffbsi_error.notna() & ffbsi_baseline_error.notna()
    ffbsi_delta = ffbsi_error - ffbsi_baseline_error

    return {
        "scope": scope,
        "scope_id": scope_id,
        "epochs": int(len(frame)),
        "reference_coverage": _rate(error_valid),
        "pass_0p5": _rate(error <= 0.5, error_valid),
        "pass_1m": _rate(error <= 1.0, error_valid),
        "pass_3m": _rate(error <= 3.0, error_valid),
        "error_p50_m": _quantile(error, 0.50),
        "error_p95_m": _quantile(error, 0.95),
        "error_p99_m": _quantile(error, 0.99),
        "mode_evaluated": int(mode_eval.sum()),
        "mode_accepted": int((mode_eval & mode_accept).sum()),
        "mode_abstention_rate": 1.0 - _rate(mode_accept, mode_eval),
        "mode_multimodal_rate": _rate(mode_count >= 2, mode_eval),
        "mode_mean_distance_p95_m": _quantile(
            _numeric(frame.loc[mode_eval], "pf_mode_weighted_mean_distance_m"), 0.95
        ),
        "mode_counterfactual_emissions": int(mode_counterfactual.sum()),
        "mode_counterfactual_improved_rate": _rate(mode_delta < 0.0, mode_counterfactual),
        "mode_counterfactual_error_delta_mean_m": _mean(mode_delta[mode_counterfactual]),
        "mode_counterfactual_error_delta_p95_m": _quantile(
            mode_delta[mode_counterfactual], 0.95
        ),
        "doppler_diagnostic_epochs": int(doppler_diag.sum()),
        "doppler_update_rate": _rate(_truth(frame, "doppler_update_applied"), doppler_diag),
        "doppler_known_rows_mean": _mean(doppler_known),
        "doppler_unknown_rows_mean": _mean(doppler_unknown),
        "doppler_clock_groups_mean": _mean(_numeric(frame, "doppler_clock_group_count")),
        "doppler_clock_fit_rms_p95_mps": _quantile(
            _numeric(frame, "doppler_clock_fit_rms_mps"), 0.95
        ),
        "doppler_clock_drift_span_p95_mps": _quantile(
            _numeric(frame, "doppler_clock_drift_span_mps"), 0.95
        ),
        "ffbsi_available": int(ffbsi_available.sum()),
        "ffbsi_applied": int(ffbsi_applied.sum()),
        "ffbsi_abstention_rate": 1.0 - _rate(ffbsi_applied, ffbsi_available),
        "ffbsi_correction_p95_m": _quantile(_numeric(frame, "pf_ffbsi_correction_m"), 0.95),
        "ffbsi_counterfactual_epochs": int(ffbsi_counterfactual.sum()),
        "ffbsi_improved_rate": _rate(ffbsi_delta < 0.0, ffbsi_counterfactual),
        "ffbsi_error_delta_mean_m": _mean(ffbsi_delta[ffbsi_counterfactual]),
        "ffbsi_error_delta_p95_m": _quantile(ffbsi_delta[ffbsi_counterfactual], 0.95),
    }


def summarize(
    epochs: pd.DataFrame,
    spans: pd.DataFrame,
    runs: pd.DataFrame | None = None,
    *,
    holdout_start_epoch: int = 200,
) -> pd.DataFrame:
    required = {"city", "run", "epoch"}
    missing = required - set(epochs.columns)
    if missing:
        raise ValueError(f"internal epoch CSV lacks columns: {sorted(missing)}")

    pooled = summarize_scope(epochs, scope="pooled", scope_id="all")
    pooled["evaluation_role"] = "diagnostic"
    rows = [pooled]
    holdout_mask = _numeric(epochs, "epoch") >= int(holdout_start_epoch)
    pooled_holdout = summarize_scope(
        epochs.loc[holdout_mask], scope="pooled_holdout", scope_id="all_after_development"
    )
    pooled_holdout["evaluation_role"] = "holdout"
    rows.append(pooled_holdout)
    for (city, run), group in epochs.groupby(["city", "run"], sort=True):
        run_row = summarize_scope(group, scope="run", scope_id=f"{city}_{run}")
        run_row["evaluation_role"] = "diagnostic"
        rows.append(run_row)
        run_holdout = summarize_scope(
            group.loc[_numeric(group, "epoch") >= int(holdout_start_epoch)],
            scope="run_holdout",
            scope_id=f"{city}_{run}_after_development",
        )
        run_holdout["evaluation_role"] = "holdout"
        rows.append(run_holdout)

    for span in spans.itertuples(index=False):
        mask = (
            (epochs["city"] == span.city)
            & (epochs["run"] == span.run)
            & (_numeric(epochs, "epoch") >= int(span.start_epoch))
            & (_numeric(epochs, "epoch") < int(span.end_epoch_exclusive))
        )
        span_row = summarize_scope(
            epochs.loc[mask], scope="blocked_span", scope_id=span.span_id
        )
        span_row["evaluation_role"] = getattr(span, "evaluation_role", "predeclared")
        rows.append(span_row)

    result = pd.DataFrame(rows)
    result["ms_per_epoch"] = np.nan
    result["runtime_scope"] = "unavailable"
    if runs is not None and "ms_per_epoch" in runs:
        runtime = runs.copy()
        runtime["scope_id"] = runtime["city"].astype(str) + "_" + runtime["run"].astype(str)
        mapping = runtime.groupby("scope_id")["ms_per_epoch"].mean().to_dict()
        run_mask = result["scope"] == "run"
        result.loc[run_mask, "ms_per_epoch"] = result.loc[run_mask, "scope_id"].map(mapping)
        result.loc[run_mask, "runtime_scope"] = "measured_full_run"
        run_holdout_mask = result["scope"] == "run_holdout"
        holdout_keys = result.loc[run_holdout_mask, "scope_id"].str.replace(
            "_after_development", "", regex=False
        )
        result.loc[run_holdout_mask, "ms_per_epoch"] = holdout_keys.map(mapping).to_numpy()
        result.loc[run_holdout_mask, "runtime_scope"] = "full_run_average_proxy"
        pooled_runtime = pd.to_numeric(runtime["ms_per_epoch"], errors="coerce").mean()
        result.loc[result["scope"].isin(["pooled", "pooled_holdout"]), "ms_per_epoch"] = (
            pooled_runtime
        )
        result.loc[
            result["scope"].isin(["pooled", "pooled_holdout"]), "runtime_scope"
        ] = "mean_of_measured_full_runs"
        if not spans.empty:
            span_to_run = {
                str(row.span_id): f"{row.city}_{row.run}"
                for row in spans.itertuples(index=False)
            }
            blocked_mask = result["scope"] == "blocked_span"
            blocked_keys = result.loc[blocked_mask, "scope_id"].map(span_to_run)
            result.loc[blocked_mask, "ms_per_epoch"] = blocked_keys.map(mapping).to_numpy()
            result.loc[blocked_mask, "runtime_scope"] = "full_run_average_proxy"
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("internal_epochs", type=Path)
    parser.add_argument("--runs-csv", type=Path)
    parser.add_argument("--spans", type=Path, default=DEFAULT_SPANS)
    parser.add_argument("--holdout-start-epoch", type=int, default=200)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    epochs = pd.read_csv(args.internal_epochs, low_memory=False)
    spans = pd.read_csv(args.spans)
    runs = pd.read_csv(args.runs_csv) if args.runs_csv else None
    result = summarize(
        epochs,
        spans,
        runs,
        holdout_start_epoch=args.holdout_start_epoch,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Saved {len(result)} scopes: {args.output}")


if __name__ == "__main__":
    main()
