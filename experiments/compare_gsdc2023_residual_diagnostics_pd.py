#!/usr/bin/env python3
"""Compare P/D MATLAB residual-diagnostics columns against bridge values."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
import sys

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from experiments.compare_gsdc2023_residual_values import build_bridge_residual_frame  # noqa: E402
from experiments.gsdc2023_audit_cli import (  # noqa: E402
    add_data_root_trip_args as _add_data_root_trip_args,
    add_max_epochs_arg as _add_max_epochs_arg,
    add_multi_gnss_arg as _add_multi_gnss_arg,
    add_output_dir_arg as _add_output_dir_arg,
    nonnegative_max_epochs as _nonnegative_max_epochs,
    resolve_trip_dir as _resolve_trip_dir,
    resolved_output_root as _resolved_output_root,
)
from experiments.gsdc2023_audit_output import (  # noqa: E402
    print_summary_and_output_dir as _print_summary_and_output_dir,
    timestamped_output_dir as _timestamped_output_dir,
    write_summary_json as _write_summary_json,
)
from experiments.gsdc2023_raw_bridge import DEFAULT_ROOT  # noqa: E402
from experiments.gsdc2023_residual_audit import (  # noqa: E402
    RESIDUAL_KEY_COLUMNS,
    matlab_residual_frame as _matlab_residual_frame,
)
from experiments.gsdc2023_trip_window import (  # noqa: E402
    settings_epoch_window_for_trip as _settings_epoch_window_for_trip,
    trim_epoch_window as _trim_epoch_window,
)


PD_VALUE_KEY_COLUMNS = ["field", "diagnostics_column", "freq", "epoch_index", "utcTimeMillis", "sys", "svid"]
_DIAGNOSTICS_KEY_COLUMNS = ["freq", "epoch_index", "utcTimeMillis", "sys", "svid"]
PD_WIDE_VALUE_KEY_COLUMNS = ["diagnostics_column", *_DIAGNOSTICS_KEY_COLUMNS]
PD_DIAGNOSTICS_VALUE_MAP: tuple[tuple[str, str, str, str], ...] = (
    ("P", "p_residual_m", "matlab_residual", "bridge_residual"),
    ("P", "p_pre_respc_m", "matlab_pre_residual", "bridge_pre_residual"),
    ("P", "p_clock_bias_m", "matlab_common_bias", "bridge_common_bias"),
    ("P", "p_corrected_m", "matlab_observation", "bridge_observation"),
    ("P", "p_range_m", "matlab_model", "bridge_model"),
    ("D", "d_residual_mps", "matlab_residual", "bridge_residual"),
    ("D", "d_pre_resd_m", "matlab_pre_residual", "bridge_pre_residual"),
    ("D", "d_clock_bias_mps", "matlab_common_bias", "bridge_common_bias"),
    ("D", "d_obs_mps", "matlab_observation", "bridge_observation"),
    ("D", "d_model_mps", "matlab_model", "bridge_model"),
)
PD_WIDE_COMPONENT_MAP: tuple[tuple[str | None, str, str], ...] = (
    (None, "sat_x_m", "bridge_sat_x"),
    (None, "sat_y_m", "bridge_sat_y"),
    (None, "sat_z_m", "bridge_sat_z"),
    (None, "sat_vx_mps", "bridge_sat_vx"),
    (None, "sat_vy_mps", "bridge_sat_vy"),
    (None, "sat_vz_mps", "bridge_sat_vz"),
    (None, "sat_clock_bias_m", "bridge_sat_clock_bias"),
    (None, "sat_clock_drift_mps", "bridge_sat_clock_drift"),
    (None, "sat_iono_m", "bridge_sat_iono"),
    (None, "sat_trop_m", "bridge_sat_trop"),
    ("P", "sat_range_m", "bridge_model"),
    ("D", "sat_rate_mps", "bridge_model"),
    (None, "sat_elevation_deg", "bridge_sat_elevation"),
    (None, "rcv_x_m", "bridge_rcv_x"),
    (None, "rcv_y_m", "bridge_rcv_y"),
    (None, "rcv_z_m", "bridge_rcv_z"),
    (None, "rcv_vx_mps", "bridge_rcv_vx"),
    (None, "rcv_vy_mps", "bridge_rcv_vy"),
    (None, "rcv_vz_mps", "bridge_rcv_vz"),
    ("P", "obs_clk_m", "bridge_obs_clk"),
    ("D", "obs_dclk_m", "bridge_obs_dclk"),
    ("P", "p_isb_m", "bridge_isb"),
)
PD_WIDE_BOOLEAN_MAP: tuple[tuple[str, str, str], ...] = (
    ("P", "p_pre_finite", "bridge_pre_finite"),
    ("D", "d_pre_finite", "bridge_pre_finite"),
    ("P", "p_factor_finite", "bridge_factor_finite"),
    ("D", "d_factor_finite", "bridge_factor_finite"),
)
PD_WIDE_EXPORT_COLUMNS = (
    "sat_col",
    "p_residual_m",
    "d_residual_mps",
    "p_pre_respc_m",
    "d_pre_resd_m",
    "p_corrected_m",
    "p_range_m",
    "d_obs_mps",
    "d_model_mps",
    "sat_x_m",
    "sat_y_m",
    "sat_z_m",
    "sat_vx_mps",
    "sat_vy_mps",
    "sat_vz_mps",
    "sat_clock_bias_m",
    "sat_clock_drift_mps",
    "sat_iono_m",
    "sat_trop_m",
    "sat_range_m",
    "sat_rate_mps",
    "sat_elevation_deg",
    "rcv_x_m",
    "rcv_y_m",
    "rcv_z_m",
    "rcv_vx_mps",
    "rcv_vy_mps",
    "rcv_vz_mps",
    "obs_clk_m",
    "obs_dclk_m",
    "p_isb_m",
    "p_clock_bias_m",
    "d_clock_bias_mps",
    "p_pre_finite",
    "d_pre_finite",
    "p_factor_finite",
    "d_factor_finite",
)
PD_WIDE_FRAME_COLUMNS = (*_DIAGNOSTICS_KEY_COLUMNS, *PD_WIDE_EXPORT_COLUMNS)
PD_WIDE_VALUE_COLUMNS = PD_WIDE_EXPORT_COLUMNS

BridgeFrameFn = Callable[..., pd.DataFrame]


def _bridge_pd_factor_finite_key_filter(
    trip_dir: Path,
    *,
    max_epochs: int,
    multi_gnss: bool,
) -> set[tuple[object, ...]]:
    from experiments.compare_gsdc2023_factor_masks import build_bridge_factor_mask

    frame = build_bridge_factor_mask(
        Path(trip_dir),
        max_epochs=max_epochs,
        multi_gnss=multi_gnss,
    )
    if frame.empty:
        return set()
    frame = frame[frame["field"].astype(str).isin(("P", "D"))].copy()
    if frame.empty:
        return set()
    return set(frame[RESIDUAL_KEY_COLUMNS].itertuples(index=False, name=None))


def _normalized_value_frame(frame: pd.DataFrame, *, value_column: str) -> pd.DataFrame:
    out = frame.copy()
    if out.empty:
        return pd.DataFrame(columns=PD_VALUE_KEY_COLUMNS + [value_column])
    for column in ("epoch_index", "utcTimeMillis", "sys", "svid"):
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0).astype(np.int64)
    out["field"] = out["field"].astype(str)
    out["diagnostics_column"] = out["diagnostics_column"].astype(str)
    out["freq"] = out["freq"].astype(str)
    out[value_column] = pd.to_numeric(out[value_column], errors="coerce")
    out = out[np.isfinite(out[value_column].to_numpy(dtype=np.float64))]
    return out.drop_duplicates(PD_VALUE_KEY_COLUMNS).reset_index(drop=True)


def _melt_pd_values(frame: pd.DataFrame, *, side: str) -> pd.DataFrame:
    value_column = f"{side}_value"
    rows: list[pd.DataFrame] = []
    for field, diagnostics_column, matlab_column, bridge_column in PD_DIAGNOSTICS_VALUE_MAP:
        source_column = matlab_column if side == "matlab" else bridge_column
        if source_column not in frame.columns:
            continue
        sub = frame.loc[frame["field"].astype(str).eq(field), RESIDUAL_KEY_COLUMNS].copy()
        if sub.empty:
            continue
        sub["diagnostics_column"] = diagnostics_column
        sub[value_column] = pd.to_numeric(frame.loc[sub.index, source_column], errors="coerce")
        rows.append(sub)
    if not rows:
        return pd.DataFrame(columns=PD_VALUE_KEY_COLUMNS + [value_column])
    return _normalized_value_frame(pd.concat(rows, ignore_index=True), value_column=value_column)


def matlab_residual_diagnostics_pd_values(diagnostics_path: Path) -> pd.DataFrame:
    """Return MATLAB P/D residual diagnostics as long ``diagnostics_column`` values."""

    frame = _matlab_residual_frame(Path(diagnostics_path))
    return _melt_pd_values(frame, side="matlab")


def _normalized_wide_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in ("epoch_index", "utcTimeMillis", "sys", "svid", "sat_col"):
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    for column in ("epoch_index", "utcTimeMillis", "sys", "svid"):
        out[column] = out[column].fillna(0).astype(np.int64)
    out["freq"] = out["freq"].astype(str)
    for column in PD_WIDE_VALUE_COLUMNS:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def _key_frame(keys: set[tuple[object, ...]]) -> pd.DataFrame:
    if not keys:
        return pd.DataFrame(columns=_DIAGNOSTICS_KEY_COLUMNS)
    out = pd.DataFrame(list(keys), columns=_DIAGNOSTICS_KEY_COLUMNS)
    return _normalized_wide_frame(out).loc[:, _DIAGNOSTICS_KEY_COLUMNS].drop_duplicates()


def matlab_residual_diagnostics_pd_wide_frame(
    diagnostics_path: Path,
    *,
    key_filter: set[tuple[object, ...]] | None = None,
) -> pd.DataFrame:
    """Return MATLAB diagnostics in the bridge writer-shaped P/D wide subset."""

    frame = pd.read_csv(Path(diagnostics_path))
    missing = sorted(set(PD_WIDE_FRAME_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"diagnostics CSV missing P/D wide subset columns {missing}: {diagnostics_path}")
    out = _normalized_wide_frame(frame.loc[:, list(PD_WIDE_FRAME_COLUMNS)])
    if key_filter is not None:
        out = out.merge(_key_frame(key_filter), on=_DIAGNOSTICS_KEY_COLUMNS, how="inner")
    out = out.drop_duplicates(_DIAGNOSTICS_KEY_COLUMNS)
    return out.sort_values(["freq", "sat_col", "sys", "svid", "epoch_index"], kind="mergesort").reset_index(drop=True)


def bridge_residual_diagnostics_pd_values(
    trip_dir: Path,
    *,
    max_epochs: int = 0,
    multi_gnss: bool = False,
    apply_observation_mask: bool = True,
    include_inactive_observations: bool = False,
    inactive_key_filter: set[tuple[object, ...]] | None = None,
    bridge_frame_fn: BridgeFrameFn = build_bridge_residual_frame,
) -> pd.DataFrame:
    """Return bridge P/D residual diagnostics as long ``diagnostics_column`` values."""

    bridge = bridge_frame_fn(
        Path(trip_dir),
        max_epochs=max_epochs,
        multi_gnss=multi_gnss,
        apply_observation_mask=apply_observation_mask,
        include_inactive_observations=include_inactive_observations,
        inactive_key_filter=inactive_key_filter,
    )
    return _melt_pd_values(bridge, side="bridge")


def bridge_residual_diagnostics_pd_wide_values(
    trip_dir: Path,
    *,
    max_epochs: int = 0,
    multi_gnss: bool = False,
    apply_observation_mask: bool = True,
    include_inactive_observations: bool = False,
    inactive_key_filter: set[tuple[object, ...]] | None = None,
    factor_finite_key_filter: set[tuple[object, ...]] | None = None,
    bridge_frame_fn: BridgeFrameFn = build_bridge_residual_frame,
) -> pd.DataFrame:
    """Return bridge P/D diagnostics in a wide MATLAB-sidecar column subset."""

    if factor_finite_key_filter is None:
        factor_finite_key_filter = _bridge_pd_factor_finite_key_filter(
            Path(trip_dir),
            max_epochs=max_epochs,
            multi_gnss=multi_gnss,
        )
    bridge = bridge_frame_fn(
        Path(trip_dir),
        max_epochs=max_epochs,
        multi_gnss=multi_gnss,
        apply_observation_mask=apply_observation_mask,
        include_inactive_observations=include_inactive_observations,
        inactive_key_filter=inactive_key_filter,
        factor_finite_key_filter=factor_finite_key_filter,
    )
    return bridge_residual_diagnostics_pd_wide_export_frame(bridge)


def _normalized_wide_value_frame(frame: pd.DataFrame, *, value_column: str) -> pd.DataFrame:
    out = frame.copy()
    if out.empty:
        return pd.DataFrame(columns=PD_WIDE_VALUE_KEY_COLUMNS + [value_column])
    for column in ("epoch_index", "utcTimeMillis", "sys", "svid"):
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0).astype(np.int64)
    out["diagnostics_column"] = out["diagnostics_column"].astype(str)
    out["freq"] = out["freq"].astype(str)
    out[value_column] = pd.to_numeric(out[value_column], errors="coerce")
    out = out[np.isfinite(out[value_column].to_numpy(dtype=np.float64))]
    return out.drop_duplicates(PD_WIDE_VALUE_KEY_COLUMNS).reset_index(drop=True)


def _melt_pd_wide_values(frame: pd.DataFrame, *, side: str) -> pd.DataFrame:
    value_column = f"{side}_value"
    rows: list[pd.DataFrame] = []
    for diagnostics_column in PD_WIDE_VALUE_COLUMNS:
        if diagnostics_column not in frame.columns:
            continue
        sub = frame.loc[:, _DIAGNOSTICS_KEY_COLUMNS].copy()
        sub["diagnostics_column"] = diagnostics_column
        sub[value_column] = pd.to_numeric(frame[diagnostics_column], errors="coerce")
        rows.append(sub)
    if not rows:
        return pd.DataFrame(columns=PD_WIDE_VALUE_KEY_COLUMNS + [value_column])
    return _normalized_wide_value_frame(pd.concat(rows, ignore_index=True), value_column=value_column)


def bridge_residual_diagnostics_pd_export_frame(values: pd.DataFrame) -> pd.DataFrame:
    """Pivot long bridge P/D values to a MATLAB-column subset export."""

    if values.empty:
        return pd.DataFrame(columns=_DIAGNOSTICS_KEY_COLUMNS + [row[1] for row in PD_DIAGNOSTICS_VALUE_MAP])
    bridge_values = values.copy()
    if "bridge_value" not in bridge_values.columns:
        raise ValueError("bridge P/D values must include bridge_value")
    wide = bridge_values.pivot_table(
        index=_DIAGNOSTICS_KEY_COLUMNS,
        columns="diagnostics_column",
        values="bridge_value",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None
    for _field, diagnostics_column, _matlab_column, _bridge_column in PD_DIAGNOSTICS_VALUE_MAP:
        if diagnostics_column not in wide.columns:
            wide[diagnostics_column] = np.nan
    return wide[_DIAGNOSTICS_KEY_COLUMNS + [row[1] for row in PD_DIAGNOSTICS_VALUE_MAP]]


def bridge_residual_diagnostics_pd_wide_export_frame(bridge_residuals: pd.DataFrame) -> pd.DataFrame:
    """Return a writer-shaped P/D diagnostics subset with shared component columns."""

    if bridge_residuals.empty:
        return pd.DataFrame(columns=list(PD_WIDE_FRAME_COLUMNS))
    rows: list[pd.DataFrame] = []
    for field in ("P", "D"):
        sub = bridge_residuals.loc[bridge_residuals["field"].astype(str).eq(field)].copy()
        if sub.empty:
            continue
        key = sub.loc[:, _DIAGNOSTICS_KEY_COLUMNS].copy()
        if "bridge_sat_col" in sub.columns:
            key["sat_col"] = pd.to_numeric(sub["bridge_sat_col"], errors="coerce")
        for map_field, diagnostics_column, _matlab_column, bridge_column in PD_DIAGNOSTICS_VALUE_MAP:
            if map_field == field and bridge_column in sub.columns:
                key[diagnostics_column] = pd.to_numeric(sub[bridge_column], errors="coerce")
        for component_field, diagnostics_column, bridge_column in PD_WIDE_COMPONENT_MAP:
            if component_field in (None, field) and bridge_column in sub.columns:
                key[diagnostics_column] = pd.to_numeric(sub[bridge_column], errors="coerce")
        for boolean_field, diagnostics_column, bridge_column in PD_WIDE_BOOLEAN_MAP:
            if boolean_field == field and bridge_column in sub.columns:
                key[diagnostics_column] = pd.to_numeric(sub[bridge_column], errors="coerce")
        rows.append(key)
    if not rows:
        return pd.DataFrame(columns=list(PD_WIDE_FRAME_COLUMNS))
    out = pd.concat(rows, ignore_index=True)
    for column in PD_WIDE_FRAME_COLUMNS:
        if column not in out.columns:
            out[column] = np.nan
    for column in ("epoch_index", "utcTimeMillis", "sys", "svid", "sat_col"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out["freq"] = out["freq"].astype(str)
    out = out.groupby(_DIAGNOSTICS_KEY_COLUMNS, sort=False, as_index=False).first()
    out = out.sort_values(["freq", "sat_col", "sys", "svid", "epoch_index"], kind="mergesort")
    return out.loc[:, list(PD_WIDE_FRAME_COLUMNS)].reset_index(drop=True)


def merge_residual_diagnostics_pd_values(matlab: pd.DataFrame, bridge: pd.DataFrame) -> pd.DataFrame:
    merged = matlab.merge(bridge, on=PD_VALUE_KEY_COLUMNS, how="outer", indicator=True)
    merged["side"] = merged["_merge"].map(
        {"left_only": "matlab_only", "right_only": "bridge_only", "both": "both"},
    )
    merged = merged.drop(columns=["_merge"])
    matched = merged["side"].eq("both")
    merged["delta"] = np.nan
    merged.loc[matched, "delta"] = (
        merged.loc[matched, "bridge_value"].to_numpy(dtype=np.float64)
        - merged.loc[matched, "matlab_value"].to_numpy(dtype=np.float64)
    )
    return merged.sort_values(PD_VALUE_KEY_COLUMNS).reset_index(drop=True)


def merge_residual_diagnostics_pd_wide_values(matlab: pd.DataFrame, bridge: pd.DataFrame) -> pd.DataFrame:
    merged = matlab.merge(bridge, on=PD_WIDE_VALUE_KEY_COLUMNS, how="outer", indicator=True)
    merged["side"] = merged["_merge"].map(
        {"left_only": "matlab_only", "right_only": "bridge_only", "both": "both"},
    )
    merged = merged.drop(columns=["_merge"])
    matched = merged["side"].eq("both")
    merged["delta"] = np.nan
    merged.loc[matched, "delta"] = (
        merged.loc[matched, "bridge_value"].to_numpy(dtype=np.float64)
        - merged.loc[matched, "matlab_value"].to_numpy(dtype=np.float64)
    )
    return merged.sort_values(PD_WIDE_VALUE_KEY_COLUMNS).reset_index(drop=True)


def residual_diagnostics_pd_summary(
    merged: pd.DataFrame,
    *,
    max_abs_delta_threshold: float = 1.0e-4,
) -> tuple[pd.DataFrame, dict[str, object]]:
    rows: list[dict[str, object]] = []
    if not merged.empty:
        for (field, diagnostics_column, freq), group in merged.groupby(
            ["field", "diagnostics_column", "freq"],
            sort=True,
        ):
            matched = group[group["side"].eq("both")]
            abs_delta = pd.to_numeric(matched.get("delta", pd.Series(dtype=float)), errors="coerce").abs()
            rows.append(
                {
                    "field": field,
                    "diagnostics_column": diagnostics_column,
                    "freq": freq,
                    "matlab_count": int(np.count_nonzero(group["side"].isin(("both", "matlab_only")))),
                    "bridge_count": int(np.count_nonzero(group["side"].isin(("both", "bridge_only")))),
                    "matched_count": int(len(matched)),
                    "matlab_only": int(np.count_nonzero(group["side"].eq("matlab_only"))),
                    "bridge_only": int(np.count_nonzero(group["side"].eq("bridge_only"))),
                    "median_abs_delta": float(abs_delta.median()) if not abs_delta.empty else None,
                    "p95_abs_delta": float(abs_delta.quantile(0.95)) if not abs_delta.empty else None,
                    "max_abs_delta": float(abs_delta.max()) if not abs_delta.empty else None,
                },
            )
    summary = pd.DataFrame(rows)
    total_matlab_only = int(np.count_nonzero(merged["side"].eq("matlab_only"))) if "side" in merged else 0
    total_bridge_only = int(np.count_nonzero(merged["side"].eq("bridge_only"))) if "side" in merged else 0
    matched_delta = pd.to_numeric(
        merged.loc[merged["side"].eq("both"), "delta"] if "side" in merged else pd.Series(dtype=float),
        errors="coerce",
    ).abs()
    max_abs_delta = float(matched_delta.max()) if not matched_delta.empty else float("nan")
    payload = {
        "total_matlab_count": int(np.count_nonzero(merged["side"].isin(("both", "matlab_only")))) if "side" in merged else 0,
        "total_bridge_count": int(np.count_nonzero(merged["side"].isin(("both", "bridge_only")))) if "side" in merged else 0,
        "total_matched_count": int(np.count_nonzero(merged["side"].eq("both"))) if "side" in merged else 0,
        "total_matlab_only": total_matlab_only,
        "total_bridge_only": total_bridge_only,
        "median_abs_delta": float(matched_delta.median()) if not matched_delta.empty else None,
        "p95_abs_delta": float(matched_delta.quantile(0.95)) if not matched_delta.empty else None,
        "max_abs_delta": max_abs_delta,
        "max_abs_delta_threshold": float(max_abs_delta_threshold),
        "passed": bool(
            total_matlab_only == 0
            and total_bridge_only == 0
            and np.isfinite(max_abs_delta)
            and max_abs_delta <= float(max_abs_delta_threshold)
        ),
    }
    return summary, payload


def residual_diagnostics_pd_wide_summary(
    merged: pd.DataFrame,
    *,
    max_abs_delta_threshold: float = 5.0e-3,
) -> tuple[pd.DataFrame, dict[str, object]]:
    rows: list[dict[str, object]] = []
    if not merged.empty:
        for diagnostics_column, group in merged.groupby("diagnostics_column", sort=True):
            matched = group[group["side"].eq("both")]
            abs_delta = pd.to_numeric(matched.get("delta", pd.Series(dtype=float)), errors="coerce").abs()
            rows.append(
                {
                    "diagnostics_column": diagnostics_column,
                    "matlab_count": int(np.count_nonzero(group["side"].isin(("both", "matlab_only")))),
                    "bridge_count": int(np.count_nonzero(group["side"].isin(("both", "bridge_only")))),
                    "matched_count": int(len(matched)),
                    "matlab_only": int(np.count_nonzero(group["side"].eq("matlab_only"))),
                    "bridge_only": int(np.count_nonzero(group["side"].eq("bridge_only"))),
                    "median_abs_delta": float(abs_delta.median()) if not abs_delta.empty else None,
                    "p95_abs_delta": float(abs_delta.quantile(0.95)) if not abs_delta.empty else None,
                    "max_abs_delta": float(abs_delta.max()) if not abs_delta.empty else None,
                },
            )
    summary = pd.DataFrame(rows)
    total_matlab_only = int(np.count_nonzero(merged["side"].eq("matlab_only"))) if "side" in merged else 0
    total_bridge_only = int(np.count_nonzero(merged["side"].eq("bridge_only"))) if "side" in merged else 0
    matched_delta = pd.to_numeric(
        merged.loc[merged["side"].eq("both"), "delta"] if "side" in merged else pd.Series(dtype=float),
        errors="coerce",
    ).abs()
    max_abs_delta = float(matched_delta.max()) if not matched_delta.empty else float("nan")
    sat_col = merged[merged["diagnostics_column"].eq("sat_col")] if "diagnostics_column" in merged else pd.DataFrame()
    sat_col_delta = pd.to_numeric(
        sat_col.loc[sat_col["side"].eq("both"), "delta"] if not sat_col.empty else pd.Series(dtype=float),
        errors="coerce",
    ).abs()
    sat_col_mismatch_count = int(np.count_nonzero(sat_col_delta.to_numpy(dtype=np.float64) > 0.0))
    payload = {
        "total_matlab_count": int(np.count_nonzero(merged["side"].isin(("both", "matlab_only")))) if "side" in merged else 0,
        "total_bridge_count": int(np.count_nonzero(merged["side"].isin(("both", "bridge_only")))) if "side" in merged else 0,
        "total_matched_count": int(np.count_nonzero(merged["side"].eq("both"))) if "side" in merged else 0,
        "total_matlab_only": total_matlab_only,
        "total_bridge_only": total_bridge_only,
        "median_abs_delta": float(matched_delta.median()) if not matched_delta.empty else None,
        "p95_abs_delta": float(matched_delta.quantile(0.95)) if not matched_delta.empty else None,
        "max_abs_delta": max_abs_delta,
        "max_abs_delta_threshold": float(max_abs_delta_threshold),
        "sat_col_mismatch_count": sat_col_mismatch_count,
        "sat_col_max_abs_delta": float(sat_col_delta.max()) if not sat_col_delta.empty else None,
        "passed": bool(
            total_matlab_only == 0
            and total_bridge_only == 0
            and sat_col_mismatch_count == 0
            and np.isfinite(max_abs_delta)
            and max_abs_delta <= float(max_abs_delta_threshold)
        ),
    }
    return summary, payload


def compare_residual_diagnostics_pd_values(
    trip_dir: Path,
    *,
    diagnostics_path: Path | None = None,
    max_epochs: int = 0,
    multi_gnss: bool = False,
    apply_observation_mask: bool = True,
    include_inactive_observations: bool = False,
    max_abs_delta_threshold: float = 1.0e-4,
    bridge_frame_fn: BridgeFrameFn = build_bridge_residual_frame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    trip_dir = Path(trip_dir)
    if diagnostics_path is None:
        diagnostics_path = trip_dir / "phone_data_residual_diagnostics.csv"
    start_epoch, bridge_max_epochs = _settings_epoch_window_for_trip(trip_dir, max_epochs)
    matlab_residuals = _trim_epoch_window(_matlab_residual_frame(Path(diagnostics_path)), start_epoch, bridge_max_epochs)
    matlab = _melt_pd_values(matlab_residuals, side="matlab")
    inactive_key_filter = (
        set(matlab_residuals[RESIDUAL_KEY_COLUMNS].itertuples(index=False, name=None))
        if include_inactive_observations
        else None
    )
    bridge = bridge_residual_diagnostics_pd_values(
        trip_dir,
        max_epochs=max_epochs,
        multi_gnss=multi_gnss,
        apply_observation_mask=apply_observation_mask,
        include_inactive_observations=include_inactive_observations,
        inactive_key_filter=inactive_key_filter,
        bridge_frame_fn=bridge_frame_fn,
    )
    merged = merge_residual_diagnostics_pd_values(matlab, bridge)
    summary, payload = residual_diagnostics_pd_summary(merged, max_abs_delta_threshold=max_abs_delta_threshold)
    payload.update(
        {
            "trip_dir": str(trip_dir),
            "diagnostics_path": str(Path(diagnostics_path)),
            "max_epochs": int(max_epochs),
            "multi_gnss": bool(multi_gnss),
            "apply_observation_mask": bool(apply_observation_mask),
            "include_inactive_observations": bool(include_inactive_observations),
        },
    )
    return merged, summary, bridge, payload


def compare_residual_diagnostics_pd_wide_values(
    trip_dir: Path,
    *,
    diagnostics_path: Path | None = None,
    max_epochs: int = 0,
    multi_gnss: bool = False,
    apply_observation_mask: bool = True,
    include_inactive_observations: bool = False,
    max_abs_delta_threshold: float = 5.0e-3,
    bridge_frame_fn: BridgeFrameFn = build_bridge_residual_frame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    trip_dir = Path(trip_dir)
    if diagnostics_path is None:
        diagnostics_path = trip_dir / "phone_data_residual_diagnostics.csv"
    start_epoch, bridge_max_epochs = _settings_epoch_window_for_trip(trip_dir, max_epochs)
    matlab_residuals = _trim_epoch_window(_matlab_residual_frame(Path(diagnostics_path)), start_epoch, bridge_max_epochs)
    residual_key_filter = (
        set(matlab_residuals[RESIDUAL_KEY_COLUMNS].itertuples(index=False, name=None))
        if include_inactive_observations
        else None
    )
    wide_key_filter = {
        tuple(row)
        for row in matlab_residuals[_DIAGNOSTICS_KEY_COLUMNS].itertuples(index=False, name=None)
    }
    matlab_wide = matlab_residual_diagnostics_pd_wide_frame(Path(diagnostics_path), key_filter=wide_key_filter)
    bridge_wide = bridge_residual_diagnostics_pd_wide_values(
        trip_dir,
        max_epochs=max_epochs,
        multi_gnss=multi_gnss,
        apply_observation_mask=apply_observation_mask,
        include_inactive_observations=include_inactive_observations,
        inactive_key_filter=residual_key_filter,
        factor_finite_key_filter=_bridge_pd_factor_finite_key_filter(
            trip_dir,
            max_epochs=max_epochs,
            multi_gnss=multi_gnss,
        ),
        bridge_frame_fn=bridge_frame_fn,
    )
    merged = merge_residual_diagnostics_pd_wide_values(
        _melt_pd_wide_values(matlab_wide, side="matlab"),
        _melt_pd_wide_values(bridge_wide, side="bridge"),
    )
    summary, payload = residual_diagnostics_pd_wide_summary(
        merged,
        max_abs_delta_threshold=max_abs_delta_threshold,
    )
    payload.update(
        {
            "trip_dir": str(trip_dir),
            "diagnostics_path": str(Path(diagnostics_path)),
            "max_epochs": int(max_epochs),
            "multi_gnss": bool(multi_gnss),
            "apply_observation_mask": bool(apply_observation_mask),
            "include_inactive_observations": bool(include_inactive_observations),
            "matlab_wide_row_count": int(len(matlab_wide)),
            "bridge_wide_row_count": int(len(bridge_wide)),
        },
    )
    return merged, summary, bridge_wide, payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _add_data_root_trip_args(parser, default_root=DEFAULT_ROOT)
    parser.add_argument("--diagnostics", type=Path, default=None)
    _add_max_epochs_arg(parser)
    _add_multi_gnss_arg(parser)
    parser.add_argument("--observation-mask", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-inactive-observations", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-abs-delta-threshold", type=float, default=1.0e-4)
    parser.add_argument("--write-bridge-pd-values", action="store_true")
    parser.add_argument("--write-bridge-pd-wide", action="store_true")
    _add_output_dir_arg(parser)
    args = parser.parse_args()

    trip_dir = _resolve_trip_dir(args)
    out_dir = _timestamped_output_dir(_resolved_output_root(args), "gsdc2023_residual_diagnostics_pd_parity")
    max_epochs = _nonnegative_max_epochs(args)
    merged, summary, bridge, payload = compare_residual_diagnostics_pd_values(
        trip_dir,
        diagnostics_path=args.diagnostics,
        max_epochs=max_epochs,
        multi_gnss=bool(args.multi_gnss),
        apply_observation_mask=bool(args.observation_mask),
        include_inactive_observations=bool(args.include_inactive_observations),
        max_abs_delta_threshold=float(args.max_abs_delta_threshold),
    )
    merged.to_csv(out_dir / "residual_diagnostics_pd_join.csv", index=False)
    merged[merged["side"].eq("matlab_only")].to_csv(out_dir / "matlab_only.csv", index=False)
    merged[merged["side"].eq("bridge_only")].to_csv(out_dir / "bridge_only.csv", index=False)
    summary.to_csv(out_dir / "summary_by_column.csv", index=False)
    if args.write_bridge_pd_values:
        bridge.to_csv(out_dir / "bridge_residual_diagnostics_pd_values.csv", index=False)
        bridge_residual_diagnostics_pd_export_frame(bridge).to_csv(
            out_dir / "bridge_residual_diagnostics_pd_subset.csv",
            index=False,
        )
    if args.write_bridge_pd_wide:
        diagnostics_path = Path(args.diagnostics) if args.diagnostics is not None else trip_dir / "phone_data_residual_diagnostics.csv"
        start_epoch, bridge_max_epochs = _settings_epoch_window_for_trip(trip_dir, max_epochs)
        matlab_residuals = _trim_epoch_window(
            _matlab_residual_frame(diagnostics_path),
            start_epoch,
            bridge_max_epochs,
        )
        inactive_key_filter = (
            set(matlab_residuals[RESIDUAL_KEY_COLUMNS].itertuples(index=False, name=None))
            if bool(args.include_inactive_observations)
            else None
        )
        bridge_residual_diagnostics_pd_wide_values(
            trip_dir,
            max_epochs=max_epochs,
            multi_gnss=bool(args.multi_gnss),
            apply_observation_mask=bool(args.observation_mask),
            include_inactive_observations=bool(args.include_inactive_observations),
            inactive_key_filter=inactive_key_filter,
        ).to_csv(out_dir / "bridge_residual_diagnostics_pd_wide_subset.csv", index=False)
    _write_summary_json(out_dir, payload)
    _print_summary_and_output_dir(payload, out_dir)


if __name__ == "__main__":
    main()
