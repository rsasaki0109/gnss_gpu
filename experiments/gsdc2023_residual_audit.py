"""Shared residual-value audit helpers for GSDC2023 diagnostics comparisons."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from experiments.gsdc2023_signal_model import constellation_to_matlab_sys


RESIDUAL_KEY_COLUMNS = ["field", "freq", "epoch_index", "utcTimeMillis", "sys", "svid"]
RESIDUAL_COMPONENT_SUMMARY_COLUMNS = (
    ("sat_position_delta_norm", "sat_position_delta_norm"),
    ("sat_velocity_delta_norm", "sat_velocity_delta_norm"),
    ("sat_clock_bias_delta", "sat_clock_bias_delta"),
    ("sat_clock_drift_delta", "sat_clock_drift_delta"),
    ("sat_iono_delta", "sat_iono_delta"),
    ("sat_trop_delta", "sat_trop_delta"),
    ("sat_elevation_delta", "sat_elevation_delta"),
    ("rcv_position_delta_norm", "rcv_position_delta_norm"),
    ("rcv_velocity_delta_norm", "rcv_velocity_delta_norm"),
)
_MATLAB_COMMON_OPTIONAL_MAP = {
    "sat_x_m": "matlab_sat_x",
    "sat_y_m": "matlab_sat_y",
    "sat_z_m": "matlab_sat_z",
    "sat_vx_mps": "matlab_sat_vx",
    "sat_vy_mps": "matlab_sat_vy",
    "sat_vz_mps": "matlab_sat_vz",
    "sat_clock_bias_m": "matlab_sat_clock_bias",
    "sat_clock_drift_mps": "matlab_sat_clock_drift",
    "sat_iono_m": "matlab_sat_iono",
    "sat_trop_m": "matlab_sat_trop",
    "sat_range_m": "matlab_sat_range",
    "sat_rate_mps": "matlab_sat_rate",
    "sat_elevation_deg": "matlab_sat_elevation",
    "rcv_x_m": "matlab_rcv_x",
    "rcv_y_m": "matlab_rcv_y",
    "rcv_z_m": "matlab_rcv_z",
    "rcv_vx_mps": "matlab_rcv_vx",
    "rcv_vy_mps": "matlab_rcv_vy",
    "rcv_vz_mps": "matlab_rcv_vz",
}
_MATLAB_NUMERIC_COLUMNS = (
    "matlab_residual",
    "matlab_pre_residual",
    "matlab_common_bias",
    "matlab_observation",
    "matlab_model",
    "matlab_sat_x",
    "matlab_sat_y",
    "matlab_sat_z",
    "matlab_sat_vx",
    "matlab_sat_vy",
    "matlab_sat_vz",
    "matlab_sat_clock_bias",
    "matlab_sat_clock_drift",
    "matlab_sat_iono",
    "matlab_sat_trop",
    "matlab_sat_range",
    "matlab_sat_rate",
    "matlab_sat_elevation",
    "matlab_rcv_x",
    "matlab_rcv_y",
    "matlab_rcv_z",
    "matlab_rcv_vx",
    "matlab_rcv_vy",
    "matlab_rcv_vz",
    "matlab_obs_clk",
    "matlab_obs_dclk",
    "matlab_isb",
)


def matlab_residual_frame(diagnostics_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(diagnostics_path)
    required = {
        "freq",
        "epoch_index",
        "utcTimeMillis",
        "sys",
        "svid",
        "p_residual_m",
        "d_residual_mps",
        "p_pre_finite",
        "d_pre_finite",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"diagnostics CSV missing required columns {missing}: {diagnostics_path}")

    rows: list[pd.DataFrame] = []
    for field, finite_col, value_col, optional_map in (
        (
            "P",
            "p_pre_finite",
            "p_residual_m",
            {
                **_MATLAB_COMMON_OPTIONAL_MAP,
                "p_pre_respc_m": "matlab_pre_residual",
                "p_clock_bias_m": "matlab_common_bias",
                "p_corrected_m": "matlab_observation",
                "p_range_m": "matlab_model",
                "obs_clk_m": "matlab_obs_clk",
                "p_isb_m": "matlab_isb",
            },
        ),
        (
            "D",
            "d_pre_finite",
            "d_residual_mps",
            {
                **_MATLAB_COMMON_OPTIONAL_MAP,
                "d_pre_resd_m": "matlab_pre_residual",
                "d_clock_bias_mps": "matlab_common_bias",
                "d_obs_mps": "matlab_observation",
                "d_model_mps": "matlab_model",
                "obs_dclk_m": "matlab_obs_dclk",
            },
        ),
    ):
        optional_cols = [col for col in optional_map if col in frame.columns]
        sub = frame.loc[
            frame[finite_col].astype(bool),
            RESIDUAL_KEY_COLUMNS[1:] + [value_col] + optional_cols,
        ].copy()
        if sub.empty:
            continue
        sub.insert(0, "field", field)
        sub = sub.rename(columns={value_col: "matlab_residual", **optional_map})
        rows.append(sub)

    if not rows:
        return pd.DataFrame(columns=RESIDUAL_KEY_COLUMNS + ["matlab_residual"])

    out = pd.concat(rows, ignore_index=True)
    for col in ("epoch_index", "utcTimeMillis", "sys", "svid"):
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(np.int64)
    out["field"] = out["field"].astype(str)
    out["freq"] = out["freq"].astype(str)
    for col in _MATLAB_NUMERIC_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.drop_duplicates(RESIDUAL_KEY_COLUMNS)


def append_bridge_residual_row(
    rows: list[dict[str, object]],
    *,
    field: str,
    freq: str,
    times_ms: np.ndarray,
    slot_keys: tuple[tuple[int, int, str], ...],
    epoch_idx: int,
    slot_idx: int,
    residual: float,
    pre_residual: float,
    common_bias: float,
    observation: float,
    model: float,
    epoch_offset: int = 0,
) -> None:
    constellation, svid, _signal_type = slot_keys[int(slot_idx)]
    rows.append(
        {
            "field": field,
            "freq": freq,
            "epoch_index": int(epoch_idx) + 1 + int(epoch_offset),
            "utcTimeMillis": int(round(float(times_ms[int(epoch_idx)]))),
            "sys": constellation_to_matlab_sys(int(constellation)),
            "svid": int(svid),
            "bridge_residual": float(residual),
            "bridge_pre_residual": float(pre_residual),
            "bridge_common_bias": float(common_bias),
            "bridge_observation": float(observation),
            "bridge_model": float(model),
        },
    )


def merge_residual_value_frames(matlab: pd.DataFrame, bridge: pd.DataFrame) -> pd.DataFrame:
    merged = matlab.merge(bridge, on=RESIDUAL_KEY_COLUMNS, how="outer", indicator=True)
    merged["side"] = merged["_merge"].map(
        {"left_only": "matlab_only", "right_only": "bridge_only", "both": "both"},
    )
    merged = merged.drop(columns=["_merge"])
    return add_residual_value_deltas(merged)


def add_residual_value_deltas(merged: pd.DataFrame) -> pd.DataFrame:
    out = merged.copy()
    matched = out["side"] == "both"
    _add_scalar_delta(out, matched, "delta", "bridge_residual", "matlab_residual")
    _add_scalar_delta(out, matched, "pre_residual_delta", "bridge_pre_residual", "matlab_pre_residual")
    _add_scalar_delta(out, matched, "common_bias_delta", "bridge_common_bias", "matlab_common_bias")
    _add_scalar_delta(out, matched, "observation_delta", "bridge_observation", "matlab_observation")
    _add_scalar_delta(out, matched, "model_delta", "bridge_model", "matlab_model")

    for delta_col, bridge_col, matlab_col in (
        ("sat_clock_bias_delta", "bridge_sat_clock_bias", "matlab_sat_clock_bias"),
        ("sat_clock_drift_delta", "bridge_sat_clock_drift", "matlab_sat_clock_drift"),
        ("sat_iono_delta", "bridge_sat_iono", "matlab_sat_iono"),
        ("sat_trop_delta", "bridge_sat_trop", "matlab_sat_trop"),
        ("sat_elevation_delta", "bridge_sat_elevation", "matlab_sat_elevation"),
    ):
        _add_scalar_delta(out, matched, delta_col, bridge_col, matlab_col)

    _add_vector_delta_norm(
        out,
        matched,
        "sat_position_delta_norm",
        ("bridge_sat_x", "bridge_sat_y", "bridge_sat_z"),
        ("matlab_sat_x", "matlab_sat_y", "matlab_sat_z"),
    )
    _add_vector_delta_norm(
        out,
        matched,
        "sat_velocity_delta_norm",
        ("bridge_sat_vx", "bridge_sat_vy", "bridge_sat_vz"),
        ("matlab_sat_vx", "matlab_sat_vy", "matlab_sat_vz"),
    )
    _add_vector_delta_norm(
        out,
        matched,
        "rcv_position_delta_norm",
        ("bridge_rcv_x", "bridge_rcv_y", "bridge_rcv_z"),
        ("matlab_rcv_x", "matlab_rcv_y", "matlab_rcv_z"),
    )
    _add_vector_delta_norm(
        out,
        matched,
        "rcv_velocity_delta_norm",
        ("bridge_rcv_vx", "bridge_rcv_vy", "bridge_rcv_vz"),
        ("matlab_rcv_vx", "matlab_rcv_vy", "matlab_rcv_vz"),
    )
    return out


def residual_value_summary_frame(merged: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    rows: list[dict[str, object]] = []
    matched = merged[merged["side"] == "both"].copy()
    if "delta" not in matched.columns and {"bridge_residual", "matlab_residual"}.issubset(matched.columns):
        matched["delta"] = matched["bridge_residual"] - matched["matlab_residual"]
    matched["epoch_median_delta"] = matched.groupby(["field", "freq", "epoch_index"])["delta"].transform("median")
    matched["delta_after_epoch_median"] = matched["delta"] - matched["epoch_median_delta"]

    for (field, freq), group in merged.groupby(["field", "freq"], sort=True):
        group_matched = matched[(matched["field"] == field) & (matched["freq"] == freq)]
        delta = group_matched["delta"].to_numpy(dtype=np.float64)
        abs_delta = np.abs(delta)
        epoch_demeaned = group_matched["delta_after_epoch_median"].to_numpy(dtype=np.float64)
        abs_epoch_demeaned = np.abs(epoch_demeaned)
        epoch_offset_abs = (
            group_matched.groupby("epoch_index")["epoch_median_delta"].first().abs().to_numpy(dtype=np.float64)
        )
        row = {
            "field": field,
            "freq": freq,
            "matlab_count": int(np.count_nonzero(group["side"].isin(("both", "matlab_only")))),
            "bridge_count": int(np.count_nonzero(group["side"].isin(("both", "bridge_only")))),
            "matched_count": int(group_matched.shape[0]),
            "matlab_only": int(np.count_nonzero(group["side"] == "matlab_only")),
            "bridge_only": int(np.count_nonzero(group["side"] == "bridge_only")),
            "mean_delta": float(np.mean(delta)) if delta.size else None,
            "median_abs_delta": float(np.median(abs_delta)) if abs_delta.size else None,
            "p95_abs_delta": float(np.percentile(abs_delta, 95)) if abs_delta.size else None,
            "max_abs_delta": float(np.max(abs_delta)) if abs_delta.size else None,
            "median_abs_delta_after_epoch_median": (
                float(np.median(abs_epoch_demeaned)) if abs_epoch_demeaned.size else None
            ),
            "p95_abs_delta_after_epoch_median": (
                float(np.percentile(abs_epoch_demeaned, 95)) if abs_epoch_demeaned.size else None
            ),
            "median_epoch_offset_abs": float(np.median(epoch_offset_abs)) if epoch_offset_abs.size else None,
            "p95_epoch_offset_abs": (
                float(np.percentile(epoch_offset_abs, 95)) if epoch_offset_abs.size else None
            ),
        }
        row.update(delta_abs_stats(group_matched, "pre_residual_delta", "pre_residual_delta"))
        row.update(delta_abs_stats(group_matched, "common_bias_delta", "common_bias_delta"))
        row.update(delta_abs_stats(group_matched, "observation_delta", "observation_delta"))
        row.update(delta_abs_stats(group_matched, "model_delta", "model_delta"))
        for column, prefix in RESIDUAL_COMPONENT_SUMMARY_COLUMNS:
            row.update(delta_abs_stats(group_matched, column, prefix))
        rows.append(row)

    summary = pd.DataFrame(rows)
    all_delta = matched["delta"].to_numpy(dtype=np.float64)
    all_abs_delta = np.abs(all_delta)
    all_epoch_demeaned = matched["delta_after_epoch_median"].to_numpy(dtype=np.float64)
    all_abs_epoch_demeaned = np.abs(all_epoch_demeaned)
    payload = {
        "total_matlab_count": int(np.count_nonzero(merged["side"].isin(("both", "matlab_only")))),
        "total_bridge_count": int(np.count_nonzero(merged["side"].isin(("both", "bridge_only")))),
        "total_matched_count": int(matched.shape[0]),
        "total_matlab_only": int(np.count_nonzero(merged["side"] == "matlab_only")),
        "total_bridge_only": int(np.count_nonzero(merged["side"] == "bridge_only")),
        "mean_delta": float(np.mean(all_delta)) if all_delta.size else None,
        "median_abs_delta": float(np.median(all_abs_delta)) if all_abs_delta.size else None,
        "p95_abs_delta": float(np.percentile(all_abs_delta, 95)) if all_abs_delta.size else None,
        "max_abs_delta": float(np.max(all_abs_delta)) if all_abs_delta.size else None,
        "median_abs_delta_after_epoch_median": (
            float(np.median(all_abs_epoch_demeaned)) if all_abs_epoch_demeaned.size else None
        ),
        "p95_abs_delta_after_epoch_median": (
            float(np.percentile(all_abs_epoch_demeaned, 95)) if all_abs_epoch_demeaned.size else None
        ),
    }
    payload.update(delta_abs_stats(matched, "pre_residual_delta", "pre_residual_delta"))
    payload.update(delta_abs_stats(matched, "common_bias_delta", "common_bias_delta"))
    payload.update(delta_abs_stats(matched, "observation_delta", "observation_delta"))
    payload.update(delta_abs_stats(matched, "model_delta", "model_delta"))
    for column, prefix in RESIDUAL_COMPONENT_SUMMARY_COLUMNS:
        payload.update(delta_abs_stats(matched, column, prefix))
    return summary, payload


def delta_abs_stats(frame: pd.DataFrame, column: str, prefix: str) -> dict[str, float | None]:
    if column not in frame.columns:
        return {
            f"mean_{prefix}": None,
            f"median_abs_{prefix}": None,
            f"p95_abs_{prefix}": None,
            f"max_abs_{prefix}": None,
        }
    values = frame[column].to_numpy(dtype=np.float64)
    values = values[np.isfinite(values)]
    abs_values = np.abs(values)
    return {
        f"mean_{prefix}": float(np.mean(values)) if values.size else None,
        f"median_abs_{prefix}": float(np.median(abs_values)) if abs_values.size else None,
        f"p95_abs_{prefix}": float(np.percentile(abs_values, 95)) if abs_values.size else None,
        f"max_abs_{prefix}": float(np.max(abs_values)) if abs_values.size else None,
    }


def _add_scalar_delta(
    frame: pd.DataFrame,
    matched: pd.Series,
    delta_col: str,
    bridge_col: str,
    matlab_col: str,
) -> None:
    if {bridge_col, matlab_col}.issubset(frame.columns):
        frame[delta_col] = np.nan
        frame.loc[matched, delta_col] = (
            frame.loc[matched, bridge_col].to_numpy(dtype=np.float64)
            - frame.loc[matched, matlab_col].to_numpy(dtype=np.float64)
        )


def _add_vector_delta_norm(
    frame: pd.DataFrame,
    matched: pd.Series,
    delta_col: str,
    bridge_cols: tuple[str, str, str],
    matlab_cols: tuple[str, str, str],
) -> None:
    if set(bridge_cols + matlab_cols).issubset(frame.columns):
        frame[delta_col] = np.nan
        delta = (
            frame.loc[matched, list(bridge_cols)].to_numpy(dtype=np.float64)
            - frame.loc[matched, list(matlab_cols)].to_numpy(dtype=np.float64)
        )
        frame.loc[matched, delta_col] = np.linalg.norm(delta, axis=1)


__all__ = [
    "RESIDUAL_COMPONENT_SUMMARY_COLUMNS",
    "RESIDUAL_KEY_COLUMNS",
    "add_residual_value_deltas",
    "append_bridge_residual_row",
    "delta_abs_stats",
    "matlab_residual_frame",
    "merge_residual_value_frames",
    "residual_value_summary_frame",
]
