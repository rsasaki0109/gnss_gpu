"""MATLAB residual diagnostics mask overlay helpers for GSDC2023."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from experiments.gsdc2023_signal_model import (
    constellation_to_matlab_sys,
    slot_frequency_label,
)


def diagnostics_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None or pd.isna(value):
        return False
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "t", "yes", "y"}:
            return True
        if normalized in {"0", "false", "f", "no", "n", ""}:
            return False
    try:
        return bool(float(value) != 0.0)
    except (TypeError, ValueError):
        return False


def apply_matlab_residual_diagnostics_mask(
    *,
    diagnostics_path: Path,
    times_ms: np.ndarray,
    slot_keys: list[tuple[int, int, str]] | tuple[tuple[int, int, str], ...],
    weights: np.ndarray,
    signal_weights: np.ndarray,
    doppler_weights: np.ndarray | None,
    signal_doppler_weights: np.ndarray | None,
    tdcp_meas: np.ndarray | None,
    tdcp_weights: np.ndarray | None,
    signal_tdcp_weights: np.ndarray | None,
) -> tuple[int, int, int]:
    diagnostics = pd.read_csv(diagnostics_path)
    required = {
        "freq",
        "utcTimeMillis",
        "sys",
        "svid",
        "p_factor_finite",
        "d_factor_finite",
        "l_factor_finite",
    }
    missing = sorted(required - set(diagnostics.columns))
    if missing:
        raise ValueError(f"MATLAB residual diagnostics missing columns {missing}: {diagnostics_path}")

    frame = diagnostics.copy()
    for col in ("utcTimeMillis", "sys", "svid"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0).astype(np.int64)
    frame["freq"] = frame["freq"].astype(str)
    time_to_epoch = {int(round(float(time_ms))): idx for idx, time_ms in enumerate(times_ms)}
    slot_lookup = {
        (constellation_to_matlab_sys(int(key[0])), int(key[1]), slot_frequency_label(str(key[2]))): idx
        for idx, key in enumerate(slot_keys)
    }

    p_keep = np.zeros_like(weights, dtype=bool)
    d_keep = np.zeros_like(doppler_weights, dtype=bool) if doppler_weights is not None else None
    l_finite = np.zeros_like(weights, dtype=bool)
    for row in frame.itertuples(index=False):
        epoch_idx = time_to_epoch.get(int(row.utcTimeMillis))
        if epoch_idx is None:
            continue
        slot_idx = slot_lookup.get((int(row.sys), int(row.svid), str(row.freq)))
        if slot_idx is None:
            continue
        if diagnostics_bool(getattr(row, "p_factor_finite")):
            p_keep[epoch_idx, slot_idx] = True
        if d_keep is not None and diagnostics_bool(getattr(row, "d_factor_finite")):
            d_keep[epoch_idx, slot_idx] = True
        if diagnostics_bool(getattr(row, "l_factor_finite")):
            l_finite[epoch_idx, slot_idx] = True

    weights[:, :] = 0.0
    weights[p_keep] = signal_weights[p_keep]
    weights[p_keep & (weights <= 0.0)] = 1.0

    if doppler_weights is not None and d_keep is not None:
        doppler_weights[:, :] = 0.0
        d_signal = (
            signal_doppler_weights
            if signal_doppler_weights is not None
            else np.ones_like(doppler_weights, dtype=np.float64)
        )
        doppler_weights[d_keep] = d_signal[d_keep]
        doppler_weights[d_keep & (doppler_weights <= 0.0)] = 1.0

    l_pair_count = 0
    if tdcp_weights is not None:
        tdcp_weights[:, :] = 0.0
        if tdcp_meas is not None:
            l_keep = l_finite[:-1] & l_finite[1:] & np.isfinite(tdcp_meas)
            tdcp_signal = (
                signal_tdcp_weights
                if signal_tdcp_weights is not None
                else np.ones_like(tdcp_weights, dtype=np.float64)
            )
            tdcp_weights[l_keep] = tdcp_signal[l_keep]
            tdcp_weights[l_keep & (tdcp_weights <= 0.0)] = 1.0
            l_pair_count = int(np.count_nonzero(l_keep))

    return int(np.count_nonzero(p_keep)), int(np.count_nonzero(d_keep)) if d_keep is not None else 0, l_pair_count


__all__ = [
    "apply_matlab_residual_diagnostics_mask",
    "diagnostics_bool",
]
