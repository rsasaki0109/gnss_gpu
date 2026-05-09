"""Shared factor-mask key construction helpers for GSDC2023 audits."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.gsdc2023_signal_model import constellation_to_matlab_sys


FACTOR_MASK_FIELDS = ("P", "D", "L", "resPc", "resD", "resL")
FACTOR_MASK_KEY_COLUMNS = [
    "field",
    "freq",
    "epoch_index",
    "utcTimeMillis",
    "next_epoch_index",
    "nextUtcTimeMillis",
    "sys",
    "svid",
]
FACTOR_MASK_INT_COLUMNS = (
    "epoch_index",
    "utcTimeMillis",
    "next_epoch_index",
    "nextUtcTimeMillis",
    "sys",
    "svid",
)


def append_factor_rows(
    rows: list[dict[str, object]],
    *,
    field_names: Iterable[str],
    freq: str,
    epoch_indices: np.ndarray,
    slot_indices: np.ndarray,
    times_ms: np.ndarray,
    slot_keys: Sequence[tuple[int, int, str]],
    next_epoch_indices: np.ndarray | None = None,
    epoch_offset: int = 0,
) -> None:
    if epoch_indices.size == 0:
        return
    if next_epoch_indices is None:
        next_epoch_indices = np.zeros_like(epoch_indices)
    for field in field_names:
        for epoch_idx, slot_idx, next_epoch_idx in zip(epoch_indices, slot_indices, next_epoch_indices):
            constellation, svid, signal_type = slot_keys[int(slot_idx)]
            epoch_i = int(epoch_idx)
            next_i = int(next_epoch_idx)
            rows.append(
                {
                    "field": field,
                    "freq": freq,
                    "epoch_index": epoch_i + 1 + int(epoch_offset),
                    "utcTimeMillis": int(round(float(times_ms[epoch_i]))),
                    "next_epoch_index": 0 if next_i <= 0 else next_i + 1 + int(epoch_offset),
                    "nextUtcTimeMillis": 0 if next_i <= 0 else int(round(float(times_ms[next_i]))),
                    "sys": constellation_to_matlab_sys(int(constellation)),
                    "svid": int(svid),
                    "signal_type": str(signal_type),
                },
            )


def normalize_factor_mask_frame(
    frame: pd.DataFrame,
    *,
    keep_extra_columns: bool = True,
    filter_fields: bool = True,
    missing_label: str = "factor mask",
) -> pd.DataFrame:
    missing = [col for col in FACTOR_MASK_KEY_COLUMNS if col not in frame.columns]
    if missing:
        raise ValueError(f"{missing_label} missing required columns: {missing}")
    out = frame.copy() if keep_extra_columns else frame[FACTOR_MASK_KEY_COLUMNS].copy()
    if filter_fields:
        out = out[out["field"].isin(FACTOR_MASK_FIELDS)].copy()
    for col in FACTOR_MASK_INT_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(np.int64)
    out["field"] = out["field"].astype(str)
    out["freq"] = out["freq"].astype(str)
    return out.drop_duplicates(FACTOR_MASK_KEY_COLUMNS)


def merge_factor_mask_keys(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    left_only_side: str,
    right_only_side: str,
) -> pd.DataFrame:
    merged = left[FACTOR_MASK_KEY_COLUMNS].merge(
        right[FACTOR_MASK_KEY_COLUMNS],
        on=FACTOR_MASK_KEY_COLUMNS,
        how="outer",
        indicator=True,
    )
    merged["side"] = merged["_merge"].map(
        {
            "left_only": left_only_side,
            "right_only": right_only_side,
            "both": "both",
        },
    )
    return merged.drop(columns=["_merge"])


def factor_mask_side_summary(
    merged: pd.DataFrame,
    *,
    left_name: str,
    right_name: str,
    left_only_side: str,
    right_only_side: str,
    include_jaccard: bool = False,
) -> tuple[pd.DataFrame, dict[str, object]]:
    summary_rows: list[dict[str, object]] = []
    for (field, freq), group in merged.groupby(["field", "freq"], sort=True):
        both = int(np.count_nonzero(group["side"] == "both"))
        left_only = int(np.count_nonzero(group["side"] == left_only_side))
        right_only = int(np.count_nonzero(group["side"] == right_only_side))
        left_count = both + left_only
        right_count = both + right_only
        denom = max(left_count, right_count)
        row: dict[str, object] = {
            "field": field,
            "freq": freq,
            f"{left_name}_count": left_count,
            f"{right_name}_count": right_count,
            "matched_count": both,
            left_only_side: left_only,
            right_only_side: right_only,
            "symmetric_parity": float(both / denom) if denom > 0 else None,
        }
        if include_jaccard:
            union = both + left_only + right_only
            row["jaccard"] = float(both / union) if union > 0 else None
        summary_rows.append(row)

    total_both = int(np.count_nonzero(merged["side"] == "both"))
    total_left_only = int(np.count_nonzero(merged["side"] == left_only_side))
    total_right_only = int(np.count_nonzero(merged["side"] == right_only_side))
    union_total = total_both + total_left_only + total_right_only
    payload: dict[str, object] = {
        f"total_{left_name}_count": int(total_both + total_left_only),
        f"total_{right_name}_count": int(total_both + total_right_only),
        "total_matched_count": total_both,
        f"total_{left_only_side}": total_left_only,
        f"total_{right_only_side}": total_right_only,
        "symmetric_parity": (
            float(total_both / max(total_both + total_left_only, total_both + total_right_only))
            if union_total > 0
            else None
        ),
    }
    if include_jaccard:
        payload["jaccard"] = float(total_both / union_total) if union_total > 0 else None
    return pd.DataFrame(summary_rows), payload


def build_factor_mask_from_residual_diagnostics(diagnostics_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(diagnostics_path)
    required = {
        "freq",
        "epoch_index",
        "utcTimeMillis",
        "sys",
        "svid",
        "p_factor_finite",
        "d_factor_finite",
        "l_factor_finite",
    }
    if not required.issubset(frame.columns):
        raise ValueError(f"diagnostics CSV missing required columns: {sorted(required - set(frame.columns))}")

    frame = frame.copy()
    for col in ("epoch_index", "utcTimeMillis", "sys", "svid"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0).astype(np.int64)
    frame["freq"] = frame["freq"].astype(str)
    frame = frame.sort_values(["freq", "sys", "svid", "epoch_index"]).drop_duplicates(
        ["freq", "epoch_index", "sys", "svid"],
    )
    time_by_epoch = frame.drop_duplicates("epoch_index").set_index("epoch_index")["utcTimeMillis"].to_dict()

    rows: list[dict[str, object]] = []

    def append_rows(sub: pd.DataFrame, fields: tuple[str, ...], *, next_epoch: pd.Series | None = None) -> None:
        if sub.empty:
            return
        if next_epoch is None:
            next_epoch = pd.Series(np.zeros(len(sub), dtype=np.int64), index=sub.index)
        for field in fields:
            for row, next_idx in zip(sub.itertuples(index=False), next_epoch.to_numpy(dtype=np.int64)):
                next_utc = int(time_by_epoch.get(int(next_idx), 0)) if next_idx > 0 else 0
                rows.append(
                    {
                        "field": field,
                        "freq": str(row.freq),
                        "epoch_index": int(row.epoch_index),
                        "utcTimeMillis": int(row.utcTimeMillis),
                        "next_epoch_index": int(next_idx),
                        "nextUtcTimeMillis": next_utc,
                        "sys": int(row.sys),
                        "svid": int(row.svid),
                    },
                )

    append_rows(frame.loc[frame["p_factor_finite"].astype(bool)], ("P", "resPc"))
    append_rows(frame.loc[frame["d_factor_finite"].astype(bool)], ("D", "resD"))

    for (_freq, _sys, _svid), group in frame.loc[frame["l_factor_finite"].astype(bool)].groupby(
        ["freq", "sys", "svid"],
        sort=False,
    ):
        epochs = set(group["epoch_index"].astype(np.int64).tolist())
        left = group.loc[group["epoch_index"].map(lambda epoch: int(epoch) + 1 in epochs)].copy()
        if left.empty:
            continue
        append_rows(left, ("L", "resL"), next_epoch=left["epoch_index"].astype(np.int64) + 1)

    if not rows:
        return pd.DataFrame(columns=FACTOR_MASK_KEY_COLUMNS)
    out = normalize_factor_mask_frame(
        pd.DataFrame(rows),
        keep_extra_columns=False,
        filter_fields=False,
    )
    return out.sort_values(FACTOR_MASK_KEY_COLUMNS).reset_index(drop=True)


__all__ = [
    "FACTOR_MASK_FIELDS",
    "FACTOR_MASK_INT_COLUMNS",
    "FACTOR_MASK_KEY_COLUMNS",
    "append_factor_rows",
    "build_factor_mask_from_residual_diagnostics",
    "factor_mask_side_summary",
    "merge_factor_mask_keys",
    "normalize_factor_mask_frame",
]
