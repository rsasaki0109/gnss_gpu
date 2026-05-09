#!/usr/bin/env python3
"""Compare MATLAB and raw-bridge base-correction series."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.gsdc2023_audit_cli import (
    add_output_dir_arg as _add_output_dir_arg,
    resolved_output_root as _resolved_output_root,
)

_KEY_COLUMNS = ["freq", "epoch_index", "utcTimeMillis", "sys", "svid"]


def _freq_label(signal_type: str) -> str:
    sig = str(signal_type).upper()
    return "L5" if "L5" in sig or "E5" in sig or "B2" in sig else "L1"


def _constellation_to_matlab_sys(constellation_type: int) -> int:
    mapping = {
        1: 1,  # GPS
        4: 4,  # QZSS
        6: 8,  # Galileo
    }
    return mapping.get(int(constellation_type), int(constellation_type))


def _load_matlab_base_correction(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = set(_KEY_COLUMNS + ["correction_m"])
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"MATLAB correction CSV missing required columns {missing}: {path}")

    out = frame[_KEY_COLUMNS + ["correction_m"]].copy()
    out = out.rename(columns={"correction_m": "matlab_correction_m"})
    for col in ("epoch_index", "utcTimeMillis", "sys", "svid"):
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(np.int64)
    out["freq"] = out["freq"].astype(str)
    out["matlab_correction_m"] = pd.to_numeric(out["matlab_correction_m"], errors="coerce")
    out = out[np.isfinite(out["matlab_correction_m"].to_numpy(dtype=np.float64))]
    return out.sort_values(_KEY_COLUMNS).drop_duplicates(_KEY_COLUMNS)


def _load_bridge_base_correction(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "UnixTimeMillis",
        "EpochIndex",
        "ConstellationType",
        "Svid",
        "SignalType",
        "CorrectionMeters",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"raw-bridge correction CSV missing required columns {missing}: {path}")

    out = pd.DataFrame(
        {
            "freq": frame["SignalType"].map(_freq_label).astype(str),
            "epoch_index": pd.to_numeric(frame["EpochIndex"], errors="coerce").fillna(-1).astype(np.int64) + 1,
            "utcTimeMillis": pd.to_numeric(frame["UnixTimeMillis"], errors="coerce").fillna(0).astype(np.int64),
            "sys": pd.to_numeric(frame["ConstellationType"], errors="coerce")
            .fillna(0)
            .astype(np.int64)
            .map(_constellation_to_matlab_sys)
            .astype(np.int64),
            "svid": pd.to_numeric(frame["Svid"], errors="coerce").fillna(0).astype(np.int64),
            "bridge_correction_m": pd.to_numeric(frame["CorrectionMeters"], errors="coerce"),
        },
    )
    if "ObservationWeightPositive" in frame.columns:
        out["bridge_observation_weight_positive"] = frame["ObservationWeightPositive"].astype(bool)
    out = out[np.isfinite(out["bridge_correction_m"].to_numpy(dtype=np.float64))]
    return out.sort_values(_KEY_COLUMNS).drop_duplicates(_KEY_COLUMNS)


def _summary_for_group(group: pd.DataFrame) -> dict[str, object]:
    matched = group[np.isfinite(group["delta_m"].to_numpy(dtype=np.float64))]
    abs_delta = np.abs(matched["delta_m"].to_numpy(dtype=np.float64))
    out: dict[str, object] = {
        "matlab_rows": int(group["matlab_correction_m"].notna().sum()),
        "bridge_rows": int(group["bridge_correction_m"].notna().sum()),
        "matched_count": int(len(matched)),
        "matlab_only_count": int((group["matlab_correction_m"].notna() & group["bridge_correction_m"].isna()).sum()),
        "bridge_only_count": int((group["matlab_correction_m"].isna() & group["bridge_correction_m"].notna()).sum()),
    }
    if abs_delta.size:
        out.update(
            {
                "mean_delta_m": float(np.mean(matched["delta_m"].to_numpy(dtype=np.float64))),
                "median_abs_delta_m": float(np.median(abs_delta)),
                "p95_abs_delta_m": float(np.percentile(abs_delta, 95)),
                "max_abs_delta_m": float(np.max(abs_delta)),
            },
        )
    else:
        out.update(
            {
                "mean_delta_m": None,
                "median_abs_delta_m": None,
                "p95_abs_delta_m": None,
                "max_abs_delta_m": None,
            },
        )
    return out


def compare_base_correction_series(
    matlab_csv: Path,
    bridge_long_csv: Path,
    output_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    matlab = _load_matlab_base_correction(Path(matlab_csv))
    bridge = _load_bridge_base_correction(Path(bridge_long_csv))
    merged = matlab.merge(bridge, on=_KEY_COLUMNS, how="outer", indicator=True)
    merged["delta_m"] = merged["bridge_correction_m"] - merged["matlab_correction_m"]
    merged = merged.sort_values(_KEY_COLUMNS).reset_index(drop=True)

    summary = _summary_for_group(merged)
    summary.update(
        {
            "matlab_csv": str(matlab_csv),
            "bridge_long_csv": str(bridge_long_csv),
            "merged_rows": int(len(merged)),
        },
    )
    rows: list[dict[str, object]] = []
    for (freq, sys), group in merged.groupby(["freq", "sys"], dropna=False, sort=True):
        row = {"freq": str(freq), "sys": int(sys)}
        row.update(_summary_for_group(group))
        rows.append(row)
    summary_by_freq_sys = pd.DataFrame(rows)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        merged.to_csv(output_dir / "base_correction_series_comparison.csv", index=False)
        summary_by_freq_sys.to_csv(output_dir / "base_correction_series_summary_by_freq_sys.csv", index=False)
        (output_dir / "base_correction_series_summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
    return merged, summary_by_freq_sys, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matlab-csv", type=Path, required=True)
    parser.add_argument("--bridge-long-csv", type=Path, required=True)
    _add_output_dir_arg(parser, required=True)
    args = parser.parse_args()

    _merged, summary_by_freq_sys, summary = compare_base_correction_series(
        args.matlab_csv,
        args.bridge_long_csv,
        _resolved_output_root(args),
    )
    print(json.dumps(summary, indent=2))
    if not summary_by_freq_sys.empty:
        print(summary_by_freq_sys.to_string(index=False))


if __name__ == "__main__":
    main()
