#!/usr/bin/env python3
"""Merge epoch-level PLATEAU NLOS mask features into selector training rows."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/media/sasaki/aiueo/ai_coding_ws/gnss_gpu")
RUNS = (
    ("tokyo", "run1"),
    ("tokyo", "run2"),
    ("tokyo", "run3"),
    ("nagoya", "run1"),
    ("nagoya", "run2"),
    ("nagoya", "run3"),
)


def _safe_float(value: object, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _mask_feature_rows(mask_dir: Path) -> list[dict[str, float | str]]:
    out: list[dict[str, float | str]] = []
    for city, run in RUNS:
        path = mask_dir / f"{city}_{run}_per_epoch_nlos.csv"
        by_tow: dict[float, list[tuple[bool, float]]] = {}
        with path.open(newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                tow = round(float(row["tow"]), 1)
                is_nlos = str(row.get("is_los", "")).strip() == "0"
                elev = _safe_float(row.get("elevation_deg"))
                by_tow.setdefault(tow, []).append((is_nlos, elev))

        run_id = f"{city}_{run}"
        for tow, rows in by_tow.items():
            total = len(rows)
            nlos_elev = [e for is_nlos, e in rows if is_nlos and np.isfinite(e)]
            nlos = sum(1 for is_nlos, _e in rows if is_nlos)
            los = total - nlos
            out.append(
                {
                    "run_id": run_id,
                    "tow": tow,
                    "nlos_n_sats": float(total),
                    "nlos_count": float(nlos),
                    "nlos_los_count": float(los),
                    "nlos_frac": float(nlos / total) if total else 0.0,
                    "nlos_min_elev_deg": float(min(nlos_elev)) if nlos_elev else 90.0,
                    "nlos_mean_elev_deg": float(np.mean(nlos_elev)) if nlos_elev else 90.0,
                }
            )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in-csv",
        type=Path,
        default=REPO / "experiments/results/selector_training_features_v3.csv",
    )
    parser.add_argument(
        "--mask-dir",
        type=Path,
        default=REPO / "experiments/results/nlos_masks",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=REPO / "experiments/results/selector_training_features_v5_nlos.csv",
    )
    args = parser.parse_args()

    print(f"loading selector features: {args.in_csv}", flush=True)
    features = pd.read_csv(args.in_csv)
    features["tow"] = features["tow"].round(1)

    print(f"loading NLOS masks: {args.mask_dir}", flush=True)
    nlos = pd.DataFrame(_mask_feature_rows(args.mask_dir))
    merged = features.merge(nlos, on=["run_id", "tow"], how="left")

    nlos_cols = [c for c in merged.columns if c.startswith("nlos_")]
    missing = int(merged[nlos_cols].isna().any(axis=1).sum())
    if missing:
        print(f"missing NLOS rows: {missing}; filling neutral values", flush=True)
        fill = {
            "nlos_n_sats": 0.0,
            "nlos_count": 0.0,
            "nlos_los_count": 0.0,
            "nlos_frac": 0.0,
            "nlos_min_elev_deg": 90.0,
            "nlos_mean_elev_deg": 90.0,
        }
        merged = merged.fillna(value=fill)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out_csv, index=False)
    print(
        f"wrote {args.out_csv} rows={len(merged)} "
        f"nlos_cols={','.join(nlos_cols)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
