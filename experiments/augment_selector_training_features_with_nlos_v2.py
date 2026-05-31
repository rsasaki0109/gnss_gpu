#!/usr/bin/env python3
"""Augment v3 features with richer NLOS aggregates (per-system + elevation bands).

v5 used 6 NLOS features (total count + min/mean elev). v7 adds:
- nlos_count_G, nlos_count_E, nlos_count_J, nlos_count_C: per-system NLOS count
- nlos_frac_G_visible, nlos_frac_E_visible, nlos_frac_J_visible, nlos_frac_C_visible: per-system NLOS / per-system visible
- nlos_n_systems_affected: distinct systems with >=1 NLOS sat
- nlos_low_elev_count_10: NLOS sat with elev < 10° (horizon scatter)
- nlos_low_elev_count_20: NLOS sat with elev < 20° (low-elev multipath)
- nlos_high_elev_count_45: NLOS sat with elev >= 45° (rare but high-bias)
- nlos_max_elev_deg: max elev of NLOS sat (rare-but-bad detector)

Combined with v5 NLOS features (kept) and v3 base features = 14 NLOS-related total.
"""

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


def _safe_float(value, default=float("nan")):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _mask_feature_rows(mask_dir: Path) -> list[dict]:
    out = []
    for city, run in RUNS:
        path = mask_dir / f"{city}_{run}_per_epoch_nlos.csv"
        # by_tow: tow -> list[(is_nlos: bool, elev: float, system: str)]
        by_tow: dict[float, list[tuple[bool, float, str]]] = {}
        with path.open(newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                tow = round(float(row["tow"]), 1)
                is_nlos = str(row.get("is_los", "")).strip() == "0"
                elev = _safe_float(row.get("elevation_deg"))
                system = str(row.get("system", "")).strip().upper()
                by_tow.setdefault(tow, []).append((is_nlos, elev, system))

        run_id = f"{city}_{run}"
        for tow, rows in by_tow.items():
            total = len(rows)
            nlos_count = sum(1 for is_nlos, _, _ in rows if is_nlos)
            los_count = total - nlos_count
            nlos_elev_list = [e for is_nlos, e, _ in rows if is_nlos and np.isfinite(e)]

            # Per-system counts
            sys_total = {}
            sys_nlos = {}
            for is_nlos, _e, s in rows:
                sys_total[s] = sys_total.get(s, 0) + 1
                if is_nlos:
                    sys_nlos[s] = sys_nlos.get(s, 0) + 1

            n_systems_affected = sum(1 for s, c in sys_nlos.items() if c > 0)

            # Elevation bands (for NLOS sats only)
            low10 = sum(1 for is_nlos, e, _ in rows if is_nlos and np.isfinite(e) and e < 10.0)
            low20 = sum(1 for is_nlos, e, _ in rows if is_nlos and np.isfinite(e) and e < 20.0)
            high45 = sum(1 for is_nlos, e, _ in rows if is_nlos and np.isfinite(e) and e >= 45.0)
            max_elev = float(max(nlos_elev_list)) if nlos_elev_list else 0.0

            row_out = {
                "run_id": run_id,
                "tow": tow,
                # v5 features (kept for backward compat)
                "nlos_n_sats": float(total),
                "nlos_count": float(nlos_count),
                "nlos_los_count": float(los_count),
                "nlos_frac": float(nlos_count / total) if total else 0.0,
                "nlos_min_elev_deg": float(min(nlos_elev_list)) if nlos_elev_list else 90.0,
                "nlos_mean_elev_deg": float(np.mean(nlos_elev_list)) if nlos_elev_list else 90.0,
                # v7 NEW: per-system counts
                "nlos_count_G": float(sys_nlos.get("G", 0)),
                "nlos_count_E": float(sys_nlos.get("E", 0)),
                "nlos_count_J": float(sys_nlos.get("J", 0)),
                "nlos_count_C": float(sys_nlos.get("C", 0)),
                # v7 NEW: per-system NLOS frac among visible of that system
                "nlos_frac_G_visible": float(sys_nlos.get("G", 0) / sys_total.get("G", 1)) if sys_total.get("G", 0) else 0.0,
                "nlos_frac_E_visible": float(sys_nlos.get("E", 0) / sys_total.get("E", 1)) if sys_total.get("E", 0) else 0.0,
                "nlos_frac_J_visible": float(sys_nlos.get("J", 0) / sys_total.get("J", 1)) if sys_total.get("J", 0) else 0.0,
                "nlos_frac_C_visible": float(sys_nlos.get("C", 0) / sys_total.get("C", 1)) if sys_total.get("C", 0) else 0.0,
                # v7 NEW: system spread + elevation bands
                "nlos_n_systems_affected": float(n_systems_affected),
                "nlos_low_elev_count_10": float(low10),
                "nlos_low_elev_count_20": float(low20),
                "nlos_high_elev_count_45": float(high45),
                "nlos_max_elev_deg": max_elev,
            }
            out.append(row_out)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-csv", type=Path, default=REPO / "experiments/results/selector_training_features_v3.csv")
    parser.add_argument("--mask-dir", type=Path, default=REPO / "experiments/results/plateau_nlos_phase33")
    parser.add_argument("--out-csv", type=Path, default=REPO / "experiments/results/selector_training_features_v7_nlos_rich.csv")
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
            "nlos_n_sats": 0.0, "nlos_count": 0.0, "nlos_los_count": 0.0, "nlos_frac": 0.0,
            "nlos_min_elev_deg": 90.0, "nlos_mean_elev_deg": 90.0,
            "nlos_count_G": 0.0, "nlos_count_E": 0.0, "nlos_count_J": 0.0, "nlos_count_C": 0.0,
            "nlos_frac_G_visible": 0.0, "nlos_frac_E_visible": 0.0,
            "nlos_frac_J_visible": 0.0, "nlos_frac_C_visible": 0.0,
            "nlos_n_systems_affected": 0.0,
            "nlos_low_elev_count_10": 0.0, "nlos_low_elev_count_20": 0.0,
            "nlos_high_elev_count_45": 0.0, "nlos_max_elev_deg": 0.0,
        }
        merged = merged.fillna(value=fill)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out_csv, index=False)
    print(f"wrote {args.out_csv} rows={len(merged)} nlos_cols={','.join(nlos_cols)}", flush=True)


if __name__ == "__main__":
    main()
