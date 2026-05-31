#!/usr/bin/env python3
"""Relate PLATEAU per-epoch NLOS masks to archived hybrid RTK errors."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "python"))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from exp_ppc_libgnss_rtk import _load_reference, _parse_pos  # noqa: E402


_DEFAULT_DATA_ROOT = Path("/media/sasaki/aiueo/ai_coding_ws/datasets/PPC-Dataset-data")
_DEFAULT_HYBRID_POS_DIR = _SCRIPT_DIR / "results" / "libgnss_rtk_pos_v5"
_DEFAULT_MASK_DIR = _SCRIPT_DIR / "results" / "nlos_masks"
_DEFAULT_RUNS = (
    ("tokyo", "run1"),
    ("tokyo", "run2"),
    ("tokyo", "run3"),
    ("nagoya", "run1"),
    ("nagoya", "run2"),
    ("nagoya", "run3"),
)


@dataclass(frozen=True)
class NlosEpochFeature:
    total: int
    nlos: int
    frac: float
    min_elev: float
    mean_nlos_elev: float


def _safe_float(value: object, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _load_mask_features(path: Path) -> dict[float, NlosEpochFeature]:
    rows: dict[float, list[tuple[bool, float]]] = {}
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            tow = round(float(row["tow"]), 1)
            is_nlos = str(row.get("is_los", "")).strip() == "0"
            elev = _safe_float(row.get("elevation_deg"))
            rows.setdefault(tow, []).append((is_nlos, elev))

    out: dict[float, NlosEpochFeature] = {}
    for tow, items in rows.items():
        total = len(items)
        nlos_elev = [elev for is_nlos, elev in items if is_nlos and np.isfinite(elev)]
        nlos = sum(1 for is_nlos, _elev in items if is_nlos)
        frac = float(nlos / total) if total else 0.0
        out[tow] = NlosEpochFeature(
            total=total,
            nlos=nlos,
            frac=frac,
            min_elev=float(min(nlos_elev)) if nlos_elev else float("nan"),
            mean_nlos_elev=float(np.mean(nlos_elev)) if nlos_elev else float("nan"),
        )
    return out


def _run_rows(
    *,
    city: str,
    run: str,
    data_root: Path,
    hybrid_pos_dir: Path,
    mask_dir: Path,
    pass_threshold_m: float,
) -> list[dict[str, object]]:
    data_dir = data_root / city / run
    ref = _load_reference(data_dir / "reference.csv")
    pos_rows = {
        tow: ecef
        for tow, ecef, _status, _ratio in _parse_pos(hybrid_pos_dir / f"{city}_{run}_full.pos")
    }
    features = _load_mask_features(mask_dir / f"{city}_{run}_per_epoch_nlos.csv")

    rows: list[dict[str, object]] = []
    for tow in sorted(ref):
        feat = features.get(tow)
        if feat is None:
            continue
        pred = pos_rows.get(tow)
        has_pos = pred is not None
        err3d = float(np.linalg.norm(pred - ref[tow])) if has_pos else float("inf")
        passed = bool(has_pos and err3d <= pass_threshold_m)
        rows.append(
            {
                "city": city,
                "run": run,
                "tow": tow,
                "has_pos": int(has_pos),
                "pass": int(passed),
                "err3d_m": err3d,
                "n_sats": feat.total,
                "n_nlos": feat.nlos,
                "nlos_frac": feat.frac,
                "nlos_min_elev_deg": feat.min_elev,
                "nlos_mean_elev_deg": feat.mean_nlos_elev,
            }
        )
    return rows


def _summarize(rows: list[dict[str, object]], *, label: str) -> None:
    if not rows:
        print(f"[summary] {label}: no rows")
        return
    pass_rate = 100.0 * sum(int(r["pass"]) for r in rows) / len(rows)
    pos_rate = 100.0 * sum(int(r["has_pos"]) for r in rows) / len(rows)
    print(f"[summary] {label}: rows={len(rows)} pos={pos_rate:.2f}% pass={pass_rate:.2f}%")

    bins = (
        (0.00, 0.05),
        (0.05, 0.10),
        (0.10, 0.20),
        (0.20, 0.35),
        (0.35, 1.01),
    )
    for lo, hi in bins:
        b = [r for r in rows if lo <= float(r["nlos_frac"]) < hi]
        if not b:
            continue
        b_pass = 100.0 * sum(int(r["pass"]) for r in b) / len(b)
        b_pos = 100.0 * sum(int(r["has_pos"]) for r in b) / len(b)
        med_err = np.median([float(r["err3d_m"]) for r in b if int(r["has_pos"])])
        print(
            f"  frac[{lo:.2f},{hi:.2f}): n={len(b):5d} "
            f"pos={b_pos:6.2f}% pass={b_pass:6.2f}% med_err={med_err:7.3f}m"
        )


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=_DEFAULT_DATA_ROOT)
    parser.add_argument("--hybrid-pos-dir", type=Path, default=_DEFAULT_HYBRID_POS_DIR)
    parser.add_argument("--mask-dir", type=Path, default=_DEFAULT_MASK_DIR)
    parser.add_argument("--pass-threshold-m", type=float, default=0.5)
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=_SCRIPT_DIR / "results" / "nlos_mask_vs_hybrid_errors.csv",
    )
    args = parser.parse_args()

    all_rows: list[dict[str, object]] = []
    for city, run in _DEFAULT_RUNS:
        rows = _run_rows(
            city=city,
            run=run,
            data_root=args.data_root,
            hybrid_pos_dir=args.hybrid_pos_dir,
            mask_dir=args.mask_dir,
            pass_threshold_m=float(args.pass_threshold_m),
        )
        _summarize(rows, label=f"{city}/{run}")
        all_rows.extend(rows)

    _summarize(all_rows, label="all")
    _write_rows(args.out_csv, all_rows)
    print(f"[write] {args.out_csv}")


if __name__ == "__main__":
    main()
