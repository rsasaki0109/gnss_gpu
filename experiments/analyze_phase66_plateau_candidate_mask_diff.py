#!/usr/bin/env python3
"""Compare PLATEAU NLOS masks generated at two candidate positions."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any


def _read_mask(path: Path) -> dict[float, dict[str, dict[str, Any]]]:
    by_tow: dict[float, dict[str, dict[str, Any]]] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            tow = round(float(row["tow"]), 1)
            prn = str(row["prn"])
            by_tow.setdefault(tow, {})[prn] = {
                "is_los": str(row.get("is_los", "")).strip() == "1",
                "system": str(row.get("system", "")),
                "elevation_deg": _safe_float(row.get("elevation_deg")),
            }
    return by_tow


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved: {path}")


def _epoch_features(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    nlos = sum(1 for row in rows.values() if not bool(row["is_los"]))
    systems = Counter(str(row["system"]) for row in rows.values() if not bool(row["is_los"]))
    return {
        "n_sats": total,
        "n_nlos": nlos,
        "nlos_frac": nlos / total if total else 0.0,
        "nlos_G": systems.get("G", 0),
        "nlos_E": systems.get("E", 0),
        "nlos_J": systems.get("J", 0),
        "nlos_C": systems.get("C", 0),
        "nlos_R": systems.get("R", 0),
    }


def _median(values: list[float]) -> float | str:
    vals = [v for v in values if isinstance(v, (int, float))]
    return float(median(vals)) if vals else ""


def _mean(values: list[float]) -> float | str:
    vals = [v for v in values if isinstance(v, (int, float))]
    return float(sum(vals) / len(vals)) if vals else ""


def analyze(label_a: str, mask_a: Path, label_b: str, mask_b: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    a = _read_mask(mask_a)
    b = _read_mask(mask_b)
    all_tows = sorted(set(a) | set(b))
    epoch_rows: list[dict[str, Any]] = []
    for tow in all_tows:
        fa = _epoch_features(a.get(tow, {}))
        fb = _epoch_features(b.get(tow, {}))
        common_prns = sorted(set(a.get(tow, {})) & set(b.get(tow, {})))
        a_only_nlos = 0
        b_only_nlos = 0
        disagree = 0
        for prn in common_prns:
            a_los = bool(a[tow][prn]["is_los"])
            b_los = bool(b[tow][prn]["is_los"])
            if a_los != b_los:
                disagree += 1
                if not a_los and b_los:
                    a_only_nlos += 1
                elif a_los and not b_los:
                    b_only_nlos += 1
        epoch_rows.append(
            {
                "tow": tow,
                f"{label_a}_available": tow in a,
                f"{label_b}_available": tow in b,
                f"{label_a}_n_sats": fa["n_sats"],
                f"{label_a}_n_nlos": fa["n_nlos"],
                f"{label_a}_nlos_frac": fa["nlos_frac"],
                f"{label_b}_n_sats": fb["n_sats"],
                f"{label_b}_n_nlos": fb["n_nlos"],
                f"{label_b}_nlos_frac": fb["nlos_frac"],
                "common_prns": len(common_prns),
                "mask_disagree_prns": disagree,
                f"{label_a}_only_nlos_prns": a_only_nlos,
                f"{label_b}_only_nlos_prns": b_only_nlos,
                "nlos_frac_delta_a_minus_b": fa["nlos_frac"] - fb["nlos_frac"] if tow in a and tow in b else "",
                "n_nlos_delta_a_minus_b": fa["n_nlos"] - fb["n_nlos"] if tow in a and tow in b else "",
            },
        )

    common = [row for row in epoch_rows if row[f"{label_a}_available"] and row[f"{label_b}_available"]]
    summary = [
        {
            "label_a": label_a,
            "label_b": label_b,
            "epochs_a": len(a),
            "epochs_b": len(b),
            "common_epochs": len(common),
            f"{label_a}_median_nlos_frac_all": _median([float(row[f"{label_a}_nlos_frac"]) for row in epoch_rows if row[f"{label_a}_available"]]),
            f"{label_b}_median_nlos_frac_all": _median([float(row[f"{label_b}_nlos_frac"]) for row in epoch_rows if row[f"{label_b}_available"]]),
            f"{label_a}_mean_nlos_frac_all": _mean([float(row[f"{label_a}_nlos_frac"]) for row in epoch_rows if row[f"{label_a}_available"]]),
            f"{label_b}_mean_nlos_frac_all": _mean([float(row[f"{label_b}_nlos_frac"]) for row in epoch_rows if row[f"{label_b}_available"]]),
            "median_nlos_frac_delta_a_minus_b_common": _median([float(row["nlos_frac_delta_a_minus_b"]) for row in common]),
            "mean_nlos_frac_delta_a_minus_b_common": _mean([float(row["nlos_frac_delta_a_minus_b"]) for row in common]),
            "median_n_nlos_delta_a_minus_b_common": _median([float(row["n_nlos_delta_a_minus_b"]) for row in common]),
            "mean_mask_disagree_prns_common": _mean([float(row["mask_disagree_prns"]) for row in common]),
            "median_mask_disagree_prns_common": _median([float(row["mask_disagree_prns"]) for row in common]),
            "mean_a_only_nlos_prns_common": _mean([float(row[f"{label_a}_only_nlos_prns"]) for row in common]),
            "mean_b_only_nlos_prns_common": _mean([float(row[f"{label_b}_only_nlos_prns"]) for row in common]),
        }
    ]
    return summary, epoch_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-a", default="xd_gici_hs")
    parser.add_argument("--mask-a", type=Path, required=True)
    parser.add_argument("--label-b", default="xd_gici_c4")
    parser.add_argument("--mask-b", type=Path, required=True)
    parser.add_argument("--out-prefix", type=Path, default=Path("experiments/results/phase66_plateau_candidate_mask_diff"))
    args = parser.parse_args()
    summary, epochs = analyze(str(args.label_a), args.mask_a, str(args.label_b), args.mask_b)
    _write_csv(args.out_prefix.with_name(args.out_prefix.name + "_summary.csv"), summary)
    _write_csv(args.out_prefix.with_name(args.out_prefix.name + "_epochs.csv"), epochs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
