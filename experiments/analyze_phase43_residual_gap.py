#!/usr/bin/env python3
"""Analyze remaining Phase43 PPC gap from internal per-epoch diagnostics."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


GICI_PREFIX = "xd_gici_"


def _float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value == "":
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return float("nan")


def _base_label(row: dict[str, str]) -> str:
    label = row.get("rtkdiag_selected_base_label", "")
    if label:
        return label
    label = row.get("rtkdiag_selected_label", "").removesuffix("+rnk")
    if label:
        return label
    source = row.get("emitted_source", "").strip()
    return source or "unknown"


def _family(label: str) -> str:
    if label.startswith(GICI_PREFIX):
        return "gici"
    if label.startswith("xd_fgo_"):
        return "fgo"
    if label == "pf_bridge":
        return "pf_bridge"
    return "non_gici"


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _distance_weights(rows: list[dict[str, str]]) -> list[float]:
    xyz = [(_float(row, "ref_x"), _float(row, "ref_y"), _float(row, "ref_z")) for row in rows]
    weights: list[float] = []
    for i, pos in enumerate(xyz):
        if i == 0:
            weights.append(0.0)
            continue
        prev = xyz[i - 1]
        if any(not math.isfinite(v) for v in (*prev, *pos)):
            weights.append(0.0)
        else:
            weights.append(float(math.dist(prev, pos)))
    return weights


def _safe_median(values: list[float]) -> float | str:
    finite = [v for v in values if math.isfinite(v)]
    return float(statistics.median(finite)) if finite else ""


def _safe_p95(values: list[float]) -> float | str:
    finite = sorted(v for v in values if math.isfinite(v))
    if not finite:
        return ""
    idx = min(int(math.ceil(0.95 * len(finite))) - 1, len(finite) - 1)
    return float(finite[idx])


def _summaries(rows: list[dict[str, str]], weights: list[float], pass_m: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_label: dict[str, list[int]] = defaultdict(list)
    by_family: dict[str, list[int]] = defaultdict(list)
    errors = [_float(row, "emit_to_ref_m") for row in rows]
    for i, row in enumerate(rows):
        label = _base_label(row)
        by_label[label].append(i)
        by_family[_family(label)].append(i)

    def build(group_name: str, idxs: list[int]) -> dict[str, Any]:
        total_m = sum(weights[i] for i in idxs)
        pass_idxs = [i for i in idxs if math.isfinite(errors[i]) and errors[i] <= pass_m]
        fail_idxs = [i for i in idxs if not math.isfinite(errors[i]) or errors[i] > pass_m]
        pass_meters = sum(weights[i] for i in pass_idxs)
        fail_meters = total_m - pass_meters
        err_values = [errors[i] for i in idxs]
        far_fail_idxs = [i for i in idxs if math.isfinite(errors[i]) and errors[i] > 3.0]
        return {
            "group": group_name,
            "n_epochs": len(idxs),
            "total_m": total_m,
            "pass_m": pass_meters,
            "fail_m": fail_meters,
            "score_pct": 100.0 * pass_meters / total_m if total_m > 0.0 else "",
            "fail_epoch_count": len(fail_idxs),
            "far_fail_epoch_count": len(far_fail_idxs),
            "median_error_m": _safe_median(err_values),
            "p95_error_m": _safe_p95(err_values),
        }

    label_rows = [build(label, idxs) for label, idxs in by_label.items()]
    family_rows = [build(family, idxs) for family, idxs in by_family.items()]
    label_rows.sort(key=lambda row: float(row["fail_m"]), reverse=True)
    family_rows.sort(key=lambda row: float(row["fail_m"]), reverse=True)
    return label_rows, family_rows


def _spans(rows: list[dict[str, str]], weights: list[float], pass_m: float) -> list[dict[str, Any]]:
    errors = [_float(row, "emit_to_ref_m") for row in rows]
    spans: list[dict[str, Any]] = []
    start = 0
    while start < len(rows):
        label = _base_label(rows[start])
        family = _family(label)
        end = start + 1
        while end < len(rows) and _base_label(rows[end]) == label:
            end += 1
        idxs = list(range(start, end))
        total_m = sum(weights[i] for i in idxs)
        pass_meters = sum(
            weights[i]
            for i in idxs
            if math.isfinite(errors[i]) and errors[i] <= pass_m
        )
        fail_meters = total_m - pass_meters
        if fail_meters > 0.0:
            spans.append(
                {
                    "label": label,
                    "family": family,
                    "start_epoch": int(float(rows[start]["epoch"])),
                    "end_epoch": int(float(rows[end - 1]["epoch"])),
                    "start_tow": _float(rows[start], "tow"),
                    "end_tow": _float(rows[end - 1], "tow"),
                    "n_epochs": len(idxs),
                    "total_m": total_m,
                    "pass_m": pass_meters,
                    "fail_m": fail_meters,
                    "score_pct": 100.0 * pass_meters / total_m if total_m > 0.0 else "",
                    "median_error_m": _safe_median([errors[i] for i in idxs]),
                    "p95_error_m": _safe_p95([errors[i] for i in idxs]),
                    "median_diag_rms": _safe_median([_float(rows[i], "rtkdiag_selected_diag_rms") for i in idxs]),
                    "median_agree_1m": _safe_median([_float(rows[i], "rtkdiag_candidate_agreement_count_1m") for i in idxs]),
                    "median_family_span_m": _safe_median([_float(rows[i], "rtkdiag_candidate_family_span_m") for i in idxs]),
                },
            )
        start = end
    spans.sort(key=lambda row: float(row["fail_m"]), reverse=True)
    return spans


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved: {path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("internal_epochs_csv", type=Path)
    parser.add_argument("--pass-m", type=float, default=0.5)
    parser.add_argument("--out-prefix", type=Path, default=Path("experiments/results/phase57_gap_nagoya_run2"))
    args = parser.parse_args(argv)

    rows = _read_rows(args.internal_epochs_csv)
    weights = _distance_weights(rows)
    label_rows, family_rows = _summaries(rows, weights, args.pass_m)
    span_rows = _spans(rows, weights, args.pass_m)

    total_m = sum(weights)
    pass_meters = sum(
        weight
        for row, weight in zip(rows, weights)
        if math.isfinite(_float(row, "emit_to_ref_m")) and _float(row, "emit_to_ref_m") <= args.pass_m
    )
    overall = [
        {
            "n_epochs": len(rows),
            "total_m": total_m,
            "pass_m": pass_meters,
            "fail_m": total_m - pass_meters,
            "score_pct": 100.0 * pass_meters / total_m if total_m > 0.0 else "",
        },
    ]
    _write_csv(Path(f"{args.out_prefix}_overall.csv"), overall)
    _write_csv(Path(f"{args.out_prefix}_by_family.csv"), family_rows)
    _write_csv(Path(f"{args.out_prefix}_by_label.csv"), label_rows)
    _write_csv(Path(f"{args.out_prefix}_spans.csv"), span_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
