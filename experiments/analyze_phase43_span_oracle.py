#!/usr/bin/env python3
"""Oracle-check existing candidates on Phase43 residual-gap spans."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "python"))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from exp_ppc_ctrbpf_fgo import (  # noqa: E402
    _load_hybrid_pos_file,
    _load_rtk_diag_file,
    _rtkdiag_candidate_gate,
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


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


def _float(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, ""))
    except ValueError:
        return float("nan")


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


def _split_csv_values(text: str) -> list[str]:
    return [item.strip() for item in str(text or "").split(",") if item.strip()]


def _load_pool(labels: list[str], dirs: list[str], city: str, run: str) -> list[dict[str, Any]]:
    if len(labels) != len(dirs):
        raise ValueError(f"labels/dirs length mismatch: {len(labels)} != {len(dirs)}")
    out: list[dict[str, Any]] = []
    for label, raw_dir in zip(labels, dirs):
        base = Path(raw_dir)
        pos_path = base / f"{city}_{run}_full.pos"
        diag_path = base / f"{city}_{run}_full.csv"
        if not pos_path.is_file():
            continue
        pos, _status = _load_hybrid_pos_file(pos_path)
        diag = _load_rtk_diag_file(diag_path) if diag_path.is_file() else {}
        out.append(
            {
                "label": label,
                "pos": pos,
                "diag": diag,
                "pos_path": str(pos_path),
                "diag_path": str(diag_path),
            },
        )
    return out


def _top_labels(counter: Counter[str], n: int = 5) -> str:
    return ",".join(f"{label}:{count}" for label, count in counter.most_common(n))


def analyze_spans(
    *,
    internal_rows: list[dict[str, str]],
    span_rows: list[dict[str, str]],
    candidates: list[dict[str, Any]],
    top: int,
    threshold_m: float,
    ratio_min: float,
    residual_rms_max: float,
    status5_residual_rms_max: float,
) -> list[dict[str, Any]]:
    weights = _distance_weights(internal_rows)
    rows_by_epoch = {int(float(row["epoch"])): row for row in internal_rows}
    out: list[dict[str, Any]] = []
    for span in span_rows[:top]:
        start = int(float(span["start_epoch"]))
        end = int(float(span["end_epoch"]))
        epoch_rows = [rows_by_epoch[i] for i in range(start, end + 1) if i in rows_by_epoch]
        current_pass_m = 0.0
        current_total_m = 0.0
        all_pass_m = 0.0
        gated_pass_m = 0.0
        all_best_errors: list[float] = []
        gated_best_errors: list[float] = []
        all_best_labels: Counter[str] = Counter()
        gated_best_labels: Counter[str] = Counter()
        available_counts: list[int] = []
        gated_counts: list[int] = []
        for row in epoch_rows:
            epoch = int(float(row["epoch"]))
            tow = round(float(row["tow"]), 1)
            truth = np.array([_float(row, "ref_x"), _float(row, "ref_y"), _float(row, "ref_z")], dtype=np.float64)
            weight = weights[epoch]
            current_error = _float(row, "emit_to_ref_m")
            current_total_m += weight
            if math.isfinite(current_error) and current_error <= threshold_m:
                current_pass_m += weight

            all_options: list[tuple[str, float]] = []
            gated_options: list[tuple[str, float]] = []
            for cand in candidates:
                pos = cand["pos"].get(tow)
                if pos is None or not np.all(np.isfinite(pos)):
                    continue
                dist = float(np.linalg.norm(np.asarray(pos, dtype=np.float64) - truth))
                all_options.append((str(cand["label"]), dist))
                diag_row = cand["diag"].get(tow)
                if _rtkdiag_candidate_gate(
                    diag_row,
                    ratio_min=ratio_min,
                    residual_rms_max=residual_rms_max,
                    status5_residual_rms_max=status5_residual_rms_max,
                ):
                    gated_options.append((str(cand["label"]), dist))
            available_counts.append(len(all_options))
            gated_counts.append(len(gated_options))
            if all_options:
                label, dist = min(all_options, key=lambda item: item[1])
                all_best_labels[label] += 1
                all_best_errors.append(dist)
                if dist <= threshold_m:
                    all_pass_m += weight
            if gated_options:
                label, dist = min(gated_options, key=lambda item: item[1])
                gated_best_labels[label] += 1
                gated_best_errors.append(dist)
                if dist <= threshold_m:
                    gated_pass_m += weight

        def mean(values: list[float]) -> float | str:
            return float(sum(values) / len(values)) if values else ""

        out.append(
            {
                "label": span["label"],
                "family": span["family"],
                "start_epoch": start,
                "end_epoch": end,
                "start_tow": span["start_tow"],
                "end_tow": span["end_tow"],
                "n_epochs": len(epoch_rows),
                "current_total_m": current_total_m,
                "current_pass_m": current_pass_m,
                "current_fail_m": current_total_m - current_pass_m,
                "all_oracle_pass_m": all_pass_m,
                "all_oracle_gain_m": all_pass_m - current_pass_m,
                "gated_oracle_pass_m": gated_pass_m,
                "gated_oracle_gain_m": gated_pass_m - current_pass_m,
                "mean_available_candidates": float(sum(available_counts) / len(available_counts)) if available_counts else "",
                "mean_gated_candidates": float(sum(gated_counts) / len(gated_counts)) if gated_counts else "",
                "all_best_mean_error_m": mean(all_best_errors),
                "gated_best_mean_error_m": mean(gated_best_errors),
                "all_best_labels": _top_labels(all_best_labels),
                "gated_best_labels": _top_labels(gated_best_labels),
            },
        )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("internal_epochs_csv", type=Path)
    parser.add_argument("spans_csv", type=Path)
    parser.add_argument("--city", default="nagoya")
    parser.add_argument("--run", default="run2")
    parser.add_argument("--labels", required=True)
    parser.add_argument("--candidate-dirs", required=True)
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--threshold-m", type=float, default=0.5)
    parser.add_argument("--ratio-min", type=float, default=1.0)
    parser.add_argument("--residual-rms-max", type=float, default=50.0)
    parser.add_argument("--status5-residual-rms-max", type=float, default=0.3)
    parser.add_argument("--out-csv", type=Path, default=Path("experiments/results/phase58_span_oracle.csv"))
    args = parser.parse_args(argv)

    labels = _split_csv_values(args.labels)
    dirs = _split_csv_values(args.candidate_dirs)
    candidates = _load_pool(labels, dirs, args.city, args.run)
    if not candidates:
        raise SystemExit("no candidates loaded")
    rows = analyze_spans(
        internal_rows=_read_csv(args.internal_epochs_csv),
        span_rows=_read_csv(args.spans_csv),
        candidates=candidates,
        top=args.top,
        threshold_m=args.threshold_m,
        ratio_min=args.ratio_min,
        residual_rms_max=args.residual_rms_max,
        status5_residual_rms_max=args.status5_residual_rms_max,
    )
    _write_csv(args.out_csv, rows)
    print(f"loaded candidates: {len(candidates)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
