#!/usr/bin/env python3
"""Oracle-bound constant bias correction on Phase58 no-oracle spans."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "python"))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from analyze_phase43_span_oracle import _load_pool, _split_csv_values  # noqa: E402
from analyze_phase61_no_oracle_span_bias import _ecef_delta_to_enu  # noqa: E402
from exp_ppc_ctrbpf_fgo import _rtkdiag_candidate_gate  # noqa: E402


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


def _int(row: dict[str, str], key: str) -> int:
    return int(float(row[key]))


def _finite(values: list[float]) -> list[float]:
    return [v for v in values if math.isfinite(v)]


def _median(values: list[float]) -> float | str:
    values = _finite(values)
    return float(median(values)) if values else ""


def _p95(values: list[float]) -> float | str:
    values = sorted(_finite(values))
    if not values:
        return ""
    idx = min(len(values) - 1, int(math.ceil(0.95 * len(values))) - 1)
    return float(values[idx])


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


def _spans_to_analyze(span_rows: list[dict[str, str]], top: int, include_recoverable: bool) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for row in span_rows:
        gain = _float(row, "gated_oracle_gain_m")
        if include_recoverable or (math.isfinite(gain) and gain <= 1e-9):
            out.append(row)
        if len(out) >= top:
            break
    return out


def _label_counts(labels: list[str]) -> str:
    return ",".join(f"{label}:{count}" for label, count in Counter(labels).most_common(8))


def _best_current_label(row: dict[str, str]) -> str:
    label = row.get("rtkdiag_selected_base_label", "")
    return label or row.get("rtkdiag_selected_label", "").removesuffix("+rnk")


def analyze(
    *,
    internal_rows: list[dict[str, str]],
    span_rows: list[dict[str, str]],
    candidates: list[dict[str, Any]],
    top: int,
    include_recoverable: bool,
    threshold_m: float,
    min_coverage_frac: float,
    gated_only: bool,
    ratio_min: float,
    residual_rms_max: float,
    status5_residual_rms_max: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows_by_epoch = {_int(row, "epoch"): row for row in internal_rows}
    weights = _distance_weights(internal_rows)
    span_out: list[dict[str, Any]] = []
    cand_out: list[dict[str, Any]] = []

    for span_index, span in enumerate(_spans_to_analyze(span_rows, top, include_recoverable), 1):
        start = _int(span, "start_epoch")
        end = _int(span, "end_epoch")
        epoch_rows = [rows_by_epoch[e] for e in range(start, end + 1) if e in rows_by_epoch]
        min_epochs = max(1, int(math.ceil(len(epoch_rows) * min_coverage_frac)))
        current_pass_m = 0.0
        current_total_m = 0.0
        selected_labels: list[str] = []
        selected_errors: list[float] = []
        for row in epoch_rows:
            epoch = _int(row, "epoch")
            current_total_m += weights[epoch]
            err = _float(row, "emit_to_ref_m")
            selected_errors.append(err)
            selected_labels.append(_best_current_label(row))
            if math.isfinite(err) and err <= threshold_m:
                current_pass_m += weights[epoch]

        best_row: dict[str, Any] | None = None
        for cand in candidates:
            label = str(cand["label"])
            epochs: list[int] = []
            truths: list[np.ndarray] = []
            positions: list[np.ndarray] = []
            raw_errors: list[float] = []
            weights_used: list[float] = []
            enu_biases: list[np.ndarray] = []
            for row in epoch_rows:
                epoch = _int(row, "epoch")
                tow = round(float(row["tow"]), 1)
                pos = cand["pos"].get(tow)
                if pos is None or not np.all(np.isfinite(pos)):
                    continue
                diag = cand["diag"].get(tow)
                if gated_only and not _rtkdiag_candidate_gate(
                    diag,
                    ratio_min=ratio_min,
                    residual_rms_max=residual_rms_max,
                    status5_residual_rms_max=status5_residual_rms_max,
                ):
                    continue
                truth = np.array([_float(row, "ref_x"), _float(row, "ref_y"), _float(row, "ref_z")], dtype=np.float64)
                pos_xyz = np.asarray(pos, dtype=np.float64)
                epochs.append(epoch)
                truths.append(truth)
                positions.append(pos_xyz)
                weights_used.append(weights[epoch])
                raw_errors.append(float(np.linalg.norm(pos_xyz - truth)))
                enu_biases.append(_ecef_delta_to_enu(pos_xyz - truth, truth))
            if len(epochs) < min_epochs:
                continue

            deltas = np.asarray(positions, dtype=np.float64) - np.asarray(truths, dtype=np.float64)
            # Truth-derived oracle constant correction. Median is robust against
            # occasional candidate jumps inside the span.
            correction_ecef = np.median(deltas, axis=0)
            corrected_positions = np.asarray(positions, dtype=np.float64) - correction_ecef
            corrected_errors = [
                float(np.linalg.norm(pos - truth))
                for pos, truth in zip(corrected_positions, truths)
            ]
            raw_pass_m = sum(w for w, err in zip(weights_used, raw_errors) if err <= threshold_m)
            corrected_pass_m = sum(w for w, err in zip(weights_used, corrected_errors) if err <= threshold_m)
            enu_arr = np.asarray(enu_biases, dtype=np.float64)
            correction_enu = np.median(enu_arr, axis=0)
            row_out = {
                "span_index": span_index,
                "span_label": span.get("label", ""),
                "start_epoch": start,
                "end_epoch": end,
                "candidate_label": label,
                "epochs_used": len(epochs),
                "coverage_frac": len(epochs) / len(epoch_rows) if epoch_rows else "",
                "raw_pass_m": raw_pass_m,
                "corrected_pass_m": corrected_pass_m,
                "gain_vs_candidate_raw_m": corrected_pass_m - raw_pass_m,
                "gain_vs_current_m": corrected_pass_m - current_pass_m,
                "median_raw_error_m": _median(raw_errors),
                "median_corrected_error_m": _median(corrected_errors),
                "p95_corrected_error_m": _p95(corrected_errors),
                "raw_pass_epochs": sum(1 for err in raw_errors if err <= threshold_m),
                "corrected_pass_epochs": sum(1 for err in corrected_errors if err <= threshold_m),
                "correction_e_m": -float(correction_enu[0]),
                "correction_n_m": -float(correction_enu[1]),
                "correction_u_m": -float(correction_enu[2]),
            }
            cand_out.append(row_out)
            if best_row is None or float(row_out["gain_vs_current_m"]) > float(best_row["gain_vs_current_m"]):
                best_row = row_out

        span_out.append(
            {
                "span_index": span_index,
                "label": span.get("label", ""),
                "start_epoch": start,
                "end_epoch": end,
                "n_epochs": len(epoch_rows),
                "current_total_m": current_total_m,
                "current_pass_m": current_pass_m,
                "current_fail_m": current_total_m - current_pass_m,
                "gated_oracle_gain_m": _float(span, "gated_oracle_gain_m"),
                "median_selected_error_m": _median(selected_errors),
                "selected_labels": _label_counts(selected_labels),
                "best_corrected_label": best_row.get("candidate_label", "") if best_row else "",
                "best_corrected_epochs_used": best_row.get("epochs_used", "") if best_row else "",
                "best_corrected_pass_m": best_row.get("corrected_pass_m", "") if best_row else "",
                "best_corrected_gain_vs_current_m": best_row.get("gain_vs_current_m", "") if best_row else "",
                "best_corrected_pass_epochs": best_row.get("corrected_pass_epochs", "") if best_row else "",
                "best_correction_e_m": best_row.get("correction_e_m", "") if best_row else "",
                "best_correction_n_m": best_row.get("correction_n_m", "") if best_row else "",
                "best_correction_u_m": best_row.get("correction_u_m", "") if best_row else "",
                "best_median_corrected_error_m": best_row.get("median_corrected_error_m", "") if best_row else "",
                "best_p95_corrected_error_m": best_row.get("p95_corrected_error_m", "") if best_row else "",
            },
        )

    cand_out.sort(key=lambda row: (int(row["span_index"]), -float(row["gain_vs_current_m"])))
    return span_out, cand_out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("internal_epochs_csv", type=Path)
    parser.add_argument("span_oracle_csv", type=Path)
    parser.add_argument("--city", default="nagoya")
    parser.add_argument("--run", default="run2")
    parser.add_argument("--labels", required=True)
    parser.add_argument("--candidate-dirs", required=True)
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--include-recoverable", action="store_true")
    parser.add_argument("--threshold-m", type=float, default=0.5)
    parser.add_argument("--min-coverage-frac", type=float, default=0.8)
    parser.add_argument("--gated-only", action="store_true")
    parser.add_argument("--ratio-min", type=float, default=1.0)
    parser.add_argument("--residual-rms-max", type=float, default=50.0)
    parser.add_argument("--status5-residual-rms-max", type=float, default=0.3)
    parser.add_argument("--out-prefix", type=Path, default=Path("experiments/results/phase62_span_bias_correction_oracle"))
    args = parser.parse_args(argv)

    candidates = _load_pool(_split_csv_values(args.labels), _split_csv_values(args.candidate_dirs), args.city, args.run)
    if not candidates:
        raise SystemExit("no candidates loaded")
    span_rows, cand_rows = analyze(
        internal_rows=_read_csv(args.internal_epochs_csv),
        span_rows=_read_csv(args.span_oracle_csv),
        candidates=candidates,
        top=int(args.top),
        include_recoverable=bool(args.include_recoverable),
        threshold_m=float(args.threshold_m),
        min_coverage_frac=float(args.min_coverage_frac),
        gated_only=bool(args.gated_only),
        ratio_min=float(args.ratio_min),
        residual_rms_max=float(args.residual_rms_max),
        status5_residual_rms_max=float(args.status5_residual_rms_max),
    )
    _write_csv(args.out_prefix.with_name(args.out_prefix.name + "_spans.csv"), span_rows)
    _write_csv(args.out_prefix.with_name(args.out_prefix.name + "_candidates.csv"), cand_rows)
    print(f"loaded candidates: {len(candidates)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
