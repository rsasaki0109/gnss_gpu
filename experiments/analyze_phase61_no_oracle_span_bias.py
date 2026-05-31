#!/usr/bin/env python3
"""Bias diagnostics for Phase58 spans with no existing-pool oracle gain."""

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
from exp_ppc_ctrbpf_fgo import _diag_float, _rtkdiag_candidate_gate  # noqa: E402

_WGS84_A = 6378137.0
_WGS84_E2 = 6.69437999014e-3


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


def _mean(values: list[float]) -> float | str:
    values = _finite(values)
    return float(sum(values) / len(values)) if values else ""


def _median(values: list[float]) -> float | str:
    values = _finite(values)
    return float(median(values)) if values else ""


def _p95(values: list[float]) -> float | str:
    values = sorted(_finite(values))
    if not values:
        return ""
    idx = min(len(values) - 1, int(math.ceil(0.95 * len(values))) - 1)
    return float(values[idx])


def _ecef_to_llh(xyz: np.ndarray) -> tuple[float, float, float]:
    x, y, z = (float(v) for v in xyz)
    lon = math.atan2(y, x)
    p = math.hypot(x, y)
    lat = math.atan2(z, p * (1.0 - _WGS84_E2))
    h = 0.0
    for _ in range(8):
        sin_lat = math.sin(lat)
        n = _WGS84_A / math.sqrt(1.0 - _WGS84_E2 * sin_lat * sin_lat)
        h = p / max(math.cos(lat), 1e-12) - n
        lat = math.atan2(z, p * (1.0 - _WGS84_E2 * n / (n + h)))
    return lat, lon, h


def _ecef_delta_to_enu(delta: np.ndarray, ref_xyz: np.ndarray) -> np.ndarray:
    lat, lon, _h = _ecef_to_llh(ref_xyz)
    sin_lat, cos_lat = math.sin(lat), math.cos(lat)
    sin_lon, cos_lon = math.sin(lon), math.cos(lon)
    rot = np.array(
        [
            [-sin_lon, cos_lon, 0.0],
            [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
            [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat],
        ],
        dtype=np.float64,
    )
    return rot @ delta


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


def _mode_status(statuses: list[float]) -> str:
    valid = [int(v) for v in statuses if math.isfinite(v)]
    if not valid:
        return ""
    mode, count = Counter(valid).most_common(1)[0]
    return f"{mode}:{count}"


def _labels(counter: Counter[str], n: int = 8) -> str:
    return ",".join(f"{label}:{count}" for label, count in counter.most_common(n))


def _span_rows_to_analyze(span_rows: list[dict[str, str]], top: int, include_recoverable: bool) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    for row in span_rows:
        gain = _float(row, "gated_oracle_gain_m")
        if include_recoverable or (math.isfinite(gain) and gain <= 1e-9):
            selected.append(row)
        if len(selected) >= top:
            break
    return selected


def analyze(
    *,
    internal_rows: list[dict[str, str]],
    span_rows: list[dict[str, str]],
    candidates: list[dict[str, Any]],
    top: int,
    include_recoverable: bool,
    threshold_m: float,
    ratio_min: float,
    residual_rms_max: float,
    status5_residual_rms_max: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows_by_epoch = {_int(row, "epoch"): row for row in internal_rows}
    weights = _distance_weights(internal_rows)
    span_summary: list[dict[str, Any]] = []
    candidate_summary: list[dict[str, Any]] = []
    epoch_summary: list[dict[str, Any]] = []

    for span_index, span in enumerate(_span_rows_to_analyze(span_rows, top, include_recoverable), 1):
        start = _int(span, "start_epoch")
        end = _int(span, "end_epoch")
        epoch_rows = [rows_by_epoch[e] for e in range(start, end + 1) if e in rows_by_epoch]
        best_all_labels: Counter[str] = Counter()
        best_gated_labels: Counter[str] = Counter()
        best_all_errors: list[float] = []
        best_gated_errors: list[float] = []
        selected_errors: list[float] = []

        per_label: dict[str, dict[str, Any]] = {}
        for cand in candidates:
            per_label[str(cand["label"])] = {
                "errors": [],
                "h_errors": [],
                "e": [],
                "n": [],
                "u": [],
                "gated_errors": [],
                "gated_e": [],
                "gated_n": [],
                "gated_u": [],
                "rms": [],
                "ratio": [],
                "status": [],
            }

        for row in epoch_rows:
            epoch = _int(row, "epoch")
            tow = round(float(row["tow"]), 1)
            truth = np.array([_float(row, "ref_x"), _float(row, "ref_y"), _float(row, "ref_z")], dtype=np.float64)
            selected_errors.append(_float(row, "emit_to_ref_m"))
            all_options: list[tuple[str, float]] = []
            gated_options: list[tuple[str, float]] = []
            for cand in candidates:
                label = str(cand["label"])
                pos = cand["pos"].get(tow)
                if pos is None or not np.all(np.isfinite(pos)):
                    continue
                pos_xyz = np.asarray(pos, dtype=np.float64)
                enu = _ecef_delta_to_enu(pos_xyz - truth, truth)
                err = float(np.linalg.norm(pos_xyz - truth))
                h_err = float(math.hypot(float(enu[0]), float(enu[1])))
                stats = per_label[label]
                stats["errors"].append(err)
                stats["h_errors"].append(h_err)
                stats["e"].append(float(enu[0]))
                stats["n"].append(float(enu[1]))
                stats["u"].append(float(enu[2]))
                all_options.append((label, err))
                diag = cand["diag"].get(tow)
                if diag:
                    stats["rms"].append(_diag_float(diag, "final_residual_rms"))
                    stats["ratio"].append(_diag_float(diag, "ambiguity_ratio"))
                    stats["status"].append(_diag_float(diag, "final_status"))
                if _rtkdiag_candidate_gate(
                    diag,
                    ratio_min=ratio_min,
                    residual_rms_max=residual_rms_max,
                    status5_residual_rms_max=status5_residual_rms_max,
                ):
                    stats["gated_errors"].append(err)
                    stats["gated_e"].append(float(enu[0]))
                    stats["gated_n"].append(float(enu[1]))
                    stats["gated_u"].append(float(enu[2]))
                    gated_options.append((label, err))
            if all_options:
                best_label, best_err = min(all_options, key=lambda item: item[1])
                best_all_labels[best_label] += 1
                best_all_errors.append(best_err)
            if gated_options:
                best_label, best_err = min(gated_options, key=lambda item: item[1])
                best_gated_labels[best_label] += 1
                best_gated_errors.append(best_err)
            epoch_summary.append(
                {
                    "span_index": span_index,
                    "span_label": span.get("label", ""),
                    "epoch": epoch,
                    "tow": row.get("tow", ""),
                    "weight_m": weights[epoch],
                    "selected_label": row.get("rtkdiag_selected_base_label", ""),
                    "selected_error_m": _float(row, "emit_to_ref_m"),
                    "best_all_error_m": min([v for _label, v in all_options], default=float("nan")),
                    "best_gated_error_m": min([v for _label, v in gated_options], default=float("nan")),
                    "n_available_candidates": len(all_options),
                    "n_gated_candidates": len(gated_options),
                },
            )

        current_fail_m = sum(
            weights[_int(row, "epoch")]
            for row in epoch_rows
            if not (math.isfinite(_float(row, "emit_to_ref_m")) and _float(row, "emit_to_ref_m") <= threshold_m)
        )
        span_summary.append(
            {
                "span_index": span_index,
                "label": span.get("label", ""),
                "start_epoch": start,
                "end_epoch": end,
                "n_epochs": len(epoch_rows),
                "current_fail_m": current_fail_m,
                "gated_oracle_gain_m": _float(span, "gated_oracle_gain_m"),
                "median_selected_error_m": _median(selected_errors),
                "median_best_all_error_m": _median(best_all_errors),
                "median_best_gated_error_m": _median(best_gated_errors),
                "p95_best_gated_error_m": _p95(best_gated_errors),
                "best_all_pass_epochs": sum(1 for v in best_all_errors if v <= threshold_m),
                "best_gated_pass_epochs": sum(1 for v in best_gated_errors if v <= threshold_m),
                "best_all_labels": _labels(best_all_labels),
                "best_gated_labels": _labels(best_gated_labels),
            },
        )

        for label, stats in per_label.items():
            errors = stats["errors"]
            if not errors:
                continue
            gated_errors = stats["gated_errors"]
            candidate_summary.append(
                {
                    "span_index": span_index,
                    "span_label": span.get("label", ""),
                    "start_epoch": start,
                    "end_epoch": end,
                    "candidate_label": label,
                    "available_epochs": len(errors),
                    "gated_epochs": len(gated_errors),
                    "pass_epochs": sum(1 for v in errors if v <= threshold_m),
                    "gated_pass_epochs": sum(1 for v in gated_errors if v <= threshold_m),
                    "median_error_m": _median(errors),
                    "mean_error_m": _mean(errors),
                    "p95_error_m": _p95(errors),
                    "median_horiz_error_m": _median(stats["h_errors"]),
                    "median_e_m": _median(stats["e"]),
                    "median_n_m": _median(stats["n"]),
                    "median_u_m": _median(stats["u"]),
                    "mean_e_m": _mean(stats["e"]),
                    "mean_n_m": _mean(stats["n"]),
                    "mean_u_m": _mean(stats["u"]),
                    "median_gated_error_m": _median(gated_errors),
                    "median_gated_e_m": _median(stats["gated_e"]),
                    "median_gated_n_m": _median(stats["gated_n"]),
                    "median_gated_u_m": _median(stats["gated_u"]),
                    "median_diag_rms": _median(stats["rms"]),
                    "median_diag_ratio": _median(stats["ratio"]),
                    "mode_diag_status": _mode_status(stats["status"]),
                },
            )
    candidate_summary.sort(key=lambda row: (int(row["span_index"]), float(row["median_error_m"])))
    return span_summary, candidate_summary, epoch_summary


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
    parser.add_argument("--ratio-min", type=float, default=1.0)
    parser.add_argument("--residual-rms-max", type=float, default=50.0)
    parser.add_argument("--status5-residual-rms-max", type=float, default=0.3)
    parser.add_argument("--out-prefix", type=Path, default=Path("experiments/results/phase61_no_oracle_span_bias"))
    args = parser.parse_args(argv)

    candidates = _load_pool(_split_csv_values(args.labels), _split_csv_values(args.candidate_dirs), args.city, args.run)
    if not candidates:
        raise SystemExit("no candidates loaded")
    span_summary, candidate_summary, epoch_summary = analyze(
        internal_rows=_read_csv(args.internal_epochs_csv),
        span_rows=_read_csv(args.span_oracle_csv),
        candidates=candidates,
        top=int(args.top),
        include_recoverable=bool(args.include_recoverable),
        threshold_m=float(args.threshold_m),
        ratio_min=float(args.ratio_min),
        residual_rms_max=float(args.residual_rms_max),
        status5_residual_rms_max=float(args.status5_residual_rms_max),
    )
    _write_csv(args.out_prefix.with_name(args.out_prefix.name + "_spans.csv"), span_summary)
    _write_csv(args.out_prefix.with_name(args.out_prefix.name + "_candidates.csv"), candidate_summary)
    _write_csv(args.out_prefix.with_name(args.out_prefix.name + "_epochs.csv"), epoch_summary)
    print(f"loaded candidates: {len(candidates)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
