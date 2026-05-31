#!/usr/bin/env python3
"""Boundary and motion-continuity cues for Phase58 no-oracle spans."""

from __future__ import annotations

import argparse
import csv
import math
import sys
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

from analyze_phase61_no_oracle_span_bias import _ecef_delta_to_enu  # noqa: E402


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


def _xyz(row: dict[str, str], prefix: str) -> np.ndarray | None:
    xyz = np.array([_float(row, f"{prefix}_x"), _float(row, f"{prefix}_y"), _float(row, f"{prefix}_z")], dtype=np.float64)
    return xyz if np.all(np.isfinite(xyz)) else None


def _ref(row: dict[str, str]) -> np.ndarray | None:
    xyz = np.array([_float(row, "ref_x"), _float(row, "ref_y"), _float(row, "ref_z")], dtype=np.float64)
    return xyz if np.all(np.isfinite(xyz)) else None


def _enu_delta(a: np.ndarray | None, b: np.ndarray | None, ref: np.ndarray | None) -> tuple[float, float, float]:
    if a is None or b is None or ref is None:
        return (float("nan"), float("nan"), float("nan"))
    enu = _ecef_delta_to_enu(a - b, ref)
    return (float(enu[0]), float(enu[1]), float(enu[2]))


def _norm3(values: tuple[float, float, float]) -> float:
    if any(not math.isfinite(v) for v in values):
        return float("nan")
    return float(math.sqrt(sum(v * v for v in values)))


def _cosine(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    if any(not math.isfinite(v) for v in (*a, *b)):
        return float("nan")
    na = _norm3(a)
    nb = _norm3(b)
    if na <= 0.0 or nb <= 0.0:
        return float("nan")
    return float(sum(x * y for x, y in zip(a, b)) / (na * nb))


def _spans_to_analyze(span_rows: list[dict[str, str]], top: int, include_recoverable: bool) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for row in span_rows:
        gain = _float(row, "gated_oracle_gain_m")
        if include_recoverable or (math.isfinite(gain) and gain <= 1e-9):
            out.append(row)
        if len(out) >= top:
            break
    return out


def _window(rows_by_epoch: dict[int, dict[str, str]], start: int, end: int) -> list[dict[str, str]]:
    return [rows_by_epoch[e] for e in range(start, end + 1) if e in rows_by_epoch]


def _stats(rows: list[dict[str, str]]) -> dict[str, Any]:
    step = [_float(row, "rtkdiag_selected_to_prev_selected_m") for row in rows]
    vel = [_float(row, "rtkdiag_selected_velocity_mps") for row in rows]
    tdcp = [_float(row, "rtkdiag_selected_to_tdcp_velocity_mps") for row in rows]
    span = [_float(row, "rtkdiag_candidate_family_span_m") for row in rows]
    agree = [_float(row, "rtkdiag_candidate_agreement_count_1m") for row in rows]
    return {
        "median_step_m": _median(step),
        "p95_step_m": _p95(step),
        "max_step_m": max(_finite(step), default=""),
        "median_velocity_mps": _median(vel),
        "median_tdcp_disagreement_mps": _median(tdcp),
        "p95_tdcp_disagreement_mps": _p95(tdcp),
        "tdcp_gt_2_mps_epochs": sum(1 for v in tdcp if math.isfinite(v) and v > 2.0),
        "step_gt_1m_epochs": sum(1 for v in step if math.isfinite(v) and v > 1.0),
        "median_family_span_m": _median(span),
        "median_agreement_1m": _median(agree),
    }


def analyze(
    internal_rows: list[dict[str, str]],
    span_rows: list[dict[str, str]],
    *,
    top: int,
    include_recoverable: bool,
    context_epochs: int,
) -> list[dict[str, Any]]:
    rows_by_epoch = {_int(row, "epoch"): row for row in internal_rows}
    out: list[dict[str, Any]] = []
    for span_index, span in enumerate(_spans_to_analyze(span_rows, top, include_recoverable), 1):
        start = _int(span, "start_epoch")
        end = _int(span, "end_epoch")
        span_rows_local = _window(rows_by_epoch, start, end)
        pre_rows = _window(rows_by_epoch, start - context_epochs, start - 1)
        post_rows = _window(rows_by_epoch, end + 1, end + context_epochs)
        start_row = rows_by_epoch.get(start)
        prev_row = rows_by_epoch.get(start - 1)
        end_row = rows_by_epoch.get(end)
        next_row = rows_by_epoch.get(end + 1)

        biases: list[np.ndarray] = []
        for row in span_rows_local:
            pos = _xyz(row, "pf_epoch_end")
            truth = _ref(row)
            if pos is None or truth is None:
                continue
            biases.append(_ecef_delta_to_enu(pos - truth, truth))
        bias_arr = np.asarray(biases, dtype=np.float64) if biases else np.empty((0, 3), dtype=np.float64)
        median_bias = np.median(bias_arr, axis=0) if len(bias_arr) else np.array([float("nan")] * 3)
        oracle_correction = tuple(float(-v) for v in median_bias)

        start_jump = _enu_delta(_xyz(start_row, "pf_epoch_end") if start_row else None, _xyz(prev_row, "pf_epoch_end") if prev_row else None, _ref(start_row) if start_row else None)
        end_jump = _enu_delta(_xyz(next_row, "pf_epoch_end") if next_row else None, _xyz(end_row, "pf_epoch_end") if end_row else None, _ref(end_row) if end_row else None)
        start_ref_step = _enu_delta(_ref(start_row) if start_row else None, _ref(prev_row) if prev_row else None, _ref(start_row) if start_row else None)
        end_ref_step = _enu_delta(_ref(next_row) if next_row else None, _ref(end_row) if end_row else None, _ref(end_row) if end_row else None)
        start_excess = tuple(a - b for a, b in zip(start_jump, start_ref_step))
        end_excess = tuple(a - b for a, b in zip(end_jump, end_ref_step))

        span_stats = _stats(span_rows_local)
        pre_stats = _stats(pre_rows)
        post_stats = _stats(post_rows)
        out.append(
            {
                "span_index": span_index,
                "label": span.get("label", ""),
                "start_epoch": start,
                "end_epoch": end,
                "n_epochs": len(span_rows_local),
                "current_fail_m": span.get("current_fail_m", ""),
                "gated_oracle_gain_m": span.get("gated_oracle_gain_m", ""),
                "oracle_correction_e_m": oracle_correction[0],
                "oracle_correction_n_m": oracle_correction[1],
                "oracle_correction_u_m": oracle_correction[2],
                "start_jump_e_m": start_jump[0],
                "start_jump_n_m": start_jump[1],
                "start_jump_u_m": start_jump[2],
                "start_jump_norm_m": _norm3(start_jump),
                "start_excess_e_m": start_excess[0],
                "start_excess_n_m": start_excess[1],
                "start_excess_u_m": start_excess[2],
                "start_excess_norm_m": _norm3(start_excess),
                "start_excess_to_correction_cos": _cosine(start_excess, oracle_correction),
                "end_jump_e_m": end_jump[0],
                "end_jump_n_m": end_jump[1],
                "end_jump_u_m": end_jump[2],
                "end_jump_norm_m": _norm3(end_jump),
                "end_excess_e_m": end_excess[0],
                "end_excess_n_m": end_excess[1],
                "end_excess_u_m": end_excess[2],
                "end_excess_norm_m": _norm3(end_excess),
                "end_excess_to_correction_cos": _cosine(end_excess, oracle_correction),
                **{f"pre_{k}": v for k, v in pre_stats.items()},
                **{f"span_{k}": v for k, v in span_stats.items()},
                **{f"post_{k}": v for k, v in post_stats.items()},
            },
        )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("internal_epochs_csv", type=Path)
    parser.add_argument("span_oracle_csv", type=Path)
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--include-recoverable", action="store_true")
    parser.add_argument("--context-epochs", type=int, default=30)
    parser.add_argument("--out-csv", type=Path, default=Path("experiments/results/phase63_motion_boundary_cues.csv"))
    args = parser.parse_args(argv)
    rows = analyze(
        _read_csv(args.internal_epochs_csv),
        _read_csv(args.span_oracle_csv),
        top=int(args.top),
        include_recoverable=bool(args.include_recoverable),
        context_epochs=int(args.context_epochs),
    )
    _write_csv(args.out_csv, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
