#!/usr/bin/env python3
"""Summarize PPC realtime fusion per-epoch failure regimes."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _write_rows(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _float_or_nan(row: dict[str, object], key: str) -> float:
    value = row.get(key, "")
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _bool_value(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "t", "yes", "y"}


def _finite_array(rows: list[dict[str, object]], key: str) -> np.ndarray:
    return np.array([_float_or_nan(row, key) for row in rows], dtype=np.float64)


def _safe_count(mask: np.ndarray) -> int:
    return int(np.count_nonzero(mask))


def _safe_percent(count: int, total: int) -> float:
    return float(100.0 * int(count) / max(int(total), 1))


def _safe_median(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if finite.size else float("nan")


def _safe_p95(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.percentile(finite, 95)) if finite.size else float("nan")


def _safe_mean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if finite.size else float("nan")


def _vertical_abs_errors(errors_2d: np.ndarray, errors_3d: np.ndarray) -> np.ndarray:
    squared = np.maximum(np.square(errors_3d) - np.square(errors_2d), 0.0)
    return np.sqrt(squared)


def _high_dd_blend_mask(rows: list[dict[str, object]], base_dd_alpha: float) -> np.ndarray:
    alphas = _finite_array(rows, "dd_anchor_effective_alpha")
    return np.isfinite(alphas) & (alphas > float(base_dd_alpha) + 1.0e-9)


def _bool_mask(rows: list[dict[str, object]], key: str) -> np.ndarray:
    return np.array([_bool_value(row.get(key, "")) for row in rows], dtype=bool)


def summarize_epochs(
    rows: list[dict[str, object]],
    *,
    label: str,
    ppc_threshold_m: float = 0.5,
    base_dd_alpha: float = 0.3,
) -> dict[str, object]:
    errors_2d = _finite_array(rows, "fused_error_2d_m")
    errors_3d = _finite_array(rows, "fused_error_3d_m")
    wls_errors_3d = _finite_array(rows, "wls_error_3d_m")
    distances = _finite_array(rows, "ppc_segment_distance_m")
    if not np.any(np.isfinite(distances)):
        distances = np.ones(len(rows), dtype=np.float64)

    valid = np.isfinite(errors_3d)
    pass_mask = valid & (errors_3d <= float(ppc_threshold_m))
    horizontal_pass_mask = np.isfinite(errors_2d) & (errors_2d <= float(ppc_threshold_m))
    vertical_limited_mask = horizontal_pass_mask & ~pass_mask
    near_mask = valid & (errors_3d > float(ppc_threshold_m)) & (errors_3d <= 1.0)
    mid_mask = valid & (errors_3d > 1.0) & (errors_3d <= 3.0)
    far_mask = valid & (errors_3d > 3.0)
    finite_distances = np.where(np.isfinite(distances), distances, 0.0)
    total_distance_m = float(np.sum(finite_distances))
    pass_distance_m = float(np.sum(finite_distances[pass_mask]))
    vertical_abs = _vertical_abs_errors(errors_2d, errors_3d)
    improved_wls_mask = np.isfinite(wls_errors_3d) & valid & (errors_3d < wls_errors_3d)

    n_epochs = len(rows)
    pass_epochs = _safe_count(pass_mask)
    return {
        "label": label,
        "n_epochs": int(n_epochs),
        "ppc_pass_epochs": int(pass_epochs),
        "ppc_epoch_pass_pct": _safe_percent(pass_epochs, n_epochs),
        "ppc_pass_distance_m": pass_distance_m,
        "ppc_total_distance_m": total_distance_m,
        "ppc_score_pct": (
            float(100.0 * pass_distance_m / total_distance_m) if total_distance_m > 0.0 else 0.0
        ),
        "horizontal_pass_epochs": _safe_count(horizontal_pass_mask),
        "horizontal_pass_pct": _safe_percent(_safe_count(horizontal_pass_mask), n_epochs),
        "vertical_limited_epochs": _safe_count(vertical_limited_mask),
        "near_3d_fail_epochs": _safe_count(near_mask),
        "mid_3d_fail_epochs": _safe_count(mid_mask),
        "far_3d_fail_epochs": _safe_count(far_mask),
        "median_error_2d_m": _safe_median(errors_2d),
        "p95_error_2d_m": _safe_p95(errors_2d),
        "median_error_3d_m": _safe_median(errors_3d),
        "p95_error_3d_m": _safe_p95(errors_3d),
        "median_vertical_abs_m": _safe_median(vertical_abs),
        "p95_vertical_abs_m": _safe_p95(vertical_abs),
        "tdcp_used_epochs": _safe_count(_bool_mask(rows, "tdcp_used")),
        "last_velocity_epochs": _safe_count(_bool_mask(rows, "tdcp_last_velocity_used")),
        "dd_pr_anchor_epochs": _safe_count(_bool_mask(rows, "dd_pr_anchor_used")),
        "high_dd_blend_epochs": _safe_count(_high_dd_blend_mask(rows, base_dd_alpha)),
        "widelane_anchor_epochs": _safe_count(_bool_mask(rows, "widelane_anchor_used")),
        "rsp_correction_epochs": _safe_count(_bool_mask(rows, "rsp_correction_used")),
        "height_released_epochs": _safe_count(
            _bool_mask(rows, "height_hold_used")
            & (_finite_array(rows, "height_hold_effective_alpha") <= 0.0)
        ),
        "fused_improves_wls_3d_epochs": _safe_count(improved_wls_mask),
        "median_dd_shift_m": _safe_median(_finite_array(rows, "dd_pr_shift_m")),
        "median_dd_robust_rms_m": _safe_median(_finite_array(rows, "dd_pr_robust_rms_m")),
    }


def _subset_stats(
    rows: list[dict[str, object]],
    mask: np.ndarray,
    *,
    label: str,
    group: str,
    value: str,
    ppc_threshold_m: float,
) -> dict[str, object]:
    errors_3d = _finite_array(rows, "fused_error_3d_m")
    selected_errors = errors_3d[mask]
    pass_mask = mask & np.isfinite(errors_3d) & (errors_3d <= float(ppc_threshold_m))
    n_epochs = _safe_count(mask)
    return {
        "label": label,
        "group": group,
        "value": value,
        "n_epochs": n_epochs,
        "ppc_pass_epochs": _safe_count(pass_mask),
        "ppc_epoch_pass_pct": _safe_percent(_safe_count(pass_mask), n_epochs),
        "mean_error_3d_m": _safe_mean(selected_errors),
        "median_error_3d_m": _safe_median(selected_errors),
        "p95_error_3d_m": _safe_p95(selected_errors),
        "median_dd_shift_m": _safe_median(_finite_array(rows, "dd_pr_shift_m")[mask]),
        "median_dd_robust_rms_m": _safe_median(_finite_array(rows, "dd_pr_robust_rms_m")[mask]),
    }


def build_group_rows(
    rows: list[dict[str, object]],
    *,
    label: str,
    ppc_threshold_m: float = 0.5,
    base_dd_alpha: float = 0.3,
) -> list[dict[str, object]]:
    groups = {
        "tdcp_used": _bool_mask(rows, "tdcp_used"),
        "last_velocity_used": _bool_mask(rows, "tdcp_last_velocity_used"),
        "dd_pr_anchor_used": _bool_mask(rows, "dd_pr_anchor_used"),
        "high_dd_blend": _high_dd_blend_mask(rows, base_dd_alpha),
        "widelane_anchor_used": _bool_mask(rows, "widelane_anchor_used"),
        "rsp_correction_used": _bool_mask(rows, "rsp_correction_used"),
        "height_released": _bool_mask(rows, "height_hold_used")
        & (_finite_array(rows, "height_hold_effective_alpha") <= 0.0),
    }
    out: list[dict[str, object]] = []
    for group, mask in groups.items():
        out.append(
            _subset_stats(
                rows,
                mask,
                label=label,
                group=group,
                value="true",
                ppc_threshold_m=ppc_threshold_m,
            )
        )
        out.append(
            _subset_stats(
                rows,
                ~mask,
                label=label,
                group=group,
                value="false",
                ppc_threshold_m=ppc_threshold_m,
            )
        )
    return out


def build_failure_spans(
    rows: list[dict[str, object]],
    *,
    label: str,
    failure_threshold_m: float = 3.0,
    base_dd_alpha: float = 0.3,
) -> list[dict[str, object]]:
    errors_2d = _finite_array(rows, "fused_error_2d_m")
    errors_3d = _finite_array(rows, "fused_error_3d_m")
    epochs = _finite_array(rows, "epoch")
    tows = _finite_array(rows, "tow")
    distances = _finite_array(rows, "ppc_segment_distance_m")
    fail_idx = np.flatnonzero(np.isfinite(errors_3d) & (errors_3d > float(failure_threshold_m)))
    if fail_idx.size == 0:
        return []

    spans: list[dict[str, object]] = []
    span_start = 0
    span_id = 0
    for pos in range(1, len(fail_idx) + 1):
        end_of_span = pos == len(fail_idx) or fail_idx[pos] != fail_idx[pos - 1] + 1
        if not end_of_span:
            continue
        idx = fail_idx[span_start:pos]
        selected_rows = [rows[int(i)] for i in idx]
        spans.append(
            {
                "label": label,
                "span_id": span_id,
                "start_epoch": int(epochs[idx[0]]) if np.isfinite(epochs[idx[0]]) else int(idx[0]),
                "end_epoch": int(epochs[idx[-1]]) if np.isfinite(epochs[idx[-1]]) else int(idx[-1]),
                "start_tow": float(tows[idx[0]]) if np.isfinite(tows[idx[0]]) else "",
                "end_tow": float(tows[idx[-1]]) if np.isfinite(tows[idx[-1]]) else "",
                "n_epochs": int(len(idx)),
                "distance_m": float(np.nansum(distances[idx])),
                "median_error_2d_m": _safe_median(errors_2d[idx]),
                "max_error_2d_m": float(np.nanmax(errors_2d[idx])),
                "median_error_3d_m": _safe_median(errors_3d[idx]),
                "max_error_3d_m": float(np.nanmax(errors_3d[idx])),
                "tdcp_used_epochs": _safe_count(_bool_mask(selected_rows, "tdcp_used")),
                "last_velocity_epochs": _safe_count(
                    _bool_mask(selected_rows, "tdcp_last_velocity_used")
                ),
                "dd_pr_anchor_epochs": _safe_count(
                    _bool_mask(selected_rows, "dd_pr_anchor_used")
                ),
                "high_dd_blend_epochs": _safe_count(
                    _high_dd_blend_mask(selected_rows, base_dd_alpha)
                ),
                "widelane_anchor_epochs": _safe_count(
                    _bool_mask(selected_rows, "widelane_anchor_used")
                ),
                "rsp_correction_epochs": _safe_count(
                    _bool_mask(selected_rows, "rsp_correction_used")
                ),
                "height_released_epochs": _safe_count(
                    _bool_mask(selected_rows, "height_hold_used")
                    & (_finite_array(selected_rows, "height_hold_effective_alpha") <= 0.0)
                ),
            }
        )
        span_id += 1
        span_start = pos
    return spans


def _default_label(path: Path) -> str:
    stem = path.stem
    return stem.removesuffix("_epochs")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize PPC per-epoch failure regimes")
    parser.add_argument("--epoch-csv", type=Path, action="append", required=True)
    parser.add_argument("--label", type=str, action="append", default=[])
    parser.add_argument("--ppc-threshold-m", type=float, default=0.5)
    parser.add_argument("--failure-threshold-m", type=float, default=3.0)
    parser.add_argument("--base-dd-alpha", type=float, default=0.3)
    parser.add_argument("--results-prefix", type=str, default="ppc_epoch_failure_diagnostic")
    args = parser.parse_args()

    paths = [path.resolve() for path in args.epoch_csv]
    if args.label and len(args.label) != len(paths):
        raise ValueError("--label must be supplied once per --epoch-csv")
    labels = args.label or [_default_label(path) for path in paths]

    summary_rows: list[dict[str, object]] = []
    group_rows: list[dict[str, object]] = []
    span_rows: list[dict[str, object]] = []

    print("=" * 72)
    print("  PPC Epoch Failure Diagnostic")
    print("=" * 72)
    for path, label in zip(paths, labels, strict=True):
        rows = _read_rows(path)
        summary = summarize_epochs(
            rows,
            label=label,
            ppc_threshold_m=args.ppc_threshold_m,
            base_dd_alpha=args.base_dd_alpha,
        )
        summary_rows.append(summary)
        group_rows.extend(
            build_group_rows(
                rows,
                label=label,
                ppc_threshold_m=args.ppc_threshold_m,
                base_dd_alpha=args.base_dd_alpha,
            )
        )
        span_rows.extend(
            build_failure_spans(
                rows,
                label=label,
                failure_threshold_m=args.failure_threshold_m,
                base_dd_alpha=args.base_dd_alpha,
            )
        )
        print(
            f"  {label:<28} ppc={summary['ppc_score_pct']:.2f}% "
            f"pass={summary['ppc_pass_epochs']}/{summary['n_epochs']} "
            f"p50_3d={summary['median_error_3d_m']:.2f}m "
            f"p95_3d={summary['p95_error_3d_m']:.2f}m "
            f"far={summary['far_3d_fail_epochs']}"
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = RESULTS_DIR / f"{args.results_prefix}_summary.csv"
    groups_path = RESULTS_DIR / f"{args.results_prefix}_groups.csv"
    spans_path = RESULTS_DIR / f"{args.results_prefix}_failure_spans.csv"
    _write_rows(summary_rows, summary_path)
    _write_rows(group_rows, groups_path)
    _write_rows(span_rows, spans_path)
    print(f"  Saved: {summary_path}")
    print(f"  Saved: {groups_path}")
    print(f"  Saved: {spans_path}")


if __name__ == "__main__":
    main()
