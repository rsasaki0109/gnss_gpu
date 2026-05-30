#!/usr/bin/env python3
"""Post-hoc PPC internal-state diagnostics from per-epoch CSV files."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


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


def _float(row: dict[str, object], key: str) -> float:
    value = row.get(key, "")
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _bool(row: dict[str, object], key: str) -> bool:
    text = str(row.get(key, "")).strip().lower()
    return text in {"1", "true", "t", "yes", "y"}


def _arr(rows: list[dict[str, object]], key: str) -> np.ndarray:
    return np.asarray([_float(row, key) for row in rows], dtype=np.float64)


def _preferred_key(rows: list[dict[str, object]], primary: str, fallback: str) -> str:
    if rows and primary not in rows[0] and fallback in rows[0]:
        return fallback
    return primary


def _safe_median(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if finite.size else float("nan")


def _safe_mean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if finite.size else float("nan")


def _safe_p95(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.percentile(finite, 95)) if finite.size else float("nan")


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    finite = np.isfinite(a) & np.isfinite(b)
    if int(np.count_nonzero(finite)) < 3:
        return float("nan")
    aa = a[finite]
    bb = b[finite]
    if float(np.std(aa)) <= 0.0 or float(np.std(bb)) <= 0.0:
        return float("nan")
    return float(np.corrcoef(aa, bb)[0, 1])


def _percent(count: int, total: int) -> float:
    return float(100.0 * int(count) / max(int(total), 1))


def _distance_weighted_pass(
    rows: list[dict[str, object]],
    pass_mask: np.ndarray,
    denominator_mask: np.ndarray | None = None,
) -> tuple[float, float, float]:
    distances = _arr(rows, "ppc_segment_distance_m")
    if not np.any(np.isfinite(distances)):
        distances = np.ones(len(rows), dtype=np.float64)
    else:
        distances = np.where(np.isfinite(distances), distances, 0.0)
    if denominator_mask is None:
        denominator_mask = np.ones(len(rows), dtype=bool)
    total = float(np.sum(distances[denominator_mask]))
    passed = float(np.sum(distances[pass_mask]))
    return passed, total, float(100.0 * passed / total) if total > 0.0 else 0.0


def _group_row(
    rows: list[dict[str, object]],
    mask: np.ndarray,
    *,
    source: str,
    group: str,
    value: str,
    threshold_m: float,
) -> dict[str, object]:
    errors_2d = _arr(rows, "fused_error_2d_m")
    errors_3d = _arr(rows, _preferred_key(rows, "fused_error_3d_m", "emit_to_ref_m"))
    wls_3d = _arr(rows, "wls_error_3d_m")
    pass_mask = mask & np.isfinite(errors_3d) & (errors_3d <= float(threshold_m))
    far_mask = mask & np.isfinite(errors_3d) & (errors_3d > 3.0)
    horiz_pass = mask & np.isfinite(errors_2d) & (errors_2d <= float(threshold_m))
    vertical_limited = horiz_pass & ~pass_mask
    pass_m, total_m, score_pct = _distance_weighted_pass(rows, pass_mask, mask)
    n = int(np.count_nonzero(mask))
    return {
        "source": source,
        "group": group,
        "value": value,
        "n_epochs": n,
        "pass_epochs": int(np.count_nonzero(pass_mask)),
        "pass_epoch_pct": _percent(int(np.count_nonzero(pass_mask)), n),
        "pass_m": pass_m,
        "total_m": total_m,
        "score_pct": score_pct,
        "far_fail_epochs": int(np.count_nonzero(far_mask)),
        "vertical_limited_epochs": int(np.count_nonzero(vertical_limited)),
        "median_error_3d_m": _safe_median(errors_3d[mask]),
        "p95_error_3d_m": _safe_p95(errors_3d[mask]),
        "median_dd_shift_m": _safe_median(_arr(rows, "dd_pr_shift_m")[mask]),
        "median_dd_rms_m": _safe_median(_arr(rows, "dd_pr_robust_rms_m")[mask]),
        "median_tdcp_postfit_m": _safe_median(_arr(rows, "tdcp_postfit_rms_m")[mask]),
        "median_wls_error_3d_m": _safe_median(wls_3d[mask]),
        "fused_better_than_wls_pct": _percent(
            int(np.count_nonzero(mask & np.isfinite(wls_3d) & np.isfinite(errors_3d) & (errors_3d < wls_3d))),
            n,
        ),
    }


def _numeric_bins(
    rows: list[dict[str, object]],
    *,
    key: str,
    source: str,
    threshold_m: float,
) -> list[dict[str, object]]:
    values = _arr(rows, key)
    finite_values = values[np.isfinite(values)]
    if finite_values.size < 10:
        return []
    quantiles = np.unique(np.percentile(finite_values, [0.0, 25.0, 50.0, 75.0, 100.0]))
    if quantiles.size < 2:
        return []
    out: list[dict[str, object]] = []
    for lo, hi in zip(quantiles[:-1], quantiles[1:]):
        if hi <= lo:
            continue
        if hi == quantiles[-1]:
            mask = np.isfinite(values) & (values >= lo) & (values <= hi)
        else:
            mask = np.isfinite(values) & (values >= lo) & (values < hi)
        label = f"{lo:.3g}..{hi:.3g}"
        out.append(
            _group_row(
                rows,
                mask,
                source=source,
                group=f"{key}_quartile",
                value=label,
                threshold_m=threshold_m,
            )
        )
    return out


def _failure_spans(
    rows: list[dict[str, object]],
    *,
    source: str,
    failure_threshold_m: float,
) -> list[dict[str, object]]:
    errors = _arr(rows, _preferred_key(rows, "fused_error_3d_m", "emit_to_ref_m"))
    fail_idx = np.flatnonzero(np.isfinite(errors) & (errors > float(failure_threshold_m)))
    if fail_idx.size == 0:
        return []
    spans: list[dict[str, object]] = []
    start = 0
    span_id = 0
    for pos in range(1, len(fail_idx) + 1):
        if pos < len(fail_idx) and fail_idx[pos] == fail_idx[pos - 1] + 1:
            continue
        idx = fail_idx[start:pos]
        selected = [rows[int(i)] for i in idx]
        dd_rms = _arr(selected, "dd_pr_robust_rms_m")
        dd_shift = _arr(selected, "dd_pr_shift_m")
        tdcp = _arr(selected, "tdcp_postfit_rms_m")
        distances = _arr(selected, "ppc_segment_distance_m")
        distances = np.where(np.isfinite(distances), distances, 0.0)
        span_id += 1
        spans.append(
            {
                "source": source,
                "span_id": span_id,
                "start_epoch": int(_float(selected[0], "epoch")),
                "end_epoch": int(_float(selected[-1], "epoch")),
                "start_tow": _float(selected[0], "tow"),
                "end_tow": _float(selected[-1], "tow"),
                "n_epochs": int(len(selected)),
                "distance_m": float(np.sum(distances)),
                "median_error_3d_m": _safe_median(errors[idx]),
                "p95_error_3d_m": _safe_p95(errors[idx]),
                "median_dd_shift_m": _safe_median(dd_shift),
                "median_dd_rms_m": _safe_median(dd_rms),
                "median_tdcp_postfit_m": _safe_median(tdcp),
                "dd_anchor_used_pct": _percent(
                    sum(1 for row in selected if _bool(row, "dd_pr_anchor_used")),
                    len(selected),
                ),
                "widelane_used_pct": _percent(
                    sum(1 for row in selected if _bool(row, "widelane_anchor_used")),
                    len(selected),
                ),
                "tdcp_used_pct": _percent(
                    sum(1 for row in selected if _bool(row, "tdcp_used")),
                    len(selected),
                ),
            }
        )
        start = pos
    spans.sort(key=lambda row: (float(row["distance_m"]), int(row["n_epochs"])), reverse=True)
    return spans


def _source_label(path: Path) -> str:
    name = path.name
    for suffix in ("_epochs.csv", ".csv"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def analyze_file(
    path: Path,
    *,
    threshold_m: float,
    failure_threshold_m: float,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    rows = _read_rows(path)
    source = _source_label(path)
    errors_2d = _arr(rows, "fused_error_2d_m")
    errors_3d = _arr(rows, _preferred_key(rows, "fused_error_3d_m", "emit_to_ref_m"))
    wls_3d = _arr(rows, "wls_error_3d_m")
    dd_shift = _arr(rows, "dd_pr_shift_m")
    dd_rms = _arr(rows, "dd_pr_robust_rms_m")
    tdcp = _arr(rows, "tdcp_postfit_rms_m")
    rsp_ess = _arr(rows, "rsp_ess_before")
    valid = np.isfinite(errors_3d)
    pass_mask = valid & (errors_3d <= float(threshold_m))
    far_mask = valid & (errors_3d > float(failure_threshold_m))
    false_conf_dd = far_mask & np.isfinite(dd_rms) & (dd_rms <= 1.0)
    false_conf_tdcp = far_mask & np.isfinite(tdcp) & (tdcp <= 0.2)
    horizontal_pass = np.isfinite(errors_2d) & (errors_2d <= float(threshold_m))
    vertical_limited = horizontal_pass & ~pass_mask
    pass_m, total_m, score_pct = _distance_weighted_pass(rows, pass_mask)

    summary = {
        "source": source,
        "n_epochs": int(len(rows)),
        "valid_epochs": int(np.count_nonzero(valid)),
        "pass_epochs": int(np.count_nonzero(pass_mask)),
        "pass_epoch_pct": _percent(int(np.count_nonzero(pass_mask)), int(np.count_nonzero(valid))),
        "score_pct": score_pct,
        "pass_m": pass_m,
        "total_m": total_m,
        "far_fail_epochs": int(np.count_nonzero(far_mask)),
        "vertical_limited_epochs": int(np.count_nonzero(vertical_limited)),
        "median_error_3d_m": _safe_median(errors_3d),
        "p95_error_3d_m": _safe_p95(errors_3d),
        "median_wls_error_3d_m": _safe_median(wls_3d),
        "fused_better_than_wls_pct": _percent(
            int(np.count_nonzero(valid & np.isfinite(wls_3d) & (errors_3d < wls_3d))),
            int(np.count_nonzero(valid)),
        ),
        "dd_shift_error_corr": _safe_corr(dd_shift, errors_3d),
        "dd_rms_error_corr": _safe_corr(dd_rms, errors_3d),
        "tdcp_postfit_error_corr": _safe_corr(tdcp, errors_3d),
        "rsp_ess_error_corr": _safe_corr(rsp_ess, errors_3d),
        "false_confident_dd_fail_epochs": int(np.count_nonzero(false_conf_dd)),
        "false_confident_tdcp_fail_epochs": int(np.count_nonzero(false_conf_tdcp)),
    }

    n = len(rows)
    group_rows: list[dict[str, object]] = []
    bool_keys = [
        "tdcp_used",
        "tdcp_last_velocity_used",
        "dd_pr_anchor_used",
        "widelane_anchor_used",
        "rsp_correction_used",
        "height_hold_reference_trusted",
        "dd_anchor_high_regime_untrusted",
        "dd_anchor_high_regime_last_velocity",
        "dd_anchor_high_regime_widelane_gap",
        "dd_anchor_high_regime_persistence",
        "dd_anchor_high_regime_big_shift",
        "dd_anchor_high_regime_tdcp_sparse",
        "doppler_update_applied",
        "dd_carrier_update_applied",
        "hybrid_available",
        "hybrid_pu_applied",
        "rtkdiag_candidate_available",
        "imu_tc_emit_pf_here",
        "ins_tc_emit_pf_here",
        "resampled_before_emit",
        "resampled_epoch_end",
        "pr_ess_guard_enabled",
    ]
    for key in bool_keys:
        if key not in rows[0]:
            continue
        mask = np.asarray([_bool(row, key) for row in rows], dtype=bool)
        if int(np.count_nonzero(mask)) == 0 and int(np.count_nonzero(~mask)) == n:
            continue
        group_rows.append(
            _group_row(rows, mask, source=source, group=key, value="true", threshold_m=threshold_m)
        )
        group_rows.append(
            _group_row(rows, ~mask, source=source, group=key, value="false", threshold_m=threshold_m)
        )

    for key in [
        "dd_pr_shift_m",
        "dd_pr_robust_rms_m",
        "tdcp_postfit_rms_m",
        "rsp_ess_before",
        "height_hold_correction_m",
        "widelane_fix_rate",
        "widelane_anchor_robust_rms_m",
        "pf_after_pr_ess_ratio",
        "pf_after_pr_to_ref_m",
        "pf_after_pr_spread_m",
        "pf_before_emit_ess_ratio",
        "pf_before_emit_to_ref_m",
        "pf_before_emit_spread_m",
        "pf_epoch_start_to_ref_m",
        "pf_after_predict_to_ref_m",
        "pf_after_doppler_to_ref_m",
        "pf_after_dd_carrier_to_ref_m",
        "pf_after_position_update_to_ref_m",
        "pf_after_hybrid_to_ref_m",
        "pf_after_rtkdiag_to_ref_m",
        "pf_epoch_end_to_ref_m",
        "pf_before_doppler_vel_speed_mps",
        "pf_after_doppler_vel_speed_mps",
        "ref_velocity_speed_mps",
        "doppler_current_pfvel_rms_mps",
        "doppler_flipped_pfvel_rms_mps",
        "doppler_current_refvel_rms_mps",
        "doppler_flipped_refvel_rms_mps",
        "doppler_current_wls_rms_mps",
        "doppler_flipped_wls_rms_mps",
        "doppler_current_wls_speed_mps",
        "doppler_flipped_wls_speed_mps",
        "doppler_current_wls_to_refvel_mps",
        "doppler_flipped_wls_to_refvel_mps",
        "doppler_raw_finite_count",
        "doppler_system_filtered_count",
        "doppler_prefit_gate_mps",
        "doppler_prefit_kept_count",
        "doppler_prefit_dropped_count",
        "dd_pr_ls_anchor_sigma_m",
        "dd_pr_ls_anchor_to_pf_after_pr_m",
        "dd_pr_ls_anchor_to_ref_m",
        "pf_before_hybrid_to_hybrid_m",
        "rtkdiag_candidate_to_hybrid_m",
        "rtkdiag_candidate_to_ref_m",
        "emit_to_hybrid_m",
        "emit_to_ref_m",
        "pr_ess_guard_alpha",
        "pr_ess_guard_pre_ratio",
        "pr_ess_guard_post_ratio",
        "pr_ess_guard_final_ratio",
        "pr_elevation_gate_deg",
        "pr_elevation_kept_count",
        "pr_elevation_dropped_count",
        "pr_atmosphere_scale",
        "pr_atmosphere_extra_zenith_m",
        "pr_slant_delay_zenith_m",
        "pr_slant_delay_median_m",
        "pr_atmosphere_delay_median_m",
        "position_update_wls_postfit_rms_m",
        "position_update_wls_postfit_absmax_m",
        "position_update_wls_pdop",
        "position_update_wls_to_pf_before_m",
        "pr_auto_robust_selected",
        "pr_auto_robust_low_sat_epochs",
    ]:
        if key in rows[0]:
            group_rows.extend(_numeric_bins(rows, key=key, source=source, threshold_m=threshold_m))

    spans = _failure_spans(rows, source=source, failure_threshold_m=failure_threshold_m)
    return summary, group_rows, spans


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze PPC per-epoch internal-state CSVs")
    parser.add_argument("epochs_csv", nargs="+", type=Path)
    parser.add_argument("--threshold-m", type=float, default=0.5)
    parser.add_argument("--failure-threshold-m", type=float, default=3.0)
    parser.add_argument("--out-prefix", type=str, default="ppc_internal_state")
    args = parser.parse_args()

    summary_rows: list[dict[str, object]] = []
    group_rows: list[dict[str, object]] = []
    span_rows: list[dict[str, object]] = []
    for path in args.epochs_csv:
        summary, groups, spans = analyze_file(
            path,
            threshold_m=float(args.threshold_m),
            failure_threshold_m=float(args.failure_threshold_m),
        )
        summary_rows.append(summary)
        group_rows.extend(groups)
        span_rows.extend(spans)

    summary_path = RESULTS_DIR / f"{args.out_prefix}_summary.csv"
    groups_path = RESULTS_DIR / f"{args.out_prefix}_groups.csv"
    spans_path = RESULTS_DIR / f"{args.out_prefix}_spans.csv"
    _write_rows(summary_rows, summary_path)
    _write_rows(group_rows, groups_path)
    _write_rows(span_rows, spans_path)

    print(f"saved {summary_path}")
    print(f"saved {groups_path}")
    print(f"saved {spans_path}")
    print()
    for row in summary_rows:
        print(
            f"{row['source']}: score={row['score_pct']:.2f}% "
            f"pass={row['pass_epoch_pct']:.1f}% "
            f"median3d={row['median_error_3d_m']:.2f}m "
            f"p95={row['p95_error_3d_m']:.2f}m "
            f"far={row['far_fail_epochs']} "
            f"falseDD={row['false_confident_dd_fail_epochs']}"
        )


if __name__ == "__main__":
    main()
