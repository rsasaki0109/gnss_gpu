"""Epoch diagnostics formatting and CSV output for PF smoother experiments."""

from __future__ import annotations

import csv
import math
from pathlib import Path

from gnss_gpu.pf_smoother_common import finite_float as _finite_float

def _format_diag_value(value: object, fmt: str) -> str:
    out = _finite_float(value)
    if out is None:
        return "n/a"
    return format(out, fmt)


def _format_diag_metric(value: object, fmt: str, unit: str) -> str:
    out = _finite_float(value)
    if out is None:
        return "n/a"
    return f"{format(out, fmt)}{unit}"


def _diagnostics_output_path(base_path: Path, run_name: str, label: str, multiple_outputs: bool) -> Path:
    if not multiple_outputs:
        return base_path
    safe_run = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in run_name)
    safe_label = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in label)
    suffix = base_path.suffix or ".csv"
    return base_path.with_name(f"{base_path.stem}_{safe_run}_{safe_label}{suffix}")


def _epoch_dd_carrier_diagnostics(
    dd_carrier_result,
    dd_gate_stats,
    anchor_attempt,
    fallback_attempt,
    *,
    dd_cp_input_pairs: int,
    dd_cp_gate_scale: float | None,
    dd_cp_raw_abs_afv_median_cycles: float | None,
    dd_cp_raw_abs_afv_max_cycles: float | None,
    dd_cp_sigma_support_scale: float,
    dd_cp_sigma_afv_scale: float,
    dd_cp_sigma_ess_scale: float,
    dd_cp_sigma_scale: float,
    dd_cp_sigma_cycles: float | None,
    dd_cp_support_skip: bool,
    carrier_anchor_sigma_m: float,
) -> dict[str, object]:
    anchor_stats = getattr(anchor_attempt, "stats", None)
    fallback_stats = getattr(fallback_attempt, "tracked_stats", None)
    fallback_afv = getattr(fallback_attempt, "afv", None)

    return {
        "used_dd_carrier": bool(
            dd_carrier_result is not None and getattr(dd_carrier_result, "n_dd", 0) >= 3
        ),
        "dd_cp_input_pairs": int(dd_cp_input_pairs),
        "dd_cp_kept_pairs": (
            int(dd_gate_stats.n_kept_pairs)
            if dd_gate_stats is not None
            else int(dd_cp_input_pairs)
        ),
        "dd_cp_pair_rejected": (
            int(dd_gate_stats.n_pair_rejected) if dd_gate_stats is not None else 0
        ),
        "dd_cp_epoch_rejected": (
            bool(dd_gate_stats.rejected_by_epoch) if dd_gate_stats is not None else False
        ),
        "dd_cp_gate_scale": _finite_float(dd_cp_gate_scale),
        "dd_cp_gate_pair_threshold_cycles": (
            _finite_float(dd_gate_stats.pair_threshold) if dd_gate_stats is not None else None
        ),
        "dd_cp_raw_abs_afv_median_cycles": _finite_float(dd_cp_raw_abs_afv_median_cycles),
        "dd_cp_raw_abs_afv_max_cycles": _finite_float(dd_cp_raw_abs_afv_max_cycles),
        "dd_cp_kept_abs_afv_median_cycles": (
            _finite_float(dd_gate_stats.metric_median) if dd_gate_stats is not None else None
        ),
        "dd_cp_kept_abs_afv_max_cycles": (
            _finite_float(dd_gate_stats.metric_max) if dd_gate_stats is not None else None
        ),
        "dd_cp_sigma_support_scale": _finite_float(dd_cp_sigma_support_scale),
        "dd_cp_sigma_afv_scale": _finite_float(dd_cp_sigma_afv_scale),
        "dd_cp_sigma_ess_scale": _finite_float(dd_cp_sigma_ess_scale),
        "dd_cp_sigma_scale": _finite_float(dd_cp_sigma_scale),
        "dd_cp_sigma_cycles": _finite_float(dd_cp_sigma_cycles),
        "dd_cp_support_skip": bool(dd_cp_support_skip),
        "carrier_anchor_propagated_rows": int(getattr(anchor_attempt, "propagated_rows", 0)),
        "carrier_anchor_n_sat": (
            int(anchor_stats["n_sat"]) if anchor_stats is not None else 0
        ),
        "carrier_anchor_sigma_m": (
            _finite_float(carrier_anchor_sigma_m)
            if getattr(anchor_attempt, "update", None) is not None
            else None
        ),
        "carrier_anchor_residual_median_m": (
            _finite_float(anchor_stats["residual_median_m"]) if anchor_stats is not None else None
        ),
        "carrier_anchor_residual_max_m": (
            _finite_float(anchor_stats["residual_max_m"]) if anchor_stats is not None else None
        ),
        "carrier_anchor_continuity_median_m": (
            _finite_float(anchor_stats["continuity_median_m"]) if anchor_stats is not None else None
        ),
        "carrier_anchor_continuity_max_m": (
            _finite_float(anchor_stats["continuity_max_m"]) if anchor_stats is not None else None
        ),
        "carrier_anchor_max_age_s": (
            _finite_float(anchor_stats["max_age_s"]) if anchor_stats is not None else None
        ),
        "used_carrier_anchor": bool(getattr(anchor_attempt, "used", False)),
        "used_dd_carrier_fallback": bool(getattr(fallback_attempt, "used", False)),
        "used_dd_carrier_fallback_weak_dd": bool(
            getattr(fallback_attempt, "used", False)
            and getattr(fallback_attempt, "replaced_weak_dd", False)
        ),
        "attempted_dd_carrier_fallback_tracked": bool(
            getattr(fallback_attempt, "attempted_tracked", False)
        ),
        "used_dd_carrier_fallback_tracked": bool(
            getattr(fallback_attempt, "used_tracked", False)
        ),
        "dd_carrier_fallback_n_sat": (
            int(fallback_afv["n_sat"]) if fallback_afv is not None else 0
        ),
        "dd_carrier_fallback_tracked_candidate_n_sat": (
            int(fallback_stats.get("n_tracked_consistent_sat", fallback_stats.get("n_sat", 0)))
            if fallback_stats is not None
            else 0
        ),
        "dd_carrier_fallback_tracked_continuity_median_m": (
            _finite_float(fallback_stats["continuity_median_m"])
            if fallback_stats is not None
            else None
        ),
        "dd_carrier_fallback_tracked_continuity_max_m": (
            _finite_float(fallback_stats["continuity_max_m"])
            if fallback_stats is not None
            else None
        ),
        "dd_carrier_fallback_tracked_stable_epochs_median": (
            _finite_float(fallback_stats["stable_epochs_median"])
            if fallback_stats is not None
            else None
        ),
        "dd_carrier_fallback_sigma_scale": _finite_float(
            getattr(fallback_attempt, "sigma_scale", None)
        ),
        "dd_carrier_fallback_sigma_cycles": _finite_float(
            getattr(fallback_attempt, "sigma_cycles", None)
        ),
    }


def _epoch_widelane_diagnostics(
    wl_stats,
    wl_gate_info: dict[str, object],
    *,
    used_widelane_epoch: bool,
    wl_input_pairs: int,
    wl_fixed_pairs: int,
    wl_fix_rate: float | None,
) -> dict[str, object]:
    return {
        "used_widelane": bool(used_widelane_epoch),
        "widelane_input_pairs": int(wl_input_pairs),
        "widelane_fixed_pairs": int(wl_fixed_pairs),
        "widelane_fix_rate": _finite_float(wl_fix_rate),
        "widelane_reason": getattr(wl_stats, "reason", None) if wl_stats is not None else None,
        "widelane_ratio_min": (
            _finite_float(getattr(wl_stats, "ratio_min")) if wl_stats is not None else None
        ),
        "widelane_ratio_median": (
            _finite_float(getattr(wl_stats, "ratio_median")) if wl_stats is not None else None
        ),
        "widelane_residual_abs_median_cycles": (
            _finite_float(getattr(wl_stats, "residual_abs_median_cycles"))
            if wl_stats is not None
            else None
        ),
        "widelane_residual_abs_max_cycles": (
            _finite_float(getattr(wl_stats, "residual_abs_max_cycles"))
            if wl_stats is not None
            else None
        ),
        "widelane_std_median_cycles": (
            _finite_float(getattr(wl_stats, "std_median_cycles")) if wl_stats is not None else None
        ),
        "widelane_gate_reason": wl_gate_info.get("reason"),
        "widelane_gate_pair_rejected": int(wl_gate_info.get("pair_rejected") or 0),
        "widelane_raw_abs_res_median_m": _finite_float(
            wl_gate_info.get("raw_abs_res_median_m")
        ),
        "widelane_raw_abs_res_max_m": _finite_float(
            wl_gate_info.get("raw_abs_res_max_m")
        ),
        "widelane_kept_abs_res_median_m": _finite_float(
            wl_gate_info.get("kept_abs_res_median_m")
        ),
        "widelane_kept_abs_res_max_m": _finite_float(
            wl_gate_info.get("kept_abs_res_max_m")
        ),
    }


def _epoch_dd_pseudorange_gate_diagnostics(
    dd_pr_gate_stats,
    *,
    dd_pr_input_pairs: int,
    dd_pr_gate_scale: float | None,
    dd_pr_raw_abs_res_median_m: float | None,
    dd_pr_raw_abs_res_max_m: float | None,
) -> dict[str, object]:
    return {
        "dd_pr_input_pairs": int(dd_pr_input_pairs),
        "dd_pr_kept_pairs": (
            int(dd_pr_gate_stats.n_kept_pairs)
            if dd_pr_gate_stats is not None
            else int(dd_pr_input_pairs)
        ),
        "dd_pr_pair_rejected": (
            int(dd_pr_gate_stats.n_pair_rejected) if dd_pr_gate_stats is not None else 0
        ),
        "dd_pr_epoch_rejected": (
            bool(dd_pr_gate_stats.rejected_by_epoch)
            if dd_pr_gate_stats is not None
            else False
        ),
        "dd_pr_gate_scale": _finite_float(dd_pr_gate_scale),
        "dd_pr_gate_pair_threshold_m": (
            _finite_float(dd_pr_gate_stats.pair_threshold)
            if dd_pr_gate_stats is not None
            else None
        ),
        "dd_pr_raw_abs_res_median_m": _finite_float(dd_pr_raw_abs_res_median_m),
        "dd_pr_raw_abs_res_max_m": _finite_float(dd_pr_raw_abs_res_max_m),
        "dd_pr_kept_abs_res_median_m": (
            _finite_float(dd_pr_gate_stats.metric_median)
            if dd_pr_gate_stats is not None
            else None
        ),
        "dd_pr_kept_abs_res_max_m": (
            _finite_float(dd_pr_gate_stats.metric_max) if dd_pr_gate_stats is not None else None
        ),
    }


def _build_epoch_diagnostic_row(
    *,
    run_name: str,
    tow: float,
    aligned_epoch_index: int,
    store_epoch_index: int | None,
    gt_index: int,
    n_measurements: int,
    used_imu: bool,
    used_tdcp: bool,
    used_tdcp_pu_epoch: bool,
    tdcp_pu_rms: float,
    tdcp_pu_spp_diff_mps: float | None,
    tdcp_pu_gate_reason: str | None,
    imu_stop_detected: bool,
    used_imu_tight_epoch: bool,
    rbpf_velocity_kf: bool,
    doppler_update_epoch,
    doppler_kf_gate_reason: str | None,
    dd_pr_result,
    dd_pr_sigma_epoch: float,
    dd_pr_gate_stats,
    dd_pr_input_pairs: int,
    dd_pr_gate_scale: float | None,
    dd_pr_raw_abs_res_median_m: float | None,
    dd_pr_raw_abs_res_max_m: float | None,
    gate_ess_ratio: float | None,
    gate_spread_m: float | None,
    wl_stats,
    wl_gate_info: dict[str, object],
    used_widelane_epoch: bool,
    wl_input_pairs: int,
    wl_fixed_pairs: int,
    wl_fix_rate: float | None,
    dd_carrier_result,
    dd_gate_stats,
    anchor_attempt,
    fallback_attempt,
    dd_cp_input_pairs: int,
    dd_cp_gate_scale: float | None,
    dd_cp_raw_abs_afv_median_cycles: float | None,
    dd_cp_raw_abs_afv_max_cycles: float | None,
    dd_cp_sigma_support_scale: float,
    dd_cp_sigma_afv_scale: float,
    dd_cp_sigma_ess_scale: float,
    dd_cp_sigma_scale: float,
    dd_cp_sigma_cycles: float | None,
    dd_cp_support_skip: bool,
    carrier_anchor_sigma_m: float,
) -> dict[str, object]:
    dd_carrier_diag = _epoch_dd_carrier_diagnostics(
        dd_carrier_result,
        dd_gate_stats,
        anchor_attempt,
        fallback_attempt,
        dd_cp_input_pairs=dd_cp_input_pairs,
        dd_cp_gate_scale=dd_cp_gate_scale,
        dd_cp_raw_abs_afv_median_cycles=dd_cp_raw_abs_afv_median_cycles,
        dd_cp_raw_abs_afv_max_cycles=dd_cp_raw_abs_afv_max_cycles,
        dd_cp_sigma_support_scale=dd_cp_sigma_support_scale,
        dd_cp_sigma_afv_scale=dd_cp_sigma_afv_scale,
        dd_cp_sigma_ess_scale=dd_cp_sigma_ess_scale,
        dd_cp_sigma_scale=dd_cp_sigma_scale,
        dd_cp_sigma_cycles=dd_cp_sigma_cycles,
        dd_cp_support_skip=dd_cp_support_skip,
        carrier_anchor_sigma_m=carrier_anchor_sigma_m,
    )
    widelane_diag = _epoch_widelane_diagnostics(
        wl_stats,
        wl_gate_info,
        used_widelane_epoch=used_widelane_epoch,
        wl_input_pairs=wl_input_pairs,
        wl_fixed_pairs=wl_fixed_pairs,
        wl_fix_rate=wl_fix_rate,
    )
    dd_pr_gate_diag = _epoch_dd_pseudorange_gate_diagnostics(
        dd_pr_gate_stats,
        dd_pr_input_pairs=dd_pr_input_pairs,
        dd_pr_gate_scale=dd_pr_gate_scale,
        dd_pr_raw_abs_res_median_m=dd_pr_raw_abs_res_median_m,
        dd_pr_raw_abs_res_max_m=dd_pr_raw_abs_res_max_m,
    )
    return {
        "run": run_name,
        "tow": float(tow),
        "aligned_epoch_index": int(aligned_epoch_index),
        "store_epoch_index": store_epoch_index,
        "gt_index": int(gt_index),
        "n_measurements": int(n_measurements),
        "used_imu": bool(used_imu),
        "used_tdcp": bool(used_tdcp),
        "used_tdcp_pu": bool(used_tdcp_pu_epoch),
        "tdcp_pu_rms": _finite_float(tdcp_pu_rms),
        "tdcp_pu_spp_diff_mps": _finite_float(tdcp_pu_spp_diff_mps),
        "tdcp_pu_gate_reason": tdcp_pu_gate_reason,
        "imu_stop_detected": bool(imu_stop_detected),
        "used_imu_tight": bool(used_imu_tight_epoch),
        "used_doppler_kf": bool(rbpf_velocity_kf and doppler_update_epoch is not None),
        "doppler_kf_gate_reason": doppler_kf_gate_reason,
        "used_dd_pseudorange": bool(
            dd_pr_result is not None and getattr(dd_pr_result, "n_dd", 0) >= 3
        ),
        **widelane_diag,
        "dd_pr_sigma_m": _finite_float(dd_pr_sigma_epoch if dd_pr_result is not None else None),
        "used_dd_carrier": bool(
            dd_carrier_result is not None and getattr(dd_carrier_result, "n_dd", 0) >= 3
        ),
        "gate_ess_ratio": _finite_float(gate_ess_ratio),
        "gate_spread_m": _finite_float(gate_spread_m),
        **dd_pr_gate_diag,
        **dd_carrier_diag,
        "forward_error_2d": None,
        "forward_error_3d": None,
        "smoothed_error_2d": None,
        "smoothed_error_3d": None,
        "smoothed_shift_3d_m": None,
        "smoothing_improvement_2d": None,
        "tail_guard_applied": False,
        "widelane_forward_guard_applied": False,
        "local_fgo_applied": False,
    }


def _write_epoch_diagnostics(rows: list[dict[str, object]], output_path: Path) -> None:
    if not rows:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _print_top_epoch_diagnostics(rows: list[dict[str, object]], top_k: int) -> None:
    if top_k <= 0 or not rows:
        return
    sort_key = "smoothed_error_2d" if any(_finite_float(r.get("smoothed_error_2d")) is not None for r in rows) else "forward_error_2d"
    ranked = sorted(
        rows,
        key=lambda row: _finite_float(row.get(sort_key)) if _finite_float(row.get(sort_key)) is not None else -1.0,
        reverse=True,
    )
    n_show = min(int(top_k), len(ranked))
    print(f"  [epoch_diag] worst {n_show} epochs by {sort_key}:")
    for row in ranked[:n_show]:
        print(
            "    "
            f"tow={_format_diag_value(row.get('tow'), '.1f')} "
            f"{sort_key}={_format_diag_metric(row.get(sort_key), '.2f', 'm')} "
            f"fwd={_format_diag_metric(row.get('forward_error_2d'), '.2f', 'm')} "
            f"smth={_format_diag_metric(row.get('smoothed_error_2d'), '.2f', 'm')} "
            f"shift={_format_diag_metric(row.get('smoothed_shift_3d_m'), '.2f', 'm')} "
            f"ess={_format_diag_value(row.get('gate_ess_ratio'), '.3f')} "
            f"spread={_format_diag_metric(row.get('gate_spread_m'), '.2f', 'm')} "
            f"dd_pr={int(row.get('dd_pr_kept_pairs') or 0)}/{int(row.get('dd_pr_input_pairs') or 0)} "
            f"dd_pr_med={_format_diag_metric(row.get('dd_pr_raw_abs_res_median_m'), '.2f', 'm')} "
            f"dd_cp={int(row.get('dd_cp_kept_pairs') or 0)}/{int(row.get('dd_cp_input_pairs') or 0)} "
            f"dd_cp_med={_format_diag_metric(row.get('dd_cp_raw_abs_afv_median_cycles'), '.3f', 'cy')}"
        )
    _print_stop_segment_diagnostics(rows, top_k)


def build_stop_segment_diagnostic_lines(
    rows: list[dict[str, object]],
    top_k: int,
    *,
    min_epochs: int = 5,
) -> list[str]:
    """Return summary lines for the worst IMU-stop segments in diagnostics."""

    if top_k <= 0 or not rows:
        return []
    summaries = _stop_segment_summaries(rows, min_epochs=min_epochs)
    if not summaries:
        return []
    ranked = sorted(
        summaries,
        key=lambda item: _sort_float(item.get("smoothed_p50_m")),
        reverse=True,
    )
    n_show = min(int(top_k), len(ranked))
    lines = [f"  [stop_diag] worst {n_show} stop segments by smoothed p50:"]
    for summary in ranked[:n_show]:
        lines.append(
            "    "
            f"seg={summary['segment_index']} "
            f"n={summary['n_epochs']} "
            f"tow={_format_diag_value(summary.get('tow_start'), '.1f')}"
            f"-{_format_diag_value(summary.get('tow_end'), '.1f')} "
            f"smth_p50={_format_diag_metric(summary.get('smoothed_p50_m'), '.2f', 'm')} "
            f"smth_p90={_format_diag_metric(summary.get('smoothed_p90_m'), '.2f', 'm')} "
            f"fwd_p50={_format_diag_metric(summary.get('forward_p50_m'), '.2f', 'm')} "
            f"radius={_format_diag_metric(summary.get('radius_p50_m'), '.2f', 'm')} "
            f"dd_cp_med={_format_diag_value(summary.get('dd_cp_kept_p50'), '.1f')} "
            f"fallback={summary['fallback_epochs']}"
        )
    return lines


def _print_stop_segment_diagnostics(rows: list[dict[str, object]], top_k: int) -> None:
    for line in build_stop_segment_diagnostic_lines(rows, top_k):
        print(line)


def _stop_segment_summaries(
    rows: list[dict[str, object]],
    *,
    min_epochs: int,
) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    start: int | None = None
    segment_index = 0
    for idx, row in enumerate(rows):
        if _bool_value(row.get("imu_stop_detected")):
            if start is None:
                start = idx
            continue
        if start is not None:
            segment_index = _append_stop_segment_summary(
                summaries,
                rows[start:idx],
                segment_index=segment_index,
                min_epochs=min_epochs,
            )
        start = None
    if start is not None:
        _append_stop_segment_summary(
            summaries,
            rows[start:],
            segment_index=segment_index,
            min_epochs=min_epochs,
        )
    return summaries


def _append_stop_segment_summary(
    summaries: list[dict[str, object]],
    segment_rows: list[dict[str, object]],
    *,
    segment_index: int,
    min_epochs: int,
) -> int:
    if len(segment_rows) < int(min_epochs):
        return segment_index
    fallback_epochs = sum(
        1 for row in segment_rows if _bool_value(row.get("used_dd_carrier_fallback"))
    )
    summaries.append(
        {
            "segment_index": int(segment_index),
            "n_epochs": int(len(segment_rows)),
            "tow_start": segment_rows[0].get("tow"),
            "tow_end": segment_rows[-1].get("tow"),
            "smoothed_p50_m": _percentile(
                row.get("smoothed_error_2d") for row in segment_rows
            ),
            "smoothed_p90_m": _percentile(
                (row.get("smoothed_error_2d") for row in segment_rows),
                90.0,
            ),
            "forward_p50_m": _percentile(
                row.get("forward_error_2d") for row in segment_rows
            ),
            "radius_p50_m": _percentile(
                row.get("stop_segment_radius_m") for row in segment_rows
            ),
            "dd_cp_kept_p50": _percentile(
                row.get("dd_cp_kept_pairs") for row in segment_rows
            ),
            "fallback_epochs": int(fallback_epochs),
        }
    )
    return segment_index + 1


def _percentile(values, percent: float = 50.0) -> float | None:
    finite = [
        float(value)
        for value in (_finite_float(value) for value in values)
        if value is not None
    ]
    if not finite:
        return None
    finite.sort()
    idx = (len(finite) - 1) * float(percent) / 100.0
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return finite[lo]
    return finite[lo] * (hi - idx) + finite[hi] * (idx - lo)


def _sort_float(value: object) -> float:
    out = _finite_float(value)
    return float(out) if out is not None else -1.0


def _bool_value(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)
