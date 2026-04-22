"""Run-summary output for PF smoother evaluations."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any


def build_pf_smoother_summary_lines(result: Mapping[str, Any]) -> list[str]:
    lines: list[str] = []
    predict_guide = str(result.get("predict_guide", ""))

    if predict_guide == "tdcp_adaptive":
        used = _int(result, "n_tdcp_used")
        fallback = _int(result, "n_tdcp_fallback")
        total = used + fallback
        threshold = _float(result, "tdcp_rms_threshold")
        lines.append(
            f"  [tdcp_adaptive] TDCP used {used}/{total} epochs, "
            f"fallback {fallback}/{total} (rms_threshold={threshold:.1f}m)"
        )

    if _bool(result, "tdcp_position_update"):
        used = _int(result, "n_tdcp_pu_used")
        skip = _int(result, "n_tdcp_pu_skip")
        total = used + skip
        lines.append(
            f"  [tdcp_position_update] used {used}/{total} epochs, "
            f"skip {skip}/{total}, "
            f"gate_skip={_int(result, 'n_tdcp_pu_gate_skip')}"
        )

    if _bool(result, "doppler_per_particle"):
        used = _int(result, "n_doppler_pp_used")
        skip = _int(result, "n_doppler_pp_skip")
        total = used + skip
        lines.append(
            f"  [doppler_per_particle] used {used}/{total} epochs, "
            f"skip {skip}/{total}"
        )

    if _bool(result, "rbpf_velocity_kf"):
        used = _int(result, "n_doppler_kf_used")
        skip = _int(result, "n_doppler_kf_skip")
        total = used + skip
        lines.append(
            f"  [rbpf_velocity_kf] Doppler KF used {used}/{total} epochs, "
            f"skip {skip}/{total}, "
            f"gate_skip={_int(result, 'n_doppler_kf_gate_skip')}"
        )

    if predict_guide in ("imu", "imu_spp_blend"):
        used = _int(result, "n_imu_used")
        fallback = _int(result, "n_imu_fallback")
        total = used + fallback
        lines.append(
            f"  [{predict_guide}] IMU used {used}/{total} epochs, "
            f"fallback {fallback}/{total}"
        )
        stop_count = _int(result, "n_imu_stop_detected")
        if stop_count > 0:
            stop_sigma = result.get("imu_stop_sigma_pos")
            stop_sigma_str = f"{stop_sigma}" if stop_sigma is not None else "default"
            lines.append(
                f"  [imu_stop_detect] stop epochs={stop_count}, "
                f"sigma_pos={stop_sigma_str}"
            )

    if _bool(result, "imu_tight_coupling"):
        used = _int(result, "n_imu_tight_used")
        skip = _int(result, "n_imu_tight_skip")
        total = used + skip
        lines.append(
            f"  [imu_tight] IMU position_update used {used}/{total} epochs, "
            f"skip {skip}/{total}"
        )

    if _bool(result, "mupf_dd"):
        used = _int(result, "n_dd_used")
        skip = _int(result, "n_dd_skip")
        total = used + skip
        lines.append(
            f"  [mupf_dd] DD-AFV used {used}/{total} epochs, "
            f"skip {skip}/{total}"
        )
        if _any_present(
            result,
            (
                "mupf_dd_gate_afv_cycles",
                "mupf_dd_gate_adaptive_floor_cycles",
                "mupf_dd_gate_adaptive_mad_mult",
                "mupf_dd_gate_epoch_median_cycles",
            ),
        ):
            lines.append(
                f"  [mupf_dd_gate] pair_rejected={_int(result, 'n_dd_gate_pairs_rejected')} "
                f"epoch_skip={_int(result, 'n_dd_gate_epoch_skip')}"
            )
        if _int(result, "n_dd_skip_support_guard") > 0:
            lines.append(
                f"  [mupf_dd_support_skip] epochs={_int(result, 'n_dd_skip_support_guard')}"
            )
        if _int(result, "n_dd_sigma_relaxed") > 0:
            lines.append(
                f"  [mupf_dd_sigma_relax] epochs={_int(result, 'n_dd_sigma_relaxed')} "
                f"mean_scale={_float(result, 'mean_dd_sigma_scale'):.3f}"
            )
        _append_positive_counter(lines, result, "carrier_anchor", "n_carrier_anchor_used")
        _append_positive_counter(
            lines,
            result,
            "carrier_anchor_tdcp",
            "n_carrier_anchor_propagated",
            noun="propagated_rows",
        )
        _append_positive_counter(
            lines, result, "mupf_dd_fallback_undiff", "n_dd_fallback_undiff_used"
        )
        _append_positive_counter(
            lines,
            result,
            "mupf_dd_fallback_tracked_attempt",
            "n_dd_fallback_tracked_attempted",
        )
        _append_positive_counter(
            lines, result, "mupf_dd_fallback_tracked", "n_dd_fallback_tracked_used"
        )
        _append_positive_counter(
            lines, result, "mupf_dd_fallback_weak_dd", "n_dd_fallback_weak_dd_replaced"
        )

    if _bool(result, "dd_pseudorange"):
        used = _int(result, "n_dd_pr_used")
        skip = _int(result, "n_dd_pr_skip")
        total = used + skip
        lines.append(
            f"  [dd_pseudorange] used {used}/{total} epochs, "
            f"skip {skip}/{total}"
        )
        if _any_present(
            result,
            (
                "dd_pseudorange_gate_residual_m",
                "dd_pseudorange_gate_adaptive_floor_m",
                "dd_pseudorange_gate_adaptive_mad_mult",
                "dd_pseudorange_gate_epoch_median_m",
            ),
        ):
            lines.append(
                f"  [dd_pseudorange_gate] pair_rejected={_int(result, 'n_dd_pr_gate_pairs_rejected')} "
                f"epoch_skip={_int(result, 'n_dd_pr_gate_epoch_skip')}"
            )

    if _bool(result, "widelane"):
        used = _int(result, "n_wl_used")
        skip = _int(result, "n_wl_skip")
        total = used + skip
        candidate_pairs = _int(result, "n_wl_candidate_pairs")
        fixed_pairs = _int(result, "n_wl_fixed_pairs")
        pair_rate = float(fixed_pairs) / float(candidate_pairs) if candidate_pairs > 0 else 0.0
        lines.append(
            f"  [widelane] used {used}/{total} epochs, "
            f"fixed_pairs={fixed_pairs}/{candidate_pairs} ({pair_rate:.1%}), "
            f"low_fix_rate_epochs={_int(result, 'n_wl_low_fix_rate')}"
        )
        if _int(result, "n_wl_gate_skip") > 0 or _int(result, "n_wl_gate_pair_rejected") > 0:
            lines.append(
                f"  [widelane_gate] epoch_skip={_int(result, 'n_wl_gate_skip')} "
                f"pair_rejected={_int(result, 'n_wl_gate_pair_rejected')}"
            )

    return lines


def print_pf_smoother_run_summary(
    result: Mapping[str, Any],
    *,
    print_func: Callable[[str], None] = print,
) -> None:
    for line in build_pf_smoother_summary_lines(result):
        print_func(line)


def _append_positive_counter(
    lines: list[str],
    result: Mapping[str, Any],
    label: str,
    key: str,
    *,
    noun: str = "epochs",
) -> None:
    count = _int(result, key)
    if count > 0:
        lines.append(f"  [{label}] {noun}={count}")


def _any_present(result: Mapping[str, Any], keys: tuple[str, ...]) -> bool:
    return any(result.get(key) is not None for key in keys)


def _bool(result: Mapping[str, Any], key: str) -> bool:
    return bool(result.get(key, False))


def _int(result: Mapping[str, Any], key: str) -> int:
    value = result.get(key, 0)
    return int(value or 0)


def _float(result: Mapping[str, Any], key: str) -> float:
    value = result.get(key, 0.0)
    return float(value or 0.0)
