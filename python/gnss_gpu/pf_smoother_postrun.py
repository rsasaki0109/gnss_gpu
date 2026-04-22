"""Post-forward metrics, smoothing, and local-FGO finalization."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from gnss_gpu.local_fgo import LambdaFixConfig, LocalFgoConfig
from gnss_gpu.local_fgo_bridge import _apply_local_fgo_postprocess, _local_fgo_enabled
from gnss_gpu.pf_smoother_runtime import ForwardRunBuffers
from gnss_gpu.smoother_postprocess import (
    _apply_smoother_tail_guard,
    _apply_smoother_widelane_forward_guard,
    _apply_stop_segment_constant_position,
)
from gnss_gpu.stop_segment_static import (
    StaticStopSegmentConfig,
    apply_static_stop_segment_gnss,
)


@dataclass(frozen=True)
class PfSmootherPostrunFinalizeResult:
    result: dict[str, object]
    forward_positions: np.ndarray
    ground_truth: np.ndarray
    smoothed_positions: np.ndarray | None = None


def finalize_pf_smoother_postrun(
    result: dict[str, object],
    pf: Any,
    buffers: ForwardRunBuffers,
    *,
    position_update_sigma: float | None,
    use_smoother: bool,
    smoother_config: Any,
    local_fgo_config: Any,
    fgo_motion_source: str,
    collect_epoch_diagnostics: bool,
    compute_metrics_func: Callable[[np.ndarray, np.ndarray], dict[str, Any]],
    ecef_errors_func: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]],
    print_func: Callable[[str], None] = print,
) -> PfSmootherPostrunFinalizeResult:
    forward_pos_full = np.asarray(buffers.forward_aligned, dtype=np.float64)
    gt_arr = np.asarray(buffers.gt_aligned, dtype=np.float64)

    if len(forward_pos_full) == 0:
        return PfSmootherPostrunFinalizeResult(
            result=result,
            forward_positions=forward_pos_full,
            ground_truth=gt_arr,
        )

    result["forward_metrics"] = compute_metrics_func(
        forward_pos_full,
        gt_arr[: len(forward_pos_full)],
    )
    if collect_epoch_diagnostics and buffers.aligned_epoch_diagnostics:
        _append_forward_diagnostic_errors(
            buffers.aligned_epoch_diagnostics,
            forward_pos_full,
            gt_arr[: len(forward_pos_full)],
            ecef_errors_func,
        )
        result["epoch_diagnostics"] = buffers.aligned_epoch_diagnostics

    smoothed_aligned_arr = None
    if use_smoother and buffers.n_stored > 0:
        smoother_position_update_sigma = (
            smoother_config.position_update_sigma
            if getattr(smoother_config, "position_update_sigma", None) is not None
            else position_update_sigma
        )
        smoothed_full, _forward_stored = pf.smooth(
            position_update_sigma=smoother_position_update_sigma,
            skip_widelane_dd_pseudorange=smoother_config.skip_widelane_dd_pseudorange,
        )
        result["smoother_position_update_sigma"] = smoother_position_update_sigma
        if buffers.aligned_indices:
            idx = np.asarray(buffers.aligned_indices, dtype=np.int64)
            smoothed_aligned_arr = np.asarray(smoothed_full, dtype=np.float64)[idx]
            smoothed_aligned_arr = _apply_smoother_postprocesses(
                result,
                buffers,
                smoothed_aligned_arr,
                forward_pos_full,
                smoother_config,
                collect_epoch_diagnostics=collect_epoch_diagnostics,
            )
            result["smoothed_metrics_before_fgo"] = compute_metrics_func(
                smoothed_aligned_arr,
                gt_arr[: len(smoothed_aligned_arr)],
            )
            smoothed_aligned_arr = _apply_optional_local_fgo(
                result,
                buffers,
                smoothed_aligned_arr,
                local_fgo_config,
                fgo_motion_source=fgo_motion_source,
                collect_epoch_diagnostics=collect_epoch_diagnostics,
                print_func=print_func,
            )
            result["smoothed_metrics"] = compute_metrics_func(
                smoothed_aligned_arr,
                gt_arr[: len(smoothed_aligned_arr)],
            )
            if collect_epoch_diagnostics and buffers.aligned_epoch_diagnostics:
                _append_smoothed_diagnostic_errors(
                    buffers.aligned_epoch_diagnostics,
                    smoothed_aligned_arr,
                    forward_pos_full[: len(smoothed_aligned_arr)],
                    gt_arr[: len(smoothed_aligned_arr)],
                    ecef_errors_func,
                )

    return PfSmootherPostrunFinalizeResult(
        result=result,
        forward_positions=forward_pos_full,
        ground_truth=gt_arr,
        smoothed_positions=smoothed_aligned_arr,
    )


def _append_forward_diagnostic_errors(
    rows: list[dict[str, object]],
    forward_positions: np.ndarray,
    ground_truth: np.ndarray,
    ecef_errors_func: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]],
) -> None:
    forward_errors_2d, forward_errors_3d = ecef_errors_func(
        forward_positions,
        ground_truth,
    )
    for row, err2d, err3d in zip(rows, forward_errors_2d, forward_errors_3d):
        row["forward_error_2d"] = float(err2d)
        row["forward_error_3d"] = float(err3d)


def _apply_smoother_postprocesses(
    result: dict[str, object],
    buffers: ForwardRunBuffers,
    smoothed_aligned: np.ndarray,
    forward_positions: np.ndarray,
    smoother_config: Any,
    *,
    collect_epoch_diagnostics: bool,
) -> np.ndarray:
    diagnostics = buffers.aligned_epoch_diagnostics if collect_epoch_diagnostics else None
    smoothed_aligned, n_tail_guard_applied = _apply_smoother_tail_guard(
        smoothed_aligned,
        forward_positions[: len(smoothed_aligned)],
        diagnostics,
        ess_max_ratio=smoother_config.tail_guard_ess_max_ratio,
        dd_carrier_max_pairs=smoother_config.tail_guard_dd_carrier_max_pairs,
        dd_pseudorange_max_pairs=smoother_config.tail_guard_dd_pseudorange_max_pairs,
        min_shift_m=smoother_config.tail_guard_min_shift_m,
        expand_epochs=smoother_config.tail_guard_expand_epochs,
        expand_min_shift_m=smoother_config.tail_guard_expand_min_shift_m,
        expand_dd_pseudorange_max_pairs=(
            smoother_config.tail_guard_expand_dd_pseudorange_max_pairs
        ),
    )
    result["n_tail_guard_applied"] = int(n_tail_guard_applied)

    if smoother_config.widelane_forward_guard:
        smoothed_aligned, n_wl_forward_guard_applied = (
            _apply_smoother_widelane_forward_guard(
                smoothed_aligned,
                forward_positions[: len(smoothed_aligned)],
                diagnostics,
                min_shift_m=smoother_config.widelane_forward_guard_min_shift_m,
            )
        )
    else:
        n_wl_forward_guard_applied = 0
    result["n_widelane_forward_guard_applied"] = int(n_wl_forward_guard_applied)

    if smoother_config.stop_segment_constant:
        smoothed_aligned, stop_segment_info = _apply_stop_segment_constant_position(
            smoothed_aligned,
            forward_positions[: len(smoothed_aligned)],
            buffers.aligned_stop_flags[: len(smoothed_aligned)],
            diagnostics,
            min_epochs=smoother_config.stop_segment_min_epochs,
            source=smoother_config.stop_segment_source,
            max_radius_m=smoother_config.stop_segment_max_radius_m,
            blend=smoother_config.stop_segment_blend,
            density_neighbors=smoother_config.stop_segment_density_neighbors,
        )
    else:
        stop_segment_info = {
            "segments": 0,
            "segments_applied": 0,
            "epochs_applied": 0,
            "source": str(smoother_config.stop_segment_source).strip().lower(),
        }
    result["stop_segment_info"] = stop_segment_info
    result["n_stop_segment_epochs_applied"] = int(
        stop_segment_info.get("epochs_applied", 0)
    )

    if getattr(smoother_config, "stop_segment_static_gnss", False):
        aligned_dd_carrier = [buffers.stored_dd_carrier[i] for i in buffers.aligned_indices]
        aligned_dd_pr = [buffers.stored_dd_pseudorange[i] for i in buffers.aligned_indices]
        aligned_undiff_pr = [buffers.stored_undiff_pr[i] for i in buffers.aligned_indices]
        smoothed_aligned, static_info = apply_static_stop_segment_gnss(
            smoothed_aligned,
            buffers.aligned_stop_flags[: len(smoothed_aligned)],
            aligned_dd_carrier[: len(smoothed_aligned)],
            aligned_dd_pr[: len(smoothed_aligned)],
            aligned_undiff_pr[: len(smoothed_aligned)],
            diagnostics,
            config=StaticStopSegmentConfig(
                min_epochs=smoother_config.stop_segment_min_epochs,
                min_observations=smoother_config.stop_segment_static_min_observations,
                prior_sigma_m=smoother_config.stop_segment_static_prior_sigma_m,
                undiff_pr_sigma_m=smoother_config.stop_segment_static_pr_sigma_m,
                dd_pr_sigma_m=smoother_config.stop_segment_static_dd_pr_sigma_m,
                dd_cp_sigma_cycles=(
                    smoother_config.stop_segment_static_dd_cp_sigma_cycles
                ),
                max_update_m=smoother_config.stop_segment_static_max_update_m,
                blend=smoother_config.stop_segment_static_blend,
            ),
        )
    else:
        static_info = {
            "segments": 0,
            "segments_applied": 0,
            "epochs_applied": 0,
        }
    result["stop_segment_static_info"] = static_info
    result["n_stop_segment_static_epochs_applied"] = int(
        static_info.get("epochs_applied", 0)
    )
    return smoothed_aligned


def _apply_optional_local_fgo(
    result: dict[str, object],
    buffers: ForwardRunBuffers,
    smoothed_aligned: np.ndarray,
    local_fgo_config: Any,
    *,
    fgo_motion_source: str,
    collect_epoch_diagnostics: bool,
    print_func: Callable[[str], None],
) -> np.ndarray:
    if not _local_fgo_enabled(local_fgo_config.window):
        return smoothed_aligned

    fgo_config = LocalFgoConfig(
        prior_sigma_m=local_fgo_config.prior_sigma_m,
        motion_sigma_m=local_fgo_config.motion_sigma_m,
        dd_sigma_cycles=local_fgo_config.dd_sigma_cycles,
        dd_pr_sigma_m=local_fgo_config.pr_sigma_m,
        undiff_pr_sigma_m=local_fgo_config.pr_sigma_m,
        dd_huber_k=local_fgo_config.dd_huber_k,
        pr_huber_k=local_fgo_config.pr_huber_k,
        max_iterations=local_fgo_config.max_iterations,
    )
    lambda_config = (
        LambdaFixConfig(
            ratio_threshold=local_fgo_config.lambda_ratio_threshold,
            fixed_sigma_cycles=local_fgo_config.lambda_sigma_cycles,
            min_epochs=local_fgo_config.lambda_min_epochs,
        )
        if local_fgo_config.lambda_enabled
        else None
    )
    updated, fgo_info = _apply_local_fgo_postprocess(
        smoothed_aligned,
        list(map(int, buffers.aligned_indices)),
        buffers.stored_motion_deltas,
        buffers.stored_tdcp_motion_deltas,
        buffers.stored_dd_carrier,
        buffers.stored_dd_pseudorange,
        buffers.stored_undiff_pr,
        buffers.aligned_epoch_diagnostics if collect_epoch_diagnostics else None,
        window_spec=str(local_fgo_config.window),
        min_epochs=local_fgo_config.window_min_epochs,
        dd_max_pairs=local_fgo_config.dd_max_pairs,
        config=fgo_config,
        lambda_config=lambda_config,
        motion_source=fgo_motion_source,
        two_step=local_fgo_config.two_step,
        stage1_prior_sigma_m=local_fgo_config.stage1_prior_sigma_m,
        stage1_motion_sigma_m=local_fgo_config.stage1_motion_sigma_m,
        stage1_pr_sigma_m=local_fgo_config.stage1_pr_sigma_m,
    )
    result["fgo_local_info"] = fgo_info
    result["fgo_local_applied"] = bool(fgo_info.get("applied"))
    _print_local_fgo_info(fgo_info, print_func)
    return updated


def _print_local_fgo_info(
    fgo_info: dict[str, object],
    print_func: Callable[[str], None],
) -> None:
    if not fgo_info.get("applied"):
        print_func(f"  [local_fgo] skipped: {fgo_info.get('reason')}")
        return

    lambda_info = fgo_info.get("lambda") or {}
    lambda_text = ""
    if lambda_info:
        lambda_text = (
            f" lambda_fixed={lambda_info.get('n_fixed', 0)}"
            f"/obs={lambda_info.get('n_fixed_observations', 0)}"
        )
    stage1 = fgo_info.get("stage1") or {}
    stage_text = ""
    if stage1:
        if stage1.get("applied"):
            stage_text = (
                f" stage1={float(stage1.get('initial_error')):.2f}"
                f"->{float(stage1.get('final_error')):.2f}"
            )
        else:
            stage_text = f" stage1_skipped={stage1.get('reason')}"
    print_func(
        "  [local_fgo] "
        f"window={fgo_info.get('window')} "
        f"solve={fgo_info.get('solve_window')} "
        f"motion={fgo_info.get('motion_source')}"
        f"/tdcp={fgo_info.get('motion_tdcp_selected_edges', 0)} "
        f"factors={fgo_info.get('factor_counts')} "
        f"error={float(fgo_info.get('initial_error')):.2f}"
        f"->{float(fgo_info.get('final_error')):.2f}"
        f"{stage_text}"
        f"{lambda_text}"
    )


def _append_smoothed_diagnostic_errors(
    rows: list[dict[str, object]],
    smoothed_positions: np.ndarray,
    forward_positions: np.ndarray,
    ground_truth: np.ndarray,
    ecef_errors_func: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]],
) -> None:
    smoothed_errors_2d, smoothed_errors_3d = ecef_errors_func(
        smoothed_positions,
        ground_truth,
    )
    smoothed_shift_m = np.linalg.norm(
        smoothed_positions - forward_positions,
        axis=1,
    )
    for row, err2d, err3d, shift_m in zip(
        rows,
        smoothed_errors_2d,
        smoothed_errors_3d,
        smoothed_shift_m,
    ):
        row["smoothed_error_2d"] = float(err2d)
        row["smoothed_error_3d"] = float(err3d)
        row["smoothed_shift_3d_m"] = float(shift_m)
        if row["forward_error_2d"] is not None:
            row["smoothing_improvement_2d"] = float(row["forward_error_2d"] - err2d)
