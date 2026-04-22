"""Bridge PF smoother epoch buffers into the local FGO solver."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from gnss_gpu.local_fgo import (
    DDCarrierEpoch,
    DDPseudorangeEpoch,
    LambdaFixConfig,
    LocalFgoConfig,
    LocalFgoProblem,
    LocalFgoWindow,
    UndiffPseudorangeEpoch,
    detect_weak_dd_window,
    inject_into_pf,
    parse_window_spec,
    solve_local_fgo,
    solve_local_fgo_with_lambda,
)

def _local_fgo_enabled(window_spec: str | None) -> bool:
    if window_spec is None:
        return False
    return str(window_spec).strip().lower() not in {"", "off", "none", "false", "0"}


def _make_undiff_pr_epoch(
    sat_ecef: np.ndarray,
    pseudoranges: np.ndarray,
    weights: np.ndarray,
    receiver_pos_ecef: np.ndarray,
) -> UndiffPseudorangeEpoch | None:
    sat = np.asarray(sat_ecef, dtype=np.float64).reshape(-1, 3)
    pr = np.asarray(pseudoranges, dtype=np.float64).ravel()
    w = np.asarray(weights, dtype=np.float64).ravel()
    pos = np.asarray(receiver_pos_ecef, dtype=np.float64).ravel()[:3]
    valid = (
        np.isfinite(pr)
        & np.isfinite(w)
        & np.isfinite(sat).all(axis=1)
        & np.isfinite(pos).all()
    )
    if not np.any(valid):
        return None
    ranges = np.linalg.norm(sat[valid] - pos, axis=1)
    good = np.isfinite(ranges) & (ranges > 1e3)
    if np.count_nonzero(good) < 4:
        return None
    valid_idx = np.flatnonzero(valid)[good]
    cb_est = float(np.median(pr[valid_idx] - ranges[good]))
    if not np.isfinite(cb_est):
        return None
    return UndiffPseudorangeEpoch(
        sat_ecef=sat[valid_idx],
        pseudoranges_m=pr[valid_idx],
        clock_bias_m=cb_est,
        weights=w[valid_idx],
    )


def _copy_dd_carrier_epoch(dd_result) -> DDCarrierEpoch | None:
    if dd_result is None or int(getattr(dd_result, "n_dd", 0)) <= 0:
        return None
    return DDCarrierEpoch.from_result(dd_result)


def _copy_dd_pseudorange_epoch(dd_pr_result) -> DDPseudorangeEpoch | None:
    if dd_pr_result is None or int(getattr(dd_pr_result, "n_dd", 0)) <= 0:
        return None
    return DDPseudorangeEpoch.from_result(dd_pr_result)


def _aligned_motion_deltas(
    aligned_indices: list[int],
    stored_motion_deltas: list[np.ndarray],
) -> np.ndarray | None:
    if len(aligned_indices) < 2:
        return None
    deltas: list[np.ndarray] = []
    for a, b in zip(aligned_indices[:-1], aligned_indices[1:]):
        if b <= a or a < 0 or b - 1 > len(stored_motion_deltas):
            deltas.append(np.full(3, np.nan, dtype=np.float64))
            continue
        span = np.asarray(stored_motion_deltas[a:b], dtype=np.float64)
        if span.ndim != 2 or span.shape[1] != 3 or not np.isfinite(span).all():
            deltas.append(np.full(3, np.nan, dtype=np.float64))
            continue
        deltas.append(np.sum(span, axis=0))
    return np.asarray(deltas, dtype=np.float64)


def _finite_motion_edge_mask(deltas: np.ndarray | None) -> np.ndarray:
    if deltas is None:
        return np.zeros(0, dtype=bool)
    arr = np.asarray(deltas, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        return np.zeros(0, dtype=bool)
    return np.isfinite(arr).all(axis=1)


def _select_local_fgo_motion_deltas(
    aligned_indices: list[int],
    stored_motion_deltas: list[np.ndarray],
    stored_tdcp_motion_deltas: list[np.ndarray] | None = None,
    *,
    motion_source: str = "predict",
) -> tuple[np.ndarray | None, dict[str, object]]:
    source = str(motion_source).strip().lower()
    if source not in {"predict", "tdcp", "prefer_tdcp"}:
        raise ValueError(f"unsupported local FGO motion source: {motion_source!r}")

    predict = _aligned_motion_deltas(aligned_indices, stored_motion_deltas)
    predict_mask = _finite_motion_edge_mask(predict)
    tdcp = _aligned_motion_deltas(aligned_indices, stored_tdcp_motion_deltas or [])
    tdcp_mask = _finite_motion_edge_mask(tdcp)

    info: dict[str, object] = {
        "motion_source": source,
        "motion_predict_edges": int(np.count_nonzero(predict_mask)),
        "motion_tdcp_edges": int(np.count_nonzero(tdcp_mask)),
        "motion_tdcp_selected_edges": 0,
    }
    if source == "predict":
        return predict, info
    if source == "tdcp":
        info["motion_tdcp_selected_edges"] = int(np.count_nonzero(tdcp_mask))
        return tdcp, info

    if predict is None:
        selected = tdcp
        selected_tdcp_edges = int(np.count_nonzero(tdcp_mask))
    else:
        selected = np.asarray(predict, dtype=np.float64).copy()
        selected_tdcp_edges = 0
        if tdcp is not None and len(tdcp_mask) == len(selected):
            selected[tdcp_mask] = np.asarray(tdcp, dtype=np.float64)[tdcp_mask]
            selected_tdcp_edges = int(np.count_nonzero(tdcp_mask))
    info["motion_tdcp_selected_edges"] = selected_tdcp_edges
    return selected, info

def _resolve_local_fgo_window(
    window_spec: str,
    n_epochs: int,
    epoch_diagnostics: list[dict[str, object]] | None,
    *,
    min_epochs: int,
    dd_max_pairs: int,
) -> LocalFgoWindow | None:
    spec = str(window_spec).strip().lower()
    if spec == "auto":
        if not epoch_diagnostics:
            return None
        return detect_weak_dd_window(
            epoch_diagnostics,
            min_epochs=min_epochs,
            dd_max_pairs=dd_max_pairs,
        )
    return parse_window_spec(window_spec, n_epochs)


def _apply_local_fgo_postprocess(
    smoothed_aligned: np.ndarray,
    aligned_indices: list[int],
    stored_motion_deltas: list[np.ndarray],
    stored_tdcp_motion_deltas: list[np.ndarray] | None,
    stored_dd_carrier: list[DDCarrierEpoch | None],
    stored_dd_pseudorange: list[DDPseudorangeEpoch | None],
    stored_undiff_pr: list[UndiffPseudorangeEpoch | None],
    epoch_diagnostics: list[dict[str, object]] | None,
    *,
    window_spec: str,
    min_epochs: int,
    dd_max_pairs: int,
    config: LocalFgoConfig,
    lambda_config: LambdaFixConfig | None = None,
    motion_source: str = "predict",
    two_step: bool = False,
    stage1_prior_sigma_m: float | None = None,
    stage1_motion_sigma_m: float | None = None,
    stage1_pr_sigma_m: float | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    smoothed = np.asarray(smoothed_aligned, dtype=np.float64)
    info: dict[str, object] = {
        "applied": False,
        "window": None,
        "solve_window": None,
        "factor_counts": None,
        "initial_error": None,
        "final_error": None,
        "lambda": None,
        "two_step": bool(two_step),
        "stage1": None,
        "motion_source": str(motion_source).strip().lower(),
        "motion_predict_edges": 0,
        "motion_tdcp_edges": 0,
        "motion_tdcp_selected_edges": 0,
        "reason": None,
    }
    if len(smoothed) == 0:
        info["reason"] = "empty_smoothed"
        return smoothed, info

    target = _resolve_local_fgo_window(
        window_spec,
        len(smoothed),
        epoch_diagnostics,
        min_epochs=min_epochs,
        dd_max_pairs=dd_max_pairs,
    )
    if target is None:
        info["reason"] = "no_window"
        return smoothed, info

    aligned_dd_carrier = [stored_dd_carrier[i] for i in aligned_indices]
    aligned_dd_pr = [stored_dd_pseudorange[i] for i in aligned_indices]
    aligned_undiff = [stored_undiff_pr[i] for i in aligned_indices]
    motion_deltas, motion_info = _select_local_fgo_motion_deltas(
        aligned_indices,
        stored_motion_deltas,
        stored_tdcp_motion_deltas,
        motion_source=motion_source,
    )
    info.update(motion_info)
    solve_window = LocalFgoWindow(
        max(0, target.start - 1),
        min(len(smoothed) - 1, target.end + 1),
    )

    initial_for_final = smoothed
    if two_step:
        n_stage1_pr = sum(
            int(obs.n)
            for i, obs in enumerate(aligned_undiff)
            if solve_window.start <= i <= solve_window.end and obs is not None
        )
        if n_stage1_pr > 0:
            stage1_config = replace(
                config,
                prior_sigma_m=(
                    float(stage1_prior_sigma_m)
                    if stage1_prior_sigma_m is not None
                    else float(config.prior_sigma_m)
                ),
                motion_sigma_m=(
                    float(stage1_motion_sigma_m)
                    if stage1_motion_sigma_m is not None
                    else float(config.motion_sigma_m)
                ),
                dd_pr_sigma_m=(
                    float(stage1_pr_sigma_m)
                    if stage1_pr_sigma_m is not None
                    else float(config.dd_pr_sigma_m)
                ),
                undiff_pr_sigma_m=(
                    float(stage1_pr_sigma_m)
                    if stage1_pr_sigma_m is not None
                    else float(config.undiff_pr_sigma_m)
                ),
            )
            stage1_problem = LocalFgoProblem(
                initial_positions_ecef=smoothed,
                prior_positions_ecef=smoothed,
                window=solve_window,
                motion_deltas_ecef=motion_deltas,
                dd_carrier=None,
                dd_pseudorange=None,
                undiff_pseudorange=aligned_undiff,
            )
            stage1_result = solve_local_fgo(stage1_problem, stage1_config)
            initial_for_final = np.asarray(
                inject_into_pf(smoothed, stage1_result.positions_ecef, solve_window),
                dtype=np.float64,
            )
            info["stage1"] = {
                "applied": True,
                "factor_counts": dict(stage1_result.factor_counts),
                "initial_error": float(stage1_result.initial_error),
                "final_error": float(stage1_result.final_error),
                "reason": "ok",
            }
        else:
            info["stage1"] = {
                "applied": False,
                "factor_counts": None,
                "initial_error": None,
                "final_error": None,
                "reason": "no_undiff_pr",
            }

    problem = LocalFgoProblem(
        initial_positions_ecef=initial_for_final,
        prior_positions_ecef=smoothed,
        window=solve_window,
        motion_deltas_ecef=motion_deltas,
        dd_carrier=aligned_dd_carrier,
        dd_pseudorange=aligned_dd_pr,
        undiff_pseudorange=aligned_undiff,
    )
    lambda_info = None
    if lambda_config is None:
        result = solve_local_fgo(problem, config)
    else:
        result, lambda_info = solve_local_fgo_with_lambda(problem, config, lambda_config)
    rel_start = target.start - solve_window.start
    rel_end = rel_start + target.size
    updated = inject_into_pf(
        smoothed,
        result.positions_ecef[rel_start:rel_end],
        target,
    )
    if epoch_diagnostics:
        for i, row in enumerate(epoch_diagnostics):
            row["local_fgo_applied"] = bool(target.start <= i <= target.end)
    info.update(
        {
            "applied": True,
            "window": f"{target.start}:{target.end}",
            "solve_window": f"{solve_window.start}:{solve_window.end}",
            "factor_counts": dict(result.factor_counts),
            "initial_error": float(result.initial_error),
            "final_error": float(result.final_error),
            "lambda": lambda_info,
            "reason": "ok",
        }
    )
    return updated, info
