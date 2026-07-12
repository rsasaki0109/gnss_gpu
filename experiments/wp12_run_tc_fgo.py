#!/usr/bin/env python3
"""WP12a/WP12b driver: stabilized TC-FGO float + carrier AR.

Extends ``experiments/wp11_run_tc_fgo.py`` with flag-guarded stabilization
(WP12a) and carrier-phase ambiguity resolution (WP12b).
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
for _p in (_PROJECT_ROOT / "python", _PROJECT_ROOT, _SCRIPT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from gnss_gpu.local_fgo import LocalFgoConfig  # noqa: E402
from gnss_gpu.tc_fgo import (  # noqa: E402
    TcAmbiguityBank,
    TcFgoConfig,
    TcFgoEpochObs,
    TcFgoNavState,
    TcFgoWindowProblem,
    TcSchurMarginal,
    compute_dd_pr_postfit_rms,
    dd_pr_position_update_from_epoch,
    ecef_to_enu,
    enu_to_ecef,
    is_static_epoch,
    naive_marginalization_prior,
    quality_scaled_marginalization_prior,
    solve_tc_fgo_window,
)
from wp5_run_anchored_fgo import (  # noqa: E402
    RtkPosRecord,
    anchor_sigma_m,
    classify_anchor_status,
    load_rtk_pos_extended,
    nearest_anchor_distance_epochs,
)
from wp11_run_tc_fgo import (  # noqa: E402
    DEFAULT_BASELINE_POS,
    build_dd_carrier_epoch_at_index,
    build_dd_pr_epoch_at_index,
    build_dd_measurements_for_epoch,
    build_ppc_imu_preintegration,
    collect_all_rtk_fixes,
    collect_rtk_fixes_while_static,
    collect_static_imu_samples,
    imu_rows_between,
    load_ppc_window_geometry,
    make_dd_computers,
    parse_rover_tows_from_obs,
    propagate_nav_state_with_imu,
    resolve_data_root,
    resolve_run_dir,
    run_two_phase_initialization,
    write_tc_pos_file,
    collapse_imu_preintegration_segment,
    _dd_measurement_kwargs,
)
from gsdc2023_imu import imu_preintegration_segment_with_bias_jacobians  # noqa: E402
from gnss_gpu.ins_ekf import INSEKF, INSConfig  # noqa: E402
from gnss_gpu.io.ppc import PPCDatasetLoader  # noqa: E402
from evaluate import ecef_to_lla  # noqa: E402


@dataclass
class EpochTelemetry:
    epoch: int = 0
    tow: float = 0.0
    pos_err_m: float = float("nan")
    n_dd_factors: int = 0
    dd_pr_rms_raw_m: float = float("inf")
    dd_pr_rms_huber_m: float = float("inf")
    lm_iterations: int = 0
    lm_converged: bool = False
    gnss_solved: bool = False
    anchor_factors: int = 0
    marginal_prior_dims: int = 0
    recovery_fired: bool = False
    imu_fill: bool = False
    n_dd_carrier: int = 0
    n_cross_window_prior: int = 0
    ar_accepted: bool = False
    epoch_fixed: bool = False
    ar_cert_passed: bool = False
    ar_cert_marginal_sigma_m: float = float("nan")
    ar_cert_dd_pr_rms_m: float = float("nan")
    ar_cert_dd_cp_rms_cyc: float = float("nan")
    epochs_since_recovery: int = 0
    ar_offered: bool = False
    fix_truth_err_m: float = float("nan")


def _telemetry_fieldnames() -> list[str]:
    return [f.name for f in EpochTelemetry.__dataclass_fields__.values()]


def build_anchor_observation(
    tow: float,
    rtk_by_tow: dict[float, RtkPosRecord],
    *,
    origin_ecef: np.ndarray,
    origin_lat: float,
    origin_lon: float,
    anchor_class: np.ndarray,
    epoch_index: int,
    fix_sigma_m: float,
    float_sigma_m: float,
    anchor_sigma_scale: float,
    anchor_quality_weight: bool,
    enable_fix: bool,
    enable_float: bool,
) -> tuple[np.ndarray | None, float | None]:
    """Return (anchor_pos_enu, sigma) for one epoch (WP12e truth-honest sigmas)."""

    cls = int(anchor_class[epoch_index])
    if cls == 2 and not enable_fix:
        return None, None
    if cls == 1 and not enable_float:
        return None, None
    if cls not in (1, 2):
        return None, None
    hit = rtk_by_tow.get(round(float(tow), 1))
    if hit is None:
        return None, None
    pos_enu = ecef_to_enu(hit.ecef, origin_ecef, origin_lat, origin_lon)
    sigma = anchor_sigma_m(
        hit,
        cls,
        fix_sigma_m=float(fix_sigma_m),
        float_sigma_m=float(float_sigma_m),
        sigma_scale=float(anchor_sigma_scale),
        quality_weight=bool(anchor_quality_weight),
    )
    return pos_enu, sigma


def run_tc_fgo_sequence_wp12(
    *,
    tows: np.ndarray,
    data: dict,
    dd_pr_computer,
    dd_cp_computer,
    imu_preint,
    imu_times_s: np.ndarray,
    imu_acc: np.ndarray,
    imu_gyro_dps: np.ndarray,
    init_state: TcFgoNavState,
    origin_ecef: np.ndarray,
    origin_lat: float,
    origin_lon: float,
    config: TcFgoConfig,
    systems: tuple[str, ...],
    phase2_idx: int,
    rtk_by_tow: dict[float, RtkPosRecord],
    anchor_class: np.ndarray,
    nearest_anchor_dist: np.ndarray,
    enable_anchors_fix: bool,
    enable_anchors_float: bool,
    anchor_fix_sigma_m: float,
    anchor_float_sigma_m: float,
    anchor_sigma_scale: float,
    anchor_quality_weight: bool,
    enable_recovery: bool,
    enable_recovery_rtk: bool,
    recovery_rms_threshold_m: float,
    recovery_persist_epochs: int,
    recovery_min_dd: int,
    recovery_max_shift_m: float,
    recovery_max_iter: int,
    use_quality_marginal: bool,
    dynamic_dd_rebuild: bool = False,
    held_ambiguities: dict | None = None,
    reference_ecef_by_tow: dict[float, np.ndarray] | None = None,
    telemetry: list[EpochTelemetry] | None = None,
) -> tuple[np.ndarray, list[TcFgoNavState], dict[str, int], list[int]]:
    n = int(tows.size)
    window = max(1, int(config.window_epochs))
    output_ecef = np.zeros((n, 3), dtype=np.float64)
    epoch_states: list[TcFgoNavState] = []
    epoch_status: list[int] = []
    marginal_prior: TcFgoNavState | None = None
    marginal_sigmas: np.ndarray | None = None
    schur_marginal: TcSchurMarginal | None = None
    last_dd_pr_rms = float("inf")
    bad_epochs = 0
    stats = {"recovery_events": 0, "bad_epoch_streak_max": 0, "ar_fix_epochs": 0}
    held_ambiguities = dict(held_ambiguities or {})
    ambiguity_bank = TcAmbiguityBank() if config.enable_persistent_ambiguities else None
    epochs_since_recovery = 10**9
    dd_kwargs = _dd_measurement_kwargs()
    fgo_config = LocalFgoConfig()

    def rover_pos_ecef_at(index: int) -> np.ndarray:
        if index < len(epoch_states):
            return enu_to_ecef(epoch_states[index].p_enu, origin_ecef, origin_lat, origin_lon)
        return enu_to_ecef(init_state.p_enu, origin_ecef, origin_lat, origin_lon)

    def dd_epoch_at(index: int):
        pos = rover_pos_ecef_at(index)
        return build_dd_pr_epoch_at_index(
            dd_pr_computer, data, index, pos, systems, **dd_kwargs
        )

    def dd_carrier_at(index: int):
        if not config.enable_dd_carrier:
            return None
        pos = rover_pos_ecef_at(index)
        return build_dd_carrier_epoch_at_index(
            dd_cp_computer, data, index, pos, systems, **dd_kwargs
        )

    # Prebuild static DD cache when not rebuilding dynamically.
    dd_epochs_cache: list = [None] * n
    if not dynamic_dd_rebuild:
        for i in range(n):
            dd_epochs_cache[i] = dd_epoch_at(i)

    for i in range(n):
        if i == 0:
            epoch_states.append(init_state.copy())
        else:
            imu_rows = imu_rows_between(
                imu_times_s,
                imu_acc,
                imu_gyro_dps,
                float(tows[i - 1]),
                float(tows[i]),
            )
            epoch_states.append(propagate_nav_state_with_imu(epoch_states[-1], imu_rows))

        start = max(0, len(epoch_states) - window)
        win_states = [s.copy() for s in epoch_states[start:]]
        imu_segments: list = []
        for j in range(len(win_states) - 1):
            g0 = start + j
            seg_raw = imu_preintegration_segment_with_bias_jacobians(imu_preint, g0, g0 + 2)
            imu_segments.append(
                collapse_imu_preintegration_segment(
                    seg_raw[0],
                    seg_raw[1],
                    seg_raw[2],
                    seg_raw[3],
                    seg_raw[4],
                    seg_raw[5],
                    seg_raw[6],
                    seg_raw[7],
                )
            )

        observations: list[TcFgoEpochObs] = []
        for j in range(len(win_states)):
            gi = start + j
            speed = float(np.linalg.norm(win_states[j].v_enu[0:2]))
            static = is_static_epoch(speed) and gi < phase2_idx
            dd = dd_epoch_at(gi) if dynamic_dd_rebuild else (dd_epochs_cache[gi] if gi < n else None)
            dd_cp = dd_carrier_at(gi) if config.enable_dd_carrier else None
            anchor_pos, anchor_sigma = build_anchor_observation(
                float(tows[gi]),
                rtk_by_tow,
                origin_ecef=origin_ecef,
                origin_lat=origin_lat,
                origin_lon=origin_lon,
                anchor_class=anchor_class,
                epoch_index=gi,
                fix_sigma_m=anchor_fix_sigma_m,
                float_sigma_m=anchor_float_sigma_m,
                anchor_sigma_scale=anchor_sigma_scale,
                anchor_quality_weight=anchor_quality_weight,
                enable_fix=enable_anchors_fix,
                enable_float=enable_anchors_float,
            )
            observations.append(
                TcFgoEpochObs(
                    dd_pseudorange=dd,
                    dd_carrier=dd_cp,
                    enable_nhc=(gi >= phase2_idx and not static),
                    enable_zupt=static,
                    anchor_pos_enu=anchor_pos,
                    anchor_sigma_m=anchor_sigma,
                )
            )

        problem = TcFgoWindowProblem(
            initial_states=win_states,
            imu_segments=imu_segments,
            observations=observations,
            origin_ecef=origin_ecef,
            origin_lat=origin_lat,
            origin_lon=origin_lon,
            marginal_prior=marginal_prior,
            marginal_prior_sigmas=marginal_sigmas,
            schur_marginal=schur_marginal,
            last_dd_pr_rms_m=last_dd_pr_rms,
            held_ambiguities=held_ambiguities,
            window_start_epoch=start,
            epochs_since_recovery=int(epochs_since_recovery),
            epochs_since_anchor=int(
                nearest_anchor_dist[i] if np.isfinite(nearest_anchor_dist[i]) else 10**9
            ),
        )
        if len(win_states) >= 2:
            result = solve_tc_fgo_window(
                problem,
                config=config,
                fgo_config=fgo_config,
                ambiguity_bank=ambiguity_bank,
            )
            for j, solved in enumerate(result.states):
                epoch_states[start + j] = solved.copy()

            dd_last = dd_epoch_at(i) if dynamic_dd_rebuild else (dd_epochs_cache[i] if i < n else None)
            fit_rms_huber, n_dd_huber = (
                compute_dd_pr_postfit_rms(
                    epoch_states[-1],
                    dd_last,
                    origin_ecef=origin_ecef,
                    origin_lat=origin_lat,
                    origin_lon=origin_lon,
                    config=config,
                    fgo_config=fgo_config,
                    huber_weighted=True,
                )
                if dd_last is not None
                else (float("inf"), 0)
            )
            fit_rms_raw, n_dd = (
                compute_dd_pr_postfit_rms(
                    epoch_states[-1],
                    dd_last,
                    origin_ecef=origin_ecef,
                    origin_lat=origin_lat,
                    origin_lon=origin_lon,
                    config=config,
                    fgo_config=fgo_config,
                    huber_weighted=False,
                )
                if dd_last is not None
                else (float("inf"), 0)
            )
            last_dd_pr_rms = fit_rms_raw

            if fit_rms_raw > float(recovery_rms_threshold_m) and n_dd >= int(recovery_min_dd):
                bad_epochs += 1
            else:
                bad_epochs = 0
            stats["bad_epoch_streak_max"] = max(stats["bad_epoch_streak_max"], bad_epochs)

            recovery_fired = False
            if enable_recovery and bad_epochs >= int(recovery_persist_epochs):
                recovered = False
                if enable_recovery_rtk:
                    hit = rtk_by_tow.get(round(float(tows[i]), 1))
                    if hit is not None:
                        ecef, status = hit.ecef, hit.status
                        if int(status) in (4, 1, 3):
                            epoch_states[-1].p_enu = ecef_to_enu(
                                ecef, origin_ecef, origin_lat, origin_lon
                            )
                            bad_epochs = 0
                            stats["recovery_events"] += 1
                            recovered = True
                            recovery_fired = True
                if not recovered and dd_last is not None and n_dd >= int(recovery_min_dd):
                    seed_ecef = enu_to_ecef(epoch_states[-1].p_enu, origin_ecef, origin_lat, origin_lon)
                    rtk_hit = rtk_by_tow.get(round(float(tows[i]), 1))
                    if rtk_hit is not None and int(rtk_hit.status) != 0:
                        seed_ecef = np.asarray(rtk_hit.ecef, dtype=np.float64).reshape(3)
                    recovered_ecef, rec_stats = dd_pr_position_update_from_epoch(
                        seed_ecef,
                        dd_last,
                        min_dd=int(recovery_min_dd),
                        dd_sigma_m=8.0,
                        prior_sigma_m=50.0,
                        max_shift_m=float(recovery_max_shift_m),
                        max_iter=int(recovery_max_iter),
                    )
                    if rec_stats.get("accepted"):
                        epoch_states[-1].p_enu = ecef_to_enu(
                            recovered_ecef, origin_ecef, origin_lat, origin_lon
                        )
                        bad_epochs = 0
                        stats["recovery_events"] += 1
                        recovery_fired = True
                if recovery_fired:
                    marginal_prior = None
                    marginal_sigmas = None
                    schur_marginal = None
                    last_dd_pr_rms = float("inf")
                    held_ambiguities.clear()
                    epochs_since_recovery = 0
                    if ambiguity_bank is not None:
                        ambiguity_bank.bump_generation()
            if not recovery_fired:
                epochs_since_recovery = int(epochs_since_recovery) + 1

            if result.ar_accepted and result.accepted_fixes and config.enable_ar_hold:
                held_ambiguities.update(result.accepted_fixes)

            epoch_is_fixed = bool(
                result.ar_accepted
                and result.epoch_fixed is not None
                and len(result.epoch_fixed) > 0
                and result.epoch_fixed[-1]
            )
            if epoch_is_fixed:
                stats["ar_fix_epochs"] = int(stats.get("ar_fix_epochs", 0)) + 1
            epoch_status.append(4 if epoch_is_fixed else 5)

            if len(result.states) >= 2 and i >= window - 1 and not recovery_fired:
                if config.enable_schur_marginalization and result.schur_marginal is not None:
                    schur_marginal = result.schur_marginal
                    marginal_prior = None
                    marginal_sigmas = None
                elif use_quality_marginal:
                    marginal_prior, marginal_sigmas = quality_scaled_marginalization_prior(
                        result.states[1],
                        config,
                        dd_pr_rms_m=fit_rms_raw,
                        n_dd=n_dd,
                    )
                    schur_marginal = None
                else:
                    marginal_prior, marginal_sigmas = naive_marginalization_prior(
                        result.states[1], config
                    )
                    schur_marginal = None

            if telemetry is not None:
                ref_hit = None
                if reference_ecef_by_tow is not None:
                    ref_hit = reference_ecef_by_tow.get(round(float(tows[i]), 1))
                pos_err = float("nan")
                if ref_hit is not None:
                    pos_err = float(
                        np.linalg.norm(
                            enu_to_ecef(epoch_states[-1].p_enu, origin_ecef, origin_lat, origin_lon)
                            - ref_hit
                        )
                    )
                anchor_n = int(result.factor_counts.get("position_anchor", 0))
                marg_dims = (
                    int(schur_marginal.mean.size)
                    if schur_marginal is not None
                    else (
                        int(marginal_sigmas.size)
                        if marginal_prior is not None and marginal_sigmas is not None
                        else 0
                    )
                )
                cert_info = (result.ar_info or {}).get("certificate", {})
                ar_offered = bool(
                    config.enable_lambda_ar
                    and (
                        cert_info.get("passed") is True
                        or (result.ar_info or {}).get("ar_info") is not None
                    )
                )
                fix_truth_err = float("nan")
                if epoch_is_fixed and ref_hit is not None:
                    fix_truth_err = pos_err
                telemetry.append(
                    EpochTelemetry(
                        epoch=i,
                        tow=float(tows[i]),
                        pos_err_m=pos_err,
                        n_dd_factors=int(result.factor_counts.get("dd_pseudorange", 0)),
                        dd_pr_rms_raw_m=float(fit_rms_raw),
                        dd_pr_rms_huber_m=float(fit_rms_huber),
                        lm_iterations=int(result.n_iterations),
                        lm_converged=bool(result.converged),
                        gnss_solved=n_dd > 0,
                        anchor_factors=anchor_n,
                        marginal_prior_dims=marg_dims,
                        recovery_fired=recovery_fired,
                        imu_fill=len(win_states) < 2,
                        n_dd_carrier=int(result.factor_counts.get("dd_carrier", 0)),
                        n_cross_window_prior=int(result.factor_counts.get("n_cross_window_prior", 0)),
                        ar_accepted=bool(result.ar_accepted),
                        epoch_fixed=epoch_is_fixed,
                        ar_cert_passed=bool(cert_info.get("passed", False)),
                        ar_cert_marginal_sigma_m=float(cert_info.get("marginal_pos_sigma_m", float("nan"))),
                        ar_cert_dd_pr_rms_m=float(cert_info.get("dd_pr_postfit_rms_m", float("nan"))),
                        ar_cert_dd_cp_rms_cyc=float(cert_info.get("dd_cp_postfit_rms_cyc", float("nan"))),
                        epochs_since_recovery=int(epochs_since_recovery),
                        ar_offered=ar_offered,
                        fix_truth_err_m=fix_truth_err,
                    )
                )
        elif telemetry is not None:
            ref_hit = (
                reference_ecef_by_tow.get(round(float(tows[i]), 1))
                if reference_ecef_by_tow is not None
                else None
            )
            pos_err = float("nan")
            if ref_hit is not None:
                pos_err = float(
                    np.linalg.norm(
                        enu_to_ecef(epoch_states[-1].p_enu, origin_ecef, origin_lat, origin_lon) - ref_hit
                    )
                )
            telemetry.append(
                EpochTelemetry(
                    epoch=i,
                    tow=float(tows[i]),
                    pos_err_m=pos_err,
                    imu_fill=True,
                )
            )
            epoch_status.append(5)
        else:
            epoch_status.append(5)

        output_ecef[i] = enu_to_ecef(epoch_states[-1].p_enu, origin_ecef, origin_lat, origin_lon)

    return output_ecef, epoch_states, stats, epoch_status


def build_config_from_args(args: argparse.Namespace) -> TcFgoConfig:
    return TcFgoConfig(
        window_epochs=int(args.window_epochs),
        optimize_imu_biases=bool(args.optimize_imu_biases),
        enable_imu_gnss_quality_scale=bool(args.imu_gnss_quality_scale),
        marginal_quality_rms_ref_m=float(args.marginal_quality_rms_ref),
        marginal_quality_min_dd=float(args.marginal_quality_min_dd),
        imu_quality_rms_ref_m=float(args.imu_quality_rms_ref),
        bias_rw_sigma_accel=float(args.bias_rw_sigma_accel),
        bias_rw_sigma_gyro_radps=float(args.bias_rw_sigma_gyro),
        doppler_body_vel_sigma_mps=float(args.doppler_sigma_mps),
        enable_dd_carrier=bool(args.dd_carrier),
        enable_persistent_ambiguities=bool(args.persistent_ambiguities),
        enable_lambda_ar=bool(args.lambda_ar),
        lambda_ratio_threshold=float(args.lambda_ratio),
        lambda_min_epochs=int(args.lambda_min_epochs),
        subset_ar_max_drop=int(args.subset_ar_max_drop),
        ddpr_reject_threshold=float(args.ddpr_reject_threshold),
        post_ar_ddpr_degrade_threshold=float(args.post_ar_ddpr_threshold),
        enable_ar_quality_gate=bool(args.ar_quality_gate),
        ar_cert_max_pos_sigma_m=float(args.ar_cert_max_pos_sigma),
        ar_cert_max_dd_pr_rms_m=float(args.ar_cert_max_dd_pr_rms),
        ar_cert_max_dd_cp_rms_cyc=float(args.ar_cert_max_dd_cp_rms),
        ar_cert_min_epochs_since_recovery=int(args.ar_cert_min_epochs_since_recovery),
        ar_cert_min_dd_carrier=int(args.ar_cert_min_dd_carrier),
        ar_cert_max_epochs_since_anchor=int(args.ar_cert_max_epochs_since_anchor),
        enable_ar_subset=bool(args.ar_subset),
        enable_ar_ddpr_crossval=bool(args.ar_ddpr_crossval),
        enable_ar_post_ar_gate=bool(args.ar_post_ar_gate),
        enable_ar_hold=bool(args.ar_hold),
        enable_schur_marginalization=bool(args.schur_marginal),
        schur_min_eigenvalue=float(args.schur_min_eigenvalue),
    )


def write_tc_pos_file_with_status(
    path: Path,
    tows: np.ndarray,
    positions_ecef: np.ndarray,
    statuses: list[int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write("% WP12 TC-FGO trajectory\n")
        fh.write(
            "%  GPST_week   tow(s)      x-ecef(m)        y-ecef(m)        z-ecef(m)"
            "   lat(deg)   lon(deg)  height(m)   Q  ns   sdx    sdy    sdz   age  ratio\n"
        )
        for i, (tow, pos) in enumerate(zip(tows, positions_ecef, strict=True)):
            lat, lon, height = ecef_to_lla(float(pos[0]), float(pos[1]), float(pos[2]))
            status = int(statuses[i]) if i < len(statuses) else 5
            fh.write(
                f"2324 {float(tow):14.4f} "
                f"{pos[0]:16.4f} {pos[1]:16.4f} {pos[2]:16.4f}  "
                f"{lat:10.6f} {lon:11.6f} {height:8.3f} {status}   0  "
                f"0.000  0.000  0.000  0.00  0.0\n"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="WP12a/WP12b TC-FGO runner")
    parser.add_argument("--run", default="tokyo/run1", help="PPC run path under dataset root")
    parser.add_argument("--max-epochs", type=int, default=0, help="Limit epochs (0 = all)")
    parser.add_argument("--export-pos", type=Path, required=True, help="Output RTKLIB-like .pos path")
    parser.add_argument("--baseline-pos", type=Path, default=DEFAULT_BASELINE_POS)
    parser.add_argument("--systems", default="G", help="Comma-separated GNSS systems")
    parser.add_argument("--window-epochs", type=int, default=5)
    parser.add_argument("--data-root", type=Path, default=None)
    # WP12a stabilization flags
    parser.add_argument("--optimize-imu-biases", action="store_true", help="Promote ba/bg into LM state")
    parser.add_argument("--imu-gnss-quality-scale", action="store_true", help="Scale IMU sigmas from DDPR RMS")
    parser.add_argument("--imu-quality-rms-ref", type=float, default=5.0)
    parser.add_argument("--quality-marginal", action="store_true", help="Scale marginal prior by GNSS quality")
    parser.add_argument("--marginal-quality-rms-ref", type=float, default=3.0)
    parser.add_argument("--marginal-quality-min-dd", type=float, default=4.0)
    parser.add_argument("--anchor-fix", action="store_true", help="Tight RTK FIX position anchors")
    parser.add_argument("--anchor-float", action="store_true", help="Loose RTK FLOAT position anchors")
    parser.add_argument("--anchor-fix-sigma-m", type=float, default=0.15)
    parser.add_argument("--anchor-float-sigma-m", type=float, default=3.0)
    parser.add_argument(
        "--anchor-sigma-scale",
        type=float,
        default=1.0,
        help="Global multiplier on per-anchor sigmas (WP12e)",
    )
    parser.add_argument(
        "--no-anchor-quality-weight",
        dest="anchor_quality_weight",
        action="store_false",
        help="Disable nsat/ratio sigma inflation on anchors",
    )
    parser.set_defaults(anchor_quality_weight=True)
    parser.add_argument("--recovery", action="store_true", help="DDPR recovery when RMS persistently bad")
    parser.add_argument("--recovery-rtk", action="store_true", help="Prefer RTK baseline reseed in recovery")
    parser.add_argument("--recovery-rms-threshold-m", type=float, default=6.0)
    parser.add_argument("--recovery-persist-epochs", type=int, default=3)
    parser.add_argument("--recovery-min-dd", type=int, default=3)
    parser.add_argument("--recovery-max-shift-m", type=float, default=5000.0)
    parser.add_argument("--recovery-max-iter", type=int, default=12)
    parser.add_argument("--telemetry-csv", type=Path, default=None, help="Per-epoch diagnostic CSV")
    parser.add_argument("--bias-rw-sigma-accel", type=float, default=0.02)
    parser.add_argument("--bias-rw-sigma-gyro", type=float, default=0.002)
    parser.add_argument("--doppler-sigma-mps", type=float, default=0.0, help="Body/ENU vel prior (0=off)")
    # WP12b carrier + AR
    parser.add_argument("--dynamic-dd-rebuild", action="store_true", help="Rebuild DD rows from float position")
    parser.add_argument("--dd-carrier", action="store_true", help="Enable DD carrier-phase factors")
    parser.add_argument(
        "--persistent-ambiguities",
        action="store_true",
        help="Carry float DD ambiguities across sliding windows (WP12b)",
    )
    parser.add_argument("--lambda-ar", action="store_true", help="LAMBDA + validation after float solve")
    parser.add_argument("--no-ar-quality-gate", dest="ar_quality_gate", action="store_false", help="Disable WP12d float certificate (naive LAMBDA)")
    parser.set_defaults(ar_quality_gate=True)
    parser.add_argument("--ar-cert-max-pos-sigma", type=float, default=0.5)
    parser.add_argument("--ar-cert-max-dd-pr-rms", type=float, default=2.0)
    parser.add_argument("--ar-cert-max-dd-cp-rms", type=float, default=1.0)
    parser.add_argument("--ar-cert-min-epochs-since-recovery", type=int, default=10)
    parser.add_argument("--ar-cert-min-dd-carrier", type=int, default=4)
    parser.add_argument(
        "--ar-cert-max-epochs-since-anchor",
        type=int,
        default=0,
        help="Only offer AR within M epochs of an enabled anchor (0=off, WP12e)",
    )
    parser.add_argument("--no-ar-subset", dest="ar_subset", action="store_false", help="Disable subset-AR")
    parser.add_argument("--no-ar-ddpr-crossval", dest="ar_ddpr_crossval", action="store_false", help="Disable DDPR cross-validation")
    parser.add_argument("--no-ar-post-ar-gate", dest="ar_post_ar_gate", action="store_false", help="Disable post-AR DDPR degrade gate")
    parser.add_argument("--no-ar-hold", dest="ar_hold", action="store_false", help="Disable fix-and-hold across windows")
    parser.set_defaults(ar_subset=True, ar_ddpr_crossval=True, ar_post_ar_gate=True, ar_hold=True)
    parser.add_argument("--lambda-ratio", type=float, default=3.0)
    parser.add_argument("--lambda-min-epochs", type=int, default=3)
    parser.add_argument("--subset-ar-max-drop", type=int, default=2)
    parser.add_argument("--ddpr-reject-threshold", type=float, default=0.05)
    parser.add_argument("--post-ar-ddpr-threshold", type=float, default=0.10)
    # WP12c Schur marginalization + bias memory
    parser.add_argument(
        "--schur-marginal",
        action="store_true",
        help="Schur-complement sliding-window marginal (WP12c)",
    )
    parser.add_argument("--schur-min-eigenvalue", type=float, default=1.0e-6)
    args = parser.parse_args(argv)

    data_root = Path(args.data_root) if args.data_root is not None else resolve_data_root()
    run_dir = resolve_run_dir(data_root, str(args.run))
    systems = tuple(s.strip() for s in str(args.systems).split(",") if s.strip())

    all_tows = parse_rover_tows_from_obs(run_dir / "rover.obs")
    tows = all_tows[: int(args.max_epochs)] if int(args.max_epochs) > 0 else all_tows
    if tows.size == 0:
        raise ValueError("no rover epochs")

    t0 = time.perf_counter()
    data = load_ppc_window_geometry(
        run_dir,
        start_tow=float(tows[0]),
        end_tow=float(tows[-1]),
        systems=systems,
    )
    if len(data["times"]) != tows.size:
        n = min(len(data["times"]), tows.size)
        tows = tows[:n]
        for key in ("times", "sat_ecef", "weights", "system_ids", "used_prns", "truth"):
            if key in data and hasattr(data[key], "__len__"):
                data[key] = data[key][:n]

    rtk_by_tow = load_rtk_pos_extended(Path(args.baseline_pos))
    rtk_status_by_tow = {k: (v.ecef, v.status) for k, v in rtk_by_tow.items()}
    anchor_class = classify_anchor_status(tows, rtk_status_by_tow)
    nearest_anchor_dist = nearest_anchor_distance_epochs(
        anchor_class,
        include_fix=bool(args.anchor_fix),
        include_float=bool(args.anchor_float),
    )
    static_fixes = collect_rtk_fixes_while_static(tows, rtk_status_by_tow)
    all_fixes = collect_all_rtk_fixes(tows, rtk_status_by_tow)
    if len(static_fixes) < 5:
        raise ValueError(f"insufficient static RTK FIX epochs for phase-1 init: {len(static_fixes)}")
    if len(all_fixes) < 5:
        raise ValueError(f"insufficient RTK FIX epochs for phase-2 heading: {len(all_fixes)}")

    loader = PPCDatasetLoader(run_dir)
    imu_data = loader.load_imu()
    imu_times_s = np.asarray(imu_data["time"], dtype=np.float64)
    imu_acc = np.column_stack([imu_data["acc_x"], imu_data["acc_y"], imu_data["acc_z"]])
    imu_gyro_dps = np.column_stack([imu_data["gyro_x"], imu_data["gyro_y"], imu_data["gyro_z"]])

    origin_ecef = np.asarray(data["base_ecef"], dtype=np.float64)
    origin_lat, origin_lon, _ = ecef_to_lla(float(origin_ecef[0]), float(origin_ecef[1]), float(origin_ecef[2]))

    seed_positions = np.vstack(
        [rtk_by_tow.get(round(float(t), 1), RtkPosRecord(origin_ecef, 0)).ecef for t in tows]
    )
    for i, tow in enumerate(tows):
        hit = rtk_by_tow.get(round(float(tow), 1))
        if hit is not None and int(hit.status) != 0:
            seed_positions[i] = hit.ecef

    imu_preint = build_ppc_imu_preintegration(
        imu_data,
        np.asarray(data["times"], dtype=np.float64),
        seed_positions,
        delta_frame="body",
    )

    static_imu = collect_static_imu_samples(
        imu_times_s,
        imu_acc,
        imu_gyro_dps,
        float(static_fixes[0][0]),
        float(static_fixes[min(len(static_fixes) - 1, 4)][0]),
    )
    ins = INSEKF(INSConfig())
    init_state, phase2_idx = run_two_phase_initialization(
        ins,
        epoch_times_s=np.asarray(data["times"], dtype=np.float64),
        rtk_fix_positions_ecef=all_fixes,
        static_fix_positions_ecef=static_fixes,
        imu_samples_static=static_imu,
        origin_ecef=origin_ecef,
        origin_lat=origin_lat,
        origin_lon=origin_lon,
    )

    dd_pr_computer, dd_cp_computer = make_dd_computers(run_dir, data, systems)
    config = build_config_from_args(args)

    reference_ecef_by_tow: dict[float, np.ndarray] | None = None
    telemetry_rows: list[EpochTelemetry] | None = [] if args.telemetry_csv is not None else None
    if telemetry_rows is not None:
        from score_vs_inuex35 import load_reference_grid  # noqa: E402

        run_name = str(args.run).split("/")[-1]
        reference_ecef_by_tow = load_reference_grid("tokyo", run_name)

    positions_ecef, _states, seq_stats, epoch_status = run_tc_fgo_sequence_wp12(
        tows=tows,
        data=data,
        dd_pr_computer=dd_pr_computer,
        dd_cp_computer=dd_cp_computer,
        imu_preint=imu_preint,
        imu_times_s=imu_times_s,
        imu_acc=imu_acc,
        imu_gyro_dps=imu_gyro_dps,
        init_state=init_state,
        origin_ecef=origin_ecef,
        origin_lat=origin_lat,
        origin_lon=origin_lon,
        config=config,
        systems=systems,
        phase2_idx=phase2_idx,
        rtk_by_tow=rtk_by_tow,
        anchor_class=anchor_class,
        nearest_anchor_dist=nearest_anchor_dist,
        enable_anchors_fix=bool(args.anchor_fix),
        enable_anchors_float=bool(args.anchor_float),
        anchor_fix_sigma_m=float(args.anchor_fix_sigma_m),
        anchor_float_sigma_m=float(args.anchor_float_sigma_m),
        anchor_sigma_scale=float(args.anchor_sigma_scale),
        anchor_quality_weight=bool(args.anchor_quality_weight),
        enable_recovery=bool(args.recovery),
        enable_recovery_rtk=bool(args.recovery_rtk),
        recovery_rms_threshold_m=float(args.recovery_rms_threshold_m),
        recovery_persist_epochs=int(args.recovery_persist_epochs),
        recovery_min_dd=int(args.recovery_min_dd),
        recovery_max_shift_m=float(args.recovery_max_shift_m),
        recovery_max_iter=int(args.recovery_max_iter),
        use_quality_marginal=bool(args.quality_marginal),
        dynamic_dd_rebuild=bool(args.dynamic_dd_rebuild),
        reference_ecef_by_tow=reference_ecef_by_tow,
        telemetry=telemetry_rows,
    )

    if telemetry_rows is not None and args.telemetry_csv is not None:
        args.telemetry_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.telemetry_csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=_telemetry_fieldnames())
            writer.writeheader()
            for row in telemetry_rows:
                writer.writerow({k: getattr(row, k) for k in _telemetry_fieldnames()})
        print(f"Wrote telemetry: {args.telemetry_csv}")

    write_tc_pos_file_with_status(Path(args.export_pos), tows, positions_ecef, epoch_status)
    elapsed = time.perf_counter() - t0
    flags = []
    if args.dynamic_dd_rebuild:
        flags.append("dyn_dd")
    if args.dd_carrier:
        flags.append("dd_cp")
    if args.persistent_ambiguities:
        flags.append("persist_amb")
    if args.lambda_ar:
        flags.append("lambda")
    if args.lambda_ar and args.ar_quality_gate:
        flags.append("cert_ar")
    if args.lambda_ar and not args.ar_subset:
        flags.append("no_subset")
    if args.lambda_ar and not args.ar_ddpr_crossval:
        flags.append("no_ddpr")
    if args.lambda_ar and not args.ar_hold:
        flags.append("no_hold")
    if args.optimize_imu_biases:
        flags.append("bias")
    if args.anchor_fix:
        flags.append("anchor_fix")
    if args.anchor_float:
        flags.append("anchor_float")
    if args.imu_gnss_quality_scale:
        flags.append("imu_scale")
    if args.quality_marginal:
        flags.append("q_marg")
    if args.schur_marginal:
        flags.append("schur")
    if args.recovery:
        flags.append("recovery")
    if args.recovery_rtk:
        flags.append("recovery_rtk")
    flag_str = "+".join(flags) if flags else "baseline"
    print(
        f"WP12 TC-FGO [{flag_str}]: {tows.size} epochs -> {args.export_pos} "
        f"(phase2@{phase2_idx}, recovery={seq_stats['recovery_events']}, "
        f"ar_fix={seq_stats.get('ar_fix_epochs', 0)}, "
        f"bad_streak_max={seq_stats['bad_epoch_streak_max']}, {elapsed:.1f}s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
