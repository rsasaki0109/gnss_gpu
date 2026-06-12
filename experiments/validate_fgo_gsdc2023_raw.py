#!/usr/bin/env python3
"""CLI wrapper for the GSDC2023 raw-data bridge."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from experiments.gsdc2023_raw_bridge import (
    BridgeConfig,
    DEFAULT_ROOT,
    DEFAULT_MOTION_SIGMA_M,
    DEFAULT_TDCP_GEOMETRY_CORRECTION,
    DEFAULT_TDCP_WEIGHT_SCALE,
    FACTOR_DT_MAX_S,
    GATED_BASELINE_THRESHOLD_DEFAULT,
    OBS_MASK_DOPPLER_RESIDUAL_THRESHOLD_MPS,
    OBS_MASK_MIN_CN0_DBHZ,
    OBS_MASK_MIN_ELEVATION_DEG,
    OBS_MASK_PSEUDORANGE_DOPPLER_THRESHOLD_M,
    OBS_MASK_RESIDUAL_THRESHOLD_M,
    OBS_MASK_RESIDUAL_THRESHOLD_L5_M,
    HEIGHT_ABSOLUTE_DIST_M,
    HEIGHT_ABSOLUTE_SIGMA_M,
    IMU_ACCEL_BIAS_BETWEEN_SIGMA_MPS2,
    IMU_ACCEL_BIAS_PRIOR_SIGMA_MPS2,
    IMU_DELTA_FRAMES,
    POSITION_SOURCES,
    _build_trip_arrays,
    _export_bridge_outputs,
    _fit_state_with_clock_bias,
    validate_raw_gsdc2023_trip,
)
from experiments.gsdc2023_bridge_config import (
    apply_taroz_fgo_preset,
    apply_taroz_full_init_pass_preset,
    apply_taroz_gnss_only_preset,
    apply_taroz_marupaku_preset,
)
from experiments.gsdc2023_output import TAROZ_FGO_CANDIDATE_SOURCES


_DEFAULT_ROOT = DEFAULT_ROOT


def _cli_option_present(argv: list[str], *names: str) -> bool:
    return any(arg == name or arg.startswith(f"{name}=") for arg in argv for name in names)


def _apply_explicit_cli_overrides(config: BridgeConfig, args: argparse.Namespace, argv: list[str]) -> BridgeConfig:
    overrides: dict[str, object] = {}
    boolean_options = {
        "dual_frequency": ("--dual-frequency", "--no-dual-frequency"),
        "apply_imu_prior": ("--imu-prior", "--no-imu-prior"),
        "imu_accel_bias_state": ("--imu-accel-bias-state", "--no-imu-accel-bias-state"),
        "apply_absolute_height": ("--absolute-height", "--no-absolute-height"),
        "apply_relative_height": ("--relative-height", "--no-relative-height"),
        "apply_position_offset": ("--position-offset", "--no-position-offset"),
        "apply_base_correction": ("--base-correction", "--no-base-correction"),
        "apply_observation_mask": ("--observation-mask", "--no-observation-mask"),
        "graph_relative_height": ("--graph-relative-height", "--no-graph-relative-height"),
        "use_vd": ("--vd", "--no-vd"),
        "multi_gnss": ("--multi-gnss", "--no-multi-gnss"),
        "tdcp_enabled": ("--tdcp", "--no-tdcp"),
        "tdcp_geometry_correction": ("--tdcp-geometry-correction", "--no-tdcp-geometry-correction"),
        "fgo_line_search": ("--fgo-line-search", "--no-fgo-line-search"),
    }
    for field, names in boolean_options.items():
        if _cli_option_present(argv, *names):
            overrides[field] = getattr(args, field if hasattr(args, field) else names[0].lstrip("-").replace("-", "_"))
    scalar_options = {
        "motion_sigma_m": ("--motion-sigma-m",),
        "clock_drift_sigma_m": ("--clock-drift-sigma-m",),
        "stop_velocity_sigma_mps": ("--stop-velocity-sigma-mps",),
        "stop_position_sigma_m": ("--stop-position-sigma-m",),
        "stop_attitude_sigma_rad": ("--stop-attitude-sigma-rad",),
        "fgo_lm_damping": ("--fgo-lm-damping",),
        "relative_height_sigma_m": ("--relative-height-sigma-m",),
        "absolute_height_sigma_m": ("--absolute-height-sigma-m",),
        "absolute_height_dist_m": ("--absolute-height-dist-m",),
        "tdcp_weight_scale": ("--tdcp-weight-scale",),
        "tdcp_l5_weight_scale": ("--tdcp-l5-weight-scale",),
        "tdcp_consistency_threshold_m": ("--tdcp-consistency-threshold-m",),
    }
    for field, names in scalar_options.items():
        if _cli_option_present(argv, *names):
            overrides[field] = getattr(args, field)
            if field == "motion_sigma_m":
                overrides["per_type_kernel_motion_enabled"] = False
    return replace(config, **overrides) if overrides else config


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--trip", type=str, required=True, help="relative trip path under data root")
    p.add_argument("--max-epochs", type=int, default=200)
    p.add_argument("--start-epoch", type=int, default=0)
    p.add_argument("--motion-sigma-m", type=float, default=DEFAULT_MOTION_SIGMA_M)
    p.add_argument(
        "--factor-dt-max-s",
        type=float,
        default=FACTOR_DT_MAX_S,
        help="max epoch spacing for motion/clock/TDCP/IMU graph factors; <=0 disables this gate",
    )
    p.add_argument("--fgo-iters", type=int, default=8)
    p.add_argument("--fgo-line-search", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--fgo-lm-damping",
        type=float,
        default=0.0,
        help=(
            "LM lambda; with line search enabled this is the adaptive initial lambda, "
            "with --no-fgo-line-search it is fixed, and 0 keeps the Gauss-Newton solver"
        ),
    )
    p.add_argument(
        "--fgo-tol",
        type=float,
        default=None,
        help="FGO solver convergence tolerance; Taroz factor parity overrides this to 1e-10 unless explicitly set.",
    )
    p.add_argument(
        "--taroz-fgo",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable the taroz parameters.m FGO preset: taroz_sn FGO weights, per-Type PR/D/L Huber and motion, clock/stop/height sigmas.",
    )
    p.add_argument(
        "--taroz-marupaku",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable the closest Taroz full-FGO preset with direct FGO output and IMU priors.",
    )
    p.add_argument(
        "--taroz-full-init-pass",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="With --taroz-marupaku, match fgo_gnss_imu(..., true): disable stop-velocity and height factors while keeping IMU and stop-pose factors.",
    )
    p.add_argument(
        "--taroz-gnss-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable the Taroz GNSS-only fgo_gnss.m preset without stop, height, or IMU priors.",
    )
    p.add_argument(
        "--taroz-factor-parity",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run the Taroz GNSS-only fixed-factor parity setup: no bridge observation mask and loose TDCP consistency gate.",
    )
    p.add_argument(
        "--taroz-fgo-seed-state-csv",
        type=Path,
        default=None,
        help="optional Taroz GNSS initial-state CSV (or export dir) used only to seed fixed-linearized FGO",
    )
    p.add_argument(
        "--taroz-pose-bias-seed-state-csv",
        type=Path,
        default=None,
        help=(
            "optional Taroz IMU state CSV (or export dir) used to seed full-FGO Pose3 "
            "translation/attitude and ConstantBias keys separately from x/v/c/d"
        ),
    )
    p.add_argument(
        "--taroz-factor-mask-csv",
        type=Path,
        default=None,
        help="optional Taroz GNSS factor-mask CSV (or export dir) used to filter FGO-only P/D/L support",
    )
    p.add_argument(
        "--taroz-imu-factor-mask-csv",
        type=Path,
        default=None,
        help="optional Taroz IMU factor-mask CSV (or export dir) used to filter full-FGO IMU interval support",
    )
    p.add_argument(
        "--taroz-imu-preintegration-csv",
        type=Path,
        default=None,
        help="optional Taroz phone_data_imu_preintegration.csv used instead of native Python IMU preintegration",
    )
    p.add_argument(
        "--taroz-fgo-candidates",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Add taroz-weight / PR-Huber / PR-D-L-Huber FGO variants as gated candidate sources.",
    )
    p.add_argument(
        "--taroz-fgo-candidate-sources",
        default=",".join(TAROZ_FGO_CANDIDATE_SOURCES),
        help="Comma-separated taroz FGO candidate sources.",
    )
    p.add_argument("--clock-drift-sigma-m", type=float, default=1.0)
    p.add_argument("--stop-velocity-sigma-mps", type=float, default=0.0)
    p.add_argument("--stop-position-sigma-m", type=float, default=0.0)
    p.add_argument("--stop-attitude-sigma-rad", type=float, default=0.0)
    p.add_argument(
        "--imu-prior",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="experimental: pass synchronized IMU epoch deltas as weak VD-FGO priors",
    )
    p.add_argument(
        "--imu-frame",
        choices=IMU_DELTA_FRAMES,
        default="body",
        help="delta frame used by --imu-prior; ecef applies yaw/mounting/gravity approximation",
    )
    p.add_argument(
        "--imu-position-sigma-m",
        type=float,
        default=25.0,
        help="std-dev (m) for --imu-prior displacement deltas",
    )
    p.add_argument(
        "--imu-velocity-sigma-mps",
        type=float,
        default=5.0,
        help="std-dev (m/s) for --imu-prior velocity deltas",
    )
    p.add_argument(
        "--imu-accel-bias-state",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="experimental: append [bax,bay,baz] VD states for first-order IMU accel-bias correction",
    )
    p.add_argument(
        "--imu-accel-bias-prior-sigma-mps2",
        type=float,
        default=IMU_ACCEL_BIAS_PRIOR_SIGMA_MPS2,
        help="initial zero-bias prior sigma (m/s^2) for --imu-accel-bias-state",
    )
    p.add_argument(
        "--imu-accel-bias-between-sigma-mps2",
        type=float,
        default=IMU_ACCEL_BIAS_BETWEEN_SIGMA_MPS2,
        help="between-epoch accel-bias smoothness sigma (m/s^2) for --imu-accel-bias-state",
    )
    p.add_argument(
        "--absolute-height",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="experimental: use ref_hight.mat/ref_height.mat as ENU-up absolute-height priors when present",
    )
    p.add_argument(
        "--absolute-height-sigma-m",
        type=float,
        default=HEIGHT_ABSOLUTE_SIGMA_M,
        help="std-dev (m) for --absolute-height priors",
    )
    p.add_argument(
        "--absolute-height-dist-m",
        type=float,
        default=HEIGHT_ABSOLUTE_DIST_M,
        help="nearest-reference horizontal distance gate (m) for --absolute-height",
    )
    p.add_argument("--signal-type", type=str, default="GPS_L1_CA")
    p.add_argument("--constellation-type", type=int, default=1, help="Kaggle enum; GPS=1")
    p.add_argument("--weight-mode", choices=("sin2el", "cn0", "taroz_sn"), default="sin2el")
    p.add_argument(
        "--fgo-weight-mode",
        choices=("sin2el", "cn0", "taroz_sn", "same"),
        default="same",
        help="FGO-only weight model; 'same' uses --weight-mode.",
    )
    p.add_argument("--fgo-huber-k-pr", type=float, default=0.0)
    p.add_argument("--fgo-huber-k-doppler", type=float, default=0.0)
    p.add_argument("--fgo-huber-k-tdcp", type=float, default=0.0)
    p.add_argument("--position-source", choices=POSITION_SOURCES, default="baseline")
    p.add_argument("--chunk-epochs", type=int, default=0, help="if >0, solve FGO in chunks of this many epochs")
    p.add_argument(
        "--relative-height",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="apply loop-aware relative height smoothing to exported positions",
    )
    p.add_argument(
        "--position-offset",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="apply MATLAB-style phone position offset to exported positions",
    )
    p.add_argument(
        "--base-correction",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="subtract smoothed base-station pseudorange residuals when Base1/RINEX/nav inputs are ready",
    )
    p.add_argument(
        "--observation-mask",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="apply experimental MATLAB-style signal/status and pseudorange residual observation masks",
    )
    p.add_argument(
        "--observation-min-cn0-dbhz",
        type=float,
        default=OBS_MASK_MIN_CN0_DBHZ,
        help="C/N0 threshold used by --observation-mask",
    )
    p.add_argument(
        "--observation-min-elevation-deg",
        type=float,
        default=OBS_MASK_MIN_ELEVATION_DEG,
        help="elevation threshold used by --observation-mask",
    )
    p.add_argument(
        "--pseudorange-residual-mask-m",
        type=float,
        default=OBS_MASK_RESIDUAL_THRESHOLD_M,
        help="baseline residual threshold used by --observation-mask; <=0 disables residual masking",
    )
    p.add_argument(
        "--pseudorange-residual-mask-l5-m",
        type=float,
        default=OBS_MASK_RESIDUAL_THRESHOLD_L5_M,
        help="L5/E5 residual threshold used by --observation-mask in dual-frequency mode",
    )
    p.add_argument(
        "--doppler-residual-mask-mps",
        type=float,
        default=OBS_MASK_DOPPLER_RESIDUAL_THRESHOLD_MPS,
        help="Doppler residual threshold used by --observation-mask; <=0 disables Doppler residual masking",
    )
    p.add_argument(
        "--pseudorange-doppler-mask-m",
        type=float,
        default=OBS_MASK_PSEUDORANGE_DOPPLER_THRESHOLD_M,
        help="pseudorange-Doppler consistency threshold used by --observation-mask; <=0 disables this mask",
    )
    p.add_argument(
        "--matlab-residual-diagnostics-mask",
        type=Path,
        default=None,
        help="optional phone_data_residual_diagnostics.csv used to force bridge P/D/L factor availability",
    )
    p.add_argument(
        "--dual-frequency",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="experimental: include L1/E1 and L5/E5 observations as separate slots",
    )
    p.add_argument(
        "--graph-relative-height",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="add loop-aware ENU-up relative height factors inside VD-FGO (uses Kaggle WLS for loop detection)",
    )
    p.add_argument(
        "--relative-height-sigma-m",
        type=float,
        default=0.5,
        help="std-dev (m) for graph relative-height equality when --graph-relative-height is on",
    )
    p.add_argument("--vd", action=argparse.BooleanOptionalAction, default=True, help="use velocity-Doppler FGO")
    p.add_argument(
        "--multi-gnss",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use GPS + Galileo + QZSS with ISB estimation",
    )
    p.add_argument(
        "--tdcp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable ADR-derived TDCP factors when available",
    )
    p.add_argument(
        "--tdcp-consistency-threshold-m",
        type=float,
        default=1.5,
        help="reject TDCP pairs when ADR and Doppler disagree by more than this threshold",
    )
    p.add_argument(
        "--tdcp-weight-scale",
        type=float,
        default=DEFAULT_TDCP_WEIGHT_SCALE,
        help="multiply final TDCP weights by this factor; <=0 keeps TDCP arrays but disables their weight",
    )
    p.add_argument(
        "--tdcp-l5-weight-scale",
        type=float,
        default=1.0,
        help="multiply final L5-slot TDCP weights by this factor",
    )
    p.add_argument(
        "--tdcp-geometry-correction",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_TDCP_GEOMETRY_CORRECTION,
        help="subtract baseline satellite-range delta from TDCP measurements, approximating MATLAB resL differencing",
    )
    p.add_argument(
        "--gated-threshold",
        type=float,
        default=GATED_BASELINE_THRESHOLD_DEFAULT,
        help="baseline_mse_pr threshold for gated source fallback",
    )
    p.add_argument(
        "--gate-fgo-baseline-gap-p95-floor-m",
        type=float,
        default=None,
        help="Override FGO-vs-baseline gap p95 floor for gated candidate acceptance.",
    )
    p.add_argument(
        "--export-bridge-dir",
        type=Path,
        default=None,
        help="optional output directory for bridge_positions.csv and bridge_metrics.json",
    )
    argv = sys.argv[1:]
    args = p.parse_args(argv)

    config = BridgeConfig(
        motion_sigma_m=args.motion_sigma_m,
        factor_dt_max_s=args.factor_dt_max_s,
        fgo_tol=1e-7 if args.fgo_tol is None else args.fgo_tol,
        clock_drift_sigma_m=args.clock_drift_sigma_m,
        stop_velocity_sigma_mps=args.stop_velocity_sigma_mps,
        stop_position_sigma_m=args.stop_position_sigma_m,
        stop_attitude_sigma_rad=args.stop_attitude_sigma_rad,
        apply_imu_prior=args.imu_prior,
        imu_frame=args.imu_frame,
        imu_position_sigma_m=args.imu_position_sigma_m,
        imu_velocity_sigma_mps=args.imu_velocity_sigma_mps,
        imu_accel_bias_state=args.imu_accel_bias_state,
        imu_accel_bias_prior_sigma_mps2=args.imu_accel_bias_prior_sigma_mps2,
        imu_accel_bias_between_sigma_mps2=args.imu_accel_bias_between_sigma_mps2,
        apply_absolute_height=args.absolute_height,
        absolute_height_sigma_m=args.absolute_height_sigma_m,
        absolute_height_dist_m=args.absolute_height_dist_m,
        fgo_iters=args.fgo_iters,
        fgo_line_search=args.fgo_line_search,
        fgo_lm_damping=args.fgo_lm_damping,
        signal_type=args.signal_type,
        constellation_type=args.constellation_type,
        weight_mode=args.weight_mode,
        fgo_weight_mode=None if args.fgo_weight_mode == "same" else args.fgo_weight_mode,
        fgo_huber_k_pr=args.fgo_huber_k_pr,
        fgo_huber_k_doppler=args.fgo_huber_k_doppler,
        fgo_huber_k_tdcp=args.fgo_huber_k_tdcp,
        position_source=args.position_source,
        chunk_epochs=args.chunk_epochs,
        gated_baseline_threshold=args.gated_threshold,
        gate_fgo_baseline_gap_p95_floor_m=args.gate_fgo_baseline_gap_p95_floor_m,
        apply_relative_height=args.relative_height,
        apply_position_offset=args.position_offset,
        apply_base_correction=args.base_correction,
        apply_observation_mask=args.observation_mask,
        observation_min_cn0_dbhz=args.observation_min_cn0_dbhz,
        observation_min_elevation_deg=args.observation_min_elevation_deg,
        pseudorange_residual_mask_m=args.pseudorange_residual_mask_m,
        pseudorange_residual_mask_l5_m=args.pseudorange_residual_mask_l5_m,
        doppler_residual_mask_mps=args.doppler_residual_mask_mps,
        pseudorange_doppler_mask_m=args.pseudorange_doppler_mask_m,
        matlab_residual_diagnostics_mask_path=args.matlab_residual_diagnostics_mask,
        dual_frequency=args.dual_frequency,
        graph_relative_height=args.graph_relative_height,
        relative_height_sigma_m=args.relative_height_sigma_m,
        use_vd=args.vd,
        multi_gnss=args.multi_gnss,
        tdcp_enabled=args.tdcp,
        tdcp_consistency_threshold_m=args.tdcp_consistency_threshold_m,
        tdcp_weight_scale=args.tdcp_weight_scale,
        tdcp_l5_weight_scale=args.tdcp_l5_weight_scale,
        tdcp_geometry_correction=args.tdcp_geometry_correction,
        taroz_fgo_candidate_enabled=args.taroz_fgo_candidates,
        taroz_fgo_candidate_sources=tuple(
            item.strip()
            for item in args.taroz_fgo_candidate_sources.split(",")
            if item.strip()
        ),
        taroz_fgo_seed_state_csv=args.taroz_fgo_seed_state_csv,
        taroz_pose_bias_seed_state_csv=args.taroz_pose_bias_seed_state_csv,
        taroz_factor_mask_csv=args.taroz_factor_mask_csv,
        taroz_imu_factor_mask_csv=args.taroz_imu_factor_mask_csv,
        taroz_imu_preintegration_csv=args.taroz_imu_preintegration_csv,
    )
    taroz_gnss_like = bool(args.taroz_gnss_only or args.taroz_factor_parity)
    if taroz_gnss_like and (args.taroz_fgo or args.taroz_marupaku):
        raise SystemExit("--taroz-gnss-only/--taroz-factor-parity and Taroz full presets are mutually exclusive")
    if args.taroz_fgo and args.taroz_marupaku:
        raise SystemExit("--taroz-fgo and --taroz-marupaku are mutually exclusive")
    if args.taroz_full_init_pass and not args.taroz_marupaku:
        raise SystemExit("--taroz-full-init-pass requires --taroz-marupaku")
    if taroz_gnss_like:
        config = apply_taroz_gnss_only_preset(config)
    elif args.taroz_marupaku:
        config = apply_taroz_marupaku_preset(config)
        if args.taroz_full_init_pass:
            config = apply_taroz_full_init_pass_preset(config)
    elif args.taroz_fgo:
        config = apply_taroz_fgo_preset(config)
    config = _apply_explicit_cli_overrides(config, args, argv)
    if args.taroz_marupaku and config.taroz_imu_factor_mask_csv is None and config.taroz_fgo_seed_state_csv is not None:
        seed_path = Path(config.taroz_fgo_seed_state_csv)
        candidate = (seed_path if seed_path.is_dir() else seed_path.parent) / "phone_data_imu_factor_mask.csv"
        if candidate.is_file():
            config = replace(config, taroz_imu_factor_mask_csv=candidate)
    if args.taroz_factor_parity:
        factor_mask_csv = config.taroz_factor_mask_csv
        if factor_mask_csv is None and config.taroz_fgo_seed_state_csv is not None:
            seed_path = Path(config.taroz_fgo_seed_state_csv)
            factor_mask_csv = (seed_path if seed_path.is_dir() else seed_path.parent) / "phone_data_gnss_factor_mask.csv"
        config = replace(
            config,
            apply_observation_mask=False,
            tdcp_consistency_threshold_m=1.0e9,
            fgo_tol=1.0e-10 if args.fgo_tol is None else config.fgo_tol,
            taroz_factor_mask_csv=factor_mask_csv,
        )

    result = validate_raw_gsdc2023_trip(
        args.data_root,
        args.trip,
        max_epochs=args.max_epochs,
        start_epoch=args.start_epoch,
        config=config,
    )
    for line in result.summary_lines():
        print(line)
    if args.export_bridge_dir is not None:
        _export_bridge_outputs(args.export_bridge_dir, result)
        print(f"  bridge out  : {args.export_bridge_dir}")


if __name__ == "__main__":
    main()
