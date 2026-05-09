#!/usr/bin/env python3
"""CLI wrapper for the GSDC2023 raw-data bridge."""

from __future__ import annotations

import argparse
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


_DEFAULT_ROOT = DEFAULT_ROOT


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
    p.add_argument("--clock-drift-sigma-m", type=float, default=1.0)
    p.add_argument("--stop-velocity-sigma-mps", type=float, default=0.0)
    p.add_argument("--stop-position-sigma-m", type=float, default=0.0)
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
    p.add_argument("--weight-mode", choices=("sin2el", "cn0"), default="sin2el")
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
        "--export-bridge-dir",
        type=Path,
        default=None,
        help="optional output directory for bridge_positions.csv and bridge_metrics.json",
    )
    args = p.parse_args()

    result = validate_raw_gsdc2023_trip(
        args.data_root,
        args.trip,
        max_epochs=args.max_epochs,
        start_epoch=args.start_epoch,
        config=BridgeConfig(
            motion_sigma_m=args.motion_sigma_m,
            factor_dt_max_s=args.factor_dt_max_s,
            clock_drift_sigma_m=args.clock_drift_sigma_m,
            stop_velocity_sigma_mps=args.stop_velocity_sigma_mps,
            stop_position_sigma_m=args.stop_position_sigma_m,
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
            signal_type=args.signal_type,
            constellation_type=args.constellation_type,
            weight_mode=args.weight_mode,
            position_source=args.position_source,
            chunk_epochs=args.chunk_epochs,
            gated_baseline_threshold=args.gated_threshold,
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
            tdcp_geometry_correction=args.tdcp_geometry_correction,
        ),
    )
    for line in result.summary_lines():
        print(line)
    if args.export_bridge_dir is not None:
        _export_bridge_outputs(args.export_bridge_dir, result)
        print(f"  bridge out  : {args.export_bridge_dir}")


if __name__ == "__main__":
    main()
