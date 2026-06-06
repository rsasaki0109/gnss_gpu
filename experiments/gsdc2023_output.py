"""Output tables and metrics payloads for GSDC2023 raw bridge experiments."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.evaluate import ecef_to_lla
from experiments.gsdc2023_height_constraints import HEIGHT_ABSOLUTE_DIST_M, HEIGHT_ABSOLUTE_SIGMA_M
from experiments.gsdc2023_imu import (
    IMU_ACCEL_BIAS_BETWEEN_SIGMA_MPS2,
    IMU_ACCEL_BIAS_PRIOR_SIGMA_MPS2,
)
from experiments.gsdc2023_tdcp import DEFAULT_TDCP_WEIGHT_SCALE


FACTOR_DT_MAX_S = 1.5
CT_RBPF_FGO_SOURCE = "fgo_ct_rbpf"
DD_CARRIER_FGO_SOURCE = "fgo_dd_carrier"
TDCP_SCALE_FGO_SOURCE = "fgo_tdcp_scale"
TAROZ_WEIGHTS_FGO_SOURCE = "fgo_taroz_weights"
TAROZ_PR_FGO_SOURCE = "fgo_taroz_pr"
TAROZ_PR_D_L_FGO_SOURCE = "fgo_taroz_pr_d_l"
TAROZ_FGO_CANDIDATE_SOURCES = (
    TAROZ_WEIGHTS_FGO_SOURCE,
    TAROZ_PR_FGO_SOURCE,
    TAROZ_PR_D_L_FGO_SOURCE,
)
POSITION_SOURCES = ("baseline", "raw_wls", "fgo", CT_RBPF_FGO_SOURCE, DD_CARRIER_FGO_SOURCE, "auto", "gated")


def validate_position_source(position_source: str) -> str:
    if position_source not in POSITION_SOURCES:
        raise ValueError(f"unsupported position source: {position_source}")
    return position_source


def metrics_summary(metrics: dict | None) -> dict | None:
    if metrics is None:
        return None
    return {
        "rms_2d_m": float(metrics["rms_2d"]),
        "rms_3d_m": float(metrics["rms_3d"]),
        "mean_2d_m": float(metrics["mean_2d"]),
        "mean_3d_m": float(metrics["mean_3d"]),
        "std_2d_m": float(metrics["std_2d"]),
        "p50_m": float(metrics["p50"]),
        "p67_m": float(metrics["p67"]),
        "p95_m": float(metrics["p95"]),
        "max_2d_m": float(metrics["max_2d"]),
        "n_epochs": int(metrics["n_epochs"]),
    }


def score_from_metrics(metrics: dict | None) -> float | None:
    if metrics is None:
        return None
    return 0.5 * (float(metrics["p50"]) + float(metrics["p95"]))


def ecef_to_llh_deg(ecef_xyz: np.ndarray) -> np.ndarray:
    ecef_xyz = np.asarray(ecef_xyz, dtype=np.float64).reshape(-1, 3)
    llh_deg = np.zeros((ecef_xyz.shape[0], 3), dtype=np.float64)
    for i, (x, y, z) in enumerate(ecef_xyz):
        lat_rad, lon_rad, alt_m = ecef_to_lla(float(x), float(y), float(z))
        llh_deg[i] = [np.rad2deg(lat_rad), np.rad2deg(lon_rad), alt_m]
    return llh_deg


def format_metrics_line(label: str, metrics: dict | None) -> str:
    if metrics is None:
        return f"  {label:14s} unavailable"
    return (
        f"  {label:14s} "
        f"RMS2D={metrics['rms_2d']:.3f}m  "
        f"P50={metrics['p50']:.3f}m  "
        f"P95={metrics['p95']:.3f}m  "
        f"RMS3D={metrics['rms_3d']:.3f}m"
    )


@dataclass
class BridgeResult:
    trip: str
    signal_type: str
    weight_mode: str
    selected_source_mode: str
    times_ms: np.ndarray
    kaggle_wls: np.ndarray
    raw_wls: np.ndarray
    fgo_state: np.ndarray
    selected_state: np.ndarray
    selected_sources: np.ndarray
    truth: np.ndarray | None
    max_sats: int
    fgo_iters: int
    failed_chunks: int
    vd_seed_guard_skipped_segments: int
    vd_seed_guard_skipped_epochs: int
    selected_mse_pr: float
    baseline_mse_pr: float
    raw_wls_mse_pr: float
    fgo_mse_pr: float
    selected_source_counts: dict[str, int]
    metrics_selected: dict | None
    metrics_kaggle: dict | None
    metrics_raw_wls: dict | None
    metrics_fgo: dict | None
    fgo_tol: float = 1e-7
    vd_seed_guard_records: list[dict[str, object]] | None = None
    chunk_selection_records: list[dict[str, object]] | None = None
    parity_audit: dict | None = None
    factor_dt_max_s: float = FACTOR_DT_MAX_S
    factor_dt_gap_count: int = 0
    stop_velocity_sigma_mps: float = 0.0
    stop_position_sigma_m: float = 0.0
    stop_attitude_sigma_rad: float = 0.0
    stop_velocity_huber_k: float = 0.0
    stop_position_huber_k: float = 0.0
    imu_prior_applied: bool = False
    imu_prior_interval_count: int = 0
    imu_frame: str = "body"
    imu_sample_dt_mode: str = "bounded"
    imu_gravity_applied: bool = False
    imu_position_sigma_m: float = 25.0
    imu_velocity_sigma_mps: float = 5.0
    imu_attitude_state_applied: bool = False
    imu_attitude_sigma_rad: float = 0.0
    imu_diagonal_covariance_applied: bool = False
    imu_preintegration_covariance_applied: bool = False
    imu_preintegration_delta_t_applied: bool = False
    imu_preintegration_bias_jacobian_applied: bool = False
    imu_factor_use_next_bias_applied: bool = False
    imu_delta_pv_gyro_bias_correction_applied: bool = False
    imu_bias_between_sample_count_scaling_applied: bool = False
    imu_accel_bias_state_applied: bool = False
    imu_accel_bias_prior_sigma_mps2: float = IMU_ACCEL_BIAS_PRIOR_SIGMA_MPS2
    imu_accel_bias_between_sigma_mps2: float = IMU_ACCEL_BIAS_BETWEEN_SIGMA_MPS2
    imu_gyro_bias_state_applied: bool = False
    imu_gyro_bias_prior_sigma_radps: float = 0.0
    imu_gyro_bias_between_sigma_radps: float = 0.0
    taroz_imu_noise_enabled: bool = False
    imu_acc_sigma_mps2_sqrt_hz: float = 0.0
    imu_gyro_sigma_radps_sqrt_hz: float = 0.0
    imu_acc_sync_coefficient: float = 1.0
    imu_gyro_sync_coefficient: float = 1.0
    imu_effective_acc_sigma_mps2_sqrt_hz: float = 0.0
    imu_effective_gyro_sigma_radps_sqrt_hz: float = 0.0
    imu_acc_bias_mean_norm_mps2: float = float("nan")
    imu_gyro_bias_mean_norm_radps: float = float("nan")
    absolute_height_applied: bool = False
    absolute_height_ref_count: int = 0
    absolute_height_sigma_m: float = HEIGHT_ABSOLUTE_SIGMA_M
    absolute_height_dist_m: float = HEIGHT_ABSOLUTE_DIST_M
    absolute_height_huber_k: float = 0.0
    relative_height_applied: bool = False
    relative_height_huber_k: float = 0.0
    position_offset_applied: bool = False
    base_correction_applied: bool = False
    base_correction_count: int = 0
    observation_mask_applied: bool = False
    observation_mask_count: int = 0
    residual_mask_count: int = 0
    doppler_residual_mask_count: int = 0
    pseudorange_doppler_mask_count: int = 0
    tdcp_consistency_mask_count: int = 0
    tdcp_weight_scale: float = DEFAULT_TDCP_WEIGHT_SCALE
    tdcp_geometry_correction_applied: bool = False
    tdcp_geometry_correction_count: int = 0
    tdcp_scale_candidate_enabled: bool = False
    tdcp_scale_candidate_weight_scale: float = 1.0e-7
    taroz_qzss_other_clock_enabled: bool = False
    fgo_weight_mode: str = "sin2el"
    fgo_robust_kernel: str = "huber"
    fgo_huber_k_pr: float = 0.0
    fgo_huber_k_doppler: float = 0.0
    fgo_huber_k_tdcp: float = 0.0
    fgo_fixed_linearization: bool = False
    effective_trip_type: str | None = None
    effective_motion_sigma_m: float = 0.0
    effective_fgo_huber_k_pr: float = 0.0
    effective_fgo_huber_k_doppler: float = 0.0
    effective_fgo_huber_k_tdcp: float = 0.0
    per_type_kernel_enabled: bool = False
    per_type_kernel_huber_enabled: bool = True
    per_type_kernel_motion_enabled: bool = False
    clock_drift_sigma_m: float = 1.0
    clock_use_average_drift: bool | None = None
    fgo_raw_wls_proxy_rescue_enabled: bool = False
    fgo_raw_wls_proxy_rescue_mse_ratio_max: float = 1.15
    fgo_raw_wls_proxy_rescue_gap_step_p95_ratio_max: float = 1.25
    fgo_raw_wls_proxy_rescue_quality_delta_max: float = -0.20
    fgo_raw_wls_proxy_rescue_mse_delta_vs_baseline_max: float = 0.0
    taroz_fgo_candidate_enabled: bool = False
    taroz_fgo_candidate_sources: tuple[str, ...] = TAROZ_FGO_CANDIDATE_SOURCES
    dual_frequency: bool = False
    graph_relative_height: bool = False
    ct_rbpf_fgo_enabled: bool = False
    ct_rbpf_motion_sigma_m: float = 0.2
    dd_carrier_fgo_enabled: bool = False
    dd_carrier_base_obs_template: str | None = None
    dd_carrier_require_base_obs_template: bool = False
    dd_carrier_accepted_anchor_epochs: int = 0
    dd_carrier_dd_epochs: int = 0
    dd_carrier_base_snapped_epochs: int = 0
    dd_carrier_dd_pairs_mean: float = 0.0
    fgo_vd_state: np.ndarray | None = None

    @property
    def n_epochs(self) -> int:
        return int(self.times_ms.size)

    def positions_table(self) -> pd.DataFrame:
        selected_llh = ecef_to_llh_deg(self.selected_state[:, :3])
        kaggle_llh = ecef_to_llh_deg(self.kaggle_wls)
        raw_wls_llh = ecef_to_llh_deg(self.raw_wls[:, :3])
        fgo_llh = ecef_to_llh_deg(self.fgo_state[:, :3])
        if self.truth is not None:
            truth_llh = ecef_to_llh_deg(self.truth)
        else:
            truth_llh = np.full((self.times_ms.size, 3), np.nan, dtype=np.float64)
        return pd.DataFrame(
            {
                "UnixTimeMillis": self.times_ms.astype(np.int64),
                "SelectedSource": self.selected_sources.astype(str),
                "BaselineLatitudeDegrees": kaggle_llh[:, 0],
                "BaselineLongitudeDegrees": kaggle_llh[:, 1],
                "BaselineAltitudeMeters": kaggle_llh[:, 2],
                "RawWlsLatitudeDegrees": raw_wls_llh[:, 0],
                "RawWlsLongitudeDegrees": raw_wls_llh[:, 1],
                "RawWlsAltitudeMeters": raw_wls_llh[:, 2],
                "FgoLatitudeDegrees": fgo_llh[:, 0],
                "FgoLongitudeDegrees": fgo_llh[:, 1],
                "FgoAltitudeMeters": fgo_llh[:, 2],
                "LatitudeDegrees": selected_llh[:, 0],
                "LongitudeDegrees": selected_llh[:, 1],
                "AltitudeMeters": selected_llh[:, 2],
                "GroundTruthLatitudeDegrees": truth_llh[:, 0],
                "GroundTruthLongitudeDegrees": truth_llh[:, 1],
                "GroundTruthAltitudeMeters": truth_llh[:, 2],
            },
        )

    @staticmethod
    def _ecef_columns(prefix: str, xyz: np.ndarray) -> dict[str, np.ndarray]:
        arr = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
        return {
            f"{prefix}EcefXMeters": arr[:, 0],
            f"{prefix}EcefYMeters": arr[:, 1],
            f"{prefix}EcefZMeters": arr[:, 2],
        }

    @staticmethod
    def _extra_state_columns(prefix: str, state: np.ndarray, n_clock: int) -> dict[str, np.ndarray]:
        arr = np.asarray(state, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] <= 3:
            return {}
        out: dict[str, np.ndarray] = {}
        n_extra = arr.shape[1] - 3
        if n_extra <= n_clock:
            for clock_idx in range(n_extra):
                out[f"{prefix}ClockBiasMeters{clock_idx}"] = arr[:, 3 + clock_idx]
            return out

        out[f"{prefix}VelocityXMps"] = arr[:, 3]
        out[f"{prefix}VelocityYMps"] = arr[:, 4]
        out[f"{prefix}VelocityZMps"] = arr[:, 5]
        clock_start = 6
        clock_end = min(clock_start + n_clock, arr.shape[1])
        for clock_idx, col_idx in enumerate(range(clock_start, clock_end)):
            out[f"{prefix}ClockBiasMeters{clock_idx}"] = arr[:, col_idx]
        if clock_end < arr.shape[1]:
            out[f"{prefix}ClockDriftMps"] = arr[:, clock_end]
        for extra_idx, col_idx in enumerate(range(clock_end + 1, arr.shape[1])):
            out[f"{prefix}StateExtra{extra_idx}"] = arr[:, col_idx]
        return out

    def states_table(self) -> pd.DataFrame:
        """Return ECEF optimizer states for MATLAB/Taroz parity audits."""

        n_clock = int(max(self.raw_wls.shape[1] - 3, 1)) if self.raw_wls.ndim == 2 else 1
        if self.truth is not None:
            truth_xyz = np.asarray(self.truth, dtype=np.float64).reshape(-1, 3)
        else:
            truth_xyz = np.full((self.times_ms.size, 3), np.nan, dtype=np.float64)
        data: dict[str, np.ndarray] = {
            "UnixTimeMillis": self.times_ms.astype(np.int64),
            "SelectedSource": self.selected_sources.astype(str),
        }
        data.update(self._ecef_columns("Baseline", self.kaggle_wls))
        data.update(self._ecef_columns("RawWls", self.raw_wls[:, :3]))
        data.update(self._extra_state_columns("RawWls", self.raw_wls, n_clock))
        data.update(self._ecef_columns("Fgo", self.fgo_state[:, :3]))
        data.update(self._extra_state_columns("Fgo", self.fgo_state, n_clock))
        data.update(self._ecef_columns("Selected", self.selected_state[:, :3]))
        data.update(self._extra_state_columns("Selected", self.selected_state, n_clock))
        data.update(self._ecef_columns("GroundTruth", truth_xyz))
        return pd.DataFrame(data)

    def fgo_vd_state_table(self) -> pd.DataFrame | None:
        """Return the raw internal VD solver state before clock refit, if available."""

        if self.fgo_vd_state is None:
            return None
        state = np.asarray(self.fgo_vd_state, dtype=np.float64)
        if state.ndim != 2 or state.shape[0] != self.times_ms.size:
            return None
        n_clock = int(max(self.raw_wls.shape[1] - 3, 1)) if self.raw_wls.ndim == 2 else 1
        data: dict[str, np.ndarray] = {
            "UnixTimeMillis": self.times_ms.astype(np.int64),
        }
        data.update(self._ecef_columns("FgoVd", state[:, :3]))
        data.update(self._extra_state_columns("FgoVd", state, n_clock))
        return pd.DataFrame(data)

    def metrics_payload(self) -> dict:
        vd_guard_records = self.vd_seed_guard_records or []
        vd_guard_reason_counts: dict[str, int] = {}
        for record in vd_guard_records:
            reason = str(record.get("reject_reason", ""))
            if reason:
                vd_guard_reason_counts[reason] = vd_guard_reason_counts.get(reason, 0) + 1
        payload = {
            "trip": self.trip,
            "signal_type": self.signal_type,
            "weight_mode": self.weight_mode,
            "selected_source_mode": self.selected_source_mode,
            "n_epochs": self.n_epochs,
            "max_sats": int(self.max_sats),
            "n_clock": int(max(self.raw_wls.shape[1] - 3, 1)),
            "fgo_iters": int(self.fgo_iters),
            "fgo_tol": float(self.fgo_tol),
            "failed_chunks": int(self.failed_chunks),
            "vd_seed_guard_skipped_segments": int(self.vd_seed_guard_skipped_segments),
            "vd_seed_guard_skipped_epochs": int(self.vd_seed_guard_skipped_epochs),
            "vd_seed_guard_records": vd_guard_records,
            "vd_seed_guard_reject_reasons": vd_guard_reason_counts,
            "mse_pr": float(self.selected_mse_pr),
            "selected_mse_pr": float(self.selected_mse_pr),
            "baseline_mse_pr": float(self.baseline_mse_pr),
            "raw_wls_mse_pr": float(self.raw_wls_mse_pr),
            "fgo_mse_pr": float(self.fgo_mse_pr),
            "selected_source_counts": {k: int(v) for k, v in self.selected_source_counts.items()},
            "selected_score_m": score_from_metrics(self.metrics_selected),
            "kaggle_wls_score_m": score_from_metrics(self.metrics_kaggle),
            "raw_wls_score_m": score_from_metrics(self.metrics_raw_wls),
            "fgo_score_m": score_from_metrics(self.metrics_fgo),
            "selected_metrics": metrics_summary(self.metrics_selected),
            "kaggle_wls_metrics": metrics_summary(self.metrics_kaggle),
            "raw_wls_metrics": metrics_summary(self.metrics_raw_wls),
            "fgo_metrics": metrics_summary(self.metrics_fgo),
            "factor_dt_max_s": float(self.factor_dt_max_s),
            "factor_dt_gap_count": int(self.factor_dt_gap_count),
            "stop_velocity_sigma_mps": float(self.stop_velocity_sigma_mps),
            "stop_position_sigma_m": float(self.stop_position_sigma_m),
            "stop_attitude_sigma_rad": float(self.stop_attitude_sigma_rad),
            "stop_velocity_huber_k": float(self.stop_velocity_huber_k),
            "stop_position_huber_k": float(self.stop_position_huber_k),
            "imu_prior_applied": bool(self.imu_prior_applied),
            "imu_prior_interval_count": int(self.imu_prior_interval_count),
            "imu_frame": str(self.imu_frame),
            "imu_sample_dt_mode": str(self.imu_sample_dt_mode),
            "imu_gravity_applied": bool(self.imu_gravity_applied),
            "imu_position_sigma_m": float(self.imu_position_sigma_m),
            "imu_velocity_sigma_mps": float(self.imu_velocity_sigma_mps),
            "imu_attitude_state_applied": bool(self.imu_attitude_state_applied),
            "imu_attitude_sigma_rad": float(self.imu_attitude_sigma_rad),
            "imu_diagonal_covariance_applied": bool(self.imu_diagonal_covariance_applied),
            "imu_preintegration_covariance_applied": bool(self.imu_preintegration_covariance_applied),
            "imu_preintegration_delta_t_applied": bool(self.imu_preintegration_delta_t_applied),
            "imu_preintegration_bias_jacobian_applied": bool(
                self.imu_preintegration_bias_jacobian_applied
            ),
            "imu_factor_use_next_bias_applied": bool(self.imu_factor_use_next_bias_applied),
            "imu_delta_pv_gyro_bias_correction_applied": bool(
                self.imu_delta_pv_gyro_bias_correction_applied
            ),
            "imu_bias_between_sample_count_scaling_applied": bool(
                self.imu_bias_between_sample_count_scaling_applied
            ),
            "imu_accel_bias_state_applied": bool(self.imu_accel_bias_state_applied),
            "imu_accel_bias_prior_sigma_mps2": float(self.imu_accel_bias_prior_sigma_mps2),
            "imu_accel_bias_between_sigma_mps2": float(self.imu_accel_bias_between_sigma_mps2),
            "imu_gyro_bias_state_applied": bool(self.imu_gyro_bias_state_applied),
            "imu_gyro_bias_prior_sigma_radps": float(self.imu_gyro_bias_prior_sigma_radps),
            "imu_gyro_bias_between_sigma_radps": float(self.imu_gyro_bias_between_sigma_radps),
            "taroz_imu_noise_enabled": bool(self.taroz_imu_noise_enabled),
            "imu_acc_sigma_mps2_sqrt_hz": float(self.imu_acc_sigma_mps2_sqrt_hz),
            "imu_gyro_sigma_radps_sqrt_hz": float(self.imu_gyro_sigma_radps_sqrt_hz),
            "imu_acc_sync_coefficient": float(self.imu_acc_sync_coefficient),
            "imu_gyro_sync_coefficient": float(self.imu_gyro_sync_coefficient),
            "imu_effective_acc_sigma_mps2_sqrt_hz": float(self.imu_effective_acc_sigma_mps2_sqrt_hz),
            "imu_effective_gyro_sigma_radps_sqrt_hz": float(self.imu_effective_gyro_sigma_radps_sqrt_hz),
            "imu_acc_bias_mean_norm_mps2": float(self.imu_acc_bias_mean_norm_mps2),
            "imu_gyro_bias_mean_norm_radps": float(self.imu_gyro_bias_mean_norm_radps),
            "absolute_height_applied": bool(self.absolute_height_applied),
            "absolute_height_ref_count": int(self.absolute_height_ref_count),
            "absolute_height_sigma_m": float(self.absolute_height_sigma_m),
            "absolute_height_dist_m": float(self.absolute_height_dist_m),
            "absolute_height_huber_k": float(self.absolute_height_huber_k),
            "relative_height_applied": bool(self.relative_height_applied),
            "relative_height_huber_k": float(self.relative_height_huber_k),
            "position_offset_applied": bool(self.position_offset_applied),
            "base_correction_applied": bool(self.base_correction_applied),
            "base_correction_count": int(self.base_correction_count),
            "observation_mask_applied": bool(self.observation_mask_applied),
            "observation_mask_count": int(self.observation_mask_count),
            "residual_mask_count": int(self.residual_mask_count),
            "doppler_residual_mask_count": int(self.doppler_residual_mask_count),
            "pseudorange_doppler_mask_count": int(self.pseudorange_doppler_mask_count),
            "tdcp_consistency_mask_count": int(self.tdcp_consistency_mask_count),
            "tdcp_weight_scale": float(self.tdcp_weight_scale),
            "tdcp_geometry_correction_applied": bool(self.tdcp_geometry_correction_applied),
            "tdcp_geometry_correction_count": int(self.tdcp_geometry_correction_count),
            "tdcp_scale_candidate_enabled": bool(self.tdcp_scale_candidate_enabled),
            "tdcp_scale_candidate_weight_scale": float(self.tdcp_scale_candidate_weight_scale),
            "taroz_qzss_other_clock_enabled": bool(self.taroz_qzss_other_clock_enabled),
            "fgo_weight_mode": str(self.fgo_weight_mode),
            "fgo_robust_kernel": str(self.fgo_robust_kernel),
            "fgo_huber_k_pr": float(self.fgo_huber_k_pr),
            "fgo_huber_k_doppler": float(self.fgo_huber_k_doppler),
            "fgo_huber_k_tdcp": float(self.fgo_huber_k_tdcp),
            "fgo_fixed_linearization": bool(self.fgo_fixed_linearization),
            "effective_trip_type": self.effective_trip_type,
            "effective_motion_sigma_m": float(self.effective_motion_sigma_m),
            "effective_fgo_huber_k_pr": float(self.effective_fgo_huber_k_pr),
            "effective_fgo_huber_k_doppler": float(self.effective_fgo_huber_k_doppler),
            "effective_fgo_huber_k_tdcp": float(self.effective_fgo_huber_k_tdcp),
            "per_type_kernel_enabled": bool(self.per_type_kernel_enabled),
            "per_type_kernel_huber_enabled": bool(self.per_type_kernel_huber_enabled),
            "per_type_kernel_motion_enabled": bool(self.per_type_kernel_motion_enabled),
            "clock_drift_sigma_m": float(self.clock_drift_sigma_m),
            "clock_use_average_drift": self.clock_use_average_drift,
            "fgo_raw_wls_proxy_rescue_enabled": bool(self.fgo_raw_wls_proxy_rescue_enabled),
            "fgo_raw_wls_proxy_rescue_mse_ratio_max": float(self.fgo_raw_wls_proxy_rescue_mse_ratio_max),
            "fgo_raw_wls_proxy_rescue_gap_step_p95_ratio_max": float(self.fgo_raw_wls_proxy_rescue_gap_step_p95_ratio_max),
            "fgo_raw_wls_proxy_rescue_quality_delta_max": float(self.fgo_raw_wls_proxy_rescue_quality_delta_max),
            "fgo_raw_wls_proxy_rescue_mse_delta_vs_baseline_max": float(
                self.fgo_raw_wls_proxy_rescue_mse_delta_vs_baseline_max,
            ),
            "taroz_fgo_candidate_enabled": bool(self.taroz_fgo_candidate_enabled),
            "taroz_fgo_candidate_sources": list(self.taroz_fgo_candidate_sources),
            "dual_frequency": bool(self.dual_frequency),
            "graph_relative_height": bool(self.graph_relative_height),
            "ct_rbpf_fgo_enabled": bool(self.ct_rbpf_fgo_enabled),
            "ct_rbpf_motion_sigma_m": float(self.ct_rbpf_motion_sigma_m),
            "dd_carrier_fgo_enabled": bool(self.dd_carrier_fgo_enabled),
            "dd_carrier_base_obs_template": self.dd_carrier_base_obs_template,
            "dd_carrier_require_base_obs_template": bool(self.dd_carrier_require_base_obs_template),
            "dd_carrier_accepted_anchor_epochs": int(self.dd_carrier_accepted_anchor_epochs),
            "dd_carrier_dd_epochs": int(self.dd_carrier_dd_epochs),
            "dd_carrier_base_snapped_epochs": int(self.dd_carrier_base_snapped_epochs),
            "dd_carrier_dd_pairs_mean": float(self.dd_carrier_dd_pairs_mean),
        }
        if self.chunk_selection_records is not None:
            payload["chunk_selection_records"] = self.chunk_selection_records
        if self.parity_audit is not None:
            payload["parity_audit"] = self.parity_audit
        return payload

    def summary_lines(self) -> list[str]:
        lines = [
            f"GSDC2023 raw validation: {self.trip}",
            f"  epochs      : {self.n_epochs}",
            f"  max sats/ep : {self.max_sats}",
            f"  signal      : {self.signal_type}",
            f"  weights      : {self.weight_mode}",
            f"  FGO iters   : {self.fgo_iters}",
            f"  output source: {self.selected_source_mode}",
            f"  wMSE pr     : {self.selected_mse_pr:.4f} (selected)",
            "  source mix  : "
            + ", ".join(f"{name}={count}" for name, count in self.selected_source_counts.items() if count > 0),
            (
                f"  candidate MSE: baseline={self.baseline_mse_pr:.4f} "
                f"raw={self.raw_wls_mse_pr:.4f} fgo={self.fgo_mse_pr:.4f}"
            ),
        ]
        if self.failed_chunks > 0:
            lines.append(f"  failed chunks: {self.failed_chunks} (raw WLS fallback)")
        if self.vd_seed_guard_skipped_segments > 0 or self.vd_seed_guard_skipped_epochs > 0:
            reason_counts: dict[str, int] = {}
            for record in self.vd_seed_guard_records or []:
                reason = str(record.get("reject_reason", ""))
                if reason:
                    reason_counts[reason] = reason_counts.get(reason, 0) + 1
            reason_text = ""
            if reason_counts:
                reason_text = " reasons=" + ",".join(
                    f"{reason}:{count}" for reason, count in sorted(reason_counts.items())
                )
            lines.append(
                f"  vd guard    : skipped_segments={self.vd_seed_guard_skipped_segments} "
                f"skipped_epochs={self.vd_seed_guard_skipped_epochs}{reason_text} (raw-backed fgo)"
            )
        if self.factor_dt_gap_count > 0:
            lines.append(
                f"  factor gaps : skipped={self.factor_dt_gap_count} "
                f"dt_max={self.factor_dt_max_s:.3f}s"
            )
        if (
            self.stop_velocity_sigma_mps > 0.0
            or self.stop_position_sigma_m > 0.0
            or self.stop_attitude_sigma_rad > 0.0
        ):
            lines.append(
                f"  stop factors : vel_sigma={self.stop_velocity_sigma_mps:.3f}m/s "
                f"pose_sigma={self.stop_position_sigma_m:.3f}m "
                f"att_sigma={self.stop_attitude_sigma_rad:.6f}rad"
            )
        if self.imu_prior_applied:
            imu_line = (
                f"  imu prior    : intervals={self.imu_prior_interval_count} "
                f"frame={self.imu_frame} sample_dt={self.imu_sample_dt_mode} "
                f"pos_sigma={self.imu_position_sigma_m:.3f}m "
                f"vel_sigma={self.imu_velocity_sigma_mps:.3f}m/s"
            )
            if np.isfinite(self.imu_acc_bias_mean_norm_mps2) or np.isfinite(self.imu_gyro_bias_mean_norm_radps):
                imu_line += (
                    f" acc_bias={self.imu_acc_bias_mean_norm_mps2:.4g}m/s^2"
                    f" gyro_bias={self.imu_gyro_bias_mean_norm_radps:.4g}rad/s"
                )
            if self.imu_accel_bias_state_applied:
                imu_line += (
                    f" accel_bias_state=on"
                    f" prior_sigma={self.imu_accel_bias_prior_sigma_mps2:.3g}m/s^2"
                    f" between_sigma={self.imu_accel_bias_between_sigma_mps2:.3g}m/s^2"
                )
            if self.imu_attitude_state_applied:
                imu_line += (
                    f" attitude_state=on"
                    f" attitude_sigma={self.imu_attitude_sigma_rad:.3g}rad"
                )
            if self.imu_diagonal_covariance_applied:
                imu_line += " diag_cov=on"
            if self.imu_preintegration_covariance_applied:
                imu_line += " preint_cov=on"
            if self.imu_preintegration_delta_t_applied:
                imu_line += " preint_dt=on"
            if self.imu_gravity_applied:
                imu_line += " body_gravity=on"
            if self.imu_preintegration_bias_jacobian_applied:
                imu_line += " preint_bias_jac=on"
            if self.imu_factor_use_next_bias_applied:
                imu_line += " imu_bias_epoch=next"
            if self.imu_delta_pv_gyro_bias_correction_applied:
                imu_line += " gyro_pv_corr=on"
            if self.imu_bias_between_sample_count_scaling_applied:
                imu_line += " bias_count_scale=on"
            if self.imu_gyro_bias_state_applied:
                imu_line += (
                    f" gyro_bias_state=on"
                    f" gyro_between_sigma={self.imu_gyro_bias_between_sigma_radps:.3g}rad/s"
                )
            if self.taroz_imu_noise_enabled:
                imu_line += (
                    f" acc_sigma={self.imu_acc_sigma_mps2_sqrt_hz:.3g}"
                    f" gyro_sigma={self.imu_gyro_sigma_radps_sqrt_hz:.3g}"
                )
            lines.append(imu_line)
        if self.absolute_height_applied:
            lines.append(
                f"  abs height  : refs={self.absolute_height_ref_count} "
                f"sigma={self.absolute_height_sigma_m:.3f}m "
                f"dist={self.absolute_height_dist_m:.1f}m"
            )
        if self.relative_height_applied:
            lines.append("  rel height  : loop-aware up smoothing enabled")
        if self.graph_relative_height:
            lines.append("  rel height  : graph factor (ENU-up loop closure) enabled")
        if self.position_offset_applied:
            lines.append("  pos offset  : phone heuristic enabled")
        if self.base_correction_applied:
            lines.append(f"  base corr   : pseudorange residual correction n={self.base_correction_count}")
        if self.observation_mask_applied:
            lines.append(
                f"  obs mask    : raw={self.observation_mask_count} "
                f"pr_res={self.residual_mask_count} dop_res={self.doppler_residual_mask_count} "
                f"pr_dop={self.pseudorange_doppler_mask_count}",
            )
        if self.tdcp_consistency_mask_count > 0:
            lines.append(f"  tdcp mask   : doppler_carrier={self.tdcp_consistency_mask_count}")
        if self.tdcp_weight_scale != DEFAULT_TDCP_WEIGHT_SCALE:
            lines.append(f"  tdcp scale  : {self.tdcp_weight_scale:g}")
        if self.tdcp_geometry_correction_applied:
            lines.append(f"  tdcp geom   : corrected_pairs={self.tdcp_geometry_correction_count}")
        if self.dual_frequency:
            lines.append("  frequency   : experimental L1/L5 slots enabled")
        if self.taroz_qzss_other_clock_enabled:
            lines.append("  taroz clk   : qzss_other_clk=on")
        if self.dd_carrier_fgo_enabled:
            lines.append(
                f"  dd carrier  : anchors={self.dd_carrier_accepted_anchor_epochs} "
                f"dd_epochs={self.dd_carrier_dd_epochs} "
                f"snapped={self.dd_carrier_base_snapped_epochs} "
                f"pairs_mean={self.dd_carrier_dd_pairs_mean:.2f}"
            )
        if self.taroz_fgo_candidate_enabled:
            lines.append("  taroz cand  : " + ",".join(self.taroz_fgo_candidate_sources))
        if self.parity_audit is not None:
            lines.append(
                "  parity      : "
                + ("base_correction_ready" if self.parity_audit.get("base_correction_ready") else "blocked")
                + f" ({self.parity_audit.get('base_correction_status', 'unknown')})"
            )
        if self.metrics_selected is not None:
            lines.extend(
                [
                    format_metrics_line("Selected", self.metrics_selected),
                    format_metrics_line("Kaggle WLS", self.metrics_kaggle),
                    format_metrics_line("Raw WLS", self.metrics_raw_wls),
                    format_metrics_line("FGO", self.metrics_fgo),
                ],
            )
            if self.metrics_selected["rms_2d"] < self.metrics_raw_wls["rms_2d"] - 1e-9:
                gain = (1.0 - self.metrics_selected["rms_2d"] / self.metrics_raw_wls["rms_2d"]) * 100.0
                lines.append(f"  -> selected output improves raw WLS by {gain:.1f}% on RMS2D")
        else:
            lines.append("  ground truth: unavailable (test/raw mode)")
        return lines


def export_bridge_outputs(export_dir: Path, result: BridgeResult, extra_metrics: dict | None = None) -> None:
    export_dir.mkdir(parents=True, exist_ok=True)
    result.positions_table().to_csv(export_dir / "bridge_positions.csv", index=False)
    result.states_table().to_csv(export_dir / "bridge_states.csv", index=False)
    fgo_vd_state = result.fgo_vd_state_table()
    if fgo_vd_state is not None:
        fgo_vd_state.to_csv(export_dir / "bridge_fgo_vd_state.csv", index=False)
    payload = result.metrics_payload()
    if extra_metrics:
        payload.update(extra_metrics)
    (export_dir / "bridge_metrics.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def load_bridge_metrics(trip_dir: Path) -> dict:
    return json.loads((trip_dir / "bridge_metrics.json").read_text(encoding="utf-8"))


def has_valid_bridge_outputs(trip_dir: Path) -> bool:
    metrics_path = trip_dir / "bridge_metrics.json"
    positions_path = trip_dir / "bridge_positions.csv"
    if not metrics_path.is_file() or not positions_path.is_file():
        return False
    if positions_path.stat().st_size <= 0:
        return False
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        fgo_iters = int(metrics["fgo_iters"])
        mse_pr = float(metrics["mse_pr"])
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False
    return fgo_iters >= 0 and np.isfinite(mse_pr)


def bridge_position_columns(
    position_source: str,
    available_columns: set[str] | list[str] | tuple[str, ...],
) -> tuple[str, str]:
    validate_position_source(position_source)
    columns = set(available_columns)
    if position_source == "baseline":
        return "BaselineLatitudeDegrees", "BaselineLongitudeDegrees"
    if position_source == "raw_wls":
        return "RawWlsLatitudeDegrees", "RawWlsLongitudeDegrees"
    if position_source == "fgo" and {"FgoLatitudeDegrees", "FgoLongitudeDegrees"}.issubset(columns):
        return "FgoLatitudeDegrees", "FgoLongitudeDegrees"
    return "LatitudeDegrees", "LongitudeDegrees"


__all__ = [
    "BridgeResult",
    "CT_RBPF_FGO_SOURCE",
    "DD_CARRIER_FGO_SOURCE",
    "TAROZ_FGO_CANDIDATE_SOURCES",
    "TAROZ_PR_D_L_FGO_SOURCE",
    "TAROZ_PR_FGO_SOURCE",
    "TAROZ_WEIGHTS_FGO_SOURCE",
    "TDCP_SCALE_FGO_SOURCE",
    "FACTOR_DT_MAX_S",
    "POSITION_SOURCES",
    "bridge_position_columns",
    "ecef_to_llh_deg",
    "export_bridge_outputs",
    "format_metrics_line",
    "has_valid_bridge_outputs",
    "load_bridge_metrics",
    "metrics_summary",
    "score_from_metrics",
    "validate_position_source",
]
