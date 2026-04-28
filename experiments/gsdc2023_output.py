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
POSITION_SOURCES = ("baseline", "raw_wls", "fgo", "auto", "gated")


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
    selected_mse_pr: float
    baseline_mse_pr: float
    raw_wls_mse_pr: float
    fgo_mse_pr: float
    selected_source_counts: dict[str, int]
    metrics_selected: dict | None
    metrics_kaggle: dict | None
    metrics_raw_wls: dict | None
    metrics_fgo: dict | None
    chunk_selection_records: list[dict[str, object]] | None = None
    parity_audit: dict | None = None
    factor_dt_max_s: float = FACTOR_DT_MAX_S
    factor_dt_gap_count: int = 0
    stop_velocity_sigma_mps: float = 0.0
    stop_position_sigma_m: float = 0.0
    imu_prior_applied: bool = False
    imu_prior_interval_count: int = 0
    imu_frame: str = "body"
    imu_position_sigma_m: float = 25.0
    imu_velocity_sigma_mps: float = 5.0
    imu_accel_bias_state_applied: bool = False
    imu_accel_bias_prior_sigma_mps2: float = IMU_ACCEL_BIAS_PRIOR_SIGMA_MPS2
    imu_accel_bias_between_sigma_mps2: float = IMU_ACCEL_BIAS_BETWEEN_SIGMA_MPS2
    imu_acc_bias_mean_norm_mps2: float = float("nan")
    imu_gyro_bias_mean_norm_radps: float = float("nan")
    absolute_height_applied: bool = False
    absolute_height_ref_count: int = 0
    absolute_height_sigma_m: float = HEIGHT_ABSOLUTE_SIGMA_M
    absolute_height_dist_m: float = HEIGHT_ABSOLUTE_DIST_M
    relative_height_applied: bool = False
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
    dual_frequency: bool = False
    graph_relative_height: bool = False

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

    def metrics_payload(self) -> dict:
        payload = {
            "trip": self.trip,
            "signal_type": self.signal_type,
            "weight_mode": self.weight_mode,
            "selected_source_mode": self.selected_source_mode,
            "n_epochs": self.n_epochs,
            "max_sats": int(self.max_sats),
            "n_clock": int(max(self.raw_wls.shape[1] - 3, 1)),
            "fgo_iters": int(self.fgo_iters),
            "failed_chunks": int(self.failed_chunks),
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
            "imu_prior_applied": bool(self.imu_prior_applied),
            "imu_prior_interval_count": int(self.imu_prior_interval_count),
            "imu_frame": str(self.imu_frame),
            "imu_position_sigma_m": float(self.imu_position_sigma_m),
            "imu_velocity_sigma_mps": float(self.imu_velocity_sigma_mps),
            "imu_accel_bias_state_applied": bool(self.imu_accel_bias_state_applied),
            "imu_accel_bias_prior_sigma_mps2": float(self.imu_accel_bias_prior_sigma_mps2),
            "imu_accel_bias_between_sigma_mps2": float(self.imu_accel_bias_between_sigma_mps2),
            "imu_acc_bias_mean_norm_mps2": float(self.imu_acc_bias_mean_norm_mps2),
            "imu_gyro_bias_mean_norm_radps": float(self.imu_gyro_bias_mean_norm_radps),
            "absolute_height_applied": bool(self.absolute_height_applied),
            "absolute_height_ref_count": int(self.absolute_height_ref_count),
            "absolute_height_sigma_m": float(self.absolute_height_sigma_m),
            "absolute_height_dist_m": float(self.absolute_height_dist_m),
            "relative_height_applied": bool(self.relative_height_applied),
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
            "dual_frequency": bool(self.dual_frequency),
            "graph_relative_height": bool(self.graph_relative_height),
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
        if self.factor_dt_gap_count > 0:
            lines.append(
                f"  factor gaps : skipped={self.factor_dt_gap_count} "
                f"dt_max={self.factor_dt_max_s:.3f}s"
            )
        if self.stop_velocity_sigma_mps > 0.0 or self.stop_position_sigma_m > 0.0:
            lines.append(
                f"  stop factors : vel_sigma={self.stop_velocity_sigma_mps:.3f}m/s "
                f"pose_sigma={self.stop_position_sigma_m:.3f}m"
            )
        if self.imu_prior_applied:
            imu_line = (
                f"  imu prior    : intervals={self.imu_prior_interval_count} "
                f"frame={self.imu_frame} "
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


def export_bridge_outputs(export_dir: Path, result: BridgeResult) -> None:
    export_dir.mkdir(parents=True, exist_ok=True)
    result.positions_table().to_csv(export_dir / "bridge_positions.csv", index=False)
    (export_dir / "bridge_metrics.json").write_text(
        json.dumps(result.metrics_payload(), indent=2),
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
