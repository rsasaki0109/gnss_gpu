"""Solver option bundles for GSDC2023 raw-bridge FGO execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from experiments.gsdc2023_bridge_config import BridgeConfig


@dataclass(frozen=True)
class FgoRunOptions:
    motion_sigma_m: float
    clock_drift_sigma_m: float
    stop_velocity_sigma_mps: float
    stop_position_sigma_m: float
    apply_imu_prior: bool
    imu_position_sigma_m: float
    imu_velocity_sigma_mps: float
    imu_accel_bias_state: bool
    imu_accel_bias_prior_sigma_mps2: float
    imu_accel_bias_between_sigma_mps2: float
    fgo_iters: int
    tol: float
    chunk_epochs: int
    use_vd: bool
    graph_relative_height: bool
    relative_height_sigma_m: float
    apply_absolute_height: bool
    absolute_height_sigma_m: float

    @classmethod
    def from_config(cls, config: BridgeConfig, *, tol: float = 1e-7) -> "FgoRunOptions":
        return cls(
            motion_sigma_m=config.motion_sigma_m,
            clock_drift_sigma_m=config.clock_drift_sigma_m,
            stop_velocity_sigma_mps=config.stop_velocity_sigma_mps,
            stop_position_sigma_m=config.stop_position_sigma_m,
            apply_imu_prior=config.apply_imu_prior,
            imu_position_sigma_m=config.imu_position_sigma_m,
            imu_velocity_sigma_mps=config.imu_velocity_sigma_mps,
            imu_accel_bias_state=config.imu_accel_bias_state,
            imu_accel_bias_prior_sigma_mps2=config.imu_accel_bias_prior_sigma_mps2,
            imu_accel_bias_between_sigma_mps2=config.imu_accel_bias_between_sigma_mps2,
            fgo_iters=config.fgo_iters,
            tol=float(tol),
            chunk_epochs=config.chunk_epochs,
            use_vd=config.use_vd,
            graph_relative_height=config.graph_relative_height,
            relative_height_sigma_m=config.relative_height_sigma_m,
            apply_absolute_height=config.apply_absolute_height,
            absolute_height_sigma_m=config.absolute_height_sigma_m,
        )

    def run_kwargs(self) -> dict[str, Any]:
        return {
            "motion_sigma_m": self.motion_sigma_m,
            "clock_drift_sigma_m": self.clock_drift_sigma_m,
            "stop_velocity_sigma_mps": self.stop_velocity_sigma_mps,
            "stop_position_sigma_m": self.stop_position_sigma_m,
            "apply_imu_prior": self.apply_imu_prior,
            "imu_position_sigma_m": self.imu_position_sigma_m,
            "imu_velocity_sigma_mps": self.imu_velocity_sigma_mps,
            "imu_accel_bias_state": self.imu_accel_bias_state,
            "imu_accel_bias_prior_sigma_mps2": self.imu_accel_bias_prior_sigma_mps2,
            "imu_accel_bias_between_sigma_mps2": self.imu_accel_bias_between_sigma_mps2,
            "fgo_iters": self.fgo_iters,
            "tol": self.tol,
            "chunk_epochs": self.chunk_epochs,
            "use_vd": self.use_vd,
            "graph_relative_height": self.graph_relative_height,
            "relative_height_sigma_m": self.relative_height_sigma_m,
            "apply_absolute_height": self.apply_absolute_height,
            "absolute_height_sigma_m": self.absolute_height_sigma_m,
        }


def fgo_run_options_from_config(config: BridgeConfig, *, tol: float = 1e-7) -> FgoRunOptions:
    return FgoRunOptions.from_config(config, tol=tol)


__all__ = ["FgoRunOptions", "fgo_run_options_from_config"]
