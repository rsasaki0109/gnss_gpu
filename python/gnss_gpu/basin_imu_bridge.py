"""Causal PPC IMU preintegration for ambiguity-basin PF prediction."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np

from gnss_gpu.imu_preintegration import GRAVITY_ENU, PreintegratedIMU
from gnss_gpu.pf_imu_preint_adapter import body_to_ecef_frame, ecef_to_enu_rotation


@dataclass(frozen=True)
class PPCImuSamples:
    tow_s: np.ndarray
    accel_body_mps2: np.ndarray
    gyro_body_radps: np.ndarray

    def __post_init__(self) -> None:
        tow = np.asarray(self.tow_s, dtype=np.float64).reshape(-1)
        accel = np.asarray(self.accel_body_mps2, dtype=np.float64).reshape(-1, 3)
        gyro = np.asarray(self.gyro_body_radps, dtype=np.float64).reshape(-1, 3)
        if tow.size != accel.shape[0] or tow.size != gyro.shape[0]:
            raise ValueError("IMU time, acceleration, and gyro lengths must match")
        if tow.size and (
            not np.all(np.isfinite(tow))
            or not np.all(np.isfinite(accel))
            or not np.all(np.isfinite(gyro))
            or np.any(np.diff(tow) <= 0.0)
        ):
            raise ValueError("IMU samples must be finite and strictly time ordered")
        object.__setattr__(self, "tow_s", tow.copy())
        object.__setattr__(self, "accel_body_mps2", accel.copy())
        object.__setattr__(self, "gyro_body_radps", gyro.copy())


@dataclass(frozen=True)
class BasinImuPrediction:
    cv_position_correction_ecef_m: np.ndarray
    delta_velocity_ecef_mps: np.ndarray
    process_covariance: np.ndarray
    sample_count: int
    covered_duration_s: float
    heading_rad: float


def load_ppc_imu_csv(path: Path) -> PPCImuSamples:
    """Load the PPC IMU CSV without opening any reference trajectory."""

    times: list[float] = []
    acceleration: list[tuple[float, float, float]] = []
    gyro: list[tuple[float, float, float]] = []
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        rows = csv.DictReader(stream, skipinitialspace=True)
        for row in rows:
            normalized = {str(key).strip(): value for key, value in row.items()}
            try:
                times.append(float(normalized["GPS TOW (s)"]))
                acceleration.append(
                    tuple(float(normalized[f"Acc {axis} (m/s^2)"]) for axis in "XYZ")
                )
                gyro.append(
                    tuple(
                        math.radians(float(normalized[f"Ang Rate {axis} (deg/s)"]))
                        for axis in "XYZ"
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("malformed PPC IMU row") from exc
    return PPCImuSamples(
        np.asarray(times, dtype=np.float64),
        np.asarray(acceleration, dtype=np.float64).reshape(-1, 3),
        np.asarray(gyro, dtype=np.float64).reshape(-1, 3),
    )


class CausalBasinImuPredictor:
    """Preintegrate only samples available before each requested GNSS epoch."""

    def __init__(
        self,
        samples: PPCImuSamples,
        *,
        sigma_accel_mps2_sqrthz: float = 0.05,
        sigma_gyro_radps_sqrthz: float = 0.005,
        position_sigma_floor_m: float = 0.05,
        velocity_sigma_floor_mps: float = 0.10,
        minimum_horizontal_speed_mps: float = 0.5,
    ) -> None:
        self.samples = samples
        self.sigma_accel = float(sigma_accel_mps2_sqrthz)
        self.sigma_gyro = float(sigma_gyro_radps_sqrthz)
        self.position_sigma_floor_m = float(position_sigma_floor_m)
        self.velocity_sigma_floor_mps = float(velocity_sigma_floor_mps)
        self.minimum_horizontal_speed_mps = float(minimum_horizontal_speed_mps)
        if min(
            self.sigma_accel,
            self.sigma_gyro,
            self.position_sigma_floor_m,
            self.velocity_sigma_floor_mps,
            self.minimum_horizontal_speed_mps,
        ) <= 0.0:
            raise ValueError("IMU predictor scales must be positive")
        self.accel_bias_body_mps2 = np.zeros(3, dtype=np.float64)
        self.gyro_bias_body_radps = np.zeros(3, dtype=np.float64)
        self.heading_rad = 0.0
        self.calibrated = False

    def calibrate_before(self, first_tow_s: float, window_s: float = 1.5) -> int:
        """Estimate stationary biases from samples strictly before first use."""

        tow = self.samples.tow_s
        mask = (tow < float(first_tow_s)) & (tow >= float(first_tow_s) - float(window_s))
        indices = np.flatnonzero(mask)
        if indices.size < 10:
            return 0
        accel_median = np.median(self.samples.accel_body_mps2[indices], axis=0)
        gyro_median = np.median(self.samples.gyro_body_radps[indices], axis=0)
        self.accel_bias_body_mps2 = accel_median - np.array([0.0, 0.0, 9.81])
        self.gyro_bias_body_radps = gyro_median
        self.calibrated = True
        return int(indices.size)

    def _heading_from_velocity(
        self, position_ecef_m: np.ndarray, velocity_ecef_mps: np.ndarray
    ) -> float:
        position = np.asarray(position_ecef_m, dtype=np.float64).reshape(3)
        velocity = np.asarray(velocity_ecef_mps, dtype=np.float64).reshape(3)
        from gnss_gpu.pf_imu_preint_adapter import ecef_to_lla_rad

        lat, lon = ecef_to_lla_rad(position)
        velocity_enu = ecef_to_enu_rotation(lat, lon) @ velocity
        if float(np.linalg.norm(velocity_enu[:2])) >= self.minimum_horizontal_speed_mps:
            self.heading_rad = math.atan2(velocity_enu[0], velocity_enu[1])
        return self.heading_rad

    def predict_interval(
        self,
        start_tow_s: float,
        end_tow_s: float,
        *,
        position_ecef_m: np.ndarray,
        velocity_ecef_mps: np.ndarray,
    ) -> BasinImuPrediction | None:
        start = float(start_tow_s)
        end = float(end_tow_s)
        if not math.isfinite(start) or not math.isfinite(end) or end <= start:
            return None
        times = self.samples.tow_s
        if times.size < 2 or end <= times[0] or start >= times[-1]:
            return None
        first = max(int(np.searchsorted(times, start, side="right")) - 1, 0)
        last = min(int(np.searchsorted(times, end, side="left")), times.size - 1)
        preint = PreintegratedIMU(
            sigma_accel_mps2_sqrthz=self.sigma_accel,
            sigma_gyro_radps_sqrthz=self.sigma_gyro,
        )
        covered = 0.0
        for index in range(first, last + 1):
            segment_start = max(start, float(times[index]))
            segment_end = min(
                end,
                float(times[index + 1]) if index + 1 < times.size else end,
            )
            sample_dt = segment_end - segment_start
            if sample_dt <= 0.0:
                continue
            preint.add_sample(
                self.samples.accel_body_mps2[index]
                - self.accel_bias_body_mps2,
                self.samples.gyro_body_radps[index]
                - self.gyro_bias_body_radps,
                sample_dt,
            )
            covered += sample_dt
        interval = end - start
        if preint.n_samples == 0 or covered < 0.8 * interval:
            return None

        position = np.asarray(position_ecef_m, dtype=np.float64).reshape(3)
        velocity = np.asarray(velocity_ecef_mps, dtype=np.float64).reshape(3)
        heading = self._heading_from_velocity(position, velocity)
        body_to_ecef, enu_to_ecef = body_to_ecef_frame(heading, position)
        gravity_ecef = enu_to_ecef @ GRAVITY_ENU
        predicted_position, predicted_velocity = preint.predict_position_velocity(
            position,
            velocity,
            body_to_ecef,
            dt=interval,
            g_enu=gravity_ecef,
        )
        correction = predicted_position - (position + velocity * interval)
        delta_velocity = predicted_velocity - velocity

        rotation6 = np.zeros((6, 6), dtype=np.float64)
        rotation6[:3, :3] = body_to_ecef
        rotation6[3:6, 3:6] = body_to_ecef
        process = rotation6 @ preint.covariance9[:6, :6] @ rotation6.T
        process[:3, :3] += np.eye(3) * self.position_sigma_floor_m**2
        process[3:6, 3:6] += np.eye(3) * self.velocity_sigma_floor_mps**2
        process = 0.5 * (process + process.T)
        return BasinImuPrediction(
            cv_position_correction_ecef_m=correction,
            delta_velocity_ecef_mps=delta_velocity,
            process_covariance=process,
            sample_count=preint.n_samples,
            covered_duration_s=covered,
            heading_rad=heading,
        )


__all__ = [
    "BasinImuPrediction",
    "CausalBasinImuPredictor",
    "PPCImuSamples",
    "load_ppc_imu_csv",
]
