"""IMU data loader and integration for particle filter prediction.

Loads UrbanNav IMU CSV (accelerometer + gyroscope + wheel velocity)
and provides velocity estimates for PF predict step.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np


def load_imu_csv(path: str | Path) -> dict:
    """Load UrbanNav IMU CSV file.

    Returns
    -------
    dict with keys:
        tow: (N,) GPS time of week [s]
        accel: (N, 3) acceleration [m/s^2] (body frame)
        gyro: (N, 3) angular rate [rad/s] (body frame)
        wheel_vel: (N,) wheel velocity [m/s] (NaN if absent)
    """
    tow, accel, gyro, wheel = [], [], [], []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if len(row) < 8:
                continue
            tow.append(float(row[0]))
            accel.append([float(row[2]), float(row[3]), float(row[4])])
            gyro.append([float(row[5]), float(row[6]), float(row[7])])
            if len(row) >= 9 and row[8].strip():
                try:
                    wheel.append(float(row[8]))
                except ValueError:
                    wheel.append(np.nan)
            else:
                wheel.append(np.nan)

    return {
        "tow": np.array(tow, dtype=np.float64),
        "accel": np.array(accel, dtype=np.float64),
        "gyro": np.array(gyro, dtype=np.float64),
        "wheel_vel": np.array(wheel, dtype=np.float64),
    }


class ComplementaryHeadingFilter:
    """Fuse gyroscope heading (short-term precise) with SPP heading (long-term stable).

    Complementary filter: heading = (1-alpha)*gyro_heading + alpha*spp_heading
    alpha=0.05 gives best results: gyro provides smooth inter-epoch tracking,
    SPP corrects long-term drift.

    Combined with wheel velocity (precise speed magnitude), this produces
    velocity estimates that beat SPP-only guide by 6% RMS.

    Reference: complementary filter for attitude estimation (Mahony et al.)
    """

    def __init__(self, imu_data: dict, alpha: float = 0.05):
        self.tow = imu_data["tow"]
        self.gyro_z = imu_data["gyro"][:, 2]  # yaw rate
        self.wheel_vel = imu_data["wheel_vel"]
        self.heading = 0.0
        self.alpha = alpha

    def update_heading_gyro(self, t_start: float, t_end: float) -> None:
        """Integrate gyroscope for heading between GNSS epochs."""
        mask = (self.tow >= t_start) & (self.tow < t_end)
        indices = np.where(mask)[0]
        for i in indices:
            dt = self.tow[min(i + 1, len(self.tow) - 1)] - self.tow[i]
            if dt <= 0:
                dt = 0.02
            self.heading += self.gyro_z[i] * dt

    def correct_heading_spp(self, spp_heading_rad: float) -> None:
        """Correct heading drift using SPP-derived heading."""
        diff = spp_heading_rad - self.heading
        diff = (diff + math.pi) % (2 * math.pi) - math.pi  # wrap to [-pi, pi]
        self.heading += self.alpha * diff

    def get_wheel_speed(self, t_start: float, t_end: float) -> float:
        """Get latest wheel velocity in the interval."""
        mask = (self.tow >= t_start) & (self.tow < t_end)
        indices = np.where(mask)[0]
        if len(indices) == 0:
            return 0.0
        ws = self.wheel_vel[indices[-1]]
        return float(ws) if np.isfinite(ws) and ws >= 0 else 0.0

    def get_velocity_enu(self, t_start: float, t_end: float) -> np.ndarray:
        """Get velocity in ENU using fused heading + wheel speed."""
        self.update_heading_gyro(t_start, t_end)
        speed = self.get_wheel_speed(t_start, t_end)
        ve = speed * math.sin(self.heading)
        vn = speed * math.cos(self.heading)
        return np.array([ve, vn, 0.0])

    @staticmethod
    def velocity_enu_to_ecef(
        vel_enu: np.ndarray, lat: float, lon: float,
    ) -> np.ndarray:
        """Convert ENU velocity to ECEF velocity."""
        sin_lat = math.sin(lat)
        cos_lat = math.cos(lat)
        sin_lon = math.sin(lon)
        cos_lon = math.cos(lon)
        vx = (-sin_lon * vel_enu[0] - sin_lat * cos_lon * vel_enu[1]
              + cos_lat * cos_lon * vel_enu[2])
        vy = (cos_lon * vel_enu[0] - sin_lat * sin_lon * vel_enu[1]
              + cos_lat * sin_lon * vel_enu[2])
        vz = cos_lat * vel_enu[1] + sin_lat * vel_enu[2]
        return np.array([vx, vy, vz])


class IMUPredictor:
    """Compute velocity from IMU for PF predict step.

    Uses simple strapdown integration:
    - Accumulate acceleration in body frame
    - Rotate to local frame using gyro-derived heading
    - Provide velocity vector in ECEF (approximate)

    For GNSS-grade urban positioning, the main value is:
    - Bridging GNSS gaps (tunnels, signal blockage)
    - Providing velocity during predict (instead of EKF/SPP guide)

    NOTE: ComplementaryHeadingFilter is preferred over this class.
    """

    def __init__(self, imu_data: dict, initial_heading: float = 0.0):
        self.tow = imu_data["tow"]
        self.accel = imu_data["accel"]
        self.gyro = imu_data["gyro"]
        self.wheel_vel = imu_data["wheel_vel"]
        self.heading = initial_heading  # radians, from north
        self.velocity_enu = np.zeros(3)  # East, North, Up
        self._idx = 0  # current read position

    def get_velocity_enu(self, t_start: float, t_end: float) -> np.ndarray | None:
        """Integrate IMU from t_start to t_end, return velocity in ENU.

        Returns (3,) velocity [m/s] in ENU frame, or None if no IMU data.
        """
        # Find IMU samples in [t_start, t_end]
        mask = (self.tow >= t_start) & (self.tow < t_end)
        indices = np.where(mask)[0]

        if len(indices) == 0:
            return None

        # Simple integration: accumulate gyro for heading, accel for velocity
        for i in indices:
            dt = (self.tow[min(i + 1, len(self.tow) - 1)] - self.tow[i])
            if dt <= 0:
                dt = 0.02  # ~50 Hz default

            # Update heading from gyro Z (yaw rate)
            self.heading += self.gyro[i, 2] * dt

            # Rotate body acceleration to ENU
            # Body frame: X=forward, Y=left, Z=up (typical IMU convention)
            ax_body = self.accel[i, 0]
            ay_body = self.accel[i, 1]

            # Remove gravity from Z
            az_body = self.accel[i, 2] + 9.81  # gravity compensation

            # Rotate to ENU using heading
            cos_h = math.cos(self.heading)
            sin_h = math.sin(self.heading)
            a_east = ax_body * sin_h + ay_body * cos_h
            a_north = ax_body * cos_h - ay_body * sin_h
            a_up = az_body

            # Integrate acceleration to velocity
            self.velocity_enu[0] += a_east * dt
            self.velocity_enu[1] += a_north * dt
            self.velocity_enu[2] += a_up * dt

        # Use wheel velocity as speed magnitude if available (more reliable)
        last_wheel = self.wheel_vel[indices[-1]]
        if np.isfinite(last_wheel) and last_wheel > 0.1:
            # Scale ENU velocity to match wheel speed
            speed_enu = np.linalg.norm(self.velocity_enu[:2])
            if speed_enu > 0.1:
                scale = last_wheel / speed_enu
                self.velocity_enu[:2] *= min(scale, 3.0)  # cap scaling

        return self.velocity_enu.copy()

    def velocity_enu_to_ecef(
        self, vel_enu: np.ndarray, lat: float, lon: float,
    ) -> np.ndarray:
        """Convert ENU velocity to ECEF velocity."""
        sin_lat = math.sin(lat)
        cos_lat = math.cos(lat)
        sin_lon = math.sin(lon)
        cos_lon = math.cos(lon)

        # ENU to ECEF rotation
        vx = (-sin_lon * vel_enu[0] - sin_lat * cos_lon * vel_enu[1]
              + cos_lat * cos_lon * vel_enu[2])
        vy = (cos_lon * vel_enu[0] - sin_lat * sin_lon * vel_enu[1]
              + cos_lat * sin_lon * vel_enu[2])
        vz = cos_lat * vel_enu[1] + sin_lat * vel_enu[2]

        return np.array([vx, vy, vz])
