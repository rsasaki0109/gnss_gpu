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


class IMUPredictor:
    """Compute velocity from IMU for PF predict step.

    Uses simple strapdown integration:
    - Accumulate acceleration in body frame
    - Rotate to local frame using gyro-derived heading
    - Provide velocity vector in ECEF (approximate)

    For GNSS-grade urban positioning, the main value is:
    - Bridging GNSS gaps (tunnels, signal blockage)
    - Providing velocity during predict (instead of EKF/SPP guide)
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
