"""15-state INS-GNSS EKF in a local ENU frame.

Conventions:
  - Body frame: x=forward, y=left, z=up (PPC IMU convention).
  - Navigation frame: local ENU centered at the run origin.
  - Gravity: g_enu = (0, 0, -9.81).
  - Quaternion: scalar-last [qx, qy, qz, qw], body-to-navigation.
  - Specific force: f_meas = R_n2b @ (a_inertial - g_enu) + b_a + n_a.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

_G_ENU = np.array([0.0, 0.0, -9.81], dtype=np.float64)
_DEG2RAD = math.pi / 180.0


@dataclass
class INSConfig:
    sigma_acc_noise: float = 0.05
    sigma_gyro_noise: float = 0.005
    sigma_acc_bias_rw: float = 1e-4
    sigma_gyro_bias_rw: float = 1e-5
    static_acc_low: float = 9.6
    static_acc_high: float = 9.95
    static_gyro_max_dps: float = 1.5
    align_min_static_samples: int = 50
    yaw_init_min_speed_mps: float = 1.0
    init_pos_sigma_m: float = 1.0
    init_vel_sigma_mps: float = 0.5
    init_attitude_sigma_rp_rad: float = math.radians(1.0)
    init_attitude_sigma_yaw_rad: float = math.radians(10.0)
    init_acc_bias_sigma: float = 0.1
    init_gyro_bias_sigma_rps: float = math.radians(0.5)


def _skew(v: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(v, dtype=np.float64).reshape(3)
    return np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=np.float64,
    )


def _wrap_pi(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def _quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    n = float(np.linalg.norm(q))
    if not math.isfinite(n) or n <= 0.0:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return q / n


def _quat_multiply(q: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Hamilton product for scalar-last quaternions."""
    qx, qy, qz, qw = np.asarray(q, dtype=np.float64).reshape(4)
    rx, ry, rz, rw = np.asarray(r, dtype=np.float64).reshape(4)
    return np.array(
        [
            qw * rx + qx * rw + qy * rz - qz * ry,
            qw * ry - qx * rz + qy * rw + qz * rx,
            qw * rz + qx * ry - qy * rx + qz * rw,
            qw * rw - qx * rx - qy * ry - qz * rz,
        ],
        dtype=np.float64,
    )


def _quat_from_axis_angle(rotvec: np.ndarray) -> np.ndarray:
    rotvec = np.asarray(rotvec, dtype=np.float64).reshape(3)
    angle = float(np.linalg.norm(rotvec))
    if not math.isfinite(angle) or angle <= 1.0e-14:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    axis = rotvec / angle
    half = 0.5 * angle
    s = math.sin(half)
    return _quat_normalize(np.array([axis[0] * s, axis[1] * s, axis[2] * s, math.cos(half)]))


def _quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Return body-to-navigation rotation matrix for scalar-last q."""
    x, y, z, w = _quat_normalize(q)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def _rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    tr = float(np.trace(R))
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(max(1.0 + R[0, 0] - R[1, 1] - R[2, 2], 0.0)) * 2.0
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(max(1.0 + R[1, 1] - R[0, 0] - R[2, 2], 0.0)) * 2.0
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(max(1.0 + R[2, 2] - R[0, 0] - R[1, 1], 0.0)) * 2.0
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    return _quat_normalize(np.array([qx, qy, qz, qw], dtype=np.float64))


def _rpy_to_quat_body_to_enu(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Build body-to-ENU q for PPC roll/pitch/yaw convention.

    Positive pitch makes static accelerometer x positive; positive roll makes
    static accelerometer y negative. This corresponds to
    R_b2n = Rz(yaw) @ Ry(-pitch) @ Rx(-roll).
    """
    cr, sr = math.cos(-roll), math.sin(-roll)
    cp, sp = math.cos(-pitch), math.sin(-pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64)
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return _rotmat_to_quat(Rz @ Ry @ Rx)


def _ecef_to_enu_rotation(lat: float, lon: float) -> np.ndarray:
    sl, cl = math.sin(float(lat)), math.cos(float(lat))
    so, co = math.sin(float(lon)), math.cos(float(lon))
    return np.array(
        [
            [-so, co, 0.0],
            [-sl * co, -sl * so, cl],
            [cl * co, cl * so, sl],
        ],
        dtype=np.float64,
    )


class INSEKF:
    """Error-state INS-GNSS EKF with 13D nominal and 15D covariance."""

    def __init__(self, config: INSConfig | None = None):
        self.config = config if config is not None else INSConfig()
        self.aligned = False
        self.yaw_initialized = False
        self._static_buffer: list[tuple[float, np.ndarray, np.ndarray]] = []

        self.p = np.zeros(3, dtype=np.float64)
        self.v = np.zeros(3, dtype=np.float64)
        self.q = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        self.b_a = np.zeros(3, dtype=np.float64)
        self.b_g = np.zeros(3, dtype=np.float64)
        self.P = np.zeros((15, 15), dtype=np.float64)

        self.last_t: float | None = None
        self._last_accel_body: np.ndarray | None = None
        self._last_gyro_body_dps: np.ndarray | None = None

    def feed_imu_for_alignment(
        self,
        t: float,
        accel_body: np.ndarray,
        gyro_body_dps: np.ndarray,
    ) -> None:
        """Buffer continuous static IMU samples and initialize attitude."""
        if self.aligned:
            return
        accel = np.asarray(accel_body, dtype=np.float64).reshape(3)
        gyro_dps = np.asarray(gyro_body_dps, dtype=np.float64).reshape(3)
        if not (np.all(np.isfinite(accel)) and np.all(np.isfinite(gyro_dps))):
            self._static_buffer.clear()
            return

        acc_norm = float(np.linalg.norm(accel))
        gyro_norm_dps = float(np.linalg.norm(gyro_dps))
        is_static = (
            float(self.config.static_acc_low) <= acc_norm <= float(self.config.static_acc_high)
            and gyro_norm_dps <= float(self.config.static_gyro_max_dps)
        )
        if not is_static:
            self._static_buffer.clear()
            return

        self._static_buffer.append((float(t), accel.copy(), gyro_dps.copy()))
        min_samples = max(1, int(self.config.align_min_static_samples))
        if len(self._static_buffer) < min_samples:
            return

        recent = self._static_buffer[-min_samples:]
        acc_avg = np.mean(np.vstack([s[1] for s in recent]), axis=0)
        gyro_avg_dps = np.mean(np.vstack([s[2] for s in recent]), axis=0)

        roll = math.atan2(-float(acc_avg[1]), float(acc_avg[2]))
        pitch = math.atan2(float(acc_avg[0]), math.hypot(float(acc_avg[1]), float(acc_avg[2])))
        self.q = _rpy_to_quat_body_to_enu(roll, pitch, 0.0)
        self.b_a = np.zeros(3, dtype=np.float64)
        self.b_g = gyro_avg_dps * _DEG2RAD
        self.v = np.zeros(3, dtype=np.float64)
        self._reset_covariance()
        self.aligned = True
        self.yaw_initialized = False
        self.last_t = float(t)
        self._last_accel_body = accel.copy()
        self._last_gyro_body_dps = gyro_dps.copy()

    def initialize_position(self, p_enu_init: np.ndarray) -> None:
        """Anchor INS position to a known local ENU coordinate."""
        p = np.asarray(p_enu_init, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(p)):
            raise ValueError("p_enu_init must be finite")
        self.p = p.copy()
        self.P[0:3, 0:3] = np.eye(3, dtype=np.float64) * float(self.config.init_pos_sigma_m) ** 2

    def initialize_yaw_from_velocity(self, v_enu: np.ndarray) -> bool:
        """Set yaw from ENU course-over-ground once planar speed is usable."""
        v = np.asarray(v_enu, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(v)):
            return False
        speed = float(math.hypot(float(v[0]), float(v[1])))
        if speed < float(self.config.yaw_init_min_speed_mps):
            return False
        yaw_meas = math.atan2(float(v[1]), float(v[0]))
        delta = _wrap_pi(yaw_meas - self.yaw_rad())
        # Left-multiply: yaw correction is around the ENU up axis.
        dq = _quat_from_axis_angle(np.array([0.0, 0.0, delta], dtype=np.float64))
        self.q = _quat_normalize(_quat_multiply(dq, self.q))
        self.v = v.copy()
        self.P[3:6, 3:6] = np.minimum(
            self.P[3:6, 3:6],
            np.eye(3, dtype=np.float64) * float(self.config.init_vel_sigma_mps) ** 2,
        )
        self.P[8, 8] = min(float(self.P[8, 8]), math.radians(5.0) ** 2) if self.P[8, 8] > 0 else math.radians(5.0) ** 2
        self.yaw_initialized = True
        return True

    def propagate(self, imu_samples: np.ndarray) -> None:
        """Propagate through IMU samples [t, ax, ay, az, gx, gy, gz].

        Gyro samples are expected in deg/s. The previous sample is held over
        each dt interval, so repeated adjacent calls can be made per GNSS epoch.
        """
        if not self.aligned:
            return
        samples = np.asarray(imu_samples, dtype=np.float64)
        if samples.size == 0:
            return
        if samples.ndim == 1:
            samples = samples.reshape(1, -1)
        if samples.shape[1] < 7:
            raise ValueError("imu_samples must have at least 7 columns")

        for row in samples:
            if not np.all(np.isfinite(row[:7])):
                continue
            t = float(row[0])
            accel = np.asarray(row[1:4], dtype=np.float64)
            gyro_dps = np.asarray(row[4:7], dtype=np.float64)
            if self.last_t is None:
                self.last_t = t
                self._last_accel_body = accel.copy()
                self._last_gyro_body_dps = gyro_dps.copy()
                continue
            if t <= self.last_t:
                if math.isclose(t, self.last_t, abs_tol=1.0e-9):
                    self._last_accel_body = accel.copy()
                    self._last_gyro_body_dps = gyro_dps.copy()
                continue

            dt = t - float(self.last_t)
            if dt > 0.5:
                self.last_t = t
                self._last_accel_body = accel.copy()
                self._last_gyro_body_dps = gyro_dps.copy()
                continue
            accel_use = self._last_accel_body if self._last_accel_body is not None else accel
            gyro_use = self._last_gyro_body_dps if self._last_gyro_body_dps is not None else gyro_dps
            self._propagate_one(float(dt), accel_use, gyro_use)
            self.last_t = t
            self._last_accel_body = accel.copy()
            self._last_gyro_body_dps = gyro_dps.copy()

    def update_position_enu(
        self,
        p_meas_enu: np.ndarray,
        sigma_pos_m: tuple[float, float, float] | np.ndarray,
    ) -> None:
        """Position-only EKF update and error-state injection."""
        z_pos = np.asarray(p_meas_enu, dtype=np.float64).reshape(3)
        sigma = np.asarray(sigma_pos_m, dtype=np.float64).reshape(3)
        if not (np.all(np.isfinite(z_pos)) and np.all(np.isfinite(sigma))):
            return
        sigma = np.maximum(sigma, 1.0e-3)

        H = np.zeros((3, 15), dtype=np.float64)
        H[:, 0:3] = np.eye(3, dtype=np.float64)
        Rm = np.diag(sigma * sigma)
        residual = z_pos - self.p
        S = H @ self.P @ H.T + Rm
        try:
            K = np.linalg.solve(S, H @ self.P).T
        except np.linalg.LinAlgError:
            K = self.P @ H.T @ np.linalg.pinv(S)
        dx = K @ residual
        eye15 = np.eye(15, dtype=np.float64)
        IKH = eye15 - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ Rm @ K.T
        self._symmetrize_covariance()

        self.p += dx[0:3]
        self.v += dx[3:6]
        dq = _quat_from_axis_angle(dx[6:9])
        self.q = _quat_normalize(_quat_multiply(dq, self.q))
        self.b_a += dx[9:12]
        self.b_g += dx[12:15]

    def position_sigma_m(self) -> float:
        tr = float(np.trace(self.P[0:3, 0:3]))
        if not math.isfinite(tr) or tr < 0.0:
            return float("inf")
        return math.sqrt(max(tr / 3.0, 0.0))

    def position_enu(self) -> np.ndarray:
        return self.p.copy()

    def velocity_enu(self) -> np.ndarray:
        return self.v.copy()

    def position_ecef(self, origin_ecef: np.ndarray, origin_lat: float, origin_lon: float) -> np.ndarray:
        origin = np.asarray(origin_ecef, dtype=np.float64).reshape(3)
        R = _ecef_to_enu_rotation(float(origin_lat), float(origin_lon))
        return origin + R.T @ self.p

    def velocity_ecef(self, origin_lat: float, origin_lon: float) -> np.ndarray:
        R = _ecef_to_enu_rotation(float(origin_lat), float(origin_lon))
        return R.T @ self.v

    def attitude_quat_body_to_ecef(self, origin_lat: float, origin_lon: float) -> np.ndarray:
        R_enu_to_ecef = _ecef_to_enu_rotation(float(origin_lat), float(origin_lon)).T
        return _rotmat_to_quat(R_enu_to_ecef @ _quat_to_rotmat(self.q))

    def accel_bias_body(self) -> np.ndarray:
        return self.b_a.copy()

    def gyro_bias_body_radps(self) -> np.ndarray:
        return self.b_g.copy()

    def yaw_rad(self) -> float:
        R = _quat_to_rotmat(self.q)
        return math.atan2(float(R[1, 0]), float(R[0, 0]))

    def _reset_covariance(self) -> None:
        cfg = self.config
        diag = np.array(
            [
                cfg.init_pos_sigma_m,
                cfg.init_pos_sigma_m,
                cfg.init_pos_sigma_m,
                cfg.init_vel_sigma_mps,
                cfg.init_vel_sigma_mps,
                cfg.init_vel_sigma_mps,
                cfg.init_attitude_sigma_rp_rad,
                cfg.init_attitude_sigma_rp_rad,
                cfg.init_attitude_sigma_yaw_rad,
                cfg.init_acc_bias_sigma,
                cfg.init_acc_bias_sigma,
                cfg.init_acc_bias_sigma,
                cfg.init_gyro_bias_sigma_rps,
                cfg.init_gyro_bias_sigma_rps,
                cfg.init_gyro_bias_sigma_rps,
            ],
            dtype=np.float64,
        )
        self.P = np.diag(diag * diag)

    def _propagate_one(self, dt: float, accel_body: np.ndarray, gyro_body_dps: np.ndarray) -> None:
        accel = np.asarray(accel_body, dtype=np.float64).reshape(3)
        gyro_rad = np.asarray(gyro_body_dps, dtype=np.float64).reshape(3) * _DEG2RAD
        omega_corr = gyro_rad - self.b_g
        f_corr = accel - self.b_a

        dq = _quat_from_axis_angle(omega_corr * dt)
        q_new = _quat_normalize(_quat_multiply(self.q, dq))
        R_bn = _quat_to_rotmat(q_new)
        a_enu = R_bn @ f_corr + _G_ENU

        self.p = self.p + self.v * dt + 0.5 * a_enu * dt * dt
        self.v = self.v + a_enu * dt
        self.q = q_new

        F = np.zeros((15, 15), dtype=np.float64)
        F[0:3, 3:6] = np.eye(3, dtype=np.float64)
        F[3:6, 6:9] = -_skew(R_bn @ f_corr)
        F[3:6, 9:12] = -R_bn
        F[6:9, 6:9] = -_skew(omega_corr)
        F[6:9, 12:15] = -np.eye(3, dtype=np.float64)

        Phi = np.eye(15, dtype=np.float64) + F * dt
        q_diag = np.zeros(15, dtype=np.float64)
        q_diag[3:6] = float(self.config.sigma_acc_noise) ** 2
        q_diag[6:9] = float(self.config.sigma_gyro_noise) ** 2
        q_diag[9:12] = float(self.config.sigma_acc_bias_rw) ** 2
        q_diag[12:15] = float(self.config.sigma_gyro_bias_rw) ** 2
        Q = np.diag(q_diag)
        self.P = Phi @ self.P @ Phi.T + Q * dt
        self._symmetrize_covariance()

    def _symmetrize_covariance(self) -> None:
        if not np.all(np.isfinite(self.P)):
            self._reset_covariance()
            return
        self.P = 0.5 * (self.P + self.P.T)
        diag = np.diag(self.P).copy()
        bad = ~np.isfinite(diag) | (diag < 1.0e-12)
        if np.any(bad):
            for idx in np.where(bad)[0]:
                self.P[idx, idx] = 1.0e-12
