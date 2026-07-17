"""Standalone, FGO-free IMU preintegration (WP21).

This module ports the on-manifold IMU preintegration recursion that backs
``experiments/gsdc2023_imu.py`` (the engine ``tc_fgo.py``'s sliding window
actually consumes via ``collapse_imu_preintegration_segment`` /
``imu_preintegration_segment_with_bias_jacobians``) into a small,
dependency-free class usable directly by the particle filter runtime.
It performs **no import of ``tc_fgo`` or anything under ``experiments/``** —
the recursion is re-derived here from first principles (Forster et al.,
"On-Manifold Preintegration for Real-Time Visual-Inertial Odometry", the
same discrete recursion implemented in ``gsdc2023_imu.preintegrate_processed_imu``
with ``sample_dt_mode="taroz"``) so this module has zero FGO-stack
dependency.

Conventions (must match ``tc_fgo.py`` / ``ins_ekf.py`` so cross-checks are
meaningful):

- **Body frame**: x = forward, y = left, z = up (PPC IMU / ``ins_ekf.py``
  convention).
- **Gravity**: ``GRAVITY_ENU = (0, 0, -9.81)`` m/s^2 -- this matches
  ``tc_fgo._G_ENU`` and ``ins_ekf._G_ENU`` (note: this differs slightly from
  the WGS84-standard ``9.80665`` used internally by
  ``experiments/gsdc2023_imu.py``'s ``IMU_GRAVITY_MPS2`` constant for its
  optional ECEF gravity-compensation path; this module never subtracts
  gravity internally -- see next point -- so the exact gravity constant only
  matters to callers of :meth:`PreintegratedIMU.predict_position_velocity`,
  which defaults to ``GRAVITY_ENU`` to match ``tc_fgo``).
- **Gravity handling**: ``delta_p`` / ``delta_v`` are raw specific-force
  integrals accumulated in the body frame at the *start* of the segment
  (frame ``i``). Gravity is **not** subtracted internally -- exactly like
  ``tc_fgo.imu_preintegration_residual``, which adds
  ``0.5 * g_enu * dt^2`` / ``g_enu * dt`` explicitly when closing the
  segment:

      p_j = p_i + v_i*dt + 0.5*g_enu*dt^2 + R_i @ delta_p_corrected
      v_j = v_i + g_enu*dt + R_i @ delta_v_corrected

  (see :meth:`PreintegratedIMU.predict_position_velocity`, which implements
  exactly this and is numerically the same formula as
  ``tc_fgo.imu_preintegration_residual``'s zero-residual condition).
- **Rotation**: ``delta_R`` is the rotation from the body frame at the start
  of the segment to the body frame at its end, integrated purely from gyro
  samples (right-multiplied SO(3) increments, i.e. the standard IMU
  preintegration manifold recursion). Gyro samples are consumed in rad/s.
- **Bias linearization**: samples are accumulated *raw* (biases are not
  subtracted before integration); first-order correction Jacobians
  (``dp_d_ba``, ``dv_d_ba``, ``dp_d_bg``, ``dv_d_bg``, ``dR_d_bg``) have the
  same formula *shape* as ``tc_fgo.bias_corrected_preintegration``
  (``dp = delta_p + dp_d_ba @ (ba - ba_lin) + dp_d_bg @ (bg - bg_lin)``), and
  each Jacobian is verified by finite differences (see
  ``tests/test_imu_preintegration.py``) to equal
  ``d(delta_p or delta_v or delta_angle) / d(constant additive offset applied
  to every accel or gyro sample in the segment)``. If your bias convention
  instead *subtracts* bias from the raw measurement (``true = measured -
  bias``), pass ``-bias`` at the call site.
- **Covariance**: a 9x9 covariance over ``[delta_p, delta_v, delta_theta]``
  (``delta_theta`` = local perturbation of ``delta_R``) is accumulated via
  the standard closed-form F/G noise-propagation recursion, separately for
  the accel- and gyro-noise-driven channels (unit noise density), then
  combined as ``sigma_accel^2 * cov9_accel + sigma_gyro^2 * cov9_gyro``.

No FGO import: this module only depends on ``numpy``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

GRAVITY_ENU = np.array([0.0, 0.0, -9.81], dtype=np.float64)

_EYE3 = np.eye(3, dtype=np.float64)
_EYE9 = np.eye(9, dtype=np.float64)


# ---------------------------------------------------------------------------
# SO(3) helpers (self-contained; mirror experiments/gsdc2023_imu.py's helpers
# of the same name/shape so numerics match, but are re-derived here rather
# than imported so this module has no dependency on experiments/).
# ---------------------------------------------------------------------------


def _skew3(v: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(v, dtype=np.float64).reshape(3)
    return np.array(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        dtype=np.float64,
    )


def rotvec_to_rotm(rotvec_rad: np.ndarray) -> np.ndarray:
    """Rodrigues rotation matrix ``Exp(phi)`` for a 3D rotation vector."""

    rv = np.asarray(rotvec_rad, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(rv))
    kx = _skew3(rv)
    if theta < 1.0e-12:
        return _EYE3 + kx
    a = np.sin(theta) / theta
    b = (1.0 - np.cos(theta)) / (theta * theta)
    return _EYE3 + a * kx + b * (kx @ kx)


def rotm_to_rotvec(rotm: np.ndarray) -> np.ndarray:
    """SO(3) logarithm ``Log(R)`` as a 3D rotation vector."""

    rot = np.asarray(rotm, dtype=np.float64).reshape(3, 3)
    cos_theta = 0.5 * (float(np.trace(rot)) - 1.0)
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    theta = float(np.arccos(cos_theta))
    vee = np.array(
        [rot[2, 1] - rot[1, 2], rot[0, 2] - rot[2, 0], rot[1, 0] - rot[0, 1]],
        dtype=np.float64,
    )
    if theta < 1.0e-12:
        return 0.5 * vee
    return theta / (2.0 * np.sin(theta)) * vee


def right_jacobian_so3(rotvec_rad: np.ndarray) -> np.ndarray:
    """Right Jacobian ``Jr(phi)`` of SO(3)."""

    phi = np.asarray(rotvec_rad, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(phi))
    kx = _skew3(phi)
    kx2 = kx @ kx
    if theta < 1.0e-8:
        return _EYE3 - 0.5 * kx + (1.0 / 6.0) * kx2
    a = (1.0 - np.cos(theta)) / (theta * theta)
    b = (theta - np.sin(theta)) / (theta * theta * theta)
    return _EYE3 - a * kx + b * kx2


def right_jacobian_inverse_so3(rotvec_rad: np.ndarray) -> np.ndarray:
    """Inverse right Jacobian ``Jr(phi)^{-1}`` of SO(3)."""

    phi = np.asarray(rotvec_rad, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(phi))
    kx = _skew3(phi)
    kx2 = kx @ kx
    if theta < 1.0e-8:
        return _EYE3 + 0.5 * kx + (1.0 / 12.0) * kx2
    half_theta = 0.5 * theta
    b = (1.0 / (theta * theta)) * (1.0 - half_theta / np.tan(half_theta))
    return _EYE3 + 0.5 * kx + b * kx2


def _gyro_bias_jacobian_so3(increment_rotvecs_rad: np.ndarray, dt_s: np.ndarray) -> np.ndarray:
    """Return ``-d Log(prod Exp((omega-bias)dt)) / d bias`` evaluated at zero bias.

    Standard chain-rule accumulation across the composed micro-rotations
    (see Forster et al. 2015, eq. 44), used for the final ``dR_d_bg``
    bias-correction Jacobian.
    """

    phis = np.asarray(increment_rotvecs_rad, dtype=np.float64).reshape(-1, 3)
    dts = np.asarray(dt_s, dtype=np.float64).reshape(-1)
    if phis.shape[0] != dts.size:
        raise ValueError("increment_rotvecs_rad and dt_s must have the same length")
    delta_rot = _EYE3.copy()
    increment_rots: list[np.ndarray] = []
    for phi in phis:
        incr = rotvec_to_rotm(phi)
        increment_rots.append(incr)
        delta_rot = delta_rot @ incr
    rho = rotm_to_rotvec(delta_rot)
    jac_right = np.zeros((3, 3), dtype=np.float64)
    suffix_rot = _EYE3.copy()
    for phi, dt_i, incr in zip(phis[::-1], dts[::-1], increment_rots[::-1]):
        jac_right += suffix_rot.T @ right_jacobian_so3(phi) * float(dt_i)
        suffix_rot = incr @ suffix_rot
    return right_jacobian_inverse_so3(rho) @ jac_right


def _rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> scalar-last quaternion ``[x, y, z, w]`` (ins_ekf.py convention)."""

    rot = np.asarray(R, dtype=np.float64).reshape(3, 3)
    tr = float(np.trace(rot))
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * s
        qx = (rot[2, 1] - rot[1, 2]) / s
        qy = (rot[0, 2] - rot[2, 0]) / s
        qz = (rot[1, 0] - rot[0, 1]) / s
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        s = np.sqrt(max(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2], 0.0)) * 2.0
        qw = (rot[2, 1] - rot[1, 2]) / s
        qx = 0.25 * s
        qy = (rot[0, 1] + rot[1, 0]) / s
        qz = (rot[0, 2] + rot[2, 0]) / s
    elif rot[1, 1] > rot[2, 2]:
        s = np.sqrt(max(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2], 0.0)) * 2.0
        qw = (rot[0, 2] - rot[2, 0]) / s
        qx = (rot[0, 1] + rot[1, 0]) / s
        qy = 0.25 * s
        qz = (rot[1, 2] + rot[2, 1]) / s
    else:
        s = np.sqrt(max(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1], 0.0)) * 2.0
        qw = (rot[1, 0] - rot[0, 1]) / s
        qx = (rot[0, 2] + rot[2, 0]) / s
        qy = (rot[1, 2] + rot[2, 1]) / s
        qz = 0.25 * s
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    n = float(np.linalg.norm(q))
    if not np.isfinite(n) or n <= 0.0:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return q / n


# ---------------------------------------------------------------------------
# Core recursion
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PreintResult:
    """Immutable result of preintegrating a batch of IMU samples."""

    delta_p: np.ndarray
    delta_v: np.ndarray
    delta_R: np.ndarray
    delta_angle: np.ndarray
    delta_t: float
    n_samples: int
    dp_d_ba: np.ndarray
    dv_d_ba: np.ndarray
    dp_d_bg: np.ndarray
    dv_d_bg: np.ndarray
    dR_d_bg: np.ndarray
    cov9_accel: np.ndarray
    cov9_gyro: np.ndarray


_EMPTY_RESULT = PreintResult(
    delta_p=np.zeros(3),
    delta_v=np.zeros(3),
    delta_R=_EYE3.copy(),
    delta_angle=np.zeros(3),
    delta_t=0.0,
    n_samples=0,
    dp_d_ba=np.zeros((3, 3)),
    dv_d_ba=np.zeros((3, 3)),
    dp_d_bg=np.zeros((3, 3)),
    dv_d_bg=np.zeros((3, 3)),
    dR_d_bg=np.zeros((3, 3)),
    cov9_accel=np.zeros((9, 9)),
    cov9_gyro=np.zeros((9, 9)),
)


def preintegrate_raw(
    accel_body: np.ndarray,
    gyro_body_radps: np.ndarray,
    dt_s: np.ndarray,
) -> PreintResult:
    """Preintegrate one batch of raw (accel, gyro, dt) samples.

    Pure function version of :class:`PreintegratedIMU`'s accumulation.
    Samples are assumed already time-ordered and each ``dt_s[k]`` is the
    forward-difference interval attributed to sample ``k`` (matches the
    ``sample_dt_mode="taroz"`` convention in ``gsdc2023_imu.py``, i.e. the
    caller supplies per-sample dt directly rather than us inferring it from
    timestamps).
    """

    acc = np.asarray(accel_body, dtype=np.float64).reshape(-1, 3)
    gyro = np.asarray(gyro_body_radps, dtype=np.float64).reshape(-1, 3)
    dts = np.asarray(dt_s, dtype=np.float64).reshape(-1)
    n = min(acc.shape[0], gyro.shape[0], dts.size)
    if n == 0:
        return _EMPTY_RESULT

    vel = np.zeros(3, dtype=np.float64)
    pos = np.zeros(3, dtype=np.float64)
    acc_jac_v = np.zeros((3, 3), dtype=np.float64)
    acc_jac_p = np.zeros((3, 3), dtype=np.float64)
    delta_rot = _EYE3.copy()
    delta_rot_bias_deriv = np.zeros((3, 3, 3), dtype=np.float64)
    vel_bias_gyro_deriv = np.zeros((3, 3), dtype=np.float64)
    pos_bias_gyro_deriv = np.zeros((3, 3), dtype=np.float64)
    accel_noise_cov = np.zeros((9, 9), dtype=np.float64)
    gyro_noise_cov = np.zeros((9, 9), dtype=np.float64)
    gyro_increments: list[np.ndarray] = []
    dts_used: list[float] = []

    for k in range(n):
        dt_k = float(dts[k])
        if not np.isfinite(dt_k) or dt_k <= 0.0:
            continue
        a_body = acc[k]
        w_body = gyro[k]
        if not (np.all(np.isfinite(a_body)) and np.all(np.isfinite(w_body))):
            continue

        gyro_increment = w_body * dt_k
        increment_rot = rotvec_to_rotm(gyro_increment)
        increment_jr = right_jacobian_so3(gyro_increment)
        rot_accel = delta_rot

        # --- unit-noise-density 9x9 covariance propagation (F, G matrices) ---
        f_cov = _EYE9.copy()
        f_cov[0:3, 3:6] = _EYE3 * dt_k
        accel_theta_jac = -rot_accel @ _skew3(a_body)
        f_cov[0:3, 6:9] = 0.5 * accel_theta_jac * dt_k * dt_k
        f_cov[3:6, 6:9] = accel_theta_jac * dt_k
        f_cov[6:9, 6:9] = increment_rot.T
        g_acc = np.zeros((9, 3), dtype=np.float64)
        g_acc[0:3, :] = 0.5 * rot_accel * dt_k * dt_k
        g_acc[3:6, :] = rot_accel * dt_k
        g_gyro = np.zeros((9, 3), dtype=np.float64)
        g_gyro[6:9, :] = increment_jr * dt_k
        inv_dt = 1.0 / dt_k
        accel_noise_cov = f_cov @ accel_noise_cov @ f_cov.T + (g_acc @ g_acc.T) * inv_dt
        gyro_noise_cov = f_cov @ gyro_noise_cov @ f_cov.T + (g_gyro @ g_gyro.T) * inv_dt

        # --- mean delta_p / delta_v / delta_R recursion ---
        acc_delta = delta_rot @ a_body
        acc_bias_jac = delta_rot

        acc_gyro_deriv = np.zeros((3, 3), dtype=np.float64)
        for axis in range(3):
            acc_gyro_deriv[:, axis] = delta_rot_bias_deriv[axis] @ a_body
        pos_bias_gyro_deriv = (
            pos_bias_gyro_deriv + vel_bias_gyro_deriv * dt_k + 0.5 * acc_gyro_deriv * dt_k * dt_k
        )
        vel_bias_gyro_deriv = vel_bias_gyro_deriv + acc_gyro_deriv * dt_k

        pos = pos + vel * dt_k + 0.5 * acc_delta * dt_k * dt_k
        vel = vel + acc_delta * dt_k
        acc_jac_p = acc_jac_p + acc_jac_v * dt_k + 0.5 * acc_bias_jac * dt_k * dt_k
        acc_jac_v = acc_jac_v + acc_bias_jac * dt_k

        gyro_increments.append(gyro_increment)
        dts_used.append(dt_k)
        next_deriv = np.zeros_like(delta_rot_bias_deriv)
        for axis in range(3):
            basis = _EYE3[:, axis]
            d_increment_db = increment_rot @ _skew3(-(increment_jr @ basis) * dt_k)
            next_deriv[axis] = delta_rot_bias_deriv[axis] @ increment_rot + delta_rot @ d_increment_db
        delta_rot = delta_rot @ increment_rot
        delta_rot_bias_deriv = next_deriv

    if not dts_used:
        return _EMPTY_RESULT

    angle = rotm_to_rotvec(delta_rot)
    gyro_jac_p = -pos_bias_gyro_deriv
    gyro_jac_v = -vel_bias_gyro_deriv
    gyro_jac_angle = _gyro_bias_jacobian_so3(np.asarray(gyro_increments), np.asarray(dts_used))

    # Map the raw local-perturbation covariance into the Log-map (rotation
    # vector) error covariance via the closing right-Jacobian-inverse block.
    angle_cov_map = _EYE9.copy()
    angle_cov_map[6:9, 6:9] = right_jacobian_inverse_so3(angle)
    accel_noise_cov = angle_cov_map @ accel_noise_cov @ angle_cov_map.T
    gyro_noise_cov = angle_cov_map @ gyro_noise_cov @ angle_cov_map.T

    return PreintResult(
        delta_p=pos,
        delta_v=vel,
        delta_R=delta_rot,
        delta_angle=angle,
        delta_t=float(np.sum(dts_used)),
        n_samples=len(dts_used),
        dp_d_ba=acc_jac_p,
        dv_d_ba=acc_jac_v,
        dp_d_bg=gyro_jac_p,
        dv_d_bg=gyro_jac_v,
        dR_d_bg=gyro_jac_angle,
        cov9_accel=0.5 * (accel_noise_cov + accel_noise_cov.T),
        cov9_gyro=0.5 * (gyro_noise_cov + gyro_noise_cov.T),
    )


# ---------------------------------------------------------------------------
# Stateful accumulator
# ---------------------------------------------------------------------------


class PreintegratedIMU:
    """Accumulate IMU samples between two GNSS epochs and expose the
    preintegrated Delta_p, Delta_v, Delta_R (+ quaternion), a 9x9 covariance,
    and first-order bias-correction Jacobians.

    Parameters
    ----------
    sigma_accel_mps2_sqrthz : float
        Continuous-time accelerometer noise density [m/s^2 / sqrt(Hz)].
        Defaults to the same value used elsewhere in this repo for IMU noise
        (``ins_ekf.INSConfig.sigma_acc_noise``).
    sigma_gyro_radps_sqrthz : float
        Continuous-time gyroscope noise density [rad/s / sqrt(Hz)]. Defaults
        to ``ins_ekf.INSConfig.sigma_gyro_noise``.
    ba_lin, bg_lin : array_like, shape (3,), optional
        Linearization point for the bias-correction Jacobians. Defaults to
        zero (matches ``tc_fgo``'s default epoch-i-bias linearization).
    """

    def __init__(
        self,
        sigma_accel_mps2_sqrthz: float = 0.05,
        sigma_gyro_radps_sqrthz: float = 0.005,
        ba_lin: np.ndarray | None = None,
        bg_lin: np.ndarray | None = None,
    ) -> None:
        self.sigma_accel = float(sigma_accel_mps2_sqrthz)
        self.sigma_gyro = float(sigma_gyro_radps_sqrthz)
        self.reset(ba_lin=ba_lin, bg_lin=bg_lin)

    def reset(self, ba_lin: np.ndarray | None = None, bg_lin: np.ndarray | None = None) -> None:
        """Clear accumulated samples and (optionally) reset the bias linearization point."""

        self._accel: list[np.ndarray] = []
        self._gyro: list[np.ndarray] = []
        self._dt: list[float] = []
        self._result: PreintResult | None = None
        self.ba_lin = (
            np.zeros(3, dtype=np.float64)
            if ba_lin is None
            else np.asarray(ba_lin, dtype=np.float64).reshape(3).copy()
        )
        self.bg_lin = (
            np.zeros(3, dtype=np.float64)
            if bg_lin is None
            else np.asarray(bg_lin, dtype=np.float64).reshape(3).copy()
        )

    def add_sample(self, accel_body: np.ndarray, gyro_body_radps: np.ndarray, dt: float) -> None:
        """Append one (accel, gyro, dt) sample. Non-finite or non-positive dt is dropped."""

        dt_f = float(dt)
        a = np.asarray(accel_body, dtype=np.float64).reshape(3)
        w = np.asarray(gyro_body_radps, dtype=np.float64).reshape(3)
        if not np.isfinite(dt_f) or dt_f <= 0.0:
            return
        if not (np.all(np.isfinite(a)) and np.all(np.isfinite(w))):
            return
        self._accel.append(a)
        self._gyro.append(w)
        self._dt.append(dt_f)
        self._result = None

    def add_samples(self, accel_body: np.ndarray, gyro_body_radps: np.ndarray, dt: np.ndarray) -> None:
        """Vectorized bulk append of (accel, gyro, dt) rows."""

        acc = np.asarray(accel_body, dtype=np.float64).reshape(-1, 3)
        gyro = np.asarray(gyro_body_radps, dtype=np.float64).reshape(-1, 3)
        dts = np.asarray(dt, dtype=np.float64).reshape(-1)
        n = min(acc.shape[0], gyro.shape[0], dts.size)
        for i in range(n):
            self.add_sample(acc[i], gyro[i], dts[i])

    @property
    def n_samples(self) -> int:
        return len(self._dt)

    def _compute(self) -> PreintResult:
        if self._result is None:
            self._result = preintegrate_raw(
                np.asarray(self._accel, dtype=np.float64).reshape(-1, 3) if self._accel else np.zeros((0, 3)),
                np.asarray(self._gyro, dtype=np.float64).reshape(-1, 3) if self._gyro else np.zeros((0, 3)),
                np.asarray(self._dt, dtype=np.float64),
            )
        return self._result

    # -- raw (uncorrected) segment properties -----------------------------

    @property
    def delta_p(self) -> np.ndarray:
        return self._compute().delta_p.copy()

    @property
    def delta_v(self) -> np.ndarray:
        return self._compute().delta_v.copy()

    @property
    def delta_R(self) -> np.ndarray:
        return self._compute().delta_R.copy()

    @property
    def delta_q(self) -> np.ndarray:
        """Scalar-last quaternion ``[x, y, z, w]`` equivalent to :attr:`delta_R`."""
        return _rotmat_to_quat(self._compute().delta_R)

    @property
    def delta_angle(self) -> np.ndarray:
        return self._compute().delta_angle.copy()

    @property
    def delta_t(self) -> float:
        return self._compute().delta_t

    # -- bias-correction Jacobians -----------------------------------------

    @property
    def dp_d_ba(self) -> np.ndarray:
        return self._compute().dp_d_ba.copy()

    @property
    def dv_d_ba(self) -> np.ndarray:
        return self._compute().dv_d_ba.copy()

    @property
    def dp_d_bg(self) -> np.ndarray:
        return self._compute().dp_d_bg.copy()

    @property
    def dv_d_bg(self) -> np.ndarray:
        return self._compute().dv_d_bg.copy()

    @property
    def dR_d_bg(self) -> np.ndarray:
        return self._compute().dR_d_bg.copy()

    # -- covariance ----------------------------------------------------------

    @property
    def covariance9(self) -> np.ndarray:
        """9x9 covariance over ``[delta_p, delta_v, delta_theta]`` [m^2, (m/s)^2, rad^2]."""

        r = self._compute()
        return (self.sigma_accel**2) * r.cov9_accel + (self.sigma_gyro**2) * r.cov9_gyro

    # -- bias-corrected outputs ---------------------------------------------

    def bias_corrected_delta_p_v(
        self, ba: np.ndarray | None = None, bg: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """First-order bias-corrected ``(delta_p, delta_v)`` around ``ba_lin``/``bg_lin``.

        Mirrors ``tc_fgo.bias_corrected_preintegration`` exactly.
        """

        r = self._compute()
        ba_arr = self.ba_lin if ba is None else np.asarray(ba, dtype=np.float64).reshape(3)
        bg_arr = self.bg_lin if bg is None else np.asarray(bg, dtype=np.float64).reshape(3)
        dba = ba_arr - self.ba_lin
        dbg = bg_arr - self.bg_lin
        dp = r.delta_p + r.dp_d_ba @ dba + r.dp_d_bg @ dbg
        dv = r.delta_v + r.dv_d_ba @ dba + r.dv_d_bg @ dbg
        return dp, dv

    def bias_corrected_delta_R(self, bg: np.ndarray | None = None) -> np.ndarray:
        """First-order bias-corrected ``delta_R`` around ``bg_lin``."""

        r = self._compute()
        bg_arr = self.bg_lin if bg is None else np.asarray(bg, dtype=np.float64).reshape(3)
        dbg = bg_arr - self.bg_lin
        correction = r.dR_d_bg @ dbg
        return r.delta_R @ rotvec_to_rotm(correction)

    def predict_position_velocity(
        self,
        p_i: np.ndarray,
        v_i: np.ndarray,
        R_i: np.ndarray,
        dt: float | None = None,
        g_enu: np.ndarray = GRAVITY_ENU,
        ba: np.ndarray | None = None,
        bg: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Close the segment into ``(p_j, v_j)`` in the navigation (ENU/ECEF) frame.

        ``R_i`` is the body-to-navigation rotation matrix at the start of the
        segment (e.g. from ``INSEKF`` / ``ComplementaryHeadingFilter``, kept
        outside the particle state per the WP21 roadmap). Implements exactly
        the zero-residual condition of ``tc_fgo.imu_preintegration_residual``:

            p_j = p_i + v_i*dt + 0.5*g*dt^2 + R_i @ delta_p_corrected
            v_j = v_i + g*dt + R_i @ delta_v_corrected
        """

        dt_eff = self.delta_t if dt is None else float(dt)
        dp, dv = self.bias_corrected_delta_p_v(ba, bg)
        R = np.asarray(R_i, dtype=np.float64).reshape(3, 3)
        p_i_arr = np.asarray(p_i, dtype=np.float64).reshape(3)
        v_i_arr = np.asarray(v_i, dtype=np.float64).reshape(3)
        g = np.asarray(g_enu, dtype=np.float64).reshape(3)
        p_j = p_i_arr + v_i_arr * dt_eff + 0.5 * g * dt_eff * dt_eff + R @ dp
        v_j = v_i_arr + g * dt_eff + R @ dv
        return p_j, v_j


__all__ = [
    "GRAVITY_ENU",
    "PreintResult",
    "PreintegratedIMU",
    "preintegrate_raw",
    "rotvec_to_rotm",
    "rotm_to_rotvec",
    "right_jacobian_so3",
    "right_jacobian_inverse_so3",
]
