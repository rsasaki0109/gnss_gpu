"""WP21 Phase A: thin adapter wiring IMU preintegration into the PF predict step.

Wires :class:`gnss_gpu.imu_preintegration.PreintegratedIMU` into
:class:`gnss_gpu.pf_device_runtime.ParticleFilterDeviceRuntime`'s *existing*
``predict()`` velocity-guide interface -- no CUDA kernel changes (per the
WP21 task spec's non-goals). ``imu_mode="preint"``: between GNSS epochs,
accumulate the 100 Hz IMU into a :class:`~gnss_gpu.imu_preintegration.PreintegratedIMU`
segment and close it into a per-epoch ECEF velocity/displacement guide plus
an isotropic ``sigma_pos`` derived from the preintegration covariance, fed to
``pf.predict(velocity=..., dt=..., sigma_pos=...)`` in place of the CV
(zero-velocity) or heuristic (``imu.IMUPredictor`` / ``ComplementaryHeadingFilter``
wheel-speed) guides.

Heading/attitude is tracked *outside* the particle state via the existing
``gnss_gpu.imu.ComplementaryHeadingFilter`` (gyro-rate integration + periodic
GNSS-bearing correction) -- attitude is never added to the per-particle
state, per the WP21 roadmap (particles stay ``{x, y, z, cb}``).

**Documented approximation** (ground-vehicle, mirrors the same assumption in
``experiments/ppc_imu_adapter.py``): roll and pitch are assumed zero (flat
vehicle); only yaw (heading) is tracked. Body frame convention:
x=forward, y=left, z=up (matches ``ins_ekf.py``).

**Documented approximation (velocity persistence)**: closing an IMU
preintegration segment into a displacement requires a starting velocity
``v_i``, but the PF's CV/heuristic predict path carries no persistent
velocity *particle* state. This module keeps a single **nominal** velocity
accumulator (ECEF, not a particle-filter state) that is propagated by the
preintegration's ``Delta_v`` each epoch and blended toward a GNSS-derived
velocity estimate (e.g. consecutive WLS-fix finite differences) with
``velocity_blend_alpha`` to bound long-run open-loop IMU drift -- the same
complementary-filter pattern already used by
``gnss_gpu.imu.ComplementaryHeadingFilter`` for heading.
"""

from __future__ import annotations

import math

import numpy as np

from gnss_gpu.imu_preintegration import GRAVITY_ENU, PreintegratedIMU

_WGS84_A = 6_378_137.0
_WGS84_E2 = 6.694379990141316e-3


def ecef_to_lla_rad(position_ecef: np.ndarray) -> tuple[float, float]:
    """ECEF -> (lat_rad, lon_rad), iterative (Bowring-style), matches ins_ekf.py usage."""

    x, y, z = np.asarray(position_ecef, dtype=np.float64).reshape(3)
    lon = math.atan2(y, x)
    p = math.hypot(x, y)
    lat = math.atan2(z, p * (1.0 - _WGS84_E2))
    for _ in range(6):
        sin_lat = math.sin(lat)
        n = _WGS84_A / math.sqrt(1.0 - _WGS84_E2 * sin_lat * sin_lat)
        lat = math.atan2(z + _WGS84_E2 * n * sin_lat, p)
    return lat, lon


def ecef_to_enu_rotation(lat_rad: float, lon_rad: float) -> np.ndarray:
    """ECEF -> ENU rotation matrix (matches ``ins_ekf._ecef_to_enu_rotation``)."""

    sl, cl = math.sin(lat_rad), math.cos(lat_rad)
    so, co = math.sin(lon_rad), math.cos(lon_rad)
    return np.array(
        [
            [-so, co, 0.0],
            [-sl * co, -sl * so, cl],
            [cl * co, cl * so, sl],
        ],
        dtype=np.float64,
    )


def body_to_ecef_frame(heading_rad: float, position_ecef: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(R_body_to_ecef, R_enu_to_ecef)`` for a flat-vehicle (roll=pitch=0) body frame.

    Body x=forward maps to ENU ``(sin h, cos h, 0)`` (h = heading from north,
    clockwise, matching ``imu.ComplementaryHeadingFilter``/``velocity_enu_to_ecef``
    convention); body y=left maps to ENU ``(-cos h, sin h, 0)``; body z=up
    maps to ENU ``(0, 0, 1)``.
    """

    lat, lon = ecef_to_lla_rad(position_ecef)
    r_enu_to_ecef = ecef_to_enu_rotation(lat, lon).T
    h = float(heading_rad)
    sin_h, cos_h = math.sin(h), math.cos(h)
    r_body_to_enu = np.array(
        [
            [sin_h, -cos_h, 0.0],
            [cos_h, sin_h, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return r_enu_to_ecef @ r_body_to_enu, r_enu_to_ecef


class ImuPreintPfGuide:
    """Accumulate IMU samples and close them into a PF predict() velocity/sigma_pos guide.

    Parameters
    ----------
    heading_filter : gnss_gpu.imu.ComplementaryHeadingFilter
        Existing heading tracker (gyro integration + GNSS-bearing
        correction), owned by the caller and shared across the whole run.
        Only ``.heading`` / ``.update_heading_gyro`` / ``.correct_heading_spp``
        are used here; its wheel-speed velocity output is not used (this
        adapter derives velocity from the IMU preintegration instead).
    sigma_accel_mps2_sqrthz, sigma_gyro_radps_sqrthz : float
        Forwarded to the underlying ``PreintegratedIMU``.
    sigma_pos_floor : float
        Minimum ``sigma_pos`` [m] fed to ``pf.predict()`` (guards against a
        near-zero covariance during very short/quiet segments).
    sigma_pos_scale : float
        Multiplicative tuning knob on the covariance-derived ``sigma_pos``.
    velocity_blend_alpha : float
        Complementary-filter blend weight (0-1) toward a GNSS-derived
        velocity estimate each epoch; see module docstring.
    g_enu : ndarray, shape (3,)
        Local ENU gravity vector, rotated into ECEF internally each epoch.
    """

    def __init__(
        self,
        heading_filter,
        sigma_accel_mps2_sqrthz: float = 0.05,
        sigma_gyro_radps_sqrthz: float = 0.005,
        sigma_pos_floor: float = 0.05,
        sigma_pos_scale: float = 1.0,
        velocity_blend_alpha: float = 0.3,
        g_enu: np.ndarray = GRAVITY_ENU,
    ) -> None:
        self.heading_filter = heading_filter
        self.preint = PreintegratedIMU(
            sigma_accel_mps2_sqrthz=sigma_accel_mps2_sqrthz,
            sigma_gyro_radps_sqrthz=sigma_gyro_radps_sqrthz,
        )
        self.sigma_pos_floor = float(sigma_pos_floor)
        self.sigma_pos_scale = float(sigma_pos_scale)
        self.velocity_blend_alpha = float(velocity_blend_alpha)
        self.g_enu = np.asarray(g_enu, dtype=np.float64).reshape(3)
        self.v_ecef: np.ndarray | None = None

    def reset_segment(self) -> None:
        self.preint.reset()

    def add_sample(self, accel_body: np.ndarray, gyro_body_radps: np.ndarray, dt: float) -> None:
        self.preint.add_sample(accel_body, gyro_body_radps, dt)

    @property
    def n_samples(self) -> int:
        return self.preint.n_samples

    def close_segment(
        self,
        p_i_ecef: np.ndarray,
        dt: float,
        v_gnss_ecef: np.ndarray | None = None,
        spp_heading_rad: float | None = None,
    ) -> tuple[np.ndarray | None, float | None]:
        """Close the buffered segment into ``(velocity_guide_ecef, sigma_pos_eff)``.

        Advances the internal nominal-velocity accumulator. Returns
        ``(None, None)`` if the segment has no usable samples or ``dt<=0``
        (caller should fall back to the CV predict for that epoch).
        """

        p_i = np.asarray(p_i_ecef, dtype=np.float64).reshape(3)
        dt_f = float(dt)
        if self.preint.n_samples == 0 or dt_f <= 0.0:
            return None, None

        if spp_heading_rad is not None and np.isfinite(spp_heading_rad):
            self.heading_filter.correct_heading_spp(float(spp_heading_rad))
        heading = float(self.heading_filter.heading)
        r_body_to_ecef, r_enu_to_ecef = body_to_ecef_frame(heading, p_i)
        g_ecef = r_enu_to_ecef @ self.g_enu

        if self.v_ecef is None:
            self.v_ecef = (
                np.asarray(v_gnss_ecef, dtype=np.float64).reshape(3).copy()
                if v_gnss_ecef is not None
                else np.zeros(3, dtype=np.float64)
            )
        elif v_gnss_ecef is not None and self.velocity_blend_alpha > 0.0:
            v_gnss = np.asarray(v_gnss_ecef, dtype=np.float64).reshape(3)
            if np.all(np.isfinite(v_gnss)):
                a = self.velocity_blend_alpha
                self.v_ecef = (1.0 - a) * self.v_ecef + a * v_gnss

        p_j, v_j = self.preint.predict_position_velocity(
            p_i, self.v_ecef, r_body_to_ecef, dt=dt_f, g_enu=g_ecef
        )
        displacement = p_j - p_i
        velocity_guide = displacement / dt_f
        self.v_ecef = v_j

        cov_p_body = self.preint.covariance9[0:3, 0:3]
        cov_p_ecef = r_body_to_ecef @ cov_p_body @ r_body_to_ecef.T
        sigma_pos_eff = math.sqrt(max(float(np.trace(cov_p_ecef)) / 3.0, 0.0))
        sigma_pos_eff = max(sigma_pos_eff * self.sigma_pos_scale, self.sigma_pos_floor)
        return velocity_guide, sigma_pos_eff


def imu_preint_predict(pf, guide: ImuPreintPfGuide, dt: float, **close_segment_kwargs) -> None:
    """Run one ``pf.predict()`` using ``guide``'s buffered segment, then reset it.

    Falls back to the plain CV predict (``velocity=None``) when the segment
    has no usable samples (e.g. an IMU dropout for this epoch).
    """

    p_i = np.asarray(pf.estimate(), dtype=np.float64)[:3]
    velocity_guide, sigma_pos_eff = guide.close_segment(p_i, dt, **close_segment_kwargs)
    if velocity_guide is None:
        pf.predict(velocity=None, dt=dt)
    else:
        pf.predict(
            velocity=velocity_guide,
            dt=dt,
            sigma_pos=sigma_pos_eff,
            sigma_vel=0.0,
            velocity_guide_alpha=1.0,
            rbpf_velocity_kf=False,
            velocity_process_noise=0.0,
        )
    guide.reset_segment()


__all__ = [
    "ImuPreintPfGuide",
    "imu_preint_predict",
    "body_to_ecef_frame",
    "ecef_to_lla_rad",
    "ecef_to_enu_rotation",
]
