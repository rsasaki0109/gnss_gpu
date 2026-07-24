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

**WP21b additions** (see ``internal_docs/task_wp21b_preint_payoff.md``):

1. Heading-uncertainty propagation into ``sigma_pos`` (opt-in via
   ``ImuPreintPfGuide(use_heading_uncertainty=True)``): folds
   ``ComplementaryHeadingFilter.heading_variance_rad2`` into ``sigma_pos``
   through the cross-track lever ``|displacement| * sigma_heading_rad``.
2. Per-particle velocity-KF feeding (:func:`imu_preint_predict_velocity_kf`):
   pushes the preintegration's delta_v covariance into every particle's
   ``Sigma_v`` via ``ParticleFilterDeviceRuntime.set_velocity_covariance``
   and runs ``predict(..., rbpf_velocity_kf=True)`` so uncertainty is
   propagated through the device's existing covariance-aware predict
   (``x_new ~ N(x + mu_v*dt, sigma_pos^2*I + dt^2*Sigma_v)``) instead of a
   scalar one-shot guide.
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
        Minimum ``sigma_pos`` [m] fed to ``pf.predict()`` (small numerical
        floor for stability only -- WP21b item 1 replaces the WP21 Phase A
        hand-tuned floor with a derived heading-uncertainty term; this
        residual floor exists only to guard a near-zero total during a
        perfectly quiet, perfectly-known-heading segment).
    sigma_pos_scale : float
        Multiplicative tuning knob on the covariance-derived ``sigma_pos``.
    velocity_blend_alpha : float
        Complementary-filter blend weight (0-1) toward a GNSS-derived
        velocity estimate each epoch; see module docstring.
    g_enu : ndarray, shape (3,)
        Local ENU gravity vector, rotated into ECEF internally each epoch.
    use_heading_uncertainty : bool
        WP21b item 1. When True, ``close_segment`` folds the heading
        filter's ``heading_variance_rad2`` into ``sigma_pos`` via the
        cross-track lever ``sigma_pos_heading = |displacement| *
        sigma_heading_rad`` (combined in quadrature with the existing
        accel/gyro-covariance-derived term), and passes a heading-
        measurement uncertainty to ``heading_filter.correct_heading_spp``
        so the heading variance itself updates correctly. Defaults to
        False so the WP21 Phase A ("preint-v1") code path and its reported
        numbers stay exactly reproducible byte-for-byte.
    sigma_spp_pos_m : float
        Only used when ``use_heading_uncertainty``. Documented (not
        per-epoch-fitted) approximation of the causal ``robust_spp`` point
        fix's horizontal position sigma [m] -- this repo's own
        characterization of raw, uncorrected single-frequency GPS-only SPP
        on this dataset is "tens of meters" (see
        ``results/wp21/WP21_REPORT.md`` Sec. 7); default 30.0 m sits inside
        that documented range. Converted into a per-epoch SPP-heading
        uncertainty via error propagation through
        ``heading = atan2(v_east, v_north)`` for
        ``v = (fix_i - fix_{i-1})/dt``: two independent fixes each with
        sigma ``sigma_spp_pos_m`` give a relative-displacement sigma of
        ``sqrt(2)*sigma_spp_pos_m``, and for a unit-vector angle,
        ``sigma_heading ~= sigma_perp_displacement / |displacement|``.
    min_heading_fix_disp_m : float
        Below this consecutive-fix displacement, the SPP-derived heading is
        treated as fully uninformative (``sigma_spp_heading = pi``) rather
        than dividing by a near-zero displacement.
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
        use_heading_uncertainty: bool = False,
        sigma_spp_pos_m: float = 30.0,
        min_heading_fix_disp_m: float = 2.0,
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
        self.use_heading_uncertainty = bool(use_heading_uncertainty)
        self.sigma_spp_pos_m = float(sigma_spp_pos_m)
        self.min_heading_fix_disp_m = float(min_heading_fix_disp_m)
        # WP21b item 2: velocity covariance (ECEF, [m^2/s^2]) from the last
        # closed segment's accel/gyro-derived delta_v covariance block,
        # exposed for feeding the per-particle velocity KF
        # (`ParticleFilterDeviceRuntime.set_velocity_covariance`).
        self._last_velocity_cov_ecef: np.ndarray | None = None

    def reset_segment(self) -> None:
        self.preint.reset()

    def add_sample(self, accel_body: np.ndarray, gyro_body_radps: np.ndarray, dt: float) -> None:
        self.preint.add_sample(accel_body, gyro_body_radps, dt)

    @property
    def n_samples(self) -> int:
        return self.preint.n_samples

    @property
    def velocity_covariance_ecef(self) -> np.ndarray | None:
        """Velocity covariance [m^2/s^2] (ECEF) from the last closed segment, or None."""
        if self._last_velocity_cov_ecef is None:
            return None
        return self._last_velocity_cov_ecef.copy()

    def _sigma_spp_heading_rad(self, spp_displacement_m: float | None) -> float:
        if spp_displacement_m is None or not np.isfinite(spp_displacement_m):
            return math.pi
        if spp_displacement_m <= self.min_heading_fix_disp_m:
            return math.pi
        sigma = math.sqrt(2.0) * self.sigma_spp_pos_m / spp_displacement_m
        return min(sigma, math.pi)

    def close_segment(
        self,
        p_i_ecef: np.ndarray,
        dt: float,
        v_gnss_ecef: np.ndarray | None = None,
        spp_heading_rad: float | None = None,
        spp_displacement_m: float | None = None,
    ) -> tuple[np.ndarray | None, float | None]:
        """Close the buffered segment into ``(velocity_guide_ecef, sigma_pos_eff)``.

        Advances the internal nominal-velocity accumulator. Returns
        ``(None, None)`` if the segment has no usable samples or ``dt<=0``
        (caller should fall back to the CV predict for that epoch).

        ``spp_displacement_m`` (WP21b item 1, only consulted when
        ``use_heading_uncertainty``): ``|fix_i - fix_{i-1}|`` for the
        consecutive causal SPP fixes that produced ``spp_heading_rad``,
        used to convert ``sigma_spp_pos_m`` into a per-epoch heading
        measurement uncertainty. When omitted (or ``use_heading_uncertainty``
        is False), the correction call is byte-identical to WP21 Phase A.
        """

        p_i = np.asarray(p_i_ecef, dtype=np.float64).reshape(3)
        dt_f = float(dt)
        if self.preint.n_samples == 0 or dt_f <= 0.0:
            return None, None

        if spp_heading_rad is not None and np.isfinite(spp_heading_rad):
            if self.use_heading_uncertainty:
                sigma_spp_heading = self._sigma_spp_heading_rad(spp_displacement_m)
                self.heading_filter.correct_heading_spp(
                    float(spp_heading_rad), sigma_spp_heading_rad=sigma_spp_heading
                )
            else:
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

        cov9 = self.preint.covariance9
        cov_p_body = cov9[0:3, 0:3]
        cov_p_ecef = r_body_to_ecef @ cov_p_body @ r_body_to_ecef.T
        sigma_pos_cov = math.sqrt(max(float(np.trace(cov_p_ecef)) / 3.0, 0.0))

        # WP21b item 2: expose the delta_v covariance block (rotated to
        # ECEF) for the caller's per-particle velocity KF.
        cov_v_body = cov9[3:6, 3:6]
        self._last_velocity_cov_ecef = r_body_to_ecef @ cov_v_body @ r_body_to_ecef.T

        sigma_pos_heading = 0.0
        if self.use_heading_uncertainty:
            heading_sigma_rad = math.sqrt(
                max(getattr(self.heading_filter, "heading_variance_rad2", 0.0), 0.0)
            )
            sigma_pos_heading = float(np.linalg.norm(displacement)) * heading_sigma_rad

        sigma_pos_combined = math.hypot(sigma_pos_cov, sigma_pos_heading)
        sigma_pos_eff = max(sigma_pos_combined * self.sigma_pos_scale, self.sigma_pos_floor)
        return velocity_guide, sigma_pos_eff


def imu_preint_predict(pf, guide: ImuPreintPfGuide, dt: float, **close_segment_kwargs) -> None:
    """Run one ``pf.predict()`` using ``guide``'s buffered segment, then reset it.

    Falls back to the plain CV predict (``velocity=None``) when the segment
    has no usable samples (e.g. an IMU dropout for this epoch). This is the
    WP21 Phase A ("preint-v1") predict path: a scalar isotropic ``sigma_pos``
    guide, no per-particle velocity KF. Kept unchanged for continuity; see
    :func:`imu_preint_predict_velocity_kf` for the WP21b ("preint-v2") path.
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


def imu_preint_predict_velocity_kf(
    pf,
    guide: ImuPreintPfGuide,
    dt: float,
    velocity_process_noise: float = 0.0,
    **close_segment_kwargs,
) -> None:
    """WP21b item 2: run one ``pf.predict()`` using the per-particle velocity KF path.

    Unlike :func:`imu_preint_predict` (which feeds the closing velocity as a
    one-shot guide with ``rbpf_velocity_kf=False``), this feeds the
    preintegrated segment's ECEF velocity covariance
    (``guide.velocity_covariance_ecef``) into every particle's ``Sigma_v``
    via ``pf.set_velocity_covariance`` (see
    ``ParticleFilterDeviceRuntime.set_velocity_covariance``) *before*
    calling ``pf.predict(..., rbpf_velocity_kf=True)``, so the device
    predict's ``x_new ~ N(x + mu_v*dt, sigma_pos^2*I + dt^2*Sigma_v)``
    propagation uses this epoch's modeled (accel/gyro-derived) velocity
    uncertainty instead of only the generic isotropic
    ``velocity_process_noise`` growth term. ``mu_v`` is reset to the
    preintegrated closing velocity via ``velocity_guide_alpha=1.0``, same as
    :func:`imu_preint_predict`.

    Requires ``pf`` to implement ``set_velocity_covariance`` (i.e. a real
    ``ParticleFilterDeviceRuntime``/``ParticleFilterDevice``, not the bare
    stub used by some adapter-only unit tests).
    """

    p_i = np.asarray(pf.estimate(), dtype=np.float64)[:3]
    velocity_guide, sigma_pos_eff = guide.close_segment(p_i, dt, **close_segment_kwargs)
    if velocity_guide is None:
        pf.predict(velocity=None, dt=dt)
    else:
        vel_cov = guide.velocity_covariance_ecef
        if vel_cov is not None:
            pf.set_velocity_covariance(vel_cov)
        pf.predict(
            velocity=velocity_guide,
            dt=dt,
            sigma_pos=sigma_pos_eff,
            velocity_guide_alpha=1.0,
            rbpf_velocity_kf=True,
            velocity_process_noise=velocity_process_noise,
        )
    guide.reset_segment()


__all__ = [
    "ImuPreintPfGuide",
    "imu_preint_predict",
    "imu_preint_predict_velocity_kf",
    "body_to_ecef_frame",
    "ecef_to_lla_rad",
    "ecef_to_enu_rotation",
]
