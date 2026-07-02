"""GPU-accelerated SVGD Particle Filter for GNSS positioning.

Replaces traditional resampling with Stein Variational Gradient Descent (SVGD)
to avoid sample impoverishment. Based on the MegaParticles approach
(Koide et al., ICRA 2024).

SVGD moves particles along the steepest descent direction of KL divergence
using a reproducing kernel, providing both attraction to high-probability
regions and repulsion to maintain particle diversity.
"""

import numpy as np


def _finite_float(name, value):
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not np.isfinite(out):
        raise ValueError(f"{name} must be finite")
    return out


def _positive_float(name, value):
    out = _finite_float(name, value)
    if out <= 0.0:
        raise ValueError(f"{name} must be positive")
    return out


def _nonnegative_float(name, value):
    out = _finite_float(name, value)
    if out < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return out


def _positive_int(name, value):
    try:
        out = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if out <= 0:
        raise ValueError(f"{name} must be positive")
    return out


def _as_position_ecef(position_ecef):
    arr = np.asarray(position_ecef, dtype=np.float64)
    if arr.shape != (3,):
        raise ValueError("position_ecef must have shape (3,)")
    if not np.all(np.isfinite(arr)):
        raise ValueError("position_ecef must be finite")
    return arr


def _as_sat_ecef_matrix(sat_ecef, n_sat):
    sat = np.asarray(sat_ecef, dtype=np.float64)
    if sat.shape != (n_sat, 3):
        raise ValueError("sat_ecef must have shape (n_sat, 3)")
    if not np.all(np.isfinite(sat)):
        raise ValueError("sat_ecef must be finite")
    return sat


def _finite_1d_array(name, values, *, min_size=1):
    arr = np.asarray(values, dtype=np.float64).ravel()
    if arr.size < min_size:
        raise ValueError(f"{name} must contain at least {min_size} value")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr


def _nonnegative_1d_array(name, values, *, min_size=1):
    arr = _finite_1d_array(name, values, min_size=min_size)
    if np.any(arr < 0.0):
        raise ValueError(f"{name} must be non-negative")
    return arr


class SVGDParticleFilter:
    """Particle filter with SVGD instead of resampling.

    Instead of the weight-resample cycle that causes sample impoverishment,
    SVGD iteratively transports particles toward the posterior distribution
    while maintaining diversity through a kernel-based repulsive force.

    Parameters
    ----------
    n_particles : int
        Number of particles (e.g., 1_000_000).
    sigma_pos : float
        Position noise standard deviation [m] for prediction.
    sigma_cb : float
        Clock bias noise standard deviation [m] for prediction.
    sigma_pr : float
        Pseudorange observation standard deviation [m].
    svgd_steps : int
        Number of SVGD iterations per update.
    step_size : float
        SVGD step size (learning rate).
    n_neighbors : int
        Number of random neighbors K for kernel computation (O(N*K) complexity).
    n_bandwidth_subsample : int
        Number of random pairs for median heuristic bandwidth estimation.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(self, n_particles=1_000_000, sigma_pos=1.0, sigma_cb=300.0,
                 sigma_pr=5.0, svgd_steps=5, step_size=0.1,
                 n_neighbors=32, n_bandwidth_subsample=1000, seed=42):
        from gnss_gpu._gnss_gpu_pf import (
            pf_initialize as _pf_initialize,
            pf_predict as _pf_predict,
            pf_get_particles as _pf_get_particles,
        )
        from gnss_gpu._gnss_gpu_svgd import (
            pf_estimate_bandwidth as _pf_estimate_bandwidth,
            pf_svgd_step as _pf_svgd_step,
            pf_svgd_estimate as _pf_svgd_estimate,
        )
        self._pf_initialize = _pf_initialize
        self._pf_predict = _pf_predict
        self._pf_get_particles = _pf_get_particles
        self._pf_estimate_bandwidth = _pf_estimate_bandwidth
        self._pf_svgd_step = _pf_svgd_step
        self._pf_svgd_estimate = _pf_svgd_estimate

        self.n_particles = _positive_int("n_particles", n_particles)
        self.sigma_pos = _positive_float("sigma_pos", sigma_pos)
        self.sigma_cb = _positive_float("sigma_cb", sigma_cb)
        self.sigma_pr = _positive_float("sigma_pr", sigma_pr)
        self.svgd_steps = _positive_int("svgd_steps", svgd_steps)
        self.step_size = _positive_float("step_size", step_size)
        self.n_neighbors = _positive_int("n_neighbors", n_neighbors)
        self.n_bandwidth_subsample = _positive_int(
            "n_bandwidth_subsample", n_bandwidth_subsample)
        self.seed = int(seed)

        self._px = None
        self._py = None
        self._pz = None
        self._pcb = None
        self._initialized = False
        self._step = 0

    def initialize(self, position_ecef, clock_bias=0.0, spread_pos=100.0,
                   spread_cb=1000.0):
        """Scatter particles around an initial estimate.

        Parameters
        ----------
        position_ecef : array_like, shape (3,)
            Initial ECEF position [m].
        clock_bias : float
            Initial receiver clock bias [m].
        spread_pos : float
            Standard deviation for initial position scatter [m].
        spread_cb : float
            Standard deviation for initial clock bias scatter [m].
        """
        pos = _as_position_ecef(position_ecef)
        clock_bias = _finite_float("clock_bias", clock_bias)
        spread_pos = _positive_float("spread_pos", spread_pos)
        spread_cb = _positive_float("spread_cb", spread_cb)
        n = self.n_particles

        self._px = np.empty(n, dtype=np.float64)
        self._py = np.empty(n, dtype=np.float64)
        self._pz = np.empty(n, dtype=np.float64)
        self._pcb = np.empty(n, dtype=np.float64)

        self._pf_initialize(
            self._px, self._py, self._pz, self._pcb,
            float(pos[0]), float(pos[1]), float(pos[2]), clock_bias,
            spread_pos, spread_cb,
            n, self.seed)

        self._initialized = True
        self._step = 0

    def predict(self, velocity=None, dt=1.0):
        """Predict step with optional velocity.

        Parameters
        ----------
        velocity : array_like, shape (3,), optional
            Velocity in ECEF [m/s]. Defaults to zero (stationary).
        dt : float
            Time step [s].
        """
        if not self._initialized:
            raise RuntimeError(
                "SVGDParticleFilter not initialized. Call initialize() first.")

        dt = _nonnegative_float("dt", dt)

        if velocity is None:
            velocity = [0.0, 0.0, 0.0]
        vel = np.asarray(velocity, dtype=np.float64).ravel()
        if vel.shape != (3,):
            raise ValueError("velocity must have shape (3,)")
        if not np.all(np.isfinite(vel)):
            raise ValueError("velocity must be finite")
        vx = np.array([vel[0]], dtype=np.float64)
        vy = np.array([vel[1]], dtype=np.float64)
        vz = np.array([vel[2]], dtype=np.float64)

        self._step += 1
        self._pf_predict(
            self._px, self._py, self._pz, self._pcb,
            vx, vy, vz,
            dt, self.sigma_pos, self.sigma_cb,
            self.n_particles, self.seed, self._step)

    def update(self, sat_ecef, pseudoranges, weights=None):
        """Update particles using SVGD instead of weight + resample.

        Performs multiple SVGD steps to transport particles toward the
        posterior distribution defined by the pseudorange likelihood.

        Parameters
        ----------
        sat_ecef : array_like, shape (n_sat, 3)
            Satellite ECEF positions [m].
        pseudoranges : array_like, shape (n_sat,)
            Observed pseudoranges [m].
        weights : array_like, shape (n_sat,), optional
            Per-satellite weights (1/sigma^2). Defaults to ones.
        """
        if not self._initialized:
            raise RuntimeError(
                "SVGDParticleFilter not initialized. Call initialize() first.")

        pr = _finite_1d_array("pseudoranges", pseudoranges, min_size=1)
        n_sat = pr.size
        sat = _as_sat_ecef_matrix(sat_ecef, n_sat)

        if weights is None:
            w = np.ones(n_sat, dtype=np.float64)
        else:
            w = np.asarray(weights, dtype=np.float64).ravel()
            if w.size != n_sat:
                raise ValueError("weights length must match pseudoranges")
            w = _nonnegative_1d_array("weights", w, min_size=n_sat)

        for i in range(self.svgd_steps):
            # Estimate bandwidth using median heuristic on random subsample
            bandwidth = self._pf_estimate_bandwidth(
                self._px, self._py, self._pz, self._pcb,
                self.n_particles, self.n_bandwidth_subsample,
                self.seed + self._step * 1000 + i)

            # Perform one SVGD step
            self._pf_svgd_step(
                self._px, self._py, self._pz, self._pcb,
                sat.ravel(), pr, w,
                self.n_particles, n_sat,
                self.sigma_pr, self.step_size,
                self.n_neighbors, bandwidth,
                self.seed, self._step * 100 + i)

    def estimate(self):
        """Compute mean position estimate (equal weights after SVGD).

        Returns
        -------
        result : ndarray, shape (4,)
            Estimated [x, y, z, clock_bias] in ECEF [m].
        """
        if not self._initialized:
            raise RuntimeError(
                "SVGDParticleFilter not initialized. Call initialize() first.")

        result = np.empty(4, dtype=np.float64)
        self._pf_svgd_estimate(
            self._px, self._py, self._pz, self._pcb,
            result, self.n_particles)
        return result

    def get_particles(self):
        """Get all particle states.

        Returns
        -------
        particles : ndarray, shape (n_particles, 4)
            Each row is [x, y, z, clock_bias].
        """
        if not self._initialized:
            raise RuntimeError(
                "SVGDParticleFilter not initialized. Call initialize() first.")

        output = np.empty(self.n_particles * 4, dtype=np.float64)
        self._pf_get_particles(
            self._px, self._py, self._pz, self._pcb,
            output, self.n_particles)
        return output.reshape(self.n_particles, 4)
