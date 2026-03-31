"""GPU-accelerated SVGD Particle Filter for GNSS positioning.

Replaces traditional resampling with Stein Variational Gradient Descent (SVGD)
to avoid sample impoverishment. Based on the MegaParticles approach
(Koide et al., ICRA 2024).

SVGD moves particles along the steepest descent direction of KL divergence
using a reproducing kernel, providing both attraction to high-probability
regions and repulsion to maintain particle diversity.
"""

import numpy as np


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

        self.n_particles = n_particles
        self.sigma_pos = sigma_pos
        self.sigma_cb = sigma_cb
        self.sigma_pr = sigma_pr
        self.svgd_steps = svgd_steps
        self.step_size = step_size
        self.n_neighbors = n_neighbors
        self.n_bandwidth_subsample = n_bandwidth_subsample
        self.seed = seed

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
        pos = np.asarray(position_ecef, dtype=np.float64).ravel()
        n = self.n_particles

        self._px = np.empty(n, dtype=np.float64)
        self._py = np.empty(n, dtype=np.float64)
        self._pz = np.empty(n, dtype=np.float64)
        self._pcb = np.empty(n, dtype=np.float64)

        self._pf_initialize(
            self._px, self._py, self._pz, self._pcb,
            float(pos[0]), float(pos[1]), float(pos[2]), float(clock_bias),
            float(spread_pos), float(spread_cb),
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

        if velocity is None:
            velocity = [0.0, 0.0, 0.0]
        vel = np.asarray(velocity, dtype=np.float64).ravel()
        vx = np.array([vel[0]], dtype=np.float64)
        vy = np.array([vel[1]], dtype=np.float64)
        vz = np.array([vel[2]], dtype=np.float64)

        self._step += 1
        self._pf_predict(
            self._px, self._py, self._pz, self._pcb,
            vx, vy, vz,
            float(dt), float(self.sigma_pos), float(self.sigma_cb),
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

        sat = np.asarray(sat_ecef, dtype=np.float64).reshape(-1, 3)
        pr = np.asarray(pseudoranges, dtype=np.float64).ravel()
        n_sat = len(pr)

        if weights is None:
            weights = np.ones(n_sat, dtype=np.float64)
        else:
            weights = np.asarray(weights, dtype=np.float64).ravel()

        for i in range(self.svgd_steps):
            # Estimate bandwidth using median heuristic on random subsample
            bandwidth = self._pf_estimate_bandwidth(
                self._px, self._py, self._pz, self._pcb,
                self.n_particles, self.n_bandwidth_subsample,
                self.seed + self._step * 1000 + i)

            # Perform one SVGD step
            self._pf_svgd_step(
                self._px, self._py, self._pz, self._pcb,
                sat.ravel(), pr, weights,
                self.n_particles, n_sat,
                float(self.sigma_pr), float(self.step_size),
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
