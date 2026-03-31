"""GPU-accelerated Mega Particle Filter for GNSS positioning.

Inspired by MegaParticles (Koide et al., ICRA 2024) but applied to GNSS
pseudorange-based positioning with 1M+ particles on GPU.
"""

import numpy as np


class ParticleFilter:
    """Mega Particle Filter for GNSS positioning in ECEF coordinates.

    Parameters
    ----------
    n_particles : int
        Number of particles (e.g., 1_000_000).
    sigma_pos : float
        Position noise standard deviation [m] for prediction.
    sigma_cb : float
        Clock bias noise standard deviation [m] for prediction.
    sigma_pr : float
        Pseudorange observation standard deviation [m] for weighting.
    resampling : str
        Resampling method: "megopolis" or "systematic".
    ess_threshold : float
        ESS ratio threshold for triggering resampling (0-1).
    seed : int
        Random seed for reproducibility.
    """

    def __init__(self, n_particles=1_000_000, sigma_pos=1.0, sigma_cb=300.0,
                 sigma_pr=5.0, resampling="megopolis", ess_threshold=0.5,
                 seed=42):
        from gnss_gpu._gnss_gpu_pf import (
            pf_initialize as _pf_initialize,
            pf_predict as _pf_predict,
            pf_weight as _pf_weight,
            pf_compute_ess as _pf_compute_ess,
            pf_resample_systematic as _pf_resample_systematic,
            pf_resample_megopolis as _pf_resample_megopolis,
            pf_estimate as _pf_estimate,
            pf_get_particles as _pf_get_particles,
        )
        self._pf_initialize = _pf_initialize
        self._pf_predict = _pf_predict
        self._pf_weight = _pf_weight
        self._pf_compute_ess = _pf_compute_ess
        self._pf_resample_systematic = _pf_resample_systematic
        self._pf_resample_megopolis = _pf_resample_megopolis
        self._pf_estimate = _pf_estimate
        self._pf_get_particles = _pf_get_particles

        self.n_particles = n_particles
        self.sigma_pos = sigma_pos
        self.sigma_cb = sigma_cb
        self.sigma_pr = sigma_pr
        self.resampling = resampling
        self.ess_threshold = ess_threshold
        self.seed = seed

        self._px = None
        self._py = None
        self._pz = None
        self._pcb = None
        self._log_weights = None
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
        self._log_weights = np.zeros(n, dtype=np.float64)

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
            raise RuntimeError("ParticleFilter not initialized. Call initialize() first.")

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
        """Weight update with pseudorange observations.

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
            raise RuntimeError("ParticleFilter not initialized. Call initialize() first.")

        sat = np.asarray(sat_ecef, dtype=np.float64).reshape(-1, 3)
        pr = np.asarray(pseudoranges, dtype=np.float64).ravel()
        n_sat = len(pr)

        if weights is None:
            weights = np.ones(n_sat, dtype=np.float64)
        else:
            weights = np.asarray(weights, dtype=np.float64).ravel()

        self._pf_weight(
            self._px, self._py, self._pz, self._pcb,
            sat.ravel(), pr, weights, self._log_weights,
            self.n_particles, n_sat, float(self.sigma_pr))

        # Adaptive resampling based on ESS
        ess = self.get_ess()
        if ess < self.ess_threshold * self.n_particles:
            self._resample()

    def _resample(self):
        """Perform resampling using the configured method."""
        if self.resampling == "megopolis":
            self._pf_resample_megopolis(
                self._px, self._py, self._pz, self._pcb,
                self._log_weights, self.n_particles, 15, self.seed)
        else:
            self._pf_resample_systematic(
                self._px, self._py, self._pz, self._pcb,
                self._log_weights, self.n_particles, self.seed)

        # After resampling, reset to uniform weights
        self._log_weights[:] = 0.0

    def estimate(self):
        """Compute weighted mean position.

        Returns
        -------
        result : ndarray, shape (4,)
            Estimated [x, y, z, clock_bias] in ECEF [m].
        """
        if not self._initialized:
            raise RuntimeError("ParticleFilter not initialized. Call initialize() first.")

        result = np.empty(4, dtype=np.float64)
        self._pf_estimate(
            self._px, self._py, self._pz, self._pcb,
            self._log_weights, result, self.n_particles)
        return result

    def get_particles(self):
        """Get all particle states.

        Returns
        -------
        particles : ndarray, shape (n_particles, 4)
            Each row is [x, y, z, clock_bias].
        """
        if not self._initialized:
            raise RuntimeError("ParticleFilter not initialized. Call initialize() first.")

        output = np.empty(self.n_particles * 4, dtype=np.float64)
        self._pf_get_particles(
            self._px, self._py, self._pz, self._pcb,
            output, self.n_particles)
        return output.reshape(self.n_particles, 4)

    def get_ess(self):
        """Compute current Effective Sample Size.

        Returns
        -------
        ess : float
            Effective Sample Size in [1, n_particles].
        """
        if not self._initialized:
            raise RuntimeError("ParticleFilter not initialized. Call initialize() first.")

        return self._pf_compute_ess(self._log_weights, self.n_particles)
