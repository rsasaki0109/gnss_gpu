"""GPU-accelerated Particle Filter with persistent device memory.

Particle state lives on GPU. No H2D/D2H transfers except:
- Satellite data per update (small: ~1KB for 8 satellites)
- Estimate output (32 bytes)
- Velocity per predict (24 bytes)
- Particle dump for visualization (on-demand)

This eliminates the #1 performance bottleneck: cudaMalloc/cudaFree and
full particle array H2D/D2H transfers on every call.
"""

import numpy as np


class ParticleFilterDevice:
    """High-performance particle filter with persistent GPU memory.

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
        from gnss_gpu._gnss_gpu_pf_device import (
            pf_device_create,
            pf_device_destroy,
            pf_device_initialize,
            pf_device_predict,
            pf_device_weight,
            pf_device_ess,
            pf_device_resample_systematic,
            pf_device_resample_megopolis,
            pf_device_estimate,
            pf_device_get_particles,
            pf_device_sync,
        )
        self._pf_device_create = pf_device_create
        self._pf_device_destroy = pf_device_destroy
        self._pf_device_initialize = pf_device_initialize
        self._pf_device_predict = pf_device_predict
        self._pf_device_weight = pf_device_weight
        self._pf_device_ess = pf_device_ess
        self._pf_device_resample_systematic = pf_device_resample_systematic
        self._pf_device_resample_megopolis = pf_device_resample_megopolis
        self._pf_device_estimate = pf_device_estimate
        self._pf_device_get_particles = pf_device_get_particles
        self._pf_device_sync = pf_device_sync

        self.n_particles = n_particles
        self.sigma_pos = sigma_pos
        self.sigma_cb = sigma_cb
        self.sigma_pr = sigma_pr
        self.resampling = resampling
        self.ess_threshold = ess_threshold
        self.seed = seed

        # Allocate GPU memory once
        self._state = self._pf_device_create(n_particles)
        self._initialized = False
        self._step = 0

    def __del__(self):
        # GPU resources are freed automatically by the pybind11 unique_ptr
        # custom deleter when the Python object is garbage collected.
        # Explicitly calling pf_device_destroy here would cause a double-free
        # since pybind11 already manages the lifetime.
        pass

    def initialize(self, position_ecef, clock_bias=0.0, spread_pos=100.0,
                   spread_cb=1000.0):
        """Scatter particles around an initial estimate.

        Runs entirely on GPU - no host-device transfer of particle data.

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

        self._pf_device_initialize(
            self._state,
            float(pos[0]), float(pos[1]), float(pos[2]), float(clock_bias),
            float(spread_pos), float(spread_cb),
            self.seed)

        self._initialized = True
        self._step = 0

    def predict(self, velocity=None, dt=1.0):
        """Predict step with optional velocity.

        Only 24 bytes (velocity) transferred to GPU.

        Parameters
        ----------
        velocity : array_like, shape (3,), optional
            Velocity in ECEF [m/s]. Defaults to zero (stationary).
        dt : float
            Time step [s].
        """
        if not self._initialized:
            raise RuntimeError("ParticleFilterDevice not initialized. Call initialize() first.")

        if velocity is None:
            vx, vy, vz = 0.0, 0.0, 0.0
        else:
            vel = np.asarray(velocity, dtype=np.float64).ravel()
            vx, vy, vz = float(vel[0]), float(vel[1]), float(vel[2])

        self._step += 1
        self._pf_device_predict(
            self._state,
            vx, vy, vz,
            float(dt), float(self.sigma_pos), float(self.sigma_cb),
            self.seed, self._step)

    def update(self, sat_ecef, pseudoranges, weights=None):
        """Weight update with pseudorange observations.

        Only satellite data (~1KB) transferred to GPU.
        Particle arrays stay on device.

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
            raise RuntimeError("ParticleFilterDevice not initialized. Call initialize() first.")

        sat = np.asarray(sat_ecef, dtype=np.float64).reshape(-1, 3)
        pr = np.asarray(pseudoranges, dtype=np.float64).ravel()
        n_sat = len(pr)

        if weights is None:
            weights = np.ones(n_sat, dtype=np.float64)
        else:
            weights = np.asarray(weights, dtype=np.float64).ravel()

        self._pf_device_weight(
            self._state,
            sat.ravel(), pr, weights,
            n_sat, float(self.sigma_pr))

        # Adaptive resampling based on ESS
        ess = self.get_ess()
        if ess < self.ess_threshold * self.n_particles:
            self._resample()

    def _resample(self):
        """Perform resampling entirely on GPU."""
        if self.resampling == "megopolis":
            self._pf_device_resample_megopolis(self._state, 15, self.seed)
        else:
            self._pf_device_resample_systematic(self._state, self.seed)

    def estimate(self):
        """Compute weighted mean position.

        Only 32 bytes (result) transferred from GPU.

        Returns
        -------
        result : ndarray, shape (4,)
            Estimated [x, y, z, clock_bias] in ECEF [m].
        """
        if not self._initialized:
            raise RuntimeError("ParticleFilterDevice not initialized. Call initialize() first.")

        return self._pf_device_estimate(self._state)

    def get_particles(self):
        """Get all particle states from GPU.

        WARNING: This is the only operation that transfers all particle data
        from GPU to host. Use sparingly (e.g., for visualization).

        Returns
        -------
        particles : ndarray, shape (n_particles, 4)
            Each row is [x, y, z, clock_bias].
        """
        if not self._initialized:
            raise RuntimeError("ParticleFilterDevice not initialized. Call initialize() first.")

        return self._pf_device_get_particles(self._state)

    def get_ess(self):
        """Compute current Effective Sample Size.

        Computed on GPU, only scalar result transferred.

        Returns
        -------
        ess : float
            Effective Sample Size in [1, n_particles].
        """
        if not self._initialized:
            raise RuntimeError("ParticleFilterDevice not initialized. Call initialize() first.")

        return self._pf_device_ess(self._state)

    def sync(self):
        """Explicitly synchronize the CUDA stream.

        Waits for all pending GPU operations (async transfers and kernel
        launches) to complete. This is called automatically before any
        D2H transfer (estimate, get_particles, get_ess), but can be
        useful when benchmarking or coordinating with other GPU work.
        """
        if hasattr(self, '_state') and self._state is not None:
            self._pf_device_sync(self._state)
