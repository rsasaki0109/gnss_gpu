"""GPU-accelerated 3D-aware Mega Particle Filter for GNSS positioning.

Extends the base ParticleFilter with building-aware LOS/NLOS classification
via ray tracing.  NLOS satellites receive a wider observation sigma and a
positive-only pseudorange bias correction, preventing multipath-contaminated
observations from dragging the position estimate.
"""

import numpy as np

from gnss_gpu.particle_filter import ParticleFilter
from gnss_gpu.raytrace import BuildingModel


class ParticleFilter3D(ParticleFilter):
    """Particle filter with 3D building-aware NLOS handling.

    For each particle position and each satellite, a ray is cast through the
    building mesh.  If the ray is blocked (NLOS), the pseudorange likelihood
    is evaluated with a larger sigma and a positive-only bias correction.

    Parameters
    ----------
    building_model : BuildingModel
        3D building mesh for LOS/NLOS classification.
    sigma_los : float
        Observation sigma for LOS satellites [m] (tight, e.g., 3.0).
    sigma_nlos : float
        Observation sigma for NLOS satellites [m] (loose, e.g., 30.0).
    nlos_bias : float
        Expected positive pseudorange bias for NLOS satellites [m] (e.g., 20.0).
    blocked_nlos_prob : float
        Prior probability of NLOS when the ray tracer says blocked.
    clear_nlos_prob : float
        Prior probability of NLOS when the ray tracer says clear.
    **kwargs
        Additional keyword arguments forwarded to ``ParticleFilter.__init__``.
    """

    def __init__(self, building_model, sigma_los=3.0, sigma_nlos=30.0,
                 nlos_bias=20.0, blocked_nlos_prob=1.0,
                 clear_nlos_prob=0.0, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(building_model, BuildingModel):
            raise TypeError("building_model must be a BuildingModel instance")
        self.building_model = building_model
        self.sigma_los = sigma_los
        self.sigma_nlos = sigma_nlos
        self.nlos_bias = nlos_bias
        self.blocked_nlos_prob = blocked_nlos_prob
        self.clear_nlos_prob = clear_nlos_prob

        from gnss_gpu._gnss_gpu_pf3d import pf_weight_3d as _pf_weight_3d
        self._pf_weight_3d = _pf_weight_3d

    def update(self, sat_ecef, pseudoranges, weights=None):
        """Weight update using 3D ray tracing for LOS/NLOS classification.

        Parameters
        ----------
        sat_ecef : array_like, shape (n_sat, 3)
            Satellite ECEF positions [m].
        pseudoranges : array_like, shape (n_sat,)
            Observed pseudoranges [m].
        weights : array_like, shape (n_sat,), optional
            Per-satellite weights. Defaults to ones.
        """
        if not self._initialized:
            raise RuntimeError(
                "ParticleFilter3D not initialized. Call initialize() first.")

        sat = np.asarray(sat_ecef, dtype=np.float64).reshape(-1, 3)
        pr = np.asarray(pseudoranges, dtype=np.float64).ravel()
        n_sat = len(pr)

        if weights is None:
            weights = np.ones(n_sat, dtype=np.float64)
        else:
            weights = np.asarray(weights, dtype=np.float64).ravel()

        tri = self.building_model.triangles.reshape(-1, 3, 3)

        self._pf_weight_3d(
            self._px, self._py, self._pz, self._pcb,
            sat.ravel(), pr, weights, tri,
            self._log_weights,
            self.n_particles, n_sat,
            float(self.sigma_los), float(self.sigma_nlos),
            float(self.nlos_bias),
            float(self.blocked_nlos_prob),
            float(self.clear_nlos_prob))

        # Adaptive resampling based on ESS
        ess = self.get_ess()
        if ess < self.ess_threshold * self.n_particles:
            self._resample()
