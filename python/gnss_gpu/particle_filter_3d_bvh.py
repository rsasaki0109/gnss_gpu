"""GPU-accelerated BVH-accelerated 3D-aware Particle Filter for GNSS positioning.

Extends ParticleFilter3D to use an O(log n) BVH ray traversal instead of the
O(n) linear triangle scan.  This makes the weight update step practical for
large urban meshes (10K+ triangles) without sacrificing accuracy.

Usage
-----
    from gnss_gpu.particle_filter_3d_bvh import ParticleFilter3DBVH
    from gnss_gpu.bvh import BVHAccelerator

    bvh = BVHAccelerator.from_building_model(building)
    pf = ParticleFilter3DBVH(bvh=bvh, n_particles=1_000_000, ...)
    pf.initialize(position_ecef=pos0, clock_bias=cb0)
    for obs in observations:
        pf.predict(dt=1.0)
        pf.update(obs.sat_ecef, obs.pseudoranges)
    pos_est = pf.estimate()
"""

import numpy as np

from gnss_gpu.particle_filter_3d import ParticleFilter3D
from gnss_gpu.bvh import BVHAccelerator


class ParticleFilter3DBVH(ParticleFilter3D):
    """Particle filter with BVH-accelerated 3D building-aware NLOS handling.

    This class has the same interface as :class:`ParticleFilter3D` but
    replaces the per-particle O(n_triangles) linear scan with an
    O(log n_triangles) BVH traversal.  It is drop-in compatible: swap
    ``building_model`` (a ``BuildingModel``) for ``bvh`` (a
    ``BVHAccelerator``) and everything else stays the same.

    Parameters
    ----------
    bvh : BVHAccelerator
        Pre-built BVH acceleration structure containing the building mesh.
    sigma_los : float
        Observation sigma for LOS satellites [m] (tight, e.g., 3.0).
    sigma_nlos : float
        Observation sigma for NLOS satellites [m] (loose, e.g., 30.0).
    nlos_bias : float
        Expected positive pseudorange bias for NLOS satellites [m]. The bias is
        only applied when the residual itself is positive.
    blocked_nlos_prob : float
        Prior probability of NLOS when the ray tracer says blocked.
    clear_nlos_prob : float
        Prior probability of NLOS when the ray tracer says clear.
    **kwargs
        Additional keyword arguments forwarded to ``ParticleFilter.__init__``.
    """

    def __init__(self, bvh, sigma_los=3.0, sigma_nlos=30.0,
                 nlos_bias=20.0, blocked_nlos_prob=1.0,
                 clear_nlos_prob=0.0, **kwargs):
        # Bypass ParticleFilter3D.__init__ which requires a BuildingModel.
        # Instead call the grandparent (ParticleFilter) directly, then
        # set up the BVH-specific attributes.
        from gnss_gpu.particle_filter import ParticleFilter
        ParticleFilter.__init__(self, **kwargs)

        if not isinstance(bvh, BVHAccelerator):
            raise TypeError("bvh must be a BVHAccelerator instance")

        self.bvh = bvh
        self.sigma_los = sigma_los
        self.sigma_nlos = sigma_nlos
        self.nlos_bias = nlos_bias
        self.blocked_nlos_prob = blocked_nlos_prob
        self.clear_nlos_prob = clear_nlos_prob

        from gnss_gpu._gnss_gpu_pf3d_bvh import pf_weight_3d_bvh as _pf_weight_3d_bvh
        self._pf_weight_3d_bvh = _pf_weight_3d_bvh

    # ------------------------------------------------------------------
    # Override ParticleFilter3D.update() to use the BVH kernel
    # ------------------------------------------------------------------

    def update(self, sat_ecef, pseudoranges, weights=None):
        """Weight update using BVH-accelerated 3D ray tracing.

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
                "ParticleFilter3DBVH not initialized. Call initialize() first.")

        sat = np.asarray(sat_ecef, dtype=np.float64).reshape(-1, 3)
        pr  = np.asarray(pseudoranges, dtype=np.float64).ravel()
        n_sat = len(pr)

        if weights is None:
            weights = np.ones(n_sat, dtype=np.float64)
        else:
            weights = np.asarray(weights, dtype=np.float64).ravel()

        self._pf_weight_3d_bvh(
            self._px, self._py, self._pz, self._pcb,
            sat.ravel(), pr, weights,
            self.bvh._nodes_flat,
            self.bvh._sorted_tris,
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
